//! Anthropic Messages request body -> [`ChatRequest`].
//!
//! Verified against specs/anthropic-openapi.json. Structural parsing only;
//! whether an upstream can honour a parameter is that provider's judgement.

use super::base::block_text;
use crate::error::{Error, Result};
use crate::ir::{
    Block, Cache, ChatRequest, ChoiceKind, Image, OutputFormat, Role, Source, Text, Thinking, Tool,
    ToolChoice, ToolResult, ToolUse, Turn, WebSearchTool,
};
use crate::json::{integer, py_str, truthy, Object};
use crate::tools::{definitions, optional_bool, parse_function};
use serde_json::Value;

/// The only server tool the proxy can serve; the rest are refused.
const WEB_SEARCH_PREFIX: &str = "web_search_";
/// No proxy implementation; refused rather than silently ignored. Fields the
/// proxy cannot act on but real clients send -- context_management, metadata --
/// are deliberately absent: they are accepted and dropped.
const REJECTED: [&str; 3] = ["container", "mcp_servers", "service_tier"];
const PARAMS: [&str; 3] = ["temperature", "top_p", "top_k"];
/// Block kinds whose cache breakpoint the IR carries as a hint.
const CACHED_KINDS: [&str; 4] = ["text", "image", "tool_use", "tool_result"];
/// Content only Messages defines: verbatim to an upstream that speaks it.
const NATIVE_KINDS: [&str; 2] = ["document", "search_result"];

/// Read `output_config.format`, or the `output_format` it replaced.
///
/// Messages names no schema and has no unenforced mode, so the neutral record
/// keeps the schema's own title as a label and is always strict.
fn output_format(current: &Value, deprecated: &Value) -> Result<Option<OutputFormat>> {
    let item = if current.is_null() {
        deprecated
    } else {
        current
    };
    let item = match item {
        Value::Null => return Ok(None),
        Value::Object(item) => item,
        _ => return Err(Error::request("output format must be an object")),
    };
    let kind = item.get("type").unwrap_or(&Value::Null);
    if kind.as_str() != Some("json_schema") {
        return Err(Error::request(format!(
            "unsupported output format: {}",
            py_str(kind)
        )));
    }
    let Some(Value::Object(schema)) = item.get("schema") else {
        return Err(Error::request("output format json_schema requires schema"));
    };
    let name = match schema.get("title") {
        Some(Value::String(name)) => name.clone(),
        _ => String::new(),
    };
    Ok(Some(OutputFormat {
        kind: "json_schema".into(),
        name,
        schema: Some(schema.clone()),
        strict: true,
    }))
}

/// Text of a content field that may be a string or a block list.
fn text(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => block_text(parts),
        _ => Ok(String::new()),
    }
}

/// A `cache_control` breakpoint as its TTL ("" for the default).
fn cache(value: &Value) -> Result<Cache> {
    let map = match value {
        Value::Null => return Ok(None),
        Value::Object(map) if map.get("type") == Some(&Value::String("ephemeral".into())) => map,
        _ => {
            return Err(Error::request(
                "cache_control must be an ephemeral cache control",
            ))
        }
    };
    let ttl = map.get("ttl").filter(|ttl| truthy(ttl));
    let ttl = match ttl {
        None => "",
        Some(Value::String(ttl)) if ttl == "5m" || ttl == "1h" => ttl,
        Some(_) => return Err(Error::request("cache_control.ttl must be 5m or 1h")),
    };
    if map.keys().any(|key| key != "type" && key != "ttl") {
        return Err(Error::request("unsupported cache_control options"));
    }
    Ok(Some(ttl.to_string()))
}

fn image(source: &Value) -> Result<Image> {
    let Value::Object(source) = source else {
        return Err(Error::request("image source must be an object"));
    };
    let truthy_field = |key: &str| source.get(key).filter(|value| truthy(value));
    match source.get("type").and_then(Value::as_str) {
        Some("url") if truthy_field("url").is_some() => Ok(Image {
            url: truthy_field("url").map(py_str).unwrap_or_default(),
            cache: None,
        }),
        Some("base64") if truthy_field("data").is_some() => {
            let media = truthy_field("media_type")
                .map(py_str)
                .unwrap_or_else(|| "image/png".into());
            let data = truthy_field("data").map(py_str).unwrap_or_default();
            Ok(Image {
                url: format!("data:{media};base64,{data}"),
                cache: None,
            })
        }
        _ => Err(Error::request(
            "image source must be a url or base64 source",
        )),
    }
}

fn string_field(part: &Object, key: &str, default: &str) -> String {
    part.get(key)
        .map(py_str)
        .unwrap_or_else(|| default.to_string())
}

/// `str(part.get(key) or "")`
fn truthy_string(part: &Object, key: &str) -> String {
    part.get(key)
        .filter(|value| truthy(value))
        .map(py_str)
        .unwrap_or_default()
}

fn block(part: &Value) -> Result<Block> {
    let Value::Object(part) = part else {
        return Err(Error::request("each content block must be an object"));
    };
    let null = Value::Null;
    let kind = part.get("type").unwrap_or(&null);
    let kind_name = kind.as_str().unwrap_or("");
    let is = |name: &str| kind.as_str() == Some(name);
    let cache = cache(part.get("cache_control").unwrap_or(&null))?;
    if is("web_search_tool_result")
        || (is("server_tool_use") && part.get("name") == Some(&Value::String("web_search".into())))
    {
        return Ok(Block::HostedSearch {
            item: part.clone(),
            source: Source::Anthropic,
        });
    }
    if is("text")
        && part
            .keys()
            .any(|key| !["type", "text", "cache_control", "citations"].contains(&key.as_str()))
    {
        return Ok(Block::NativeAnthropicBlock(part.clone()));
    }
    let source = part.get("source").unwrap_or(&null);
    let file_image = is("image") && source.get("type") == Some(&Value::String("file".into()));
    if NATIVE_KINDS.contains(&kind_name)
        || file_image
        || (cache.is_some() && !CACHED_KINDS.contains(&kind_name))
    {
        // No IR equivalent: the block, breakpoint included, stays verbatim.
        return Ok(Block::NativeAnthropicBlock(part.clone()));
    }
    match kind.as_str() {
        Some("text") => {
            let citations = match part.get("citations") {
                None | Some(Value::Null) => None,
                Some(Value::Array(citations)) => Some(citations.clone()),
                Some(_) => return Err(Error::request("text citations must be an array")),
            };
            Ok(Block::Text(Text {
                text: string_field(part, "text", ""),
                cache,
                citations,
            }))
        }
        Some("image") => Ok(Block::Image(Image {
            cache,
            ..image(source)?
        })),
        Some("tool_use") => Ok(Block::ToolUse(ToolUse {
            id: truthy_string(part, "id"),
            name: truthy_string(part, "name"),
            arguments: part
                .get("input")
                .cloned()
                .unwrap_or_else(|| Value::Object(Object::new())),
            namespace: String::new(),
            cache,
        })),
        Some("tool_result") => tool_result(part, cache),
        Some("thinking") => {
            // Must survive verbatim or the upstream refuses the turn.
            Ok(Block::Thinking(Thinking {
                text: string_field(part, "thinking", ""),
                signature: string_field(part, "signature", ""),
                redacted: String::new(),
            }))
        }
        Some("redacted_thinking") => Ok(Block::Thinking(Thinking {
            text: String::new(),
            signature: String::new(),
            redacted: string_field(part, "data", ""),
        })),
        // Other server tools exist only in this format; keep them verbatim.
        Some("server_tool_use") => Ok(Block::NativeAnthropicBlock(part.clone())),
        _ => Err(Error::request(format!(
            "unsupported content block: {}",
            py_str(kind)
        ))),
    }
}

fn tool_result(part: &Object, mut cache: Cache) -> Result<Block> {
    let Some(tool_use_id) = part.get("tool_use_id").filter(|id| truthy(id)) else {
        return Err(Error::request("tool_result is missing tool_use_id"));
    };
    let content = part.get("content").unwrap_or(&Value::Null);
    if let Value::Array(items) = content {
        let plain_text = |item: &Value| match item {
            Value::Object(map) => {
                map.keys()
                    .all(|key| ["type", "text", "cache_control"].contains(&key.as_str()))
                    && map.get("type") == Some(&Value::String("text".into()))
            }
            _ => false,
        };
        if !items.iter().all(plain_text) {
            return Ok(Block::NativeAnthropicBlock(part.clone()));
        }
        if cache.is_none() {
            // A breakpoint inside plain text content marks the same prefix to
            // within that content; the block keeps it.
            let mut nested = Vec::with_capacity(items.len());
            for item in items {
                nested.push(self::cache(
                    item.get("cache_control").unwrap_or(&Value::Null),
                )?);
            }
            cache = nested.into_iter().rev().flatten().next();
        }
    }
    Ok(Block::ToolResult(ToolResult {
        tool_use_id: py_str(tool_use_id),
        text: text(content)?,
        is_error: part.get("is_error").is_some_and(truthy),
        cache,
    }))
}

fn turn(message: &Value) -> Result<Turn> {
    let Value::Object(message) = message else {
        return Err(Error::request("each message must be an object"));
    };
    let role = match message.get("role").and_then(Value::as_str) {
        Some("user" | "system") => Role::User,
        Some("assistant") => Role::Assistant,
        _ => {
            return Err(Error::request(format!(
                "unsupported message role: {}",
                py_str(message.get("role").unwrap_or(&Value::Null))
            )))
        }
    };
    // Spec allows a system role inside messages and Claude Code uses it;
    // neither upstream has a third role, so keep the text in place.
    let blocks = match message.get("content") {
        Some(Value::String(content)) if content.is_empty() => Vec::new(),
        Some(Value::String(content)) => vec![Block::Text(Text::new(content.clone()))],
        Some(Value::Array(parts)) => parts.iter().map(block).collect::<Result<_>>()?,
        _ => return Err(Error::request("message content must be a string or array")),
    };
    Ok(Turn { role, blocks })
}

fn tools(value: &Value) -> Result<Vec<Tool>> {
    let mut tools = Vec::new();
    for item in definitions(value)? {
        let kind = truthy_string(item, "type");
        if kind.starts_with(WEB_SEARCH_PREFIX) {
            tools.push(Tool::WebSearch(WebSearchTool {
                native: item.clone(),
                source: Source::Anthropic,
            }));
            continue;
        }
        if !kind.is_empty() && kind != "custom" {
            return Err(Error::request(format!("unsupported server tool: {kind}")));
        }
        let fields: Object = item
            .iter()
            .filter(|(key, _)| key.as_str() != "cache_control")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        let mut tool = parse_function(&fields, "anthropic", "input_schema")?;
        tool.cache = cache(item.get("cache_control").unwrap_or(&Value::Null))?;
        tools.push(Tool::Function(tool));
    }
    Ok(tools)
}

/// The choice, and whether parallel calls were disabled.
fn tool_choice(value: &Value) -> Result<(Option<ToolChoice>, Option<bool>)> {
    let map = match value {
        Value::Null => return Ok((None, None)),
        Value::Object(map) => map,
        _ => return Err(Error::request("unsupported tool_choice")),
    };
    let kind = match map.get("type").and_then(Value::as_str) {
        Some("auto") => ChoiceKind::Auto,
        Some("any") => ChoiceKind::Required,
        Some("tool") => ChoiceKind::Tool,
        Some("none") => ChoiceKind::None,
        _ => return Err(Error::request("unsupported tool_choice")),
    };
    let disabled = optional_bool(
        map.get("disable_parallel_tool_use").unwrap_or(&Value::Null),
        "disable_parallel_tool_use",
    )?;
    let name = map.get("name");
    let name = match name {
        Some(Value::String(name)) if !name.is_empty() => name.clone(),
        _ if kind == ChoiceKind::Tool => return Err(Error::request("tool_choice requires a name")),
        Some(Value::String(name)) => name.clone(),
        _ => String::new(),
    };
    Ok((Some(ToolChoice { kind, name }), disabled.map(|flag| !flag)))
}

/// System blocks in order, preserving cache breakpoints.
fn system(value: &Value) -> Result<Vec<Text>> {
    let parts = match value {
        Value::String(text) if text.is_empty() => return Ok(Vec::new()),
        Value::String(text) => return Ok(vec![Text::new(text.clone())]),
        Value::Null => return Ok(Vec::new()),
        Value::Array(parts) => parts,
        _ => return Err(Error::request("system must be a string or array")),
    };
    let blocks = parts.iter().map(block).collect::<Result<Vec<_>>>()?;
    blocks
        .into_iter()
        .map(|block| match block {
            Block::Text(text) => Ok(text),
            _ => Err(Error::request("system content must contain text blocks")),
        })
        .collect()
}

pub fn parse(body: &Object, session: &str) -> Result<ChatRequest> {
    parse_body(body, session, true)
}

/// Parse a count_tokens body, which has no max_tokens.
pub fn parse_count(body: &Object, session: &str) -> Result<ChatRequest> {
    parse_body(body, session, false)
}

fn parse_body(body: &Object, session: &str, generating: bool) -> Result<ChatRequest> {
    let null = Value::Null;
    let get = |key: &str| body.get(key).unwrap_or(&null);
    let model = match get("model") {
        Value::String(model) if !model.is_empty() => model.clone(),
        _ => return Err(Error::request("model is required")),
    };
    let messages = match get("messages") {
        Value::Array(messages) if !messages.is_empty() => messages,
        _ => return Err(Error::request("messages must be a non-empty array")),
    };
    let mut max_tokens = Value::Null;
    if generating {
        // Zero is legal: it pre-warms the prompt cache without generating.
        let Some(count) = integer(get("max_tokens")) else {
            return Err(Error::request(
                "max_tokens is required and must be an integer",
            ));
        };
        if count < 0 {
            return Err(Error::request("max_tokens must not be negative"));
        }
        max_tokens = get("max_tokens").clone();
    }
    for name in REJECTED {
        if !get(name).is_null() {
            return Err(Error::request(format!("unsupported parameter: {name}")));
        }
    }

    let turns = messages.iter().map(turn).collect::<Result<Vec<_>>>()?;
    let (choice, parallel) = tool_choice(get("tool_choice"))?;
    let thinking = get("thinking");
    let mode = match thinking {
        Value::Null => None,
        Value::Object(map) => match map.get("type").and_then(Value::as_str) {
            Some(mode @ ("enabled" | "disabled" | "adaptive")) => Some(mode),
            _ => None,
        },
        _ => None,
    };
    if !thinking.is_null() && mode.is_none() {
        return Err(Error::request(
            "thinking must specify enabled, disabled, or adaptive",
        ));
    }
    let mut budget = None;
    if mode == Some("enabled") {
        budget = integer(thinking.get("budget_tokens").unwrap_or(&null));
        if budget.is_none() {
            return Err(Error::request("thinking.budget_tokens must be an integer"));
        }
    }
    let display = match thinking.get("display") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(display)) if display == "summarized" || display == "omitted" => {
            display.clone()
        }
        Some(_) => {
            return Err(Error::request(
                "thinking.display must be summarized or omitted",
            ))
        }
    };
    // The provider validates this against the selected model's live catalog.
    let empty = Object::new();
    let config = match get("output_config") {
        Value::Null => &empty,
        Value::Object(config) => config,
        _ => return Err(Error::request("output_config must be an object")),
    };
    let mut extra: Vec<&str> = config
        .keys()
        .map(String::as_str)
        .filter(|key| !["effort", "format"].contains(key))
        .collect();
    extra.sort_unstable();
    if !extra.is_empty() {
        // Dropping these would answer as if the client had never asked.
        return Err(Error::request(format!(
            "unsupported output_config options: {}",
            extra.join(", ")
        )));
    }
    let effort = config.get("effort").cloned().unwrap_or(Value::Null);
    let output_format = output_format(config.get("format").unwrap_or(&null), get("output_format"))?;

    let system = system(get("system"))?;
    let tools = tools(get("tools"))?;
    let stream = optional_bool(get("stream"), "stream")?.unwrap_or(false);
    let cache = cache(get("cache_control"))?;
    let mut params: Object = PARAMS
        .iter()
        .filter_map(|name| body.get(*name).map(|v| (name.to_string(), v.clone())))
        .collect();
    if let Some(stop) = body.get("stop_sequences") {
        params.insert("stop".into(), stop.clone());
    }
    Ok(ChatRequest {
        model,
        system,
        turns: turns.into_iter().filter(|t| !t.blocks.is_empty()).collect(),
        tools,
        tool_choice: choice,
        max_tokens,
        reasoning_effort: effort,
        thinking_budget: budget,
        thinking_mode: mode
            .filter(|mode| *mode != "enabled")
            .unwrap_or_default()
            .to_string(),
        thinking_display: display,
        parallel_tool_calls: parallel,
        stream,
        session: session.to_string(),
        cache,
        params,
        output_format,
        ..ChatRequest::default()
    })
}
