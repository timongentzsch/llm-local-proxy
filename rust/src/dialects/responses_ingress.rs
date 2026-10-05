//! OpenAI Responses request body -> dialect-neutral request.

use super::openai_output::{enum_value, format_of, VERBOSITY};
use super::openai_reasoning::options as reasoning_options;
use crate::error::{Error, Result};
use crate::ir::{
    Block, ChatRequest, Image, NamespaceMember, OutputFormat, Role, Source, Text, Tool,
    ToolNamespace, ToolResult, ToolUse, Turn, WebSearchTool,
};
use crate::json::{py_str, truthy, Object};
use crate::tools::{definitions, optional_bool, parse_choice, parse_function};
use serde_json::Value;
use std::collections::BTreeSet;

/// `include` values this proxy already satisfies. Encrypted reasoning always
/// reaches the client (from a Responses upstream as issued, from others wrapped
/// in `encrypted_content`); every other value asks for payloads no upstream is
/// told to produce.
const INCLUDE_SUPPORTED: [&str; 1] = ["reasoning.encrypted_content"];
const REASONING_CONTEXTS: [&str; 3] = ["auto", "current_turn", "all_turns"];

fn content(value: &Value) -> Result<Vec<Block>> {
    let parts = match value {
        Value::String(text) if text.is_empty() => return Ok(Vec::new()),
        Value::String(text) => return Ok(vec![Block::Text(Text::new(text.clone()))]),
        Value::Array(parts) => parts,
        _ => return Err(Error::request("message content must be a string or array")),
    };
    let mut blocks = Vec::new();
    for part in parts {
        let Value::Object(part) = part else {
            return Err(Error::request("message content parts must be objects"));
        };
        let kind = part.get("type").unwrap_or(&Value::Null);
        let image = part.get("image_url").filter(|url| truthy(url));
        match (kind.as_str(), image) {
            (Some("input_text" | "output_text" | "text"), _) => {
                let text = part.get("text").map(py_str).unwrap_or_default();
                blocks.push(Block::Text(Text::new(text)));
            }
            (Some("input_image"), Some(url)) => blocks.push(Block::Image(Image {
                url: py_str(url),
                cache: None,
            })),
            _ => {
                return Err(Error::request(format!(
                    "unsupported Responses content type: {}",
                    py_str(kind)
                )))
            }
        }
    }
    Ok(blocks)
}

fn append(turns: &mut Vec<Turn>, role: Role, blocks: Vec<Block>) {
    if blocks.is_empty() {
        return;
    }
    match turns.last_mut() {
        Some(last) if last.role == role => last.blocks.extend(blocks),
        _ => turns.push(Turn { role, blocks }),
    }
}

fn input(
    value: &Value,
    system: &mut Vec<Text>,
    additional_tools: &mut Vec<Tool>,
) -> Result<Vec<Turn>> {
    let items = match value {
        Value::String(text) if text.is_empty() => return Ok(Vec::new()),
        Value::String(text) => {
            return Ok(vec![Turn {
                role: Role::User,
                blocks: vec![Block::Text(Text::new(text.clone()))],
            }])
        }
        Value::Array(items) => items,
        _ => return Err(Error::request("input must be a string or array")),
    };
    let null = Value::Null;
    let mut turns = Vec::new();
    for item in items {
        let Value::Object(item) = item else {
            return Err(Error::request("input items must be objects"));
        };
        let kind = item.get("type").unwrap_or(&null);
        let has_role = item.get("role").is_some_and(truthy);
        if kind.as_str() == Some("additional_tools") {
            additional_tools.extend(tools(item.get("tools").unwrap_or(&null))?);
        } else if kind.as_str() == Some("message") || (kind.is_null() && has_role) {
            let Some(role) = item.get("role").and_then(Value::as_str) else {
                return Err(Error::request("message role must be a string"));
            };
            let blocks = content(item.get("content").unwrap_or(&Value::Array(Vec::new())))?;
            match role {
                "system" | "developer" => {
                    system.extend(blocks.into_iter().filter_map(|b| match b {
                        Block::Text(text) => Some(text),
                        _ => None,
                    }))
                }
                "user" => append(&mut turns, Role::User, blocks),
                "assistant" => append(&mut turns, Role::Assistant, blocks),
                _ => {
                    return Err(Error::request(format!(
                        "unsupported Responses role: {role}"
                    )))
                }
            }
        } else if kind.as_str() == Some("reasoning") {
            // Keep all provider-owned fields opaque. Only type is interpreted.
            append(
                &mut turns,
                Role::Assistant,
                vec![Block::Reasoning(item.clone())],
            );
        } else if kind.as_str() == Some("function_call") {
            let call_id = [item.get("call_id"), item.get("id")]
                .into_iter()
                .flatten()
                .find(|id| truthy(id));
            let name = item.get("name").filter(|name| truthy(name));
            let (Some(call_id), Some(name)) = (call_id, name) else {
                return Err(Error::request("function_call requires call_id and name"));
            };
            let namespace = item.get("namespace").filter(|n| truthy(n));
            let call = ToolUse {
                id: py_str(call_id),
                name: py_str(name),
                arguments: item
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| Value::String("{}".into())),
                namespace: namespace.map(py_str).unwrap_or_default(),
                cache: None,
            };
            append(&mut turns, Role::Assistant, vec![Block::ToolUse(call)]);
        } else if kind.as_str() == Some("function_call_output") {
            let Some(call_id) = item.get("call_id").filter(|id| truthy(id)) else {
                return Err(Error::request("function_call_output requires call_id"));
            };
            match item.get("output") {
                None => append(
                    &mut turns,
                    Role::User,
                    vec![Block::ToolResult(ToolResult {
                        tool_use_id: py_str(call_id),
                        ..ToolResult::default()
                    })],
                ),
                Some(Value::String(output)) => append(
                    &mut turns,
                    Role::User,
                    vec![Block::ToolResult(ToolResult {
                        tool_use_id: py_str(call_id),
                        text: output.clone(),
                        ..ToolResult::default()
                    })],
                ),
                // Structured text/image/file outputs have no lossless legacy IR
                // representation; keep the entire item for a Responses upstream.
                Some(_) => append(
                    &mut turns,
                    Role::User,
                    vec![Block::NativeResponseItem(item.clone())],
                ),
            }
        } else if kind.as_str() == Some("web_search_call") {
            let hosted = Block::HostedSearch {
                item: item.clone(),
                source: Source::Responses,
            };
            append(&mut turns, Role::Assistant, vec![hosted]);
        } else {
            // Current Responses adds native item kinds regularly (custom calls,
            // programs, shell/patch calls, tool search, compaction). Preserve
            // unknown typed items for a native upstream instead of dropping them.
            let kind = match kind.as_str() {
                Some(kind) if !kind.is_empty() => kind,
                _ => return Err(Error::request("Responses input item requires type")),
            };
            let role = if kind.ends_with("_output") {
                Role::User
            } else {
                Role::Assistant
            };
            append(
                &mut turns,
                role,
                vec![Block::NativeResponseItem(item.clone())],
            );
        }
    }
    Ok(turns)
}

fn tools(value: &Value) -> Result<Vec<Tool>> {
    let mut tools = Vec::new();
    for item in definitions(value)? {
        let Some(kind) = item.get("type").and_then(Value::as_str) else {
            return Err(Error::request("tool type must be a string"));
        };
        let named = item.get("name").is_some_and(truthy);
        match kind {
            "web_search" | "web_search_preview" => tools.push(Tool::WebSearch(WebSearchTool {
                native: item.clone(),
                source: Source::Responses,
            })),
            "function" if named => tools.push(Tool::Function(parse_function(
                item,
                "responses",
                "parameters",
            )?)),
            "namespace" => tools.push(namespace(item)?),
            "custom" | "tool_search" => {
                // These newer Responses definitions have no Chat Completions
                // equivalent. Providers that speak Responses can forward them;
                // other providers reject them rather than silently weakening tools.
                tools.push(Tool::Native(item.clone()))
            }
            _ => {
                return Err(Error::request(format!(
                    "unsupported Responses tool: {kind}"
                )))
            }
        }
    }
    Ok(tools)
}

fn namespace(item: &Object) -> Result<Tool> {
    let name = match item.get("name") {
        Some(Value::String(name)) if !name.is_empty() => name.clone(),
        _ => return Err(Error::request("namespace tool requires a name")),
    };
    let mut members = Vec::new();
    for member in tools(item.get("tools").unwrap_or(&Value::Null))? {
        match member {
            Tool::Function(function) => members.push(NamespaceMember::Function(function)),
            Tool::Native(native) => members.push(NamespaceMember::Native(native)),
            _ => {
                return Err(Error::request(format!(
                    "namespace {name} may hold only function and custom tools"
                )))
            }
        }
    }
    Ok(Tool::Namespace(ToolNamespace {
        name,
        tools: members,
        item: item.clone(),
    }))
}

/// Read `text` into the neutral IR, refusing options we cannot honour.
fn text(value: &Value) -> Result<(Option<OutputFormat>, String)> {
    let map = match value {
        Value::Null => return Ok((None, String::new())),
        Value::Object(map) => map,
        _ => return Err(Error::request("text must be an object")),
    };
    let mut extra: Vec<&str> = map
        .keys()
        .map(String::as_str)
        .filter(|key| !["format", "verbosity"].contains(key))
        .collect();
    extra.sort_unstable();
    if !extra.is_empty() {
        // Silently ignoring these would return unconstrained output that looks
        // like a compliant answer, which is the failure this proxy refuses.
        return Err(Error::request(format!(
            "unsupported text options: {}",
            extra.join(", ")
        )));
    }
    let verbosity = enum_value(
        map.get("verbosity").unwrap_or(&Value::Null),
        &VERBOSITY,
        "text.verbosity",
    )?;
    match map.get("format") {
        None | Some(Value::Null) => Ok((None, verbosity)),
        Some(Value::Object(item)) => {
            let kind = item.get("type").unwrap_or(&Value::Null);
            Ok((format_of(kind, item)?, verbosity))
        }
        Some(_) => Err(Error::request("text.format must be an object")),
    }
}

fn check_include(value: &Value) -> Result<()> {
    let items = match value {
        Value::Null => return Ok(()),
        Value::Array(items) => items,
        _ => return Err(Error::request("include must be an array")),
    };
    let seen: BTreeSet<String> = items.iter().map(py_str).collect();
    let extra: Vec<&str> = seen
        .iter()
        .map(String::as_str)
        .filter(|item| !INCLUDE_SUPPORTED.contains(item))
        .collect();
    if extra.is_empty() {
        return Ok(());
    }
    Err(Error::request(format!(
        "unsupported include values: {}",
        extra.join(", ")
    )))
}

pub fn parse(body: &Object, session: &str) -> Result<ChatRequest> {
    let null = Value::Null;
    let get = |key: &str| body.get(key).unwrap_or(&null);
    if get("store") == &Value::Bool(true) {
        return Err(Error::request(
            "only stateless Responses requests are supported; set store to false",
        ));
    }
    if !get("previous_response_id").is_null() {
        return Err(Error::request(
            "previous_response_id is not supported; resend complete input history",
        ));
    }
    if !get("conversation").is_null() {
        return Err(Error::request(
            "conversation is not supported; resend complete input history",
        ));
    }
    if truthy(get("background")) {
        return Err(Error::request(
            "background is not supported; this proxy streams one response and \
             keeps no job to poll",
        ));
    }
    check_include(get("include"))?;
    let model = match get("model") {
        Value::String(model) if !model.is_empty() => model.clone(),
        _ => return Err(Error::request("model is required")),
    };
    let instructions = body.get("instructions").filter(|value| truthy(value));
    let mut system: Vec<Text> = instructions
        .map(|text| Text::new(py_str(text)))
        .into_iter()
        .collect();
    let mut additional_tools = Vec::new();
    let empty = Value::String(String::new());
    let turns = input(
        body.get("input").unwrap_or(&empty),
        &mut system,
        &mut additional_tools,
    )?;
    let reasoning = get("reasoning");
    let (effort, summary) = reasoning_options(reasoning)?;
    let context = enum_value(
        reasoning.get("context").unwrap_or(&null),
        &REASONING_CONTEXTS,
        "reasoning.context",
    )?;
    let (output_format, verbosity) = text(get("text"))?;
    let cache_key = match get("prompt_cache_key") {
        key if truthy(key) => py_str(key),
        _ => String::new(),
    };
    let params: Object = ["temperature", "top_p", "stop"]
        .iter()
        .filter_map(|key| body.get(*key).map(|v| (key.to_string(), v.clone())))
        .collect();
    let mut all_tools = tools(get("tools"))?;
    all_tools.extend(additional_tools);
    let tool_choice = parse_choice(get("tool_choice"), false)?;
    let parallel_tool_calls = optional_bool(get("parallel_tool_calls"), "parallel_tool_calls")?;
    let stream = optional_bool(get("stream"), "stream")?.unwrap_or(false);
    Ok(ChatRequest {
        model,
        system,
        turns,
        tools: all_tools,
        tool_choice,
        max_tokens: get("max_output_tokens").clone(),
        reasoning_effort: effort,
        reasoning_summary: summary,
        reasoning_context: context,
        verbosity,
        parallel_tool_calls,
        stream,
        session: if session.is_empty() {
            cache_key.clone()
        } else {
            session.to_string()
        },
        cache_key,
        params,
        output_format,
        ..ChatRequest::default()
    })
}
