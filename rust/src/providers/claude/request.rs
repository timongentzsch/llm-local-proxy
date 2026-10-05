//! ChatRequest -> the Claude Messages request body.

use super::strip;
use super::subscription::CLAUDE_CODE_SYSTEM_MARKER;
use super::thinking::{unpack, Outcome, Unpacked};
use crate::error::{Error, Result};
use crate::ids::{self, Ids};
use crate::ir::{Block, ChatRequest, ChoiceKind, Role, Source, Text, Tool, ToolChoice, Turn};
use crate::json::{get, integer, py_str, truthy, Object};
use crate::obj;
use crate::reasoning::ReasoningCache;
use crate::tools::{anthropic_web_search, arguments, flatten, qualified_name, render_function};
use serde_json::{json, Value};

const WEB_SEARCH_BETA: &str = "web-search-2025-03-05";
/// Structured outputs remain gated; `output_config.format` needs this header.
const STRUCTURED_OUTPUTS_BETA: &str = "structured-outputs-2025-11-13";

/// Chat Completions knobs the Messages API has no equivalent for.
const UNSUPPORTED: [&str; 6] = [
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "top_logprobs",
    "seed",
    "logit_bias",
];

/// What the catalog knows about the model being asked.
#[derive(Default, Clone, Copy)]
pub struct Options<'a> {
    /// The model's maximum output, used when the client names no max_tokens.
    pub max_output: Option<i64>,
    /// How the model takes thinking ("adaptive", "enabled", ...), if known.
    pub thinking: Option<&'a str>,
    /// The effort tiers the catalog lists; None when it lists none at all.
    pub reasoning_efforts: Option<&'a [String]>,
    pub reasoning_cache: Option<&'a ReasoningCache>,
}

/// Block kinds Claude signs, and therefore will not accept rebuilt.
const SIGNED: [&str; 2] = ["thinking", "redacted_thinking"];

/// Why a reasoning item could not be replayed, as the operator reads it.
const DROP_REASONS: [(Outcome, &str); 4] = [
    (
        Outcome::Foreign,
        "this proxy did not write and cannot replay",
    ),
    (Outcome::Malformed, "whose envelope arrived damaged"),
    (
        Outcome::BadVersion,
        "whose envelope an unsupported version wrote",
    ),
    (
        Outcome::Withheld,
        "whose thinking text the upstream never streamed",
    ),
];

fn number(value: &Value, name: &str, low: f64, high: f64, closed: bool) -> Result<()> {
    let ok = match value {
        Value::Number(n) => n
            .as_f64()
            .is_some_and(|v| (if closed { low <= v } else { low < v }) && v <= high),
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(Error::request(format!(
            "{name} must be a number between {low} and {high}"
        )))
    }
}

/// `params.get(name)` where None (a missing or null value) means unset.
fn param<'a>(request: &'a ChatRequest, name: &str) -> Option<&'a Value> {
    request.params.get(name).filter(|value| !value.is_null())
}

fn check(request: &ChatRequest) -> Result<()> {
    for name in UNSUPPORTED {
        if param(request, name).is_some() {
            return Err(Error::request(format!("unsupported parameter: {name}")));
        }
    }
    if !request.verbosity.is_empty() {
        return Err(Error::request("unsupported parameter: verbosity"));
    }
    if !matches!(request.reasoning_context.as_str(), "" | "auto") {
        return Err(Error::request("unsupported parameter: reasoning.context"));
    }
    if let Some(temperature) = param(request, "temperature") {
        number(temperature, "temperature", 0.0, 1.0, true)?;
    }
    if let Some(top_p) = param(request, "top_p") {
        number(top_p, "top_p", 0.0, 1.0, false)?;
    }
    if let Some(top_k) = param(request, "top_k") {
        if !integer(top_k).is_some_and(|k| k > 0) {
            return Err(Error::request("top_k must be a positive integer"));
        }
    }
    Ok(())
}

/// The `cache_control` field for a breakpoint, or none.
fn cached(ttl: Option<&str>) -> Object {
    let Some(ttl) = ttl else {
        return Object::new();
    };
    let mut control = obj! { "type": "ephemeral" };
    if !ttl.is_empty() {
        control.insert("ttl".into(), json!(ttl));
    }
    obj! { "cache_control": control }
}

fn merged(mut base: Object, extra: Object) -> Object {
    base.extend(extra);
    base
}

/// Whether the client placed a breakpoint anywhere Messages allows one.
fn has_breakpoint(body: &Object) -> bool {
    let list = |key: &str| -> Vec<&Value> {
        body.get(key)
            .and_then(Value::as_array)
            .map(|items| items.iter().collect())
            .unwrap_or_default()
    };
    let mut blocks = list("system");
    blocks.extend(list("tools"));
    for message in list("messages") {
        for block in get(message, "content").as_array().into_iter().flatten() {
            blocks.push(block);
            if get(block, "type").as_str() == Some("tool_result") {
                blocks.extend(get(block, "content").as_array().into_iter().flatten());
            }
        }
    }
    blocks
        .iter()
        .any(|block| block.get("cache_control").is_some())
}

fn image(url: &str) -> Result<Object> {
    const MESSAGE: &str = "image_url must be a data URL or an http(s) URL";
    if url.starts_with("data:") {
        let (header, data) = url.split_once(',').unwrap_or((url, ""));
        if data.is_empty() {
            return Err(Error::request(MESSAGE));
        }
        let media_type = header[5..].split(';').next().unwrap_or("");
        let media_type = if media_type.is_empty() {
            "image/png"
        } else {
            media_type
        };
        return Ok(obj! {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        });
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return Ok(obj! { "type": "image", "source": {"type": "url", "url": url} });
    }
    Err(Error::request(MESSAGE))
}

fn text_block(block: &Text) -> Object {
    // Claude verifies search citations by their encrypted index; one this proxy
    // wrote for another upstream's search has none and would be refused.
    let citations: Vec<Value> = block
        .citations
        .iter()
        .flatten()
        .filter(|citation| match citation {
            Value::Object(map) => {
                map.get("type").and_then(Value::as_str) != Some("web_search_result_location")
                    || map.contains_key("encrypted_index")
            }
            _ => true,
        })
        .cloned()
        .collect();
    let mut result = merged(
        obj! { "type": "text", "text": block.text },
        cached(block.cache.as_deref()),
    );
    if !citations.is_empty() {
        result.insert("citations".into(), Value::Array(citations));
    }
    result
}

fn native_thinking(block: &crate::ir::Thinking) -> Object {
    if !block.redacted.is_empty() {
        return obj! { "type": "redacted_thinking", "data": block.redacted };
    }
    obj! { "type": "thinking", "thinking": block.text, "signature": block.signature }
}

/// `str(block.get(key, ""))`.
fn field(block: &Object, key: &str) -> String {
    block.get(key).map(py_str).unwrap_or_default()
}

/// False for a signed block Claude will not take back.
///
/// A thinking block whose text never arrived is one. Histories written before
/// that was understood hold them by the hundred, so they are refused on the
/// way in as well as on the way out.
fn replayable(block: Option<&Object>) -> bool {
    let Some(block) = block else {
        return false;
    };
    if block.get("type").and_then(Value::as_str) != Some("thinking") {
        return true;
    }
    !field(block, "thinking").is_empty() && !field(block, "signature").is_empty()
}

fn kind_name(block: &Block) -> &'static str {
    match block {
        Block::Text(_) => "Text",
        Block::Image(_) => "Image",
        Block::ToolUse(_) => "ToolUse",
        Block::ToolResult(_) => "ToolResult",
        Block::Thinking(_) => "Thinking",
        Block::Reasoning(_) => "Reasoning",
        Block::NativeResponseItem(_) => "NativeResponseItem",
        Block::NativeAnthropicBlock(_) => "NativeAnthropicBlock",
        Block::HostedSearch { .. } => "HostedSearch",
    }
}

/// One turn, in the order its blocks actually occurred.
///
/// Claude interleaves thinking with the tool calls it precedes, and verifies
/// what it gets back, so position is part of the payload: grouping blocks by
/// kind would rewrite a turn Claude signed.
fn blocks(
    turn: &Turn,
    cache: Option<&ReasoningCache>,
    dropped: &mut Vec<Outcome>,
    ids: &dyn Ids,
) -> Result<Vec<Object>> {
    let role = turn.role.as_str();
    let user = turn.role == Role::User;
    let mut out: Vec<Object> = Vec::new();
    let mut ordinals: Vec<i64> = Vec::new();
    let mut lost = 0;
    let mut has_native_thinking = false;
    for block in &turn.blocks {
        if let Block::NativeResponseItem(item) = block {
            return Err(Error::request(format!(
                "Claude upstream cannot faithfully represent Responses items: {}",
                item.get("type").map_or_else(|| "unknown".into(), py_str)
            )));
        }
        if user
            && matches!(
                block,
                Block::Thinking(_) | Block::Reasoning(_) | Block::ToolUse(_)
            )
        {
            return Err(Error::request(format!(
                "unsupported {role} content: {}",
                kind_name(block)
            )));
        }
        match block {
            Block::Thinking(thinking) => {
                has_native_thinking = true;
                // A client can hand back a block it was given, including one
                // this upstream signed without ever streaming its text.
                let native = native_thinking(thinking);
                if replayable(Some(&native)) {
                    out.push(native);
                } else {
                    lost += 1;
                    dropped.push(Outcome::Withheld);
                }
            }
            Block::Reasoning(item) => {
                let mut recovered = unpack(item.get("encrypted_content").unwrap_or(&Value::Null));
                if recovered.outcome == Outcome::Ok && !replayable(recovered.block.as_ref()) {
                    recovered = Unpacked::without_block(Outcome::Withheld);
                }
                match recovered {
                    Unpacked {
                        outcome: Outcome::Ok,
                        block: Some(block),
                        ordinal,
                    } => {
                        out.push(block);
                        ordinals.push(ordinal);
                    }
                    Unpacked { outcome, .. } => {
                        lost += 1;
                        dropped.push(outcome);
                    }
                }
            }
            Block::Text(text) => {
                if user || !strip(&text.text).is_empty() {
                    out.push(text_block(text));
                }
            }
            Block::Image(picture) if user => {
                out.push(merged(
                    image(&picture.url)?,
                    cached(picture.cache.as_deref()),
                ));
            }
            Block::ToolResult(result) if user => {
                let mut entry = obj! {
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id,
                    "content": result.text,
                };
                if result.is_error {
                    entry.insert("is_error".into(), json!(true));
                }
                out.push(merged(entry, cached(result.cache.as_deref())));
            }
            Block::ToolUse(call) => {
                let id = if call.id.is_empty() {
                    format!("toolu_{}", ids::hex24(ids))
                } else {
                    call.id.clone()
                };
                let name = qualified_name(&call.namespace, &call.name);
                let input = arguments(&call.arguments, Error::request)?;
                out.push(merged(
                    obj! { "type": "tool_use", "id": id, "name": name, "input": input },
                    cached(call.cache.as_deref()),
                ));
            }
            Block::NativeAnthropicBlock(item) => out.push(item.clone()),
            Block::HostedSearch { item, source } => {
                if *source == Source::Anthropic {
                    out.push(item.clone());
                }
            }
            other => {
                return Err(Error::request(format!(
                    "unsupported {role} content: {}",
                    kind_name(other)
                )));
            }
        }
    }
    let uses: Vec<&str> = turn
        .blocks
        .iter()
        .filter_map(|block| match block {
            Block::ToolUse(call) => Some(call.id.as_str()),
            _ => None,
        })
        .collect();
    let replay: Vec<Object> = match cache {
        Some(cache) if !uses.is_empty() => {
            let ids: Vec<String> = uses
                .iter()
                .filter(|id| !id.is_empty())
                .map(|id| id.to_string())
                .collect();
            cache
                .get(&ids)
                .into_iter()
                .filter_map(|value| match value {
                    Value::Object(map) => Some(map),
                    _ => None,
                })
                .collect()
        }
        _ => Vec::new(),
    };
    if has_native_thinking {
        // Anthropic clients return native thinking blocks themselves. The
        // cache exists for dialects such as Chat Completions that cannot carry
        // those blocks; prepending it here would duplicate a signed block and
        // Claude rejects the modified assistant turn.
        return Ok(out);
    }
    if !ordinals.is_empty() {
        // A fraction of a signed turn is altered, where none of it is merely
        // thinner, so a turn that lost any block sends none. The cache gives
        // the count -- never the blocks, which it holds without their
        // positions -- and that is what catches a dropped trailing block,
        // whose ordinals still read 0..n-1.
        if lost > 0 || (!replay.is_empty() && ordinals.len() < replay.len()) {
            out.retain(|block| {
                !block
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| SIGNED.contains(&kind))
            });
            return Ok(out);
        }
        if ordinals.iter().enumerate().any(|(i, n)| *n != i as i64) {
            return Err(Error::request(
                "cannot replay Claude reasoning: the assistant turn's signed \
                 blocks arrived out of order or incomplete",
            ));
        }
        return Ok(out);
    }
    // No envelopes: a dialect that cannot carry reasoning, or a history older
    // than the envelope. Claude accepts a turn with no thinking.
    let mut combined = replay;
    combined.extend(out);
    Ok(combined)
}

fn tool_choice(choice: Option<&ToolChoice>) -> Result<Object> {
    match choice {
        None => Ok(obj! { "type": "auto" }),
        Some(choice) => match choice.kind {
            ChoiceKind::Auto => Ok(obj! { "type": "auto" }),
            ChoiceKind::Required => Ok(obj! { "type": "any" }),
            ChoiceKind::Tool => Ok(obj! { "type": "tool", "name": choice.name }),
            ChoiceKind::None => Err(Error::request("unsupported tool_choice")),
        },
    }
}

fn stop_sequences(stop: Option<&Value>) -> Vec<Value> {
    match stop {
        Some(Value::String(text)) if !text.is_empty() => vec![json!(text)],
        Some(Value::Array(items)) => items
            .iter()
            .filter(|item| truthy(item))
            .map(|item| json!(py_str(item)))
            .collect(),
        _ => Vec::new(),
    }
}

/// `float(value)` for a number `check` has already accepted.
fn float(value: &Value) -> Value {
    value
        .as_f64()
        .and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

/// The upstream body and the beta features it needs.
pub fn build(
    request: &ChatRequest,
    model: &str,
    options: Options<'_>,
    ids: &dyn Ids,
) -> Result<(Object, Vec<String>)> {
    check(request)?;

    let mut messages = Vec::new();
    let mut dropped: Vec<Outcome> = Vec::new();
    for turn in &request.turns {
        let content = blocks(turn, options.reasoning_cache, &mut dropped, ids)?;
        if !content.is_empty() {
            messages.push(json!({"role": turn.role.as_str(), "content": content}));
        }
    }
    // Visible rather than silent: the turn still runs, but Claude is no longer
    // seeing reasoning it signed, and each reason is a different operator
    // problem -- a foreign history, a damaged blob, a version skew.
    for (outcome, reason) in DROP_REASONS {
        let count = dropped.iter().filter(|seen| **seen == outcome).count();
        if count > 0 {
            let plural = if count == 1 { "" } else { "s" };
            eprintln!("claude: dropped {count} reasoning item{plural} {reason}");
        }
    }
    if messages.first().map(|first| get(first, "role").as_str()) != Some(Some("user")) {
        return Err(Error::request("first message must be a user message"));
    }

    let max_tokens = match &request.max_tokens {
        Value::Null => match options.max_output {
            Some(max) if max > 0 => json!(max),
            _ => {
                return Err(Error::request(
                    "model catalog did not report max output tokens; provide max_tokens",
                ))
            }
        },
        other => other.clone(),
    };
    let max_tokens = match integer(&max_tokens) {
        Some(count) if count >= 0 => count,
        _ => return Err(Error::request("max_tokens must not be negative")),
    };

    // Must be first to bill against the subscription pool; real clients
    // already send it.
    let kept: Vec<&Text> = request
        .system
        .iter()
        .filter(|block| !strip(&block.text).is_empty())
        .collect();
    let is_marker = |block: &Text| strip(&block.text) == CLAUDE_CODE_SYSTEM_MARKER;
    let default_marker = Text::new(CLAUDE_CODE_SYSTEM_MARKER);
    let marker = kept
        .iter()
        .copied()
        .find(|block| is_marker(block))
        .unwrap_or(&default_marker);
    let mut system = vec![Value::Object(text_block(marker))];
    system.extend(
        kept.iter()
            .filter(|block| !is_marker(block))
            .map(|block| Value::Object(text_block(block))),
    );
    let mut body = obj! {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": true,
        "system": system,
    };
    if let Some(temperature) = param(request, "temperature") {
        body.insert("temperature".into(), float(temperature));
    }
    if let Some(top_p) = param(request, "top_p") {
        body.insert("top_p".into(), float(top_p));
    }
    if let Some(top_k) = param(request, "top_k") {
        body.insert("top_k".into(), top_k.clone());
    }
    let sequences = stop_sequences(request.params.get("stop"));
    if !sequences.is_empty() {
        body.insert("stop_sequences".into(), Value::Array(sequences));
    }

    let mut betas: Vec<String> = Vec::new();
    let mut tools = Vec::new();
    for tool in flatten(&request.tools)?.0 {
        match &tool {
            Tool::Function(function) => {
                if function.options.get("defer_loading").is_some_and(truthy) {
                    return Err(Error::request(
                        "Anthropic deferred tools require unsupported tool search",
                    ));
                }
                tools.push(Value::Object(merged(
                    render_function(function, "anthropic", "input_schema")?,
                    cached(function.cache.as_deref()),
                )));
            }
            Tool::WebSearch(search) => {
                tools.push(Value::Object(anthropic_web_search(search)?));
                if !betas.iter().any(|beta| beta == WEB_SEARCH_BETA) {
                    betas.push(WEB_SEARCH_BETA.into());
                }
            }
            Tool::Native(item) | Tool::Namespace(crate::ir::ToolNamespace { item, .. }) => {
                return Err(Error::request(format!(
                    "Claude upstream cannot faithfully represent Responses tools: {}",
                    item.get("type").map_or_else(|| "unknown".into(), py_str)
                )));
            }
        }
    }
    let choice = request.tool_choice.as_ref();
    if !tools.is_empty() && choice.map(|c| c.kind) != Some(ChoiceKind::None) {
        body.insert("tools".into(), Value::Array(tools));
        let mut selection = tool_choice(choice)?;
        if let Some(parallel) = request.parallel_tool_calls {
            selection.insert("disable_parallel_tool_use".into(), json!(!parallel));
        }
        body.insert("tool_choice".into(), Value::Object(selection));
    }
    // The client's own automatic breakpoint; otherwise ours, unless the client
    // placed breakpoints itself: the upstream accepts at most four.
    if request.cache.is_some() || !has_breakpoint(&body) {
        body.extend(cached(Some(request.cache.as_deref().unwrap_or(""))));
    }

    if let Some(format) = &request.output_format {
        if format.kind != "json_schema" {
            // Messages constrains output with a schema or not at all; a bare
            // "must be JSON" mode would have to be faked in the prompt.
            return Err(Error::request(
                "Claude upstream can constrain output only with a JSON schema; \
                 json_object has no Messages equivalent",
            ));
        }
        let schema = format.schema.clone().map_or(Value::Null, Value::Object);
        body.insert(
            "output_config".into(),
            json!({"format": {"type": "json_schema", "schema": schema}}),
        );
        betas.push(STRUCTURED_OUTPUTS_BETA.into());
    }

    // Anthropic's zero-token prewarm generates nothing. The transport switches
    // just this request to non-streaming and synthesizes the ordinary event
    // lifecycle, so generation-only controls have no work to do here.
    if max_tokens == 0 {
        return Ok((body, betas));
    }

    let mut effort: Option<String> = None;
    // A model that declares no effort tiers (reasoning_efforts == []) cannot
    // express one, so the effort stays a preference, like a cache hint; do not
    // approximate named tiers with fabricated token budgets.
    let tiers = options.reasoning_efforts;
    if truthy(&request.reasoning_effort) && tiers.is_none_or(|tiers| !tiers.is_empty()) {
        let wanted = py_str(&request.reasoning_effort).to_lowercase();
        let supported: Vec<String> = tiers
            .unwrap_or(&[])
            .iter()
            .map(|item| item.to_lowercase())
            .collect();
        if !supported.is_empty() && !supported.contains(&wanted) {
            return Err(Error::request(format!(
                "unsupported reasoning_effort: {}",
                py_str(&request.reasoning_effort)
            )));
        }
        // Claude's native effort control is independent of its thinking mode.
        let config = body
            .entry("output_config")
            .or_insert_with(|| json!({}))
            .as_object_mut();
        if let Some(config) = config {
            config.insert("effort".into(), json!(wanted));
        }
        effort = Some(wanted);
    }

    // Every OpenAI summary mode asks for readable reasoning, which Messages
    // calls "summarized"; "none" asks for none.
    let display = if !request.thinking_display.is_empty() {
        request.thinking_display.as_str()
    } else if request.reasoning_summary == "none" {
        "omitted"
    } else {
        "summarized"
    };
    if request.thinking_mode == "disabled" {
        return Ok((body, betas));
    }
    if request.thinking_mode == "adaptive" {
        // The client asked the model to size its own reasoning.
        body.insert(
            "thinking".into(),
            json!({"type": "adaptive", "display": display}),
        );
        return Ok((body, betas));
    }
    if let Some(budget) = request.thinking_budget {
        let budget = budget.min(max_tokens - 1);
        if budget < 1024 {
            return Err(Error::request(
                "max_tokens is too small for the requested thinking budget",
            ));
        }
        // The catalog can report enabled as unsupported yet honour it.
        body.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": budget, "display": display}),
        );
    } else if options.thinking != Some("enabled")
        && (options.thinking == Some("adaptive")
            || effort.is_some()
            || !request.thinking_display.is_empty()
            || !request.reasoning_summary.is_empty())
    {
        // Some live catalog entries advertise effort but omit their thinking
        // capability even though the model accepts adaptive thinking, so an
        // OpenAI-shaped reasoning request activates it unless the catalog says
        // the model only takes a budget, which such a request does not name.
        body.insert(
            "thinking".into(),
            json!({"type": "adaptive", "display": display}),
        );
    }
    Ok((body, betas))
}
