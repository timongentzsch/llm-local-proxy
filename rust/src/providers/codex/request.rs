//! ChatRequest -> a Codex Responses API request body.

use super::thinking::unpack as unpack_thinking;
use crate::error::{Error, Result};
use crate::ir::{Block, ChatRequest, OutputFormat, Role, Source, Text, Thinking, ToolUse, Turn};
use crate::json::{dumps, py_str, truthy, Dumps, Object};
use crate::obj;
use crate::reasoning::ReasoningCache;
use crate::tools::{arguments, responses_choice, responses_tool};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Knobs Codex does not expose, sorted so the error names them in order.
const UNSUPPORTED: [&str; 10] = [
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "presence_penalty",
    "seed",
    "stop",
    "temperature",
    "top_k",
    "top_logprobs",
    "top_p",
];

/// Whether `value` equals the one value of `name` that means "unset"
/// (temperature 1, top_p 1, logprobs False), by Python's `==`: `True == 1`.
fn is_neutral(name: &str, value: &Value) -> bool {
    let number = |target: f64| match value {
        Value::Number(n) => n.as_f64() == Some(target),
        Value::Bool(flag) => f64::from(u8::from(*flag)) == target,
        _ => false,
    };
    match name {
        "temperature" | "top_p" => number(1.0),
        "logprobs" => number(0.0),
        _ => false,
    }
}

fn reject_unsupported(params: &Object) -> Result<()> {
    let named: Vec<&str> = UNSUPPORTED
        .into_iter()
        .filter(|name| {
            params
                .get(*name)
                .is_some_and(|value| !value.is_null() && !is_neutral(name, value))
        })
        .collect();
    if named.is_empty() {
        return Ok(());
    }
    Err(Error::request(format!(
        "unsupported parameters: {}",
        named.join(", ")
    )))
}

/// The Responses `text.format` item for a neutral output format.
fn output_format(format: &OutputFormat) -> Object {
    if format.kind == "json_object" {
        return obj! { "type": "json_object" };
    }
    obj! {
        "type": "json_schema",
        // Responses requires a label Messages never sends; the schema is what
        // constrains the model, so a placeholder costs the client nothing.
        "name": if format.name.is_empty() { "response" } else { &format.name },
        "schema": format.schema,
        "strict": format.strict,
    }
}

fn flush_content(items: &mut Vec<Value>, pending: &mut Vec<Value>, role: Role) {
    if !pending.is_empty() {
        items.push(Value::Object(
            obj! { "role": role.as_str(), "content": std::mem::take(pending) },
        ));
    }
}

fn thinking_item(block: &Thinking) -> Result<Object> {
    // The envelope's ValueError reaches the client as a request error.
    let bridged = unpack_thinking(&block.signature)
        .map_err(|error| Error::request(error.message().to_string()))?;
    let Some(bridged) = bridged else {
        return Err(Error::request(
            "Codex upstream cannot faithfully represent Anthropic signed thinking",
        ));
    };
    if block.text != bridged.thinking {
        return Err(Error::request("Codex reasoning thinking text was modified"));
    }
    Ok(bridged.item)
}

fn function_call(block: &ToolUse) -> Result<Object> {
    let value = arguments(&block.arguments, Error::request)?;
    // Keep existing wire bytes stable for replay and prompt caching.
    let encoded = match &block.arguments {
        Value::String(text) if !text.trim().is_empty() => text.clone(),
        _ => dumps(&Value::Object(value), Dumps::COMPACT.unicode()),
    };
    let mut item = obj! { "type": "function_call", "call_id": block.id };
    if !block.namespace.is_empty() {
        item.insert("namespace".into(), block.namespace.clone().into());
    }
    item.insert("name".into(), block.name.clone().into());
    item.insert("arguments".into(), encoded.into());
    Ok(item)
}

fn turn_items(turn: &Turn, cache: &ReasoningCache) -> Result<Vec<Value>> {
    let mut items = Vec::new();
    let mut pending = Vec::new();
    let has_reasoning = turn
        .blocks
        .iter()
        .any(|block| matches!(block, Block::Reasoning(_) | Block::Thinking(_)));
    let cached = if has_reasoning {
        Vec::new()
    } else {
        let ids: Vec<String> = turn
            .blocks
            .iter()
            .filter_map(|block| match block {
                Block::ToolUse(call) if !call.id.is_empty() => Some(call.id.clone()),
                _ => None,
            })
            .collect();
        cache.get(&ids)
    };
    let mut cache_inserted = false;
    for block in &turn.blocks {
        match block {
            Block::Text(Text { text, .. }) => {
                let kind = if turn.role == Role::Assistant {
                    "output_text"
                } else {
                    "input_text"
                };
                pending.push(Value::Object(obj! { "type": kind, "text": text }));
                continue;
            }
            Block::Image(image) => {
                if turn.role == Role::Assistant {
                    return Err(Error::request(
                        "unsupported assistant content type: image_url",
                    ));
                }
                pending.push(Value::Object(
                    obj! { "type": "input_image", "image_url": image.url },
                ));
                continue;
            }
            _ => {}
        }
        flush_content(&mut items, &mut pending, turn.role);
        match block {
            Block::Reasoning(item) | Block::NativeResponseItem(item) => {
                items.push(Value::Object(item.clone()));
            }
            Block::HostedSearch { item, source } => {
                if *source == Source::Responses {
                    items.push(Value::Object(item.clone()));
                }
            }
            Block::NativeAnthropicBlock(item) => {
                let kind = item.get("type").map_or_else(|| "unknown".into(), py_str);
                return Err(Error::request(format!(
                    "Codex upstream cannot faithfully represent Anthropic content block: {kind}"
                )));
            }
            Block::Thinking(thinking) => items.push(Value::Object(thinking_item(thinking)?)),
            Block::ToolUse(call) => {
                let item = function_call(call)?;
                if !cache_inserted {
                    items.extend(cached.iter().cloned());
                    cache_inserted = true;
                }
                items.push(Value::Object(item));
            }
            Block::ToolResult(result) => items.push(Value::Object(obj! {
                "type": "function_call_output",
                "call_id": result.tool_use_id,
                "output": result.text,
            })),
            Block::Text(_) | Block::Image(_) => {}
        }
    }
    flush_content(&mut items, &mut pending, turn.role);
    Ok(items)
}

/// What the opening user turn contributes to the fallback cache key.
fn user_text(turn: &Turn) -> String {
    turn.blocks
        .iter()
        .filter_map(|block| match block {
            Block::Text(text) => Some(text.text.as_str()),
            Block::Image(image) => Some(image.url.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The upstream body and the prompt-cache key it was given.
pub fn build(
    request: &ChatRequest,
    cache: &ReasoningCache,
    reasoning_efforts: Option<&[String]>,
) -> Result<(Object, String)> {
    if request.model.is_empty() {
        return Err(Error::request("model is required"));
    }
    reject_unsupported(&request.params)?;

    // Codex caches prefixes implicitly and refuses explicit breakpoints, so
    // cache hints are dropped; the cache key is what keeps a prefix warm.
    let instructions = request
        .system
        .iter()
        .map(|block| block.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut items = Vec::new();
    // The opening user turn seeds the fallback cache key, and empty text is a
    // legitimate value for an image-only turn: a sentinel that cannot tell
    // "empty" from "not seen yet" would re-seed the key from a later turn and
    // move the whole conversation to a different upstream cache mid-flight.
    let mut first_user: Option<String> = None;
    for turn in &request.turns {
        if turn.role == Role::User && first_user.is_none() {
            first_user = Some(user_text(turn));
        }
        items.extend(turn_items(turn, cache)?);
    }

    let mut cache_key = if request.cache_key.is_empty() {
        request.session.clone()
    } else {
        request.cache_key.clone()
    };
    if cache_key.is_empty() {
        let seed = format!("{instructions}\0{}", first_user.unwrap_or_default());
        let digest = Sha256::digest(seed.as_bytes());
        let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        cache_key = format!("proxy-{}", &hex[..24]);
    }

    let mut body = obj! {
        "model": request.model,
        "instructions": instructions,
        "input": items,
        "store": false,
        "stream": true,
        "prompt_cache_key": cache_key,
    };
    let tools = request
        .tools
        .iter()
        .map(responses_tool)
        .collect::<Result<Vec<_>>>()?;
    if !tools.is_empty() {
        body.insert(
            "tools".into(),
            tools.into_iter().map(Value::Object).collect(),
        );
        body.insert(
            "tool_choice".into(),
            responses_choice(request.tool_choice.as_ref()),
        );
        body.insert(
            "parallel_tool_calls".into(),
            (request.parallel_tool_calls != Some(false)).into(),
        );
    }
    if let Some(format) = &request.output_format {
        body.insert(
            "text".into(),
            Value::Object(obj! { "format": output_format(format) }),
        );
    }
    if !request.verbosity.is_empty() {
        setdefault_object(&mut body, "text")
            .insert("verbosity".into(), request.verbosity.clone().into());
    }
    if request.thinking_budget.is_some() {
        return Err(Error::request(
            "Codex upstream cannot faithfully represent an Anthropic thinking budget; \
             use output_config.effort",
        ));
    }
    if request.thinking_mode == "disabled" {
        return Err(Error::request(
            "Codex upstream cannot guarantee that reasoning is disabled",
        ));
    }
    let mut effort = String::new();
    if truthy(&request.reasoning_effort) {
        effort = py_str(&request.reasoning_effort).to_lowercase();
        let supported: HashSet<String> = reasoning_efforts
            .unwrap_or_default()
            .iter()
            .map(|item| item.to_lowercase())
            .collect();
        if !supported.is_empty() && !supported.contains(&effort) {
            return Err(Error::request(format!(
                "unsupported reasoning_effort: {}",
                py_str(&request.reasoning_effort)
            )));
        }
    }
    // A client that asks for reasoning, or to see it, gets its summaries; the
    // mode it named reaches Codex as named.
    let wants_summary = !effort.is_empty()
        || !request.reasoning_summary.is_empty()
        || request.thinking_display == "summarized"
        || request.thinking_mode == "adaptive";
    let mut summary = "";
    if request.thinking_display != "omitted" && request.reasoning_summary != "none" && wants_summary
    {
        summary = if request.reasoning_summary.is_empty() {
            "auto"
        } else {
            &request.reasoning_summary
        };
    }
    if !effort.is_empty() || !summary.is_empty() {
        let mut reasoning = Object::new();
        if !effort.is_empty() {
            reasoning.insert("effort".into(), effort.into());
        }
        if !summary.is_empty() {
            reasoning.insert("summary".into(), summary.into());
        }
        body.insert("reasoning".into(), Value::Object(reasoning));
    }
    if !request.reasoning_context.is_empty() {
        setdefault_object(&mut body, "reasoning")
            .insert("context".into(), request.reasoning_context.clone().into());
    }
    // Models can reason at their catalog default even when the client omits an
    // explicit effort, so always request the completed encrypted item.
    body.insert("include".into(), vec!["reasoning.encrypted_content"].into());
    let key = body
        .get("prompt_cache_key")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok((body, key))
}

/// `body.setdefault(key, {})`, as the object it holds.
fn setdefault_object<'a>(body: &'a mut Object, key: &str) -> &'a mut Object {
    let entry = body
        .entry(key.to_string())
        .or_insert_with(|| Value::Object(Object::new()));
    if !entry.is_object() {
        *entry = Value::Object(Object::new());
    }
    entry.as_object_mut().expect("just ensured to be an object")
}
