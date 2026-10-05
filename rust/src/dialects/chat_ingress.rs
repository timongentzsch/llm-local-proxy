//! Chat Completions request body -> [`ChatRequest`].
//!
//! Structural parsing only. Whether a parameter is *supported* depends on the
//! upstream that will serve it, so providers make that call; this module only
//! rejects bodies that are not valid Chat Completions at all.

use super::base::block_text;
use super::openai_output::{enum_value, format_of, VERBOSITY};
use crate::error::{Error, Result};
use crate::ir::{
    Block, ChatRequest, Image, OutputFormat, Role, Source, Text, Tool, ToolResult, ToolUse, Turn,
    WebSearchTool,
};
use crate::json::{py_str, truthy, Object};
use crate::tools::{definitions, optional_bool, parse_choice, parse_function};
use serde_json::Value;

const PARAMS: [&str; 10] = [
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "top_logprobs",
    "seed",
    "logit_bias",
    "stop",
];

/// Read `response_format`, whose schema sits one level deeper than Responses.
fn output_format(value: &Value) -> Result<Option<OutputFormat>> {
    let map = match value {
        Value::Null => return Ok(None),
        Value::Object(map) => map,
        _ => return Err(Error::request("response_format must be an object")),
    };
    let kind = map.get("type").unwrap_or(&Value::Null);
    let mut nested = Object::new();
    if kind.as_str() == Some("json_schema") {
        match map.get("json_schema") {
            Some(Value::Object(schema)) => nested = schema.clone(),
            _ => {
                return Err(Error::request(
                    "response_format.json_schema must be an object",
                ))
            }
        }
    }
    format_of(kind, &nested)
}

/// Flatten a content field that may be a string or a list of parts.
fn text(content: &Value) -> Result<String> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => block_text(parts),
        _ => Err(Error::request("message content must be a string or array")),
    }
}

fn content(value: &Value, role: &str) -> Result<Vec<Block>> {
    let parts = match value {
        Value::Null => return Ok(Vec::new()),
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
        match kind.as_str() {
            Some("text") => {
                let text = part.get("text").map(py_str).unwrap_or_default();
                blocks.push(Block::Text(Text::new(text)));
            }
            Some("image_url") => {
                let empty = Value::Object(Object::new());
                let image = part.get("image_url").unwrap_or(&empty);
                let url = match image {
                    Value::Object(map) => map.get("url").unwrap_or(&Value::Null),
                    other => other,
                };
                if truthy(url) {
                    blocks.push(Block::Image(Image {
                        url: py_str(url),
                        cache: None,
                    }));
                }
            }
            _ => {
                return Err(Error::request(format!(
                    "unsupported {role} content type: {}",
                    py_str(kind)
                )))
            }
        }
    }
    Ok(blocks)
}

fn tool_calls(message: &Object) -> Result<Vec<Block>> {
    let calls = match message.get("tool_calls") {
        None => return Ok(Vec::new()),
        Some(Value::Array(calls)) => calls,
        Some(_) => return Err(Error::request("tool_calls must be an array")),
    };
    let mut blocks = Vec::new();
    for call in calls {
        let function = call.get("function").and_then(Value::as_object);
        let Some(function) = function.filter(|f| f.get("name").is_some_and(truthy)) else {
            return Err(Error::request("invalid assistant tool call"));
        };
        let id = call.get("id").filter(|id| truthy(id)).map(py_str);
        blocks.push(Block::ToolUse(ToolUse {
            id: id.unwrap_or_default(),
            name: function.get("name").map(py_str).unwrap_or_default(),
            arguments: function
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| Value::String("{}".into())),
            ..ToolUse::default()
        }));
    }
    Ok(blocks)
}

/// Group consecutive tool results into one user turn.
///
/// Messages wants every tool_result of a turn in a single user message;
/// Responses lists them one item at a time, so grouping costs it nothing
/// and leaves one representation for both.
fn add_tool_result(turns: &mut Vec<Turn>, message: &Object) -> Result<()> {
    let tool_use_id = ["tool_call_id", "tool_use_id"]
        .iter()
        .filter_map(|key| message.get(*key))
        .find(|id| truthy(id));
    let Some(tool_use_id) = tool_use_id else {
        return Err(Error::request("tool message is missing tool_call_id"));
    };
    let block = Block::ToolResult(ToolResult {
        tool_use_id: py_str(tool_use_id),
        text: text(message.get("content").unwrap_or(&Value::Null))?,
        is_error: message.get("is_error").is_some_and(truthy),
        cache: None,
    });
    if let Some(last) = turns.last_mut() {
        if last.role == Role::User
            && last
                .blocks
                .iter()
                .all(|b| matches!(b, Block::ToolResult(_)))
        {
            last.blocks.push(block);
            return Ok(());
        }
    }
    turns.push(Turn {
        role: Role::User,
        blocks: vec![block],
    });
    Ok(())
}

/// OpenAI's Chat Completions search parameter, as the Responses search tool.
///
/// Both describe one search: the same context size, and the same approximate
/// location, which Chat Completions nests one level deeper.
fn web_search_options(value: &Value) -> Result<Vec<Tool>> {
    let options = match value {
        Value::Null => return Ok(Vec::new()),
        Value::Object(options) => options,
        _ => return Err(Error::request("web_search_options must be an object")),
    };
    let mut extra: Vec<&str> = options
        .keys()
        .map(String::as_str)
        .filter(|key| !["search_context_size", "user_location"].contains(key))
        .collect();
    extra.sort_unstable();
    if !extra.is_empty() {
        return Err(Error::request(format!(
            "unsupported web_search_options: {}",
            extra.join(", ")
        )));
    }
    let mut tool = crate::obj! { "type": "web_search" };
    if let Some(size) = options.get("search_context_size").filter(|v| !v.is_null()) {
        if !size
            .as_str()
            .is_some_and(|s| ["low", "medium", "high"].contains(&s))
        {
            return Err(Error::request(
                "web_search_options.search_context_size must be low, medium or high",
            ));
        }
        tool.insert("search_context_size".into(), size.clone());
    }
    if let Some(location) = options.get("user_location").filter(|v| !v.is_null()) {
        let detail = location
            .as_object()
            .filter(|l| l.get("type") == Some(&Value::String("approximate".into())))
            .and_then(|l| l.get("approximate"))
            .and_then(Value::as_object);
        let Some(detail) = detail else {
            return Err(Error::request(
                "web_search_options.user_location must be approximate",
            ));
        };
        let mut merged = crate::obj! { "type": "approximate" };
        merged.extend(detail.clone());
        tool.insert("user_location".into(), Value::Object(merged));
    }
    Ok(vec![Tool::WebSearch(WebSearchTool {
        native: tool,
        source: Source::Responses,
    })])
}

fn tools(value: &Value) -> Result<Vec<Tool>> {
    let mut tools = Vec::new();
    for item in definitions(value)? {
        let function = item.get("function").and_then(Value::as_object);
        let Some(function) = function.filter(|_| item.get("type") == Some(&"function".into()))
        else {
            return Err(Error::request(
                "only function tools are supported; request web search with web_search_options",
            ));
        };
        tools.push(Tool::Function(parse_function(
            function,
            "chat_completions",
            "parameters",
        )?));
    }
    Ok(tools)
}

/// Python's `value != 1`, where `1.0` and `True` equal 1.
fn is_one(value: &Value) -> bool {
    match value {
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64() == Some(1.0),
        _ => false,
    }
}

pub fn parse(body: &Object, session: &str) -> Result<ChatRequest> {
    let messages = match body.get("messages") {
        Some(Value::Array(messages)) if !messages.is_empty() => messages,
        _ => return Err(Error::request("messages must be a non-empty array")),
    };
    if body.get("n").is_some_and(|n| !is_one(n)) {
        return Err(Error::request("n must be 1"));
    }

    let mut system: Vec<String> = Vec::new();
    let mut turns: Vec<Turn> = Vec::new();
    for message in messages {
        let Value::Object(message) = message else {
            return Err(Error::request("each message must be an object"));
        };
        let Some(role) = message.get("role").and_then(Value::as_str) else {
            return Err(Error::request("message role must be a string"));
        };
        let body_content = message.get("content").unwrap_or(&Value::Null);
        match role {
            "system" | "developer" => {
                let text = text(body_content)?;
                if !text.is_empty() {
                    system.push(text);
                }
            }
            "user" => {
                let blocks = content(body_content, "user")?;
                if !blocks.is_empty() {
                    turns.push(Turn {
                        role: Role::User,
                        blocks,
                    });
                }
            }
            "assistant" => {
                let mut blocks = content(body_content, "assistant")?;
                blocks.extend(tool_calls(message)?);
                if !blocks.is_empty() {
                    turns.push(Turn {
                        role: Role::Assistant,
                        blocks,
                    });
                }
            }
            "tool" => add_tool_result(&mut turns, message)?,
            _ => return Err(Error::request(format!("unsupported message role: {role}"))),
        }
    }

    let null = Value::Null;
    let get = |key: &str| body.get(key).unwrap_or(&null);
    let cache_key = match get("prompt_cache_key") {
        key if truthy(key) => py_str(key),
        _ => String::new(),
    };
    let model = match body.get("model") {
        Some(Value::String(model)) => model.clone(),
        _ => String::new(),
    };
    let mut all_tools = tools(get("tools"))?;
    all_tools.extend(web_search_options(get("web_search_options"))?);
    let tool_choice = parse_choice(get("tool_choice"), true)?;
    let verbosity = enum_value(get("verbosity"), &VERBOSITY, "verbosity")?;
    let parallel_tool_calls = optional_bool(get("parallel_tool_calls"), "parallel_tool_calls")?;
    let stream = optional_bool(get("stream"), "stream")?.unwrap_or(false);
    let params: Object = PARAMS
        .iter()
        .filter_map(|name| body.get(*name).map(|v| (name.to_string(), v.clone())))
        .collect();
    let output_format = output_format(get("response_format"))?;
    Ok(ChatRequest {
        model,
        // One block: Chat Completions has no cache breakpoints to preserve, and
        // every system and developer turn is one prompt to the upstream.
        system: if system.is_empty() {
            Vec::new()
        } else {
            vec![Text::new(system.join("\n\n"))]
        },
        turns,
        tools: all_tools,
        tool_choice,
        max_tokens: body
            .get("max_tokens")
            .or_else(|| body.get("max_completion_tokens"))
            .cloned()
            .unwrap_or(Value::Null),
        reasoning_effort: get("reasoning_effort").clone(),
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
