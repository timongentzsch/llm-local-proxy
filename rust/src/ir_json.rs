//! The IR as JSON, in the shape the Python reference's dataclasses dump to.
//!
//! Only the conformance replay reads and writes this: it lets each layer --
//! ingress, request builders, decoders, encoders -- be checked against the
//! reference on its own, with the IR as the recorded hand-off between them.
//! Nothing on the request path depends on it.

use crate::ir::*;
use crate::json::Object;
use serde_json::{json, Value};

fn opt_str(value: &Option<String>) -> Value {
    value.as_ref().map(|v| json!(v)).unwrap_or(Value::Null)
}

fn opt_int(value: Option<i64>) -> Value {
    value.map(|v| json!(v)).unwrap_or(Value::Null)
}

fn text_json(text: &Text) -> Value {
    json!({
        "type": "Text",
        "text": text.text,
        "cache": opt_str(&text.cache),
        "citations": text.citations.clone().map(Value::Array).unwrap_or(Value::Null),
    })
}

fn block_json(block: &Block) -> Value {
    match block {
        Block::Text(text) => text_json(text),
        Block::Image(image) => {
            json!({"type": "Image", "url": image.url, "cache": opt_str(&image.cache)})
        }
        Block::ToolUse(call) => json!({
            "type": "ToolUse",
            "id": call.id,
            "name": call.name,
            "arguments": call.arguments,
            "namespace": call.namespace,
            "cache": opt_str(&call.cache),
        }),
        Block::ToolResult(result) => json!({
            "type": "ToolResult",
            "tool_use_id": result.tool_use_id,
            "text": result.text,
            "is_error": result.is_error,
            "cache": opt_str(&result.cache),
        }),
        Block::Thinking(thinking) => json!({
            "type": "Thinking",
            "text": thinking.text,
            "signature": thinking.signature,
            "redacted": thinking.redacted,
        }),
        Block::Reasoning(item) => json!({"type": "Reasoning", "item": item}),
        Block::NativeResponseItem(item) => json!({"type": "NativeResponseItem", "item": item}),
        Block::NativeAnthropicBlock(item) => {
            json!({"type": "NativeAnthropicBlock", "item": item})
        }
        Block::HostedSearch { item, source } => {
            json!({"type": "HostedSearch", "item": item, "source": source.as_str()})
        }
    }
}

fn function_json(tool: &FunctionTool) -> Value {
    json!({
        "type": "FunctionTool",
        "name": tool.name,
        "parameters": tool.parameters,
        "description": tool.description,
        "strict": tool.strict,
        "source": tool.source,
        "options": tool.options,
        "cache": opt_str(&tool.cache),
    })
}

fn tool_json(tool: &Tool) -> Value {
    match tool {
        Tool::Function(function) => function_json(function),
        Tool::WebSearch(search) => json!({
            "type": "WebSearchTool",
            "native": search.native,
            "source": search.source.as_str(),
        }),
        Tool::Native(item) => json!({"type": "NativeTool", "item": item}),
        Tool::Namespace(namespace) => json!({
            "type": "ToolNamespace",
            "name": namespace.name,
            "tools": namespace.tools.iter().map(|member| match member {
                NamespaceMember::Function(function) => function_json(function),
                NamespaceMember::Native(item) => json!({"type": "NativeTool", "item": item}),
            }).collect::<Vec<_>>(),
            "item": namespace.item,
        }),
    }
}

pub fn request_json(request: &ChatRequest) -> Value {
    json!({
        "type": "ChatRequest",
        "model": request.model,
        "system": request.system.iter().map(text_json).collect::<Vec<_>>(),
        "turns": request.turns.iter().map(|turn| json!({
            "type": "Turn",
            "role": turn.role.as_str(),
            "blocks": turn.blocks.iter().map(block_json).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "tools": request.tools.iter().map(tool_json).collect::<Vec<_>>(),
        "tool_choice": request.tool_choice.as_ref().map(|choice| json!({
            "type": "ToolChoice",
            "kind": choice.kind.as_str(),
            "name": choice.name,
        })),
        "max_tokens": request.max_tokens,
        "reasoning_effort": request.reasoning_effort,
        "thinking_budget": opt_int(request.thinking_budget),
        "thinking_mode": request.thinking_mode,
        "thinking_display": request.thinking_display,
        "reasoning_summary": request.reasoning_summary,
        "reasoning_context": request.reasoning_context,
        "verbosity": request.verbosity,
        "parallel_tool_calls": request.parallel_tool_calls,
        "stream": request.stream,
        "session": request.session,
        "caller": request.caller,
        "cache_key": request.cache_key,
        "cache": opt_str(&request.cache),
        "params": request.params,
        "output_format": request.output_format.as_ref().map(|format| json!({
            "type": "OutputFormat",
            "kind": format.kind,
            "name": format.name,
            "schema": format.schema,
            "strict": format.strict,
        })),
    })
}

pub fn event_json(event: &StreamEvent) -> Value {
    match event {
        StreamEvent::TextDelta { text, span } => {
            json!({"type": "TextDelta", "text": text, "span": span})
        }
        StreamEvent::ThinkingDelta { text, item_id } => {
            json!({"type": "ThinkingDelta", "text": text, "item_id": item_id})
        }
        StreamEvent::ThinkingSignature { signature } => {
            json!({"type": "ThinkingSignature", "signature": signature})
        }
        StreamEvent::RedactedThinkingDelta { data } => {
            json!({"type": "RedactedThinkingDelta", "data": data})
        }
        StreamEvent::ReasoningItem { item } => json!({"type": "ReasoningItem", "item": item}),
        StreamEvent::NativeItem { item } => json!({"type": "NativeItem", "item": item}),
        StreamEvent::ToolCallStart {
            index,
            id,
            name,
            arguments,
            namespace,
        } => json!({
            "type": "ToolCallStart",
            "index": index,
            "id": id,
            "name": name,
            "arguments": arguments,
            "namespace": namespace,
        }),
        StreamEvent::ToolCallArgs { index, fragment } => {
            json!({"type": "ToolCallArgs", "index": index, "fragment": fragment})
        }
        StreamEvent::ToolCallEnd {
            index,
            id,
            name,
            arguments,
            namespace,
        } => json!({
            "type": "ToolCallEnd",
            "index": index,
            "id": id,
            "name": name,
            "arguments": arguments,
            "namespace": namespace,
        }),
        StreamEvent::HostedTool(event) => json!({
            "type": "HostedToolEvent",
            "tool": event.tool,
            "id": event.id,
            "phase": event.phase,
            "query": event.query,
            "error_code": event.error_code,
            "result": event.result,
        }),
        StreamEvent::Citation(citation) => json!({
            "type": "Citation",
            "url": citation.url,
            "title": citation.title,
            "start_index": citation.start_index,
            "end_index": citation.end_index,
            "native": citation.native,
            "span": citation.span,
        }),
        StreamEvent::Usage(usage) => json!({
            "type": "Usage",
            "prompt": usage.prompt,
            "completion": usage.completion,
            "total": opt_int(usage.total),
            "cache_read": usage.cache_read,
            "cache_write": usage.cache_write,
            "cache_write_1h": opt_int(usage.cache_write_1h),
            "thinking": usage.thinking,
            "web_searches": usage.web_searches,
        }),
        StreamEvent::Finish(finish) => json!({
            "type": "Finish",
            "reason": finish.reason,
            "incomplete_reason": opt_str(&finish.incomplete_reason),
            "stop_sequence": opt_str(&finish.stop_sequence),
        }),
    }
}

// -- reading -------------------------------------------------------------------

fn s(value: &Value, key: &str) -> String {
    value[key].as_str().unwrap_or_default().to_string()
}

fn os(value: &Value, key: &str) -> Option<String> {
    value[key].as_str().map(str::to_string)
}

fn o(value: &Value, key: &str) -> Object {
    value[key].as_object().cloned().unwrap_or_default()
}

fn source(value: &Value) -> Source {
    match value["source"].as_str() {
        Some("anthropic") => Source::Anthropic,
        _ => Source::Responses,
    }
}

fn text_from(value: &Value) -> Text {
    Text {
        text: s(value, "text"),
        cache: os(value, "cache"),
        citations: value["citations"].as_array().cloned(),
    }
}

fn block_from(value: &Value) -> Block {
    match value["type"].as_str().unwrap_or_default() {
        "Text" => Block::Text(text_from(value)),
        "Image" => Block::Image(Image {
            url: s(value, "url"),
            cache: os(value, "cache"),
        }),
        "ToolUse" => Block::ToolUse(ToolUse {
            id: s(value, "id"),
            name: s(value, "name"),
            arguments: value["arguments"].clone(),
            namespace: s(value, "namespace"),
            cache: os(value, "cache"),
        }),
        "ToolResult" => Block::ToolResult(ToolResult {
            tool_use_id: s(value, "tool_use_id"),
            text: s(value, "text"),
            is_error: value["is_error"].as_bool().unwrap_or(false),
            cache: os(value, "cache"),
        }),
        "Thinking" => Block::Thinking(Thinking {
            text: s(value, "text"),
            signature: s(value, "signature"),
            redacted: s(value, "redacted"),
        }),
        "Reasoning" => Block::Reasoning(o(value, "item")),
        "NativeResponseItem" => Block::NativeResponseItem(o(value, "item")),
        "NativeAnthropicBlock" => Block::NativeAnthropicBlock(o(value, "item")),
        "HostedSearch" => Block::HostedSearch {
            item: o(value, "item"),
            source: source(value),
        },
        other => panic!("unknown block type {other}"),
    }
}

fn function_from(value: &Value) -> FunctionTool {
    FunctionTool {
        name: s(value, "name"),
        parameters: o(value, "parameters"),
        description: s(value, "description"),
        strict: value["strict"].as_bool(),
        source: s(value, "source"),
        options: o(value, "options"),
        cache: os(value, "cache"),
    }
}

fn tool_from(value: &Value) -> Tool {
    match value["type"].as_str().unwrap_or_default() {
        "FunctionTool" => Tool::Function(function_from(value)),
        "WebSearchTool" => Tool::WebSearch(WebSearchTool {
            native: o(value, "native"),
            source: source(value),
        }),
        "NativeTool" => Tool::Native(o(value, "item")),
        "ToolNamespace" => Tool::Namespace(ToolNamespace {
            name: s(value, "name"),
            tools: value["tools"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|member| match member["type"].as_str() {
                    Some("FunctionTool") => NamespaceMember::Function(function_from(member)),
                    _ => NamespaceMember::Native(o(member, "item")),
                })
                .collect(),
            item: o(value, "item"),
        }),
        other => panic!("unknown tool type {other}"),
    }
}

pub fn request_from(value: &Value) -> ChatRequest {
    let list = |key: &str| value[key].as_array().cloned().unwrap_or_default();
    ChatRequest {
        model: s(value, "model"),
        system: list("system").iter().map(text_from).collect(),
        turns: list("turns")
            .iter()
            .map(|turn| Turn {
                role: match turn["role"].as_str() {
                    Some("assistant") => Role::Assistant,
                    _ => Role::User,
                },
                blocks: turn["blocks"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .map(block_from)
                    .collect(),
            })
            .collect(),
        tools: list("tools").iter().map(tool_from).collect(),
        tool_choice: value["tool_choice"].as_object().map(|choice| ToolChoice {
            kind: match choice["kind"].as_str() {
                Some("none") => ChoiceKind::None,
                Some("required") => ChoiceKind::Required,
                Some("tool") => ChoiceKind::Tool,
                _ => ChoiceKind::Auto,
            },
            name: choice["name"].as_str().unwrap_or_default().to_string(),
        }),
        max_tokens: value["max_tokens"].clone(),
        reasoning_effort: value["reasoning_effort"].clone(),
        thinking_budget: value["thinking_budget"].as_i64(),
        thinking_mode: s(value, "thinking_mode"),
        thinking_display: s(value, "thinking_display"),
        reasoning_summary: s(value, "reasoning_summary"),
        reasoning_context: s(value, "reasoning_context"),
        verbosity: s(value, "verbosity"),
        parallel_tool_calls: value["parallel_tool_calls"].as_bool(),
        stream: value["stream"].as_bool().unwrap_or(false),
        session: s(value, "session"),
        caller: s(value, "caller"),
        cache_key: s(value, "cache_key"),
        cache: os(value, "cache"),
        params: o(value, "params"),
        output_format: value["output_format"]
            .as_object()
            .map(|format| OutputFormat {
                kind: format["kind"].as_str().unwrap_or_default().to_string(),
                name: format["name"].as_str().unwrap_or_default().to_string(),
                schema: format["schema"].as_object().cloned(),
                strict: format["strict"].as_bool().unwrap_or(false),
            }),
    }
}

pub fn event_from(value: &Value) -> StreamEvent {
    match value["type"].as_str().unwrap_or_default() {
        "TextDelta" => StreamEvent::TextDelta {
            text: s(value, "text"),
            span: s(value, "span"),
        },
        "ThinkingDelta" => StreamEvent::ThinkingDelta {
            text: s(value, "text"),
            item_id: s(value, "item_id"),
        },
        "ThinkingSignature" => StreamEvent::ThinkingSignature {
            signature: s(value, "signature"),
        },
        "RedactedThinkingDelta" => StreamEvent::RedactedThinkingDelta {
            data: s(value, "data"),
        },
        "ReasoningItem" => StreamEvent::ReasoningItem {
            item: o(value, "item"),
        },
        "NativeItem" => StreamEvent::NativeItem {
            item: o(value, "item"),
        },
        "ToolCallStart" => StreamEvent::ToolCallStart {
            index: value["index"].clone(),
            id: s(value, "id"),
            name: s(value, "name"),
            arguments: s(value, "arguments"),
            namespace: s(value, "namespace"),
        },
        "ToolCallArgs" => StreamEvent::ToolCallArgs {
            index: value["index"].clone(),
            fragment: s(value, "fragment"),
        },
        "ToolCallEnd" => StreamEvent::ToolCallEnd {
            index: value["index"].clone(),
            id: s(value, "id"),
            name: s(value, "name"),
            arguments: s(value, "arguments"),
            namespace: s(value, "namespace"),
        },
        "HostedToolEvent" => StreamEvent::HostedTool(HostedToolEvent {
            tool: s(value, "tool"),
            id: s(value, "id"),
            phase: s(value, "phase"),
            query: s(value, "query"),
            error_code: s(value, "error_code"),
            result: match &value["result"] {
                Value::Null => None,
                other => Some(other.clone()),
            },
        }),
        "Citation" => StreamEvent::Citation(Citation {
            url: s(value, "url"),
            title: value["title"].clone(),
            start_index: value["start_index"].clone(),
            end_index: value["end_index"].clone(),
            native: value["native"].as_object().cloned(),
            span: s(value, "span"),
        }),
        "Usage" => StreamEvent::Usage(Usage {
            prompt: value["prompt"].as_i64().unwrap_or(0),
            completion: value["completion"].as_i64().unwrap_or(0),
            total: value["total"].as_i64(),
            cache_read: value["cache_read"].as_i64().unwrap_or(0),
            cache_write: value["cache_write"].as_i64().unwrap_or(0),
            cache_write_1h: value["cache_write_1h"].as_i64(),
            thinking: value["thinking"].as_i64().unwrap_or(0),
            web_searches: value["web_searches"].as_i64().unwrap_or(0),
        }),
        "Finish" => StreamEvent::Finish(Finish {
            reason: s(value, "reason"),
            incomplete_reason: os(value, "incomplete_reason"),
            stop_sequence: os(value, "stop_sequence"),
        }),
        other => panic!("unknown event type {other}"),
    }
}
