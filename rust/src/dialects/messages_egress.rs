//! Canonical stream events -> Anthropic Messages frames and messages.
//!
//! Two constraints from specs/anthropic-openapi.json shape this file. Exactly
//! one content block may be open at a time, under monotonically increasing
//! indices, so the encoder is a small state machine that closes the open block
//! whenever the kind changes. And message_start carries a Message whose
//! usage.input_tokens is non-nullable, while an upstream may report no input
//! count until it ends: the opening frame therefore claims zero and the
//! authoritative totals arrive in message_delta.

use crate::dialects::base::{Driver, Encoder};
use crate::error::{Error, Result};
use crate::ids::{self, SharedIds};
use crate::ir::{Citation, Decoder, HostedToolEvent, StreamEvent, Usage};
use crate::json::Object;
use crate::obj;
use crate::tools::arguments;
use serde_json::{json, Value};
use std::collections::HashMap;

#[derive(PartialEq, Clone, Copy)]
enum Kind {
    Text,
    Thinking,
    RedactedThinking,
    ToolUse,
    ServerToolUse,
    WebSearchToolResult,
}

/// The content block being written, and what its closing needs.
struct OpenBlock {
    kind: Kind,
    block: Object,
    /// Tool arguments accumulated from fragments.
    json: String,
    /// The upstream text block a text block mirrors.
    span: String,
}

impl OpenBlock {
    fn set_text(&mut self, key: &str, more: &str) {
        if let Some(Value::String(text)) = self.block.get_mut(key) {
            text.push_str(more);
        }
    }
}

/// Turns one provider's decoded stream into Anthropic Messages output.
pub struct MessageEncoder {
    driver: Driver,
    id: String,
    model: String,
    blocks: Vec<Value>,
    usage: Option<Usage>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    index: i64,
    searches: HashMap<String, String>,
    open: Option<OpenBlock>,
}

impl MessageEncoder {
    pub fn new(model: &str, decoder: Box<dyn Decoder>, ids: &SharedIds) -> Self {
        MessageEncoder {
            driver: Driver::new(decoder),
            id: format!("msg_{}", ids::hex24(&**ids)),
            model: model.to_string(),
            blocks: Vec::new(),
            usage: None,
            stop_reason: None,
            stop_sequence: None,
            index: -1,
            searches: HashMap::new(),
            open: None,
        }
    }

    // -- block state machine ---------------------------------------------

    fn open_block(&mut self, kind: Kind, block: Object) -> Result<Vec<Value>> {
        let mut frames = self.close()?;
        self.index += 1;
        // A copy: the retained block keeps accumulating, and the opening
        // frame must show the block as it was at the start.
        frames.push(json!({
            "type": "content_block_start",
            "index": self.index,
            "content_block": block.clone(),
        }));
        self.open = Some(OpenBlock {
            kind,
            block,
            json: String::new(),
            span: String::new(),
        });
        Ok(frames)
    }

    fn close(&mut self) -> Result<Vec<Value>> {
        let Some(mut open) = self.open.take() else {
            return Ok(Vec::new());
        };
        if open.kind == Kind::ToolUse {
            let input = arguments(&Value::String(open.json.clone()), Error::upstream);
            match input {
                Ok(input) => {
                    open.block.insert("input".into(), Value::Object(input));
                }
                Err(error) => {
                    self.open = Some(open);
                    return Err(error);
                }
            }
        }
        self.blocks.push(Value::Object(open.block));
        Ok(vec![
            json!({"type": "content_block_stop", "index": self.index}),
        ])
    }

    fn delta(&self, delta: Value) -> Value {
        json!({"type": "content_block_delta", "index": self.index, "delta": delta})
    }

    fn open_mut(&mut self) -> Result<&mut OpenBlock> {
        self.open
            .as_mut()
            .ok_or_else(|| Error::upstream("no content block is open"))
    }

    fn hosted(&mut self, event: HostedToolEvent) -> Result<Vec<Value>> {
        // A provider-run search as Anthropic's own server-tool blocks.
        //
        // Never `tool_use`: that block obliges the client to run the search and
        // return a result, and this one has already run upstream. `stop()`
        // matches `tool_use` exactly, so `stop_reason` is unaffected.
        let mut frames = Vec::new();
        if !self.searches.contains_key(&event.id) || !event.query.is_empty() {
            self.searches.insert(event.id.clone(), event.query.clone());
        }
        if matches!(event.phase.as_str(), "completed" | "failed") {
            // Do not expose an orphaned server_tool_use when the model switches
            // to a client tool call before its hosted search finishes. Once
            // written into client history Anthropic requires the matching
            // result, so emit the native pair atomically at the terminal step.
            let query = self.searches.get(&event.id).cloned().unwrap_or_default();
            let input = if query.is_empty() {
                json!({})
            } else {
                json!({"query": query})
            };
            frames.extend(self.open_block(
                Kind::ServerToolUse,
                obj! {
                    "type": "server_tool_use",
                    "id": event.id,
                    "name": "web_search",
                    "input": input,
                },
            )?);
            // The result block is mandatory history for every server-tool use.
            // Successful source records are not available across every lane;
            // failed native searches retain their upstream error code.
            let mut content = event.result.unwrap_or_else(|| json!([]));
            if event.phase == "failed" && !content.is_object() {
                let code = if event.error_code.is_empty() {
                    "unavailable"
                } else {
                    &event.error_code
                };
                content = json!({"type": "web_search_tool_result_error", "error_code": code});
            }
            frames.extend(self.open_block(
                Kind::WebSearchToolResult,
                obj! {
                    "type": "web_search_tool_result",
                    "tool_use_id": event.id,
                    "content": content,
                },
            )?);
            frames.extend(self.close()?);
        }
        Ok(frames)
    }

    /// The text block for `span`; a new span is a new block, as upstream.
    fn text(&mut self, span: &str) -> Result<Vec<Value>> {
        if let Some(open) = &self.open {
            if open.kind == Kind::Text && open.span == span {
                return Ok(Vec::new());
            }
        }
        let frames = self.open_block(Kind::Text, obj! {"type": "text", "text": ""})?;
        self.open_mut()?.span = span.to_string();
        Ok(frames)
    }

    fn ensure_thinking(&mut self) -> Result<Vec<Value>> {
        if matches!(&self.open, Some(open) if open.kind == Kind::Thinking) {
            return Ok(Vec::new());
        }
        self.open_block(
            Kind::Thinking,
            obj! {"type": "thinking", "thinking": "", "signature": ""},
        )
    }

    fn citation_frames(&mut self, event: Citation) -> Result<Vec<Value>> {
        // A citation can open its block before the text it cites arrives.
        let mut frames = if !event.span.is_empty() {
            self.text(&event.span)?
        } else if !matches!(&self.open, Some(open) if open.kind == Kind::Text) {
            return Ok(Vec::new());
        } else {
            Vec::new()
        };
        let citation = citation(&event);
        let open = self.open_mut()?;
        let slot = open
            .block
            .entry("citations")
            .or_insert_with(|| Value::Array(Vec::new()));
        let Value::Array(citations) = slot else {
            return Ok(frames);
        };
        if citations.contains(&citation) {
            return Ok(frames);
        }
        citations.push(citation.clone());
        frames.push(self.delta(json!({"type": "citations_delta", "citation": citation})));
        Ok(frames)
    }

    // -- message assembly -------------------------------------------------

    fn stop(&self) -> String {
        match &self.stop_reason {
            Some(reason) if !reason.is_empty() => reason.clone(),
            _ => {
                let used = self
                    .blocks
                    .iter()
                    .any(|block| block.get("type") == Some(&json!("tool_use")));
                if used { "tool_use" } else { "end_turn" }.to_string()
            }
        }
    }

    fn message(&self, streaming: bool) -> Value {
        // Every field below is required by the schema; the nullable ones must
        // still be present, so a client SDK can read them unconditionally.
        let usage = if streaming { None } else { self.usage.as_ref() };
        json!({
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": if streaming { Vec::new() } else { self.blocks.clone() },
            "stop_reason": if streaming { Value::Null } else { json!(self.stop()) },
            "stop_sequence": if streaming { Value::Null } else { json!(self.stop_sequence) },
            "stop_details": null,
            "container": null,
            "usage": usage_json(usage, true),
        })
    }
}

impl Encoder for MessageEncoder {
    fn driver(&mut self) -> &mut Driver {
        &mut self.driver
    }

    fn one(&mut self, event: StreamEvent) -> Result<Vec<Value>> {
        match event {
            StreamEvent::NativeItem { .. } => Err(Error::provider(
                502,
                "native Responses output requires the Responses endpoint",
            )),
            StreamEvent::TextDelta { text, span } => {
                let mut frames = self.text(&span)?;
                self.open_mut()?.set_text("text", &text);
                frames.push(self.delta(json!({"type": "text_delta", "text": text})));
                Ok(frames)
            }
            StreamEvent::ThinkingDelta { text, .. } => {
                let mut frames = self.ensure_thinking()?;
                self.open_mut()?.set_text("thinking", &text);
                frames.push(self.delta(json!({"type": "thinking_delta", "thinking": text})));
                Ok(frames)
            }
            StreamEvent::ThinkingSignature { signature } => {
                let mut frames = self.ensure_thinking()?;
                self.open_mut()?
                    .block
                    .insert("signature".into(), json!(signature));
                frames.push(self.delta(json!({"type": "signature_delta", "signature": signature})));
                // The signature ends its block: thinking that follows is a new
                // block with its own signature, never appended to this one.
                frames.extend(self.close()?);
                Ok(frames)
            }
            StreamEvent::RedactedThinkingDelta { data } => {
                let mut frames = self.open_block(
                    Kind::RedactedThinking,
                    obj! {"type": "redacted_thinking", "data": data},
                )?;
                frames.extend(self.close()?);
                Ok(frames)
            }
            StreamEvent::ToolCallStart {
                id,
                name,
                arguments,
                ..
            } => {
                let mut frames = self.open_block(
                    Kind::ToolUse,
                    obj! {"type": "tool_use", "id": id, "name": name, "input": {}},
                )?;
                if !arguments.is_empty() {
                    // A provider may hand over a complete call; Anthropic clients
                    // still expect the arguments to arrive as a delta.
                    self.open_mut()?.json.push_str(&arguments);
                    frames.push(
                        self.delta(json!({"type": "input_json_delta", "partial_json": arguments})),
                    );
                }
                Ok(frames)
            }
            StreamEvent::ToolCallArgs { fragment, .. } => {
                match &mut self.open {
                    Some(open) if open.kind == Kind::ToolUse => open.json.push_str(&fragment),
                    _ => return Ok(Vec::new()),
                }
                Ok(vec![self.delta(
                    json!({"type": "input_json_delta", "partial_json": fragment}),
                )])
            }
            StreamEvent::ToolCallEnd { .. } => self.close(),
            StreamEvent::HostedTool(event) => self.hosted(event),
            StreamEvent::Citation(event) => self.citation_frames(event),
            StreamEvent::Usage(event) => {
                self.usage = Some(event);
                Ok(Vec::new())
            }
            StreamEvent::Finish(event) => {
                self.stop_reason = Some(event.reason);
                self.stop_sequence = event.stop_sequence;
                Ok(Vec::new())
            }
            _ => Ok(Vec::new()),
        }
    }

    fn start(&mut self) -> Value {
        json!({"type": "message_start", "message": self.message(true)})
    }

    fn finish(&mut self) -> Result<Vec<Value>> {
        let mut frames = self.drain()?;
        frames.extend(self.close()?);
        frames.push(json!({
            "type": "message_delta",
            "delta": {"stop_reason": self.stop(), "stop_sequence": self.stop_sequence},
            "usage": usage_json(self.usage.as_ref(), false),
        }));
        frames.push(json!({"type": "message_stop"}));
        Ok(frames)
    }

    fn result(&mut self) -> Result<Value> {
        self.drain()?;
        self.close()?;
        Ok(self.message(false))
    }

    fn set_id(&mut self, id: String) {
        self.id = id;
    }
}

fn citation(event: &Citation) -> Value {
    if let Some(native) = &event.native {
        return Value::Object(native.clone());
    }
    let mut citation = obj! {"type": "web_search_result_location", "url": event.url};
    if !event.title.is_null() {
        citation.insert("title".into(), event.title.clone());
    }
    Value::Object(citation)
}

fn usage_json(usage: Option<&Usage>, full: bool) -> Value {
    let cache_read = usage.map_or(0, |u| u.cache_read);
    let cache_write = usage.map_or(0, |u| u.cache_write);
    // The canonical prompt count includes cache; Anthropic reports the three
    // apart, so back the plain input tokens out of the total.
    let mut result = obj! {
        "input_tokens": (usage.map_or(0, |u| u.prompt) - cache_read - cache_write).max(0),
        "output_tokens": usage.map_or(0, |u| u.completion),
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    };
    if let Some(usage) = usage {
        if usage.thinking != 0 {
            result.insert(
                "output_tokens_details".into(),
                json!({"thinking_tokens": usage.thinking}),
            );
        }
        if usage.web_searches != 0 {
            result.insert(
                "server_tool_use".into(),
                json!({"web_search_requests": usage.web_searches}),
            );
        }
        // Only a whole Message reports cache writes by TTL; a delta cannot.
        if let (true, Some(one_hour)) = (full, usage.cache_write_1h) {
            result.insert(
                "cache_creation".into(),
                json!({
                    "ephemeral_5m_input_tokens": (cache_write - one_hour).max(0),
                    "ephemeral_1h_input_tokens": one_hour,
                }),
            );
        }
    }
    if full {
        result.entry("cache_creation").or_insert(Value::Null);
        result.entry("service_tier").or_insert(Value::Null);
    }
    Value::Object(result)
}
