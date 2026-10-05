//! Canonical stream events -> Chat Completions chunks and completions.

use crate::dialects::base::{Driver, Encoder};
use crate::error::{Error, Result};
use crate::ids::{self, SharedIds};
use crate::ir::{Citation, Decoder, StreamEvent, Usage};
use crate::obj;
use serde_json::{json, Value};

/// Anthropic's seven stop reasons narrowed onto Chat Completions' four.
fn finish_reason(reason: &str) -> &'static str {
    match reason {
        "end_turn" | "stop_sequence" | "pause_turn" => "stop",
        "tool_use" => "tool_calls",
        "max_tokens" | "model_context_window_exceeded" => "length",
        "refusal" => "content_filter",
        _ => "stop",
    }
}

/// Turns one provider's decoded stream into Chat Completions output.
///
/// Holds only wire shaping; upstream specifics live in the decoder.
pub struct ChunkEncoder {
    driver: Driver,
    id: String,
    created: i64,
    model: String,
    content: String,
    reasoning: String,
    calls: Vec<Value>,
    /// Upstream call index -> its position among this response's calls.
    /// Clients accumulate deltas by a `tool_calls` index counted from 0,
    /// while an upstream may number calls among its other content.
    positions: Vec<(Value, usize)>,
    annotations: Vec<Value>,
    usage: Option<Value>,
    finish: Option<String>,
}

impl ChunkEncoder {
    /// `now` is the creation time in whole seconds since the epoch.
    pub fn new(model: &str, decoder: Box<dyn Decoder>, ids: &SharedIds, now: i64) -> Self {
        ChunkEncoder {
            driver: Driver::new(decoder),
            id: format!("chatcmpl-{}", ids::hex(&**ids)),
            created: now,
            model: model.to_string(),
            content: String::new(),
            reasoning: String::new(),
            calls: Vec::new(),
            positions: Vec::new(),
            annotations: Vec::new(),
            usage: None,
            finish: None,
        }
    }

    fn chunk(&self, delta: Value, finish: Option<&str>) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        })
    }

    fn finish_reason(&self) -> String {
        match &self.finish {
            Some(reason) => reason.clone(),
            None if !self.calls.is_empty() => "tool_calls".into(),
            None => "stop".into(),
        }
    }

    fn position(&self, index: &Value) -> Option<usize> {
        self.positions
            .iter()
            .find(|(key, _)| key == index)
            .map(|(_, position)| *position)
    }

    fn citation(&mut self, event: Citation) -> Vec<Value> {
        if event.url.is_empty() {
            return Vec::new();
        }
        let mut fields = serde_json::Map::new();
        fields.insert("url".into(), Value::String(event.url));
        for (key, value) in [
            ("title", event.title),
            ("start_index", event.start_index),
            ("end_index", event.end_index),
        ] {
            if !value.is_null() {
                fields.insert(key.into(), value);
            }
        }
        let annotation = json!({"type": "url_citation", "url_citation": fields});
        if self.annotations.contains(&annotation) {
            return Vec::new();
        }
        self.annotations.push(annotation.clone());
        vec![self.chunk(json!({"annotations": [annotation]}), None)]
    }
}

impl Encoder for ChunkEncoder {
    fn driver(&mut self) -> &mut Driver {
        &mut self.driver
    }

    fn one(&mut self, event: StreamEvent) -> Result<Vec<Value>> {
        match event {
            StreamEvent::NativeItem { .. } => Err(Error::provider(
                502,
                "native Responses output requires the Responses endpoint",
            )),
            StreamEvent::TextDelta { text, .. } => {
                self.content.push_str(&text);
                Ok(vec![self.chunk(json!({"content": text}), None)])
            }
            StreamEvent::ThinkingDelta { text, .. } => {
                self.reasoning.push_str(&text);
                Ok(vec![self.chunk(json!({"reasoning_content": text}), None)])
            }
            StreamEvent::ToolCallStart {
                index,
                id,
                name,
                arguments,
                ..
            } => {
                let position = match self.position(&index) {
                    Some(position) => position,
                    None => {
                        let position = self.positions.len();
                        self.positions.push((index, position));
                        position
                    }
                };
                Ok(vec![self.chunk(
                    json!({"tool_calls": [{
                        "index": position,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }]}),
                    None,
                )])
            }
            StreamEvent::ToolCallArgs { index, fragment } => {
                let position = self.position(&index).unwrap_or(0);
                Ok(vec![self.chunk(
                    json!({"tool_calls": [{
                        "index": position,
                        "function": {"arguments": fragment},
                    }]}),
                    None,
                )])
            }
            StreamEvent::ToolCallEnd {
                id,
                name,
                arguments,
                ..
            } => {
                self.calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }));
                Ok(Vec::new())
            }
            StreamEvent::Citation(event) => Ok(self.citation(event)),
            StreamEvent::Usage(event) => {
                self.usage = Some(usage(&event));
                Ok(Vec::new())
            }
            StreamEvent::Finish(event) => {
                self.finish = Some(finish_reason(&event.reason).to_string());
                Ok(Vec::new())
            }
            // Signed reasoning and hosted tool lifecycles have no Chat representation.
            _ => Ok(Vec::new()),
        }
    }

    fn start(&mut self) -> Value {
        self.chunk(json!({"role": "assistant", "content": ""}), None)
    }

    fn finish(&mut self) -> Result<Vec<Value>> {
        let mut chunks = self.drain()?;
        chunks.push(self.chunk(json!({}), Some(&self.finish_reason())));
        if let Some(usage) = &self.usage {
            chunks.push(json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage,
            }));
        }
        Ok(chunks)
    }

    fn result(&mut self) -> Result<Value> {
        self.drain()?;
        let mut message = obj! {
            "role": "assistant",
            "content": if self.content.is_empty() { Value::Null } else { json!(self.content) },
        };
        if !self.calls.is_empty() {
            message.insert("tool_calls".into(), Value::Array(self.calls.clone()));
        }
        if !self.annotations.is_empty() {
            message.insert("annotations".into(), Value::Array(self.annotations.clone()));
        }
        if !self.reasoning.is_empty() {
            message.insert("reasoning_content".into(), json!(self.reasoning));
        }
        Ok(json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": self.finish_reason(),
            }],
            "usage": self.usage,
        }))
    }

    fn set_id(&mut self, id: String) {
        self.id = id;
    }

    fn set_created(&mut self, created: i64) {
        self.created = created;
    }
}

fn usage(event: &Usage) -> Value {
    let mut prompt_details = obj! {"cached_tokens": event.cache_read};
    if event.cache_write != 0 {
        prompt_details.insert("cache_write_tokens".into(), json!(event.cache_write));
    }
    let mut result = obj! {
        "prompt_tokens": event.prompt,
        "completion_tokens": event.completion,
        "total_tokens": event.total.unwrap_or(event.prompt + event.completion),
        "prompt_tokens_details": prompt_details,
        "completion_tokens_details": {"reasoning_tokens": event.thinking},
    };
    if event.web_searches != 0 {
        result.insert(
            "server_tool_use".into(),
            json!({"web_search_requests": event.web_searches}),
        );
    }
    Value::Object(result)
}
