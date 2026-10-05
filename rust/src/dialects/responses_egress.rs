//! Canonical stream events -> Responses events and response objects.
//!
//! The open message, reasoning item, function calls and searches are all
//! elements of `output`; they are tracked by their position in it.

use crate::dialects::base::{Driver, Encoder};
use crate::error::{Error, Result};
use crate::ids::{self, SharedIds};
use crate::ir::{ChatRequest, Citation, Decoder, HostedToolEvent, StreamEvent};
use crate::json::{py_str, truthy, Object};
use crate::obj;
use crate::tools::{responses_choice, responses_tool};
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct ResponseEncoder {
    driver: Driver,
    ids: SharedIds,
    id: String,
    created: i64,
    model: String,
    request: Option<ChatRequest>,
    output: Vec<Value>,
    usage: Option<Value>,
    sequence: i64,
    message: Option<usize>,
    reasoning: Option<usize>,
    reasoning_part_open: bool,
    /// Upstream call index -> its item's position in `output`.
    calls: Vec<(Value, usize)>,
    searches: HashMap<String, usize>,
    incomplete_reason: Option<String>,
    terminal: bool,
}

impl ResponseEncoder {
    /// `request` is what the client sent, echoed back in every response
    /// object; `now` is the creation time in whole seconds since the epoch.
    pub fn new(
        model: &str,
        decoder: Box<dyn Decoder>,
        request: Option<ChatRequest>,
        ids: &SharedIds,
        now: i64,
    ) -> Self {
        ResponseEncoder {
            driver: Driver::new(decoder),
            ids: ids.clone(),
            id: format!("resp_{}", ids::hex(&**ids)),
            created: now,
            model: model.to_string(),
            request,
            output: Vec::new(),
            usage: None,
            sequence: 0,
            message: None,
            reasoning: None,
            reasoning_part_open: false,
            calls: Vec::new(),
            searches: HashMap::new(),
            incomplete_reason: None,
            terminal: false,
        }
    }

    fn response(&self, status: &str, with_output: bool) -> Result<Value> {
        let request = self.request.as_ref();
        let mut tools = Vec::new();
        if let Some(request) = request {
            for tool in &request.tools {
                tools.push(Value::Object(responses_tool(tool)?));
            }
        }
        let incomplete = if status == "incomplete" {
            json!({"reason": self.incomplete_reason})
        } else {
            Value::Null
        };
        Ok(json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": if with_output { self.output.clone() } else { Vec::new() },
            "parallel_tool_calls": request.map_or(true, |r| r.parallel_tool_calls != Some(false)),
            "tool_choice": responses_choice(request.and_then(|r| r.tool_choice.as_ref())),
            "tools": tools,
            "usage": if with_output { json!(self.usage) } else { Value::Null },
            "error": null,
            "incomplete_details": incomplete,
        }))
    }

    fn event(&mut self, kind: &str, fields: Object) -> Value {
        let mut event = obj! {"type": kind, "sequence_number": self.sequence};
        event.extend(fields);
        self.sequence += 1;
        Value::Object(event)
    }

    // -- accessors on the items in `output` ----------------------------------

    fn item_id(&self, index: usize) -> Value {
        self.output[index].get("id").cloned().unwrap_or(Value::Null)
    }

    fn set_field(&mut self, index: usize, key: &str, value: Value) {
        if let Some(item) = self.output[index].as_object_mut() {
            item.insert(key.to_string(), value);
        }
    }

    /// `item[...] += text` at a JSON pointer inside an output item.
    fn append_text(&mut self, index: usize, pointer: &str, more: &str) -> Result<()> {
        match self.output[index].pointer_mut(pointer) {
            Some(Value::String(text)) => {
                text.push_str(more);
                Ok(())
            }
            _ => Err(Error::upstream("response item has no text to extend")),
        }
    }

    fn text_part(&self, index: usize) -> Value {
        self.output[index]
            .pointer("/content/0")
            .cloned()
            .unwrap_or(Value::Null)
    }

    // -- events ----------------------------------------------------------------

    fn one_thinking(&mut self, text: &str, item_id: &str) -> Result<Vec<Value>> {
        let mut chunks = self.complete_message();
        if let Some(open) = self.reasoning {
            if !item_id.is_empty() && self.item_id(open) != json!(item_id) {
                chunks.extend(self.complete_reasoning_open());
            }
        }
        chunks.extend(self.ensure_reasoning(true, item_id));
        let Some(open) = self.reasoning else {
            return Err(Error::upstream("no reasoning item is open"));
        };
        self.append_text(open, "/summary/0/text", text)?;
        let fields = obj! {
            "item_id": self.item_id(open),
            "output_index": open,
            "summary_index": 0,
            "delta": text,
        };
        chunks.push(self.event("response.reasoning_summary_text.delta", fields));
        Ok(chunks)
    }

    fn one_native(&mut self, item: Object) -> Vec<Value> {
        let mut chunks = self.close_open_items();
        self.output.push(Value::Object(item.clone()));
        let index = self.output.len() - 1;
        let mut added = item.clone();
        if added.contains_key("status") {
            added.insert("status".into(), json!("in_progress"));
        }
        for (kind, item) in [
            ("response.output_item.added", added),
            ("response.output_item.done", item),
        ] {
            let fields = obj! {"output_index": index, "item": item};
            chunks.push(self.event(kind, fields));
        }
        chunks
    }

    fn one_text(&mut self, text: &str) -> Result<Vec<Value>> {
        let mut chunks = self.complete_reasoning_open();
        chunks.extend(self.ensure_message());
        let Some(message) = self.message else {
            return Err(Error::upstream("no message is open"));
        };
        self.append_text(message, "/content/0/text", text)?;
        let fields = obj! {
            "item_id": self.item_id(message),
            "output_index": message,
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        };
        chunks.push(self.event("response.output_text.delta", fields));
        Ok(chunks)
    }

    fn one_call_start(
        &mut self,
        index: Value,
        call_id: String,
        namespace: String,
        name: String,
        arguments: String,
    ) -> Vec<Value> {
        let mut chunks = self.close_open_items();
        let mut item = obj! {
            "type": "function_call",
            "id": format!("fc_{}", ids::hex(&*self.ids)),
            "call_id": call_id,
        };
        if !namespace.is_empty() {
            item.insert("namespace".into(), json!(namespace));
        }
        item.insert("name".into(), json!(name));
        item.insert("arguments".into(), json!(arguments));
        item.insert("status".into(), json!("in_progress"));
        self.output.push(Value::Object(item.clone()));
        let position = self.output.len() - 1;
        match self.calls.iter_mut().find(|(key, _)| *key == index) {
            Some(entry) => entry.1 = position,
            None => self.calls.push((index, position)),
        }
        let fields = obj! {"output_index": position, "item": item};
        chunks.push(self.event("response.output_item.added", fields));
        chunks
    }

    fn call(&self, index: &Value) -> Option<usize> {
        self.calls
            .iter()
            .find(|(key, _)| key == index)
            .map(|(_, position)| *position)
    }

    fn one_call_args(&mut self, index: &Value, fragment: &str) -> Result<Vec<Value>> {
        let Some(item) = self.call(index) else {
            return Ok(Vec::new());
        };
        self.append_text(item, "/arguments", fragment)?;
        let fields = obj! {
            "item_id": self.item_id(item),
            "output_index": item,
            "delta": fragment,
        };
        Ok(vec![
            self.event("response.function_call_arguments.delta", fields)
        ])
    }

    fn one_call_end(&mut self, index: &Value, arguments: String) -> Vec<Value> {
        let Some(item) = self.call(index) else {
            return Vec::new();
        };
        self.set_field(item, "arguments", json!(arguments));
        self.set_field(item, "status", json!("completed"));
        let name = self.output[item]
            .get("name")
            .cloned()
            .unwrap_or(Value::Null);
        let done = obj! {
            "item_id": self.item_id(item),
            "output_index": item,
            "name": name,
            "arguments": arguments,
        };
        let finished = obj! {"output_index": item, "item": self.output[item].clone()};
        vec![
            self.event("response.function_call_arguments.done", done),
            self.event("response.output_item.done", finished),
        ]
    }

    fn one_citation(&mut self, event: Citation) -> Result<Vec<Value>> {
        let Some(message) = self.message else {
            return Ok(Vec::new());
        };
        if event.url.is_empty() {
            return Ok(Vec::new());
        }
        let or = |value: Value, default: Value| if truthy(&value) { value } else { default };
        let annotation = Value::Object(obj! {
            "type": "url_citation",
            "url": event.url,
            "title": or(event.title, json!("")),
            "start_index": or(event.start_index, json!(0)),
            "end_index": or(event.end_index, json!(0)),
        });
        let Some(Value::Array(annotations)) =
            self.output[message].pointer_mut("/content/0/annotations")
        else {
            return Err(Error::upstream("message has no annotations"));
        };
        // An upstream may repeat an annotation on the finished item and in
        // the completed response; the client is told once.
        if annotations.contains(&annotation) {
            return Ok(Vec::new());
        }
        annotations.push(annotation.clone());
        let annotation_index = annotations.len() - 1;
        let fields = obj! {
            "item_id": self.item_id(message),
            "output_index": message,
            "content_index": 0,
            "annotation_index": annotation_index,
            "annotation": annotation,
        };
        Ok(vec![
            self.event("response.output_text.annotation.added", fields)
        ])
    }

    /// A hosted search as the Responses lifecycle a client already knows.
    ///
    /// Deliberately not the `NativeItem` path, which emits `added` and `done`
    /// back to back: the gap between them is the whole signal here, and a
    /// client showing "searching" needs the item to stay open across it.
    fn hosted(&mut self, event: HostedToolEvent) -> Vec<Value> {
        let (index, mut chunks) = match self.searches.get(&event.id) {
            Some(&index) => (index, Vec::new()),
            None => {
                let mut chunks = self.close_open_items();
                let id = if event.id.is_empty() {
                    format!("ws_{}", ids::hex(&*self.ids))
                } else {
                    event.id.clone()
                };
                let item = obj! {"type": "web_search_call", "id": id, "status": "in_progress"};
                self.output.push(Value::Object(item.clone()));
                let index = self.output.len() - 1;
                self.searches.insert(event.id.clone(), index);
                let fields = obj! {"output_index": index, "item": item};
                chunks.push(self.event("response.output_item.added", fields));
                (index, chunks)
            }
        };
        if event.phase == "searching" {
            let fields = obj! {"output_index": index, "item_id": self.item_id(index)};
            chunks.push(self.event("response.web_search_call.searching", fields));
            return chunks;
        }
        if !matches!(event.phase.as_str(), "completed" | "failed") {
            return chunks;
        }
        self.set_field(index, "status", json!(event.phase));
        if event.phase == "completed" {
            let fields = obj! {"output_index": index, "item_id": self.item_id(index)};
            chunks.push(self.event("response.web_search_call.completed", fields));
        }
        let fields = obj! {"output_index": index, "item": self.output[index].clone()};
        chunks.push(self.event("response.output_item.done", fields));
        chunks
    }

    fn ensure_reasoning(&mut self, summary: bool, item_id: &str) -> Vec<Value> {
        if self.reasoning.is_some() {
            return Vec::new();
        }
        let id = if item_id.is_empty() {
            format!("rs_{}", ids::hex(&*self.ids))
        } else {
            item_id.to_string()
        };
        let item = obj! {"type": "reasoning", "id": id, "summary": []};
        self.output.push(Value::Object(item.clone()));
        let index = self.output.len() - 1;
        self.reasoning = Some(index);
        let fields = obj! {"output_index": index, "item": item};
        let mut chunks = vec![self.event("response.output_item.added", fields)];
        if summary {
            self.set_field(
                index,
                "summary",
                json!([{"type": "summary_text", "text": ""}]),
            );
            self.reasoning_part_open = true;
            let fields = obj! {
                "item_id": self.item_id(index),
                "output_index": index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""},
            };
            chunks.push(self.event("response.reasoning_summary_part.added", fields));
        }
        chunks
    }

    fn complete_reasoning(&mut self, opaque: Object) -> Vec<Value> {
        let Some(index) = self.reasoning else {
            self.output.push(Value::Object(opaque.clone()));
            let index = self.output.len() - 1;
            let mut added = opaque.clone();
            added.insert("status".into(), json!("in_progress"));
            let added = obj! {"output_index": index, "item": added};
            let done = obj! {"output_index": index, "item": opaque};
            return vec![
                self.event("response.output_item.added", added),
                self.event("response.output_item.done", done),
            ];
        };
        // `added` already announced this item's id; `done` must match it.
        let id = self.item_id(index);
        if let Some(item) = self.output[index].as_object_mut() {
            for (key, value) in opaque {
                item.insert(key, value);
            }
            item.insert("id".into(), id);
        }
        let mut chunks = Vec::new();
        let first = self.output[index].pointer("/summary/0").cloned();
        if let (true, Some(first)) = (self.reasoning_part_open, first) {
            let text = match first.get("text") {
                Some(text) => py_str(text),
                None => String::new(),
            };
            let fields = obj! {
                "item_id": self.item_id(index),
                "output_index": index,
                "summary_index": 0,
                "text": text,
            };
            chunks.push(self.event("response.reasoning_summary_text.done", fields));
            let fields = obj! {
                "item_id": self.item_id(index),
                "output_index": index,
                "summary_index": 0,
                "part": first,
            };
            chunks.push(self.event("response.reasoning_summary_part.done", fields));
        }
        let fields = obj! {"output_index": index, "item": self.output[index].clone()};
        chunks.push(self.event("response.output_item.done", fields));
        self.reasoning = None;
        self.reasoning_part_open = false;
        chunks
    }

    fn complete_reasoning_open(&mut self) -> Vec<Value> {
        let Some(index) = self.reasoning else {
            return Vec::new();
        };
        let snapshot = self.output[index].as_object().cloned().unwrap_or_default();
        self.complete_reasoning(snapshot)
    }

    fn ensure_message(&mut self) -> Vec<Value> {
        if self.message.is_some() {
            return Vec::new();
        }
        let item = obj! {
            "type": "message",
            "id": format!("msg_{}", ids::hex(&*self.ids)),
            "status": "in_progress",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "", "annotations": [], "logprobs": []}],
        };
        let part = item["content"][0].clone();
        let id = item["id"].clone();
        self.output.push(Value::Object(item.clone()));
        let index = self.output.len() - 1;
        self.message = Some(index);
        let added = obj! {"output_index": index, "item": item};
        let part_added = obj! {
            "item_id": id,
            "output_index": index,
            "content_index": 0,
            "part": part,
        };
        vec![
            self.event("response.output_item.added", added),
            self.event("response.content_part.added", part_added),
        ]
    }

    fn complete_message(&mut self) -> Vec<Value> {
        let Some(index) = self.message.take() else {
            return Vec::new();
        };
        self.set_field(index, "status", json!("completed"));
        let part = self.text_part(index);
        let text = part.get("text").cloned().unwrap_or(Value::Null);
        let text_done = obj! {
            "item_id": self.item_id(index),
            "output_index": index,
            "content_index": 0,
            "text": text,
            "logprobs": [],
        };
        let part_done = obj! {
            "item_id": self.item_id(index),
            "output_index": index,
            "content_index": 0,
            "part": part,
        };
        let item_done = obj! {"output_index": index, "item": self.output[index].clone()};
        vec![
            self.event("response.output_text.done", text_done),
            self.event("response.content_part.done", part_done),
            self.event("response.output_item.done", item_done),
        ]
    }

    fn close_open_items(&mut self) -> Vec<Value> {
        let mut chunks = self.complete_reasoning_open();
        chunks.extend(self.complete_message());
        chunks
    }
}

impl Encoder for ResponseEncoder {
    fn driver(&mut self) -> &mut Driver {
        &mut self.driver
    }

    fn one(&mut self, event: StreamEvent) -> Result<Vec<Value>> {
        match event {
            StreamEvent::ThinkingDelta { text, item_id } => self.one_thinking(&text, &item_id),
            StreamEvent::ReasoningItem { item } => Ok(self.complete_reasoning(item)),
            StreamEvent::NativeItem { item } => Ok(self.one_native(item)),
            StreamEvent::TextDelta { text, .. } => self.one_text(&text),
            StreamEvent::ToolCallStart {
                index,
                id,
                name,
                arguments,
                namespace,
            } => Ok(self.one_call_start(index, id, namespace, name, arguments)),
            StreamEvent::ToolCallArgs { index, fragment } => self.one_call_args(&index, &fragment),
            StreamEvent::ToolCallEnd {
                index, arguments, ..
            } => Ok(self.one_call_end(&index, arguments)),
            StreamEvent::HostedTool(event) => Ok(self.hosted(event)),
            StreamEvent::Citation(event) => self.one_citation(event),
            StreamEvent::Usage(event) => {
                self.usage = Some(json!({
                    "input_tokens": event.prompt,
                    "output_tokens": event.completion,
                    "total_tokens": event.total.unwrap_or(event.prompt + event.completion),
                    "input_tokens_details": {
                        "cached_tokens": event.cache_read,
                        "cache_write_tokens": event.cache_write,
                    },
                    "output_tokens_details": {"reasoning_tokens": event.thinking},
                }));
                Ok(Vec::new())
            }
            StreamEvent::Finish(event) => {
                self.incomplete_reason = match event.incomplete_reason {
                    Some(reason) if !reason.is_empty() => Some(reason),
                    _ if matches!(
                        event.reason.as_str(),
                        "max_tokens" | "model_context_window_exceeded"
                    ) =>
                    {
                        Some("max_output_tokens".into())
                    }
                    _ => None,
                };
                Ok(Vec::new())
            }
            _ => Ok(Vec::new()),
        }
    }

    fn start(&mut self) -> Value {
        let response = match self.response("in_progress", false) {
            Ok(response) => response,
            // The request's tools are validated when it is parsed; if one
            // still cannot be echoed, say so rather than open the stream.
            Err(error) => return self.error(error.message()).unwrap_or(Value::Null),
        };
        self.event("response.created", obj! {"response": response})
    }

    fn error(&mut self, message: &str) -> Option<Value> {
        self.terminal = true;
        Some(self.event(
            "error",
            obj! {"code": "upstream_error", "message": message, "param": null},
        ))
    }

    fn finish(&mut self) -> Result<Vec<Value>> {
        let mut chunks = self.drain()?;
        chunks.extend(self.close_open_items());
        if self.terminal {
            return Ok(chunks);
        }
        self.terminal = true;
        let incomplete = self.incomplete_reason.is_some();
        let (status, kind) = if incomplete {
            ("incomplete", "response.incomplete")
        } else {
            ("completed", "response.completed")
        };
        let response = self.response(status, true)?;
        chunks.push(self.event(kind, obj! {"response": response}));
        Ok(chunks)
    }

    fn result(&mut self) -> Result<Value> {
        self.drain()?;
        self.close_open_items();
        self.terminal = true;
        let status = if self.incomplete_reason.is_some() {
            "incomplete"
        } else {
            "completed"
        };
        self.response(status, true)
    }

    fn set_id(&mut self, id: String) {
        self.id = id;
    }

    fn set_created(&mut self, created: i64) {
        self.created = created;
    }
}
