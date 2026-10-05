//! Codex Responses API SSE events -> canonical stream events.

use super::thinking::pack;
use super::usage::read_usage;
use crate::error::{Error, Result};
use crate::ir::{hosted_tool_step, Citation, Decoder, Finish, HostedToolEvent, StreamEvent, Usage};
use crate::json::{dumps, get, py_str, truthy, Dumps, Object};
use crate::reasoning::ReasoningCache;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Decodes one Codex response stream.
///
/// Tool calls arrive whole, so each yields a start and an immediate end.
pub struct CodexDecoder {
    cache: Arc<ReasoningCache>,
    calls: Vec<String>,
    reasoning_items: Vec<Value>,
    web_searches: HashSet<String>,
    search_phase: HashMap<String, String>,
    native_seen: HashSet<String>,
    cited_items: HashSet<String>,
    thinking: String,
    usage: Option<Usage>,
    stop: Option<String>,
    incomplete_reason: Option<String>,
}

/// Responses names the middle of a hosted search in its own events; the ends
/// arrive as ordinary output items.
fn search_phase(kind: &str) -> Option<&'static str> {
    match kind {
        "response.web_search_call.in_progress" => Some("started"),
        "response.web_search_call.searching" => Some("searching"),
        "response.web_search_call.completed" => Some("completed"),
        _ => None,
    }
}

/// `str(event.get(key, default))`.
fn string_or(event: &Value, key: &str, default: &str) -> String {
    event.get(key).map_or_else(|| default.to_string(), py_str)
}

/// `str(event.get(key) or "")`.
fn string_if_truthy(event: &Value, key: &str) -> String {
    let value = get(event, key);
    if truthy(value) {
        py_str(value)
    } else {
        String::new()
    }
}

impl CodexDecoder {
    pub fn new(cache: Arc<ReasoningCache>) -> Self {
        CodexDecoder {
            cache,
            calls: Vec::new(),
            reasoning_items: Vec::new(),
            web_searches: HashSet::new(),
            search_phase: HashMap::new(),
            native_seen: HashSet::new(),
            cited_items: HashSet::new(),
            thinking: String::new(),
            usage: None,
            stop: None,
            incomplete_reason: None,
        }
    }

    fn terminal_response(&mut self, event: &Value, kind: &str) -> Vec<StreamEvent> {
        let default = Value::Object(Object::new());
        let response = event.get("response").unwrap_or(&default);
        if !response.is_object() {
            return Vec::new();
        }
        let mut events = Vec::new();
        if let Some(output) = get(response, "output").as_array() {
            for item in output {
                events.extend(self.item(item));
            }
        }
        self.usage = read_usage(event, self.web_searches.len() as i64);
        if kind == "response.incomplete" {
            let reason = get(get(response, "incomplete_details"), "reason");
            self.incomplete_reason = Some(if truthy(reason) {
                py_str(reason)
            } else {
                "max_output_tokens".to_string()
            });
            self.stop = Some(
                if reason.as_str() == Some("content_filter") {
                    "refusal"
                } else {
                    "max_tokens"
                }
                .to_string(),
            );
        }
        events
    }

    fn item(&mut self, item: &Value) -> Vec<StreamEvent> {
        let Some(map) = item.as_object() else {
            return Vec::new();
        };
        let mut events = self.web_search(item, terminal(item));
        // The completed response lists every item again. A message's citations
        // are read once: by then a later message may be the one being written,
        // and they would attach to it.
        let item_id = string_if_truthy(item, "id");
        if !self.cited_items.contains(&item_id) {
            let parts = get(item, "content").as_array().map(Vec::as_slice);
            for part in parts.unwrap_or_default() {
                let annotations = get(part, "annotations").as_array().map(Vec::as_slice);
                for annotation in annotations.unwrap_or_default() {
                    events.extend(citation(annotation));
                }
            }
        }
        let kind = get(item, "type").as_str().unwrap_or("");
        if !item_id.is_empty() && kind == "message" {
            self.cited_items.insert(item_id);
        }
        match kind {
            "reasoning" => {
                events.extend(self.reasoning(map));
                return events;
            }
            "function_call" => {
                let call_id = if truthy(get(item, "call_id")) {
                    string_if_truthy(item, "call_id")
                } else {
                    string_if_truthy(item, "id")
                };
                if call_id.is_empty() || self.calls.contains(&call_id) {
                    return events;
                }
                let name = string_or(item, "name", "");
                let arguments = string_or(item, "arguments", "{}");
                let namespace = string_if_truthy(item, "namespace");
                let index = Value::from(self.calls.len());
                self.calls.push(call_id.clone());
                events.push(StreamEvent::ToolCallStart {
                    index: index.clone(),
                    id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                    namespace: namespace.clone(),
                });
                events.push(StreamEvent::ToolCallEnd {
                    index,
                    id: call_id,
                    name,
                    arguments,
                    namespace,
                });
                return events;
            }
            "message" | "web_search_call" => return events,
            _ => {}
        }
        let mut key = string_if_truthy(item, "id");
        if key.is_empty() {
            key = string_if_truthy(item, "call_id");
        }
        if key.is_empty() {
            key = dumps(item, Dumps::CANONICAL);
        }
        if self.native_seen.insert(key) {
            events.push(StreamEvent::NativeItem { item: map.clone() });
        }
        events
    }

    /// Keep one opaque item and bridge it through Anthropic's signature.
    fn reasoning(&mut self, item: &Object) -> Vec<StreamEvent> {
        if !item.get("encrypted_content").is_some_and(truthy) {
            self.thinking.clear();
            return Vec::new();
        }
        let kept: Object = ["type", "id", "summary", "encrypted_content"]
            .into_iter()
            .filter_map(|key| item.get(key).map(|value| (key.to_string(), value.clone())))
            .collect();
        let kept_value = Value::Object(kept.clone());
        if self.reasoning_items.contains(&kept_value) {
            self.thinking.clear();
            return Vec::new();
        }
        self.reasoning_items.push(kept_value);
        let mut events = Vec::new();
        let summary = summary_text(&kept);
        if !summary.is_empty() && self.thinking.is_empty() {
            self.thinking = summary.clone();
            events.push(StreamEvent::ThinkingDelta {
                text: summary,
                item_id: string_if_truthy(&Value::Object(kept.clone()), "id"),
            });
        }
        events.push(StreamEvent::ReasoningItem { item: kept.clone() });
        events.push(StreamEvent::ThinkingSignature {
            signature: pack(&kept, &self.thinking),
        });
        self.thinking.clear();
        events
    }

    fn web_search(&mut self, item: &Value, phase: &str) -> Vec<StreamEvent> {
        if get(item, "type").as_str() != Some("web_search_call") {
            return Vec::new();
        }
        let id = if truthy(get(item, "id")) {
            py_str(get(item, "id"))
        } else {
            "web_search".to_string()
        };
        self.web_searches.insert(id.clone());
        self.hosted(&id, phase)
    }

    /// One lifecycle step, dropped unless it advances this search.
    fn hosted(&mut self, search_id: &str, phase: &str) -> Vec<StreamEvent> {
        if search_id.is_empty() || !hosted_tool_step(&mut self.search_phase, search_id, phase) {
            return Vec::new();
        }
        vec![StreamEvent::HostedTool(HostedToolEvent {
            tool: "web_search".into(),
            id: search_id.into(),
            phase: phase.into(),
            ..HostedToolEvent::default()
        })]
    }
}

impl Decoder for CodexDecoder {
    fn decode(&mut self, event: &Value) -> Result<Vec<StreamEvent>> {
        let kind = get(event, "type").as_str().unwrap_or("");
        match kind {
            "response.output_text.delta" => {
                let text = string_or(event, "delta", "");
                Ok(if text.is_empty() {
                    Vec::new()
                } else {
                    vec![StreamEvent::TextDelta {
                        text,
                        span: String::new(),
                    }]
                })
            }
            "response.reasoning_summary_text.delta" => {
                let text = string_or(event, "delta", "");
                if text.is_empty() {
                    return Ok(Vec::new());
                }
                self.thinking.push_str(&text);
                Ok(vec![StreamEvent::ThinkingDelta {
                    text,
                    item_id: string_if_truthy(event, "item_id"),
                }])
            }
            "response.output_text.annotation.added" => Ok(citation(get(event, "annotation"))),
            "response.output_item.added" => Ok(self.web_search(get(event, "item"), "started")),
            "response.output_item.done" => Ok(self.item(get(event, "item"))),
            "response.completed" | "response.incomplete" => Ok(self.terminal_response(event, kind)),
            "response.failed" | "error" => {
                let mut detail = get(event, "error");
                if !truthy(detail) {
                    detail = get(event, "response");
                }
                if !truthy(detail) {
                    detail = event;
                }
                Err(Error::upstream(format!(
                    "Codex response failed: {}",
                    py_str(detail)
                )))
            }
            other => match search_phase(other) {
                Some(phase) => {
                    let id = string_if_truthy(event, "item_id");
                    Ok(self.hosted(&id, phase))
                }
                None => Ok(Vec::new()),
            },
        }
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        self.cache
            .put(&self.calls, std::mem::take(&mut self.reasoning_items));
        let reason = self.stop.clone().unwrap_or_else(|| {
            if self.calls.is_empty() {
                "end_turn".into()
            } else {
                "tool_use".into()
            }
        });
        let mut events = vec![StreamEvent::Finish(Finish {
            reason,
            incomplete_reason: self.incomplete_reason.clone(),
            stop_sequence: None,
        })];
        if let Some(usage) = &self.usage {
            events.push(StreamEvent::Usage(usage.clone()));
        }
        Ok(events)
    }
}

/// How a finished search item ended. Absent status means it simply did.
fn terminal(item: &Value) -> &'static str {
    match get(item, "status").as_str() {
        Some("failed" | "incomplete") => "failed",
        _ => "completed",
    }
}

fn summary_text(item: &Object) -> String {
    let Some(summary) = item.get("summary").and_then(Value::as_array) else {
        return String::new();
    };
    summary
        .iter()
        .filter(|part| get(part, "type").as_str() == Some("summary_text"))
        .map(|part| string_or(part, "text", ""))
        .collect()
}

fn citation(value: &Value) -> Vec<StreamEvent> {
    if get(value, "type").as_str() != Some("url_citation") {
        return Vec::new();
    }
    let url = match get(value, "url").as_str() {
        Some(url) if !url.is_empty() => url,
        _ => return Vec::new(),
    };
    vec![StreamEvent::Citation(Citation {
        url: url.to_string(),
        title: get(value, "title").clone(),
        start_index: get(value, "start_index").clone(),
        end_index: get(value, "end_index").clone(),
        ..Citation::default()
    })]
}
