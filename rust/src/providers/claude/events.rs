//! Claude Messages SSE events -> canonical stream events.

use super::strip;
use super::thinking::pack;
use super::usage::ClaudeUsage;
use crate::error::{Error, Result};
use crate::ids::{self, SharedIds};
use crate::ir::{hosted_tool_step, Citation, Decoder, Finish, HostedToolEvent, StreamEvent};
use crate::json::{get, py_str, truthy, Object};
use crate::obj;
use crate::reasoning::ReasoningCache;
use crate::tools::{Names, ANTHROPIC_WEB_SEARCH};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

struct OpenCall {
    index: Value,
    id: String,
    name: String,
    namespace: String,
    arguments: String,
}

#[derive(Default)]
struct OpenThinking {
    thinking: String,
    signature: String,
}

/// Decodes one Claude response stream.
///
/// Tool calls arrive in pieces. Signed thinking is accumulated rather than
/// forwarded: it exists to be replayed upstream on the next turn.
pub struct ClaudeDecoder {
    cache: Option<Arc<ReasoningCache>>,
    /// Flattened tool name -> (namespace, name), from `tools::flatten`.
    names: Names,
    ids: SharedIds,
    reasoning_blocks: Vec<Value>,
    calls: Vec<String>,
    /// Search ids, so the same search counted from its request and from its
    /// result stays one search.
    web_searches: HashSet<String>,
    search_phase: HashMap<String, String>,
    open_search: String,
    open_call: Option<OpenCall>,
    open_thinking: Option<OpenThinking>,
    thinking_id: String,
    text_span: String,
    open_redacted: Option<String>,
    stop: Option<Finish>,
    usage: ClaudeUsage,
}

impl ClaudeDecoder {
    /// `names` restores flattened tool names, from `tools::flatten`.
    pub fn new(cache: Option<Arc<ReasoningCache>>, names: Names, ids: SharedIds) -> Self {
        ClaudeDecoder {
            cache,
            names,
            ids,
            reasoning_blocks: Vec::new(),
            calls: Vec::new(),
            web_searches: HashSet::new(),
            search_phase: HashMap::new(),
            open_search: String::new(),
            open_call: None,
            open_thinking: None,
            thinking_id: String::new(),
            text_span: String::new(),
            open_redacted: None,
            stop: None,
            usage: ClaudeUsage::new(),
        }
    }

    fn block_start(&mut self, event: &Value) -> Vec<StreamEvent> {
        let block = get(event, "content_block");
        let kind = get(block, "type").as_str().unwrap_or("");
        let index = get(event, "index").clone();
        if kind == "text" {
            self.text_span = py_str(&index);
        }
        match kind {
            "tool_use" => return self.tool_start(block, index),
            "thinking" => {
                self.open_thinking = Some(OpenThinking::default());
                self.thinking_id = format!("rs_{}", ids::hex(self.ids.as_ref()));
            }
            "redacted_thinking" => {
                // Opaque, safety-flagged portion; replay verbatim on round-trip.
                let data = or_empty(get(block, "data"));
                self.open_redacted = Some(data.clone());
                return if data.is_empty() {
                    vec![]
                } else {
                    vec![StreamEvent::RedactedThinkingDelta { data }]
                };
            }
            // The stream says `server_tool_use`; the versioned spelling is the
            // tool *definition*, accepted here because both have been seen.
            "server_tool_use" | ANTHROPIC_WEB_SEARCH => return self.search_start(block, &index),
            "web_search_tool_result" => return self.search_result(block, &index),
            _ => {}
        }
        vec![]
    }

    fn tool_start(&mut self, block: &Value, index: Value) -> Vec<StreamEvent> {
        let id = get(block, "id");
        let call_id = if truthy(id) {
            py_str(id)
        } else {
            format!("toolu_{}", ids::hex24(self.ids.as_ref()))
        };
        let flat = block.get("name").map_or_else(String::new, py_str);
        let (namespace, name) = self
            .names
            .get(&flat)
            .cloned()
            .unwrap_or_else(|| (String::new(), flat));
        let event = StreamEvent::ToolCallStart {
            index: index.clone(),
            id: call_id.clone(),
            name: name.clone(),
            arguments: String::new(),
            namespace: namespace.clone(),
        };
        self.open_call = Some(OpenCall {
            index,
            id: call_id,
            name,
            namespace,
            arguments: String::new(),
        });
        // Announced immediately so time-to-first-token is not stalled.
        vec![event]
    }

    fn search_start(&mut self, block: &Value, index: &Value) -> Vec<StreamEvent> {
        let name = block
            .get("name")
            .map_or_else(|| "web_search".into(), py_str);
        if name != "web_search" {
            return vec![];
        }
        let id = get(block, "id");
        self.open_search = if truthy(id) {
            py_str(id)
        } else {
            format!("web_search_{}", py_str(index))
        };
        // Best effort: Claude may stream the query as input_json_delta
        // instead, and waiting for it would cost the whole visible search.
        let given = get(block, "input");
        let query = match given {
            Value::Object(map) => map.get("query").map_or_else(String::new, py_str),
            _ => String::new(),
        };
        let id = self.open_search.clone();
        self.hosted(&id, "searching", query, String::new(), None)
    }

    fn search_result(&mut self, block: &Value, index: &Value) -> Vec<StreamEvent> {
        let found = get(block, "tool_use_id");
        let search_id = if truthy(found) {
            py_str(found)
        } else if !self.open_search.is_empty() {
            self.open_search.clone()
        } else {
            format!("web_search_{}", py_str(index))
        };
        let content = get(block, "content");
        let phase = result_phase(content);
        let error_code = if phase == "failed" && content.is_object() {
            or_empty(get(content, "error_code"))
        } else {
            String::new()
        };
        let result = (!content.is_null()).then(|| content.clone());
        self.hosted(&search_id, phase, String::new(), error_code, result)
    }

    fn block_stop(&mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if let Some(open) = self.open_thinking.take() {
            // Claude verifies kind, text and signature together on replay, so
            // the envelope keeps all three in the one opaque Responses slot.
            // A signature without text is unreplayable -- the subscription
            // edge signs reasoning it never streams, and the signature covers
            // what Claude wrote, not the empty string left here -- so packing
            // one would only store a turn that can never be sent back.
            if !open.signature.is_empty() && !open.thinking.is_empty() {
                let block = obj! {
                    "type": "thinking",
                    "thinking": open.thinking,
                    "signature": open.signature,
                };
                let envelope = pack(&block, self.reasoning_blocks.len());
                self.reasoning_blocks.push(Value::Object(block));
                events.push(StreamEvent::ThinkingSignature {
                    signature: open.signature.clone(),
                });
                events.push(StreamEvent::ReasoningItem {
                    item: obj! {
                        "type": "reasoning",
                        "id": self.thinking_id,
                        "summary": [{"type": "summary_text", "text": open.thinking}],
                        "encrypted_content": envelope,
                    },
                });
            }
        }
        if let Some(data) = self.open_redacted.take() {
            let block = obj! { "type": "redacted_thinking", "data": data };
            let envelope = pack(&block, self.reasoning_blocks.len());
            self.reasoning_blocks.push(Value::Object(block));
            events.push(StreamEvent::ReasoningItem {
                item: obj! {
                    "type": "reasoning",
                    "id": format!("rs_{}", ids::hex(self.ids.as_ref())),
                    "summary": [],
                    "encrypted_content": envelope,
                },
            });
        }
        events.extend(self.close_call());
        events
    }

    fn delta(&mut self, delta: &Value) -> Vec<StreamEvent> {
        if !delta.is_object() {
            return vec![];
        }
        let field = |key: &str| delta.get(key).map_or_else(String::new, py_str);
        match get(delta, "type").as_str().unwrap_or("") {
            "text_delta" => {
                let text = field("text");
                if text.is_empty() {
                    return vec![];
                }
                vec![StreamEvent::TextDelta {
                    text,
                    span: self.text_span.clone(),
                }]
            }
            "thinking_delta" => {
                let text = field("thinking");
                if text.is_empty() {
                    return vec![];
                }
                if let Some(open) = self.open_thinking.as_mut() {
                    open.thinking.push_str(&text);
                }
                vec![StreamEvent::ThinkingDelta {
                    text,
                    item_id: self.thinking_id.clone(),
                }]
            }
            "signature_delta" => {
                let signature = field("signature");
                if let Some(open) = self.open_thinking.as_mut() {
                    open.signature.push_str(&signature);
                }
                vec![]
            }
            "redacted_thinking_delta" => {
                let data = field("data");
                if data.is_empty() {
                    return vec![];
                }
                if let Some(open) = self.open_redacted.as_mut() {
                    open.push_str(&data);
                }
                vec![StreamEvent::RedactedThinkingDelta { data }]
            }
            "input_json_delta" => {
                let piece = field("partial_json");
                match self.open_call.as_mut() {
                    Some(call) if !piece.is_empty() => {
                        call.arguments.push_str(&piece);
                        vec![StreamEvent::ToolCallArgs {
                            index: call.index.clone(),
                            fragment: piece,
                        }]
                    }
                    _ => vec![],
                }
            }
            "citations_delta" => citation(get(delta, "citation"), &self.text_span),
            _ => vec![],
        }
    }

    fn close_call(&mut self) -> Vec<StreamEvent> {
        let Some(call) = self.open_call.take() else {
            return vec![];
        };
        if call.name.is_empty() || self.calls.contains(&call.id) {
            return vec![];
        }
        let streamed = strip(&call.arguments).to_string();
        self.calls.push(call.id.clone());
        let mut events = Vec::new();
        if streamed.is_empty() {
            // No input_json_delta arrives, so the client would see "".
            events.push(StreamEvent::ToolCallArgs {
                index: call.index.clone(),
                fragment: "{}".into(),
            });
        }
        events.push(StreamEvent::ToolCallEnd {
            index: call.index,
            id: call.id,
            name: call.name,
            arguments: if streamed.is_empty() {
                "{}".into()
            } else {
                streamed
            },
            namespace: call.namespace,
        });
        events
    }

    /// One lifecycle step, dropped unless it advances this search.
    fn hosted(
        &mut self,
        search_id: &str,
        phase: &str,
        query: String,
        error_code: String,
        result: Option<Value>,
    ) -> Vec<StreamEvent> {
        self.web_searches.insert(search_id.to_string());
        if !hosted_tool_step(&mut self.search_phase, search_id, phase) {
            return vec![];
        }
        vec![StreamEvent::HostedTool(HostedToolEvent {
            tool: "web_search".into(),
            id: search_id.to_string(),
            phase: phase.to_string(),
            query,
            error_code,
            result,
        })]
    }
}

impl Decoder for ClaudeDecoder {
    fn decode(&mut self, event: &Value) -> Result<Vec<StreamEvent>> {
        match get(event, "type").as_str().unwrap_or("") {
            "message_start" => {
                self.usage.read(event);
                Ok(vec![])
            }
            "content_block_start" => Ok(self.block_start(event)),
            "content_block_delta" => Ok(self.delta(get(event, "delta"))),
            "content_block_stop" => Ok(self.block_stop()),
            "message_delta" => {
                let delta = get(event, "delta");
                let reason = get(delta, "stop_reason");
                if delta.is_object() && truthy(reason) {
                    self.stop = Some(Finish {
                        reason: py_str(reason),
                        incomplete_reason: None,
                        stop_sequence: get(delta, "stop_sequence").as_str().map(String::from),
                    });
                }
                self.usage.read(event);
                Ok(vec![])
            }
            "error" => {
                let detail = match get(event, "error") {
                    error if truthy(error) => error,
                    _ => event,
                };
                Err(Error::upstream(format!(
                    "Claude response failed: {}",
                    py_str(detail)
                )))
            }
            _ => Ok(vec![]),
        }
    }

    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        if let Some(cache) = &self.cache {
            cache.put(&self.calls, self.reasoning_blocks.clone());
        }
        let stop = self.stop.clone().unwrap_or_else(|| Finish {
            reason: if self.calls.is_empty() {
                "end_turn"
            } else {
                "tool_use"
            }
            .into(),
            ..Finish::default()
        });
        let mut events = vec![StreamEvent::Finish(stop)];
        if let Some(usage) = self.usage.snapshot(self.web_searches.len() as i64) {
            events.push(StreamEvent::Usage(usage));
        }
        Ok(events)
    }
}

/// `str(value or "")`.
fn or_empty(value: &Value) -> String {
    if truthy(value) {
        py_str(value)
    } else {
        String::new()
    }
}

/// An error record is a failed search, not a completed one.
fn result_phase(content: &Value) -> &'static str {
    if get(content, "type").as_str() == Some("web_search_tool_result_error") {
        "failed"
    } else {
        "completed"
    }
}

fn citation(value: &Value, span: &str) -> Vec<StreamEvent> {
    // Web citations carry a url; citations into documents and search results
    // the client supplied carry a location instead.
    let Value::Object(native) = value else {
        return vec![];
    };
    let title = match native.get("title") {
        Some(title @ Value::String(_)) => title.clone(),
        _ => Value::Null,
    };
    let url = native
        .get("url")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    vec![StreamEvent::Citation(Citation {
        url,
        title,
        start_index: Value::Null,
        end_index: Value::Null,
        native: Some(Object::clone(native)),
        span: span.to_string(),
    })]
}
