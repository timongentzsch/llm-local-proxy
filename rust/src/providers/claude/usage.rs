//! Anthropic's cumulative usage, shared by translation and accounting.

use crate::ir::Usage;
use crate::json::{get, integer};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Default)]
pub struct ClaudeUsage {
    counts: HashMap<&'static str, i64>,
}

impl ClaudeUsage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn read(&mut self, event: &Value) -> Option<Usage> {
        let value = match get(event, "type").as_str() {
            Some("message_start") => get(get(event, "message"), "usage"),
            Some("message_delta") => get(event, "usage"),
            _ => return None,
        };
        if !value.is_object() {
            return None;
        }
        // These are snapshots, not increments. An omitted field retains its
        // previous value; an explicit zero replaces it.
        for name in [
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ] {
            self.record(name, get(value, name));
        }
        self.record(
            "cache_write_1h",
            get(get(value, "cache_creation"), "ephemeral_1h_input_tokens"),
        );
        self.record(
            "thinking_tokens",
            get(get(value, "output_tokens_details"), "thinking_tokens"),
        );
        self.snapshot(0)
    }

    fn record(&mut self, name: &'static str, count: &Value) {
        if let Some(count) = integer(count).filter(|count| *count >= 0) {
            self.counts.insert(name, count);
        }
    }

    pub fn snapshot(&self, web_searches: i64) -> Option<Usage> {
        if self.counts.is_empty() {
            return None;
        }
        let count = |name: &str| self.counts.get(name).copied().unwrap_or(0);
        let read = count("cache_read_input_tokens");
        let write = count("cache_creation_input_tokens");
        Some(Usage {
            prompt: count("input_tokens") + read + write,
            completion: count("output_tokens"),
            total: None,
            cache_read: read,
            cache_write: write,
            cache_write_1h: self.counts.get("cache_write_1h").copied(),
            thinking: count("thinking_tokens"),
            web_searches,
        })
    }
}
