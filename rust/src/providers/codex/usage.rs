//! Codex terminal usage, shared by translation and accounting.

use crate::ir::Usage;
use crate::json::{get, truthy};
use serde_json::Value;

const TERMINAL_EVENTS: [&str; 3] = [
    "response.completed",
    "response.incomplete",
    "response.failed",
];

/// Python's `int(value or 0)`; what `int()` would reject reads as 0.
fn count(value: &Value) -> i64 {
    if !truthy(value) {
        return 0;
    }
    match value {
        Value::Bool(flag) => i64::from(*flag),
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|float| float as i64))
            .unwrap_or(0),
        Value::String(text) => text.trim().parse().unwrap_or(0),
        _ => 0,
    }
}

pub fn read_usage(event: &Value, web_searches: i64) -> Option<Usage> {
    let kind = get(event, "type").as_str()?;
    if !TERMINAL_EVENTS.contains(&kind) {
        return None;
    }
    let value = get(get(event, "response"), "usage");
    if !value.is_object() {
        return None;
    }
    let details = |key: &str| {
        let details = get(value, key);
        if details.is_object() {
            details.clone()
        } else {
            Value::Null
        }
    };
    let input_details = details("input_tokens_details");
    let output_details = details("output_tokens_details");
    let prompt = count(get(value, "input_tokens"));
    let completion = count(get(value, "output_tokens"));
    let total = match get(value, "total_tokens") {
        Value::Null => prompt + completion,
        // `int()` of a non-empty value, so zero stays zero.
        other => count(other),
    };
    Some(Usage {
        prompt,
        completion,
        total: Some(total),
        cache_read: count(get(&input_details, "cached_tokens")),
        cache_write: count(get(&input_details, "cache_write_tokens")),
        thinking: count(get(&output_details, "reasoning_tokens")),
        web_searches,
        ..Usage::default()
    })
}
