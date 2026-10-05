//! Reasoning controls shared by the OpenAI-shaped request dialects.

use crate::error::{Error, Result};
use crate::json::truthy;
use serde_json::Value;

/// Responses summary modes, plus "none", which Codex CLI sends to ask for none.
pub const SUMMARY_MODES: [&str; 4] = ["auto", "concise", "detailed", "none"];

/// The requested effort and summary mode.
pub fn options(value: &Value) -> Result<(Value, String)> {
    let Some(map) = value.as_object() else {
        return Ok((Value::Null, String::new()));
    };
    let summary = map.get("summary").unwrap_or(&Value::Null);
    let known = summary
        .as_str()
        .is_some_and(|mode| SUMMARY_MODES.contains(&mode));
    if !summary.is_null() && !known {
        return Err(Error::request(
            "reasoning.summary must be auto, concise, detailed or none",
        ));
    }
    let effort = map.get("effort").cloned().unwrap_or(Value::Null);
    let summary = match summary {
        Value::String(mode) if truthy(summary) => mode.clone(),
        _ => String::new(),
    };
    Ok((effort, summary))
}
