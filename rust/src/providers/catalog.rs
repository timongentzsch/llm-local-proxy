//! The model record both providers report to the model listings.

use serde_json::{json, Value};

/// Every model the proxy serves accepts these; providers add their own.
const PARAMETERS: [&str; 6] = [
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "reasoning_effort",
    "web_search",
];

/// Resolve an optionally provider-prefixed id against a live catalog.
pub fn match_model(value: &str, models: &[Value]) -> Option<String> {
    if value.is_empty() {
        return None;
    }
    let model = value.split_once('/').map_or(value, |(_, model)| model);
    models
        .iter()
        .any(|item| item.get("id").and_then(Value::as_str) == Some(model))
        .then(|| model.to_string())
}

#[derive(Default)]
pub struct ModelInfo<'a> {
    pub model: &'a str,
    pub name: Value,
    pub owned_by: &'a str,
    /// Missing capability metadata is unknown, not evidence of image support.
    pub modalities: Option<Vec<String>>,
    pub extra_parameters: &'a [&'a str],
    pub default_parameters: Option<Value>,
    pub reasoning_efforts: Vec<Value>,
    pub context_length: i64,
    pub created: i64,
    pub is_default: bool,
}

pub fn model_info(info: ModelInfo<'_>) -> Value {
    let modalities = info
        .modalities
        .filter(|list| !list.is_empty())
        .unwrap_or_else(|| vec!["text".to_string()]);
    let parameters: Vec<&str> = PARAMETERS
        .iter()
        .chain(info.extra_parameters)
        .copied()
        .collect();
    let mut value = json!({
        "id": info.model,
        "canonical_slug": info.model,
        "object": "model",
        "created": info.created,
        "owned_by": info.owned_by,
        "name": info.name,
        "architecture": {
            "modality": format!("{}->text", modalities.join("+")),
            "input_modalities": modalities,
            "output_modalities": ["text"],
        },
        "supported_parameters": parameters,
        "default_parameters": info.default_parameters,
        "per_request_limits": null,
        "is_default": info.is_default,
        "supported_reasoning_efforts": info.reasoning_efforts,
    });
    if info.context_length > 0 {
        value["context_length"] = json!(info.context_length);
    }
    value
}
