//! The Claude model catalog.

use crate::json::{get, integer, py_str, truthy};
use crate::providers::catalog::{model_info as shared, ModelInfo};
use serde_json::{json, Value};

/// A normalised upstream model as the listing entry clients see.
pub fn model_info(item: &Value) -> Value {
    let max_output = get(item, "max_output_tokens");
    shared(ModelInfo {
        model: get(item, "id").as_str().unwrap_or_default(),
        name: get(item, "name").clone(),
        owned_by: "anthropic",
        modalities: get(item, "modalities")
            .as_array()
            .map(|values| values.iter().map(py_str).collect()),
        extra_parameters: &["temperature", "top_p"],
        default_parameters: truthy(max_output).then(|| json!({ "max_tokens": max_output })),
        reasoning_efforts: get(item, "reasoning_efforts")
            .as_array()
            .cloned()
            .unwrap_or_default(),
        context_length: integer(get(item, "context_length"))
            .filter(|n| *n > 0)
            .unwrap_or(0),
        created: get(item, "created").as_i64().unwrap_or(0),
        is_default: false,
    })
}
