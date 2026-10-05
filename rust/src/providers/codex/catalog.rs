//! The Codex model catalog.

use crate::json::{get, py_str, truthy};
use crate::providers::catalog::{model_info as shared, ModelInfo};
use serde_json::{json, Value};
use std::collections::HashMap;

/// One `model/list` entry as the listing entry clients see. Effort tiers are
/// intersected with what the transport accepts, when it said.
pub fn model_info(
    item: &Value,
    context_windows: &HashMap<String, i64>,
    transport_efforts: Option<&[String]>,
) -> Option<Value> {
    let model = [get(item, "model"), get(item, "id")]
        .into_iter()
        .find(|value| truthy(value))
        .map(py_str)?;
    let accepts = |effort: &Value| match transport_efforts {
        None => true,
        Some(accepted) => effort
            .as_str()
            .is_some_and(|e| accepted.iter().any(|a| a == e)),
    };
    let efforts: Vec<Value> = get(item, "supportedReasoningEfforts")
        .as_array()
        .into_iter()
        .flatten()
        .map(|effort| get(effort, "reasoningEffort"))
        .filter(|effort| truthy(effort) && accepts(effort))
        .cloned()
        .collect();
    let default_effort = get(item, "defaultReasoningEffort");
    let display = get(item, "displayName");
    Some(shared(ModelInfo {
        model: &model,
        name: if truthy(display) {
            display.clone()
        } else {
            json!(model)
        },
        owned_by: "openai",
        modalities: get(item, "inputModalities")
            .as_array()
            .filter(|list| !list.is_empty())
            .map(|list| list.iter().map(py_str).collect()),
        default_parameters: (truthy(default_effort) && accepts(default_effort))
            .then(|| json!({ "reasoning_effort": default_effort })),
        reasoning_efforts: efforts,
        context_length: context_windows.get(&model).copied().unwrap_or(0),
        is_default: truthy(get(item, "isDefault")),
        ..Default::default()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn efforts_the_transport_rejects_are_left_out() {
        let item = json!({
            "model": "gpt-x",
            "displayName": "GPT X",
            "supportedReasoningEfforts": [
                {"reasoningEffort": "low"},
                {"reasoningEffort": "xhigh"},
            ],
            "defaultReasoningEffort": "xhigh",
            "inputModalities": ["text", "image"],
            "isDefault": true,
        });
        let contexts = HashMap::from([("gpt-x".to_string(), 400_000)]);
        let accepted = ["low".to_string(), "medium".to_string()];
        let model = model_info(&item, &contexts, Some(&accepted)).unwrap();
        assert_eq!(model["supported_reasoning_efforts"], json!(["low"]));
        assert_eq!(model["default_parameters"], Value::Null);
        assert_eq!(model["context_length"], 400_000);
        assert_eq!(model["architecture"]["modality"], "text+image->text");
        let open = model_info(&item, &contexts, None).unwrap();
        assert_eq!(open["supported_reasoning_efforts"], json!(["low", "xhigh"]));
        assert_eq!(open["default_parameters"]["reasoning_effort"], "xhigh");
        assert!(model_info(&json!({}), &contexts, None).is_none());
    }
}
