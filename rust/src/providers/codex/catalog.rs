//! The Codex model catalog.

use crate::json::{get, py_str, truthy};
use crate::providers::catalog::{model_info as shared, ModelInfo};
use serde_json::{json, Value};

/// The backend's model records as listing entries, in its own order of
/// preference. Hidden models are not offered; the first one is the default.
/// Effort tiers are intersected with what the transport accepts, when it said.
pub fn catalog(models: &[Value], transport_efforts: Option<&[String]>) -> Vec<Value> {
    let mut listed: Vec<&Value> = models
        .iter()
        .filter(|item| get(item, "visibility") == "list")
        .collect();
    listed.sort_by_key(|item| get(item, "priority").as_i64().unwrap_or(i64::MAX));
    listed
        .iter()
        .enumerate()
        .filter_map(|(index, item)| model_info(item, transport_efforts, index == 0))
        .collect()
}

fn model_info(
    item: &Value,
    transport_efforts: Option<&[String]>,
    is_default: bool,
) -> Option<Value> {
    let slug = get(item, "slug");
    let model = truthy(slug).then(|| py_str(slug))?;
    let accepts = |effort: &Value| match transport_efforts {
        None => true,
        Some(accepted) => effort
            .as_str()
            .is_some_and(|e| accepted.iter().any(|a| a == e)),
    };
    let efforts: Vec<Value> = get(item, "supported_reasoning_levels")
        .as_array()
        .into_iter()
        .flatten()
        .map(|level| get(level, "effort"))
        .filter(|effort| truthy(effort) && accepts(effort))
        .cloned()
        .collect();
    let default_effort = get(item, "default_reasoning_level");
    let display = get(item, "display_name");
    Some(shared(ModelInfo {
        model: &model,
        name: if truthy(display) {
            display.clone()
        } else {
            json!(model)
        },
        owned_by: "openai",
        modalities: get(item, "input_modalities")
            .as_array()
            .filter(|list| !list.is_empty())
            .map(|list| list.iter().map(py_str).collect()),
        default_parameters: (truthy(default_effort) && accepts(default_effort))
            .then(|| json!({ "reasoning_effort": default_effort })),
        reasoning_efforts: efforts,
        context_length: get(item, "context_window")
            .as_i64()
            .filter(|n| *n > 0)
            .unwrap_or(0),
        is_default,
        ..Default::default()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_models_are_left_out_and_the_preferred_one_is_the_default() {
        let models = [
            json!({"slug": "b", "display_name": "B", "visibility": "list", "priority": 5}),
            json!({"slug": "hidden", "visibility": "hide", "priority": 0}),
            json!({
                "slug": "a",
                "display_name": "A",
                "visibility": "list",
                "priority": 1,
                "supported_reasoning_levels": [{"effort": "low"}, {"effort": "xhigh"}],
                "default_reasoning_level": "xhigh",
                "input_modalities": ["text", "image"],
                "context_window": 272000,
            }),
        ];
        let accepted = ["low".to_string(), "medium".to_string()];
        let listed = catalog(&models, Some(&accepted));
        let ids: Vec<&str> = listed.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, ["a", "b"]);
        assert_eq!(listed[0]["is_default"], true);
        assert_eq!(listed[1]["is_default"], false);
        assert_eq!(listed[0]["supported_reasoning_efforts"], json!(["low"]));
        assert_eq!(listed[0]["default_parameters"], Value::Null);
        assert_eq!(listed[0]["context_length"], 272000);
        assert_eq!(listed[0]["architecture"]["modality"], "text+image->text");
        let open = catalog(&models, None);
        assert_eq!(
            open[0]["supported_reasoning_efforts"],
            json!(["low", "xhigh"])
        );
        assert_eq!(open[0]["default_parameters"]["reasoning_effort"], "xhigh");
    }
}
