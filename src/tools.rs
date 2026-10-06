//! Shared tool contracts and wire encoders, independent of any provider.
//!
//! Common fields cross formats unchanged. Extra options retain their origin
//! and are rejected on a different format unless an adapter maps them.

use crate::error::{Error, Result};
use crate::ir::{
    ChoiceKind, FunctionTool, NamespaceMember, Source, Tool, ToolChoice, WebSearchTool,
};
use crate::json::{get, py_str, truthy, Object};
use indexmap::IndexMap;
use serde_json::{json, Value};
use sha1::{Digest, Sha1};
use std::collections::BTreeSet;

/// The Messages web search tool version the proxy requests.
pub const ANTHROPIC_WEB_SEARCH: &str = "web_search_20250305";
/// The longest tool name Anthropic accepts (`^[a-zA-Z0-9_-]{1,64}$`).
pub const MAX_TOOL_NAME: usize = 64;

/// Flattened tool name -> (namespace, name), in declaration order.
pub type Names = IndexMap<String, (String, String)>;

/// `tools` as a list of objects; absent is none.
pub fn definitions(value: &Value) -> Result<Vec<&Object>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_object()
                    .ok_or_else(|| Error::request("tools must be an array of objects"))
            })
            .collect(),
        _ => Err(Error::request("tools must be an array of objects")),
    }
}

pub fn optional_bool(value: &Value, name: &str) -> Result<Option<bool>> {
    match value {
        Value::Null => Ok(None),
        Value::Bool(flag) => Ok(Some(*flag)),
        _ => Err(Error::request(format!("{name} must be a boolean"))),
    }
}

/// A function tool from its wire definition. `schema_key` is "parameters",
/// or "input_schema" for Anthropic.
pub fn parse_function(value: &Object, source: &str, schema_key: &str) -> Result<FunctionTool> {
    let name = match value.get("name") {
        Some(Value::String(name)) if !name.trim().is_empty() => name.clone(),
        _ => return Err(Error::request("function tool name is required")),
    };
    let parameters = match value.get(schema_key) {
        None => crate::obj! { "type": "object" },
        Some(Value::Object(schema)) => schema.clone(),
        Some(_) => {
            return Err(Error::request(format!(
                "tool {schema_key} must be an object"
            )))
        }
    };
    let description = match value.get("description") {
        Some(description) if truthy(description) => py_str(description),
        _ => String::new(),
    };
    let strict = optional_bool(get_field(value, "strict"), "strict")?;
    let common = ["type", "name", schema_key, "description", "strict"];
    let options: Object = value
        .iter()
        .filter(|(key, _)| !common.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Ok(FunctionTool {
        name,
        parameters,
        description,
        strict,
        source: source.to_string(),
        options,
        cache: None,
    })
}

fn get_field<'a>(value: &'a Object, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    value.get(key).unwrap_or(&NULL)
}

/// A function tool for `target`; options from another format are refused.
pub fn render_function(tool: &FunctionTool, target: &str, schema_key: &str) -> Result<Object> {
    if !tool.options.is_empty() && tool.source != target {
        let mut names: Vec<&str> = tool.options.keys().map(String::as_str).collect();
        names.sort_unstable();
        return Err(Error::request(format!(
            "{target} cannot faithfully represent {} function tool fields: {}",
            tool.source,
            names.join(", ")
        )));
    }
    let mut result = Object::new();
    result.insert("name".into(), json!(tool.name));
    result.insert(schema_key.into(), Value::Object(tool.parameters.clone()));
    for (key, value) in &tool.options {
        result.insert(key.clone(), value.clone());
    }
    if !tool.description.is_empty() {
        result.insert("description".into(), json!(tool.description));
    }
    if let Some(strict) = tool.strict {
        result.insert("strict".into(), json!(strict));
    }
    Ok(result)
}

/// Require a JSON object; never turn a broken call into an empty call.
///
/// Request renderers pass `Error::request` (400); response translators pass
/// `Error::upstream` (502). Empty input stays valid for parameterless tools.
pub fn arguments(value: &Value, error: fn(String) -> Error) -> Result<Object> {
    const MESSAGE: &str = "tool call arguments must be a JSON object";
    let parsed;
    let value = match value {
        Value::String(text) if text.trim().is_empty() => return Ok(Object::new()),
        Value::String(text) => {
            parsed = parse_json(text).ok_or_else(|| error(MESSAGE.into()))?;
            &parsed
        }
        other => other,
    };
    match value {
        Value::Object(map) => Ok(map.clone()),
        _ => Err(error(MESSAGE.into())),
    }
}

/// `json.loads`: unlike serde, Python also accepts NaN and the infinities.
pub fn parse_json(text: &str) -> Option<Value> {
    serde_json::from_str(text).ok()
}

/// One flat, deterministic tool name for a namespaced tool.
///
/// Too-long names keep a readable prefix and end in a hash of the full name,
/// so a request and every later replay of it agree without shared state.
pub fn qualified_name(namespace: &str, name: &str) -> String {
    if namespace.is_empty() {
        return name.to_string();
    }
    let full = format!("{namespace}__{name}");
    if full.chars().count() <= MAX_TOOL_NAME {
        return full;
    }
    let digest = Sha1::digest(full.as_bytes());
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    let prefix: String = full.chars().take(MAX_TOOL_NAME - 9).collect();
    format!("{prefix}_{}", &hex[..8])
}

/// Namespaced tools as plain function tools, for targets without namespaces.
///
/// Returns the flat tool list and `{qualified name: (namespace, name)}` to
/// restore calls. A member that is not a function tool, or two tools that
/// would share a name, cannot be represented and is refused.
pub fn flatten(tools: &[Tool]) -> Result<(Vec<Tool>, Names)> {
    let mut flat = Vec::new();
    let mut names = Names::new();
    let add = |tool: &FunctionTool, namespace: &str, names: &mut Names| -> Result<Tool> {
        let name = qualified_name(namespace, &tool.name);
        if names.contains_key(&name) {
            return Err(Error::request(format!("duplicate tool name: {name}")));
        }
        names.insert(name.clone(), (namespace.to_string(), tool.name.clone()));
        Ok(Tool::Function(FunctionTool {
            name,
            ..tool.clone()
        }))
    };
    for tool in tools {
        match tool {
            Tool::Namespace(namespace) => {
                for member in &namespace.tools {
                    match member {
                        NamespaceMember::Function(function) => {
                            flat.push(add(function, &namespace.name, &mut names)?)
                        }
                        NamespaceMember::Native(item) => {
                            let kind = item
                                .get("type")
                                .map(py_str)
                                .unwrap_or_else(|| "unknown".into());
                            return Err(Error::request(format!(
                                "namespace {} contains a non-function tool: {kind}",
                                namespace.name
                            )));
                        }
                    }
                }
            }
            Tool::Function(function) => flat.push(add(function, "", &mut names)?),
            other => flat.push(other.clone()),
        }
    }
    names.retain(|_, (namespace, _)| !namespace.is_empty());
    Ok((flat, names))
}

pub fn responses_tool(tool: &Tool) -> Result<Object> {
    match tool {
        Tool::Native(item) => Ok(item.clone()),
        Tool::Namespace(namespace) => Ok(namespace.item.clone()),
        Tool::WebSearch(search) => responses_web_search(search),
        Tool::Function(function) => {
            let mut result = crate::obj! { "type": "function" };
            result.extend(render_function(function, "responses", "parameters")?);
            Ok(result)
        }
    }
}

fn unknown_keys(native: &Object, known: &[&str]) -> Vec<String> {
    native
        .keys()
        .filter(|key| !known.contains(&key.as_str()))
        .cloned()
        .collect::<BTreeSet<String>>()
        .into_iter()
        .collect()
}

/// A search tool as Responses defines it.
pub fn responses_web_search(tool: &WebSearchTool) -> Result<Object> {
    let native = &tool.native;
    if tool.source == Source::Responses {
        return Ok(native.clone());
    }
    // Anthropic `max_uses` caps how often the model searches; Responses has
    // no cap, and exceeding it changes cost, not what is searched. A cache
    // breakpoint is a hint as well.
    let mut unsupported = unknown_keys(
        native,
        &[
            "type",
            "name",
            "max_uses",
            "cache_control",
            "allowed_domains",
            "blocked_domains",
            "user_location",
        ],
    );
    if truthy(get_field(native, "blocked_domains")) {
        unsupported.push("blocked_domains".into());
    }
    if !unsupported.is_empty() {
        return Err(Error::request(format!(
            "Responses cannot faithfully represent Anthropic web_search options: {}",
            unsupported.join(", ")
        )));
    }
    let mut item = crate::obj! { "type": "web_search" };
    let allowed = get_field(native, "allowed_domains");
    if truthy(allowed) {
        item.insert("filters".into(), json!({ "allowed_domains": allowed }));
    }
    let location = get_field(native, "user_location");
    if truthy(location) {
        item.insert("user_location".into(), location.clone());
    }
    Ok(item)
}

/// A search tool as the Messages API defines it.
pub fn anthropic_web_search(tool: &WebSearchTool) -> Result<Object> {
    let native = &tool.native;
    if tool.source == Source::Anthropic {
        return Ok(native.clone());
    }
    // The context size is a hint Messages has no control for, and live
    // access is its only mode; options that change what is searched must map.
    let filters = match get_field(native, "filters") {
        value if truthy(value) => value.clone(),
        _ => json!({}),
    };
    let mut unsupported = unknown_keys(
        native,
        &[
            "type",
            "search_context_size",
            "external_web_access",
            "filters",
            "user_location",
        ],
    );
    if native.get("external_web_access") == Some(&Value::Bool(false)) {
        unsupported.push("external_web_access".into());
    }
    let filters_ok = filters
        .as_object()
        .map(|map| map.keys().all(|key| key == "allowed_domains"))
        .unwrap_or(false);
    if !filters_ok {
        unsupported.push("filters".into());
    }
    if !unsupported.is_empty() {
        return Err(Error::request(format!(
            "Messages cannot faithfully represent Responses web_search options: {}",
            unsupported.join(", ")
        )));
    }
    let mut item = crate::obj! { "type": ANTHROPIC_WEB_SEARCH, "name": "web_search" };
    let allowed = get(&filters, "allowed_domains");
    if truthy(allowed) {
        item.insert("allowed_domains".into(), allowed.clone());
    }
    let location = get_field(native, "user_location");
    if truthy(location) {
        item.insert("user_location".into(), location.clone());
    }
    Ok(item)
}

/// OpenAI tool choice. `nested` removes the Chat Completions function wrapper.
pub fn parse_choice(value: &Value, nested: bool) -> Result<Option<ToolChoice>> {
    match value {
        Value::Null => return Ok(None),
        Value::String(kind) => {
            let kind = match kind.as_str() {
                "auto" => Some(ChoiceKind::Auto),
                "none" => Some(ChoiceKind::None),
                "required" => Some(ChoiceKind::Required),
                _ => None,
            };
            if let Some(kind) = kind {
                return Ok(Some(ToolChoice {
                    kind,
                    name: String::new(),
                }));
            }
        }
        Value::Object(map) if map.get("type") == Some(&json!("function")) => {
            let function = if nested {
                get_field(map, "function")
            } else {
                value
            };
            if let Some(Value::String(name)) = function.get("name") {
                if !name.is_empty() {
                    return Ok(Some(ToolChoice {
                        kind: ChoiceKind::Tool,
                        name: name.clone(),
                    }));
                }
            }
        }
        _ => {}
    }
    Err(Error::request("unsupported tool_choice"))
}

pub fn responses_choice(choice: Option<&ToolChoice>) -> Value {
    match choice {
        None => json!("auto"),
        Some(choice) if choice.kind == ChoiceKind::Tool => {
            json!({ "type": "function", "name": choice.name })
        }
        Some(choice) => json!(choice.kind.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_qualified_names_end_in_a_hash() {
        // Pinned: a request and every later replay of it must agree.
        assert_eq!(qualified_name("", "f"), "f");
        assert_eq!(qualified_name("ns", "f"), "ns__f");
        let name = qualified_name(
            "mcp__codex_apps__codex_document_control",
            "_get_document_tool_schemas",
        );
        assert_eq!(name.len(), MAX_TOOL_NAME);
        assert_eq!(
            name,
            "mcp__codex_apps__codex_document_control___get_document__97702ab1"
        );
    }
}
