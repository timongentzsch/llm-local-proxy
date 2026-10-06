//! Output formats shared by the OpenAI-shaped request dialects.

use crate::error::{Error, Result};
use crate::ir::OutputFormat;
use crate::json::{py_str, Object};
use crate::tools::optional_bool;
use serde_json::Value;

pub const VERBOSITY: [&str; 3] = ["low", "medium", "high"];

/// An optional string option restricted to its specified values.
pub fn enum_value(value: &Value, allowed: &[&str], name: &str) -> Result<String> {
    if value.is_null() {
        return Ok(String::new());
    }
    match value.as_str() {
        Some(text) if allowed.contains(&text) => Ok(text.to_string()),
        _ => {
            let mut sorted = allowed.to_vec();
            sorted.sort_unstable();
            Err(Error::request(format!(
                "{name} must be one of: {}",
                sorted.join(", ")
            )))
        }
    }
}

/// One output format, however its dialect wrapped the schema.
///
/// Chat Completions nests the schema under `response_format.json_schema`
/// while Responses spreads the same fields across `text.format`; only the
/// wrapper differs, so both hand the unwrapped fields to this check. A plain
/// text format constrains nothing and is reported as no format at all.
pub fn format_of(kind: &Value, fields: &Object) -> Result<Option<OutputFormat>> {
    match kind.as_str() {
        Some("text") => return Ok(None),
        Some("json_object") => {
            return Ok(Some(OutputFormat {
                kind: "json_object".into(),
                ..OutputFormat::default()
            }))
        }
        Some("json_schema") => {}
        _ => {
            return Err(Error::request(format!(
                "unsupported output format type: {}",
                py_str(kind)
            )))
        }
    }
    let invalid = || Error::request("json_schema output format requires name and schema");
    let name = match fields.get("name") {
        Some(Value::String(name)) if !name.is_empty() => name.clone(),
        _ => return Err(invalid()),
    };
    let Some(Value::Object(schema)) = fields.get("schema") else {
        return Err(invalid());
    };
    let strict = optional_bool(fields.get("strict").unwrap_or(&Value::Null), "strict")?;
    Ok(Some(OutputFormat {
        kind: "json_schema".into(),
        name,
        schema: Some(schema.clone()),
        strict: strict.unwrap_or(false),
    }))
}
