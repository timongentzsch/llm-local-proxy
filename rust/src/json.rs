//! JSON helpers that keep the Python reference's observable behaviour.
//!
//! The wire formats were pinned against an implementation that leaned on
//! Python's truthiness, `str()` and `json.dumps`. Where those leak into
//! output -- an id built from `str(value)`, an envelope whose bytes a client
//! carries between turns -- the port has to agree byte for byte.

use serde_json::{Map, Number, Value};

pub type Object = Map<String, Value>;

/// Python truthiness: None, False, 0, "", [] and {} are false.
pub fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().map(|n| n != 0.0).unwrap_or(true),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

/// `value.get(key)` on something that may not be an object; missing is Null.
pub fn get<'a>(value: &'a Value, key: &str) -> &'a Value {
    static NULL: Value = Value::Null;
    value.get(key).unwrap_or(&NULL)
}

/// `value.get(key) or ""` for a string field.
pub fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

/// A JSON integer that is not a bool, as `isinstance(x, int) and not bool`.
pub fn integer(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) if !number.is_f64() => number.as_i64(),
        _ => None,
    }
}

/// Python's `str(value)`.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => py_repr(other),
    }
}

/// Python's `repr(value)` for JSON-shaped data.
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(number) => number_text(number),
        Value::String(text) => str_repr(text),
        Value::Array(items) => {
            let inner: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", inner.join(", "))
        }
        Value::Object(map) => {
            let inner: Vec<String> = map
                .iter()
                .map(|(key, value)| format!("{}: {}", str_repr(key), py_repr(value)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
    }
}

fn str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A number as Python prints it: floats keep a fraction or an exponent.
fn number_text(number: &Number) -> String {
    if !number.is_f64() {
        return number.to_string();
    }
    let value = number.as_f64().unwrap_or(0.0);
    let text = format!("{value:?}");
    match text.split_once('e') {
        // Rust writes 1e16 and 1e-7 where Python writes 1e+16 and 1e-07.
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(digits) => ('-', digits),
                None => ('+', exponent),
            };
            let mantissa = mantissa.strip_suffix(".0").unwrap_or(mantissa);
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => text,
    }
}

/// How `json.dumps` was called.
#[derive(Clone, Copy)]
pub struct Dumps {
    pub sort_keys: bool,
    /// `separators=(",", ":")`; otherwise Python's default `", "` and `": "`.
    pub compact: bool,
    /// Python's default: everything outside ASCII is written as `\uXXXX`.
    pub ensure_ascii: bool,
}

impl Dumps {
    /// `json.dumps(value)`.
    pub const DEFAULT: Dumps = Dumps {
        sort_keys: false,
        compact: false,
        ensure_ascii: true,
    };
    /// `json.dumps(value, separators=(",", ":"))`.
    pub const COMPACT: Dumps = Dumps {
        sort_keys: false,
        compact: true,
        ensure_ascii: true,
    };
    /// `json.dumps(value, sort_keys=True, separators=(",", ":"))`.
    pub const CANONICAL: Dumps = Dumps {
        sort_keys: true,
        compact: true,
        ensure_ascii: true,
    };

    pub fn unicode(mut self) -> Dumps {
        self.ensure_ascii = false;
        self
    }

    pub fn sorted(mut self) -> Dumps {
        self.sort_keys = true;
        self
    }
}

/// `json.dumps`, byte for byte.
pub fn dumps(value: &Value, style: Dumps) -> String {
    let mut out = String::new();
    write(value, style, &mut out);
    out
}

fn write(value: &Value, style: Dumps, out: &mut String) {
    let (item, key) = if style.compact {
        (",", ":")
    } else {
        (", ", ": ")
    };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => out.push_str(&number_text(number)),
        Value::String(text) => write_string(text, style.ensure_ascii, out),
        Value::Array(items) => {
            out.push('[');
            for (index, entry) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(item);
                }
                write(entry, style, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if style.sort_keys {
                // Python sorts str keys by code point, as Rust orders them.
                entries.sort_by(|a, b| a.0.chars().cmp(b.0.chars()));
            }
            for (index, (name, entry)) in entries.into_iter().enumerate() {
                if index > 0 {
                    out.push_str(item);
                }
                write_string(name, style.ensure_ascii, out);
                out.push_str(key);
                write(entry, style, out);
            }
            out.push('}');
        }
    }
}

fn write_string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && !c.is_ascii() => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// An object literal: `obj! { "type": "text", "text": text }`.
#[macro_export]
macro_rules! obj {
    ($($tokens:tt)*) => {
        match serde_json::json!({ $($tokens)* }) {
            serde_json::Value::Object(map) => map,
            _ => unreachable!(),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_matches_python() {
        let value = json!({"b": [1, 1.0, 1e16, 1e-7, 0.5, true, null], "a": "ß\n\"😀"});
        assert_eq!(
            dumps(&value, Dumps::DEFAULT),
            "{\"b\": [1, 1.0, 1e+16, 1e-07, 0.5, true, null], \"a\": \"\\u00df\\n\\\"\\ud83d\\ude00\"}"
        );
        assert_eq!(
            dumps(&value, Dumps::CANONICAL.unicode()),
            "{\"a\":\"ß\\n\\\"😀\",\"b\":[1,1.0,1e+16,1e-07,0.5,true,null]}"
        );
    }

    #[test]
    fn str_matches_python() {
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(3)), "3");
        assert_eq!(py_str(&json!(2.0)), "2.0");
        assert_eq!(py_str(&json!("x")), "x");
        assert_eq!(
            py_str(&json!({"a": ["it's", null]})),
            "{'a': [\"it's\", None]}"
        );
    }

    #[test]
    fn truthiness_matches_python() {
        for falsy in [
            json!(null),
            json!(false),
            json!(0),
            json!(0.0),
            json!(""),
            json!([]),
            json!({}),
        ] {
            assert!(!truthy(&falsy), "{falsy}");
        }
        for truth in [json!(1), json!("0"), json!([0]), json!({"a": null})] {
            assert!(truthy(&truth), "{truth}");
        }
    }
}
