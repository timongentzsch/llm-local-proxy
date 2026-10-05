//! Codex reasoning items carried through Anthropic thinking blocks.
//!
//! Anthropic requires every thinking block to have an opaque signature and to
//! be returned byte-for-byte on the next turn. Codex instead returns one
//! opaque Responses reasoning item. This envelope uses Anthropic's signature
//! slot to carry that item without inventing an Anthropic signature or
//! dropping Codex's encrypted content.

use crate::error::{Error, Result};
use crate::json::{dumps, Dumps, Object};
use crate::obj;
use base64::Engine;
use serde_json::Value;

pub const ENVELOPE_PREFIX: &str = "llpv1-codex-reasoning:";

#[derive(Debug, Clone, PartialEq)]
pub struct Unpacked {
    pub item: Object,
    pub thinking: String,
}

pub fn pack(item: &Object, thinking: &str) -> String {
    let payload = dumps(
        &Value::Object(obj! { "item": item, "thinking": thinking }),
        Dumps::CANONICAL,
    );
    format!(
        "{ENVELOPE_PREFIX}{}",
        base64::engine::general_purpose::URL_SAFE.encode(payload.as_bytes())
    )
}

fn malformed() -> Error {
    Error::upstream("malformed Codex reasoning signature")
}

/// `base64.urlsafe_b64decode`: both alphabets are accepted and stray
/// characters are skipped, but the padding must be right.
fn decode(text: &str) -> Option<Vec<u8>> {
    let cleaned: String = text
        .chars()
        .map(|ch| match ch {
            '-' => '+',
            '_' => '/',
            other => other,
        })
        .filter(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '='))
        .collect();
    base64::engine::general_purpose::STANDARD
        .decode(cleaned)
        .ok()
}

/// Return a bridge item, None for a real Anthropic signature.
///
/// A signature claiming our prefix but carrying a damaged payload is an
/// error, not a foreign signature: silently accepting it would lose required
/// Codex reasoning on the following tool-result turn.
pub fn unpack(signature: &str) -> Result<Option<Unpacked>> {
    let Some(rest) = signature.strip_prefix(ENVELOPE_PREFIX) else {
        return Ok(None);
    };
    let payload = decode(rest).ok_or_else(malformed)?;
    let text = String::from_utf8(payload).map_err(|_| malformed())?;
    let envelope: Value = serde_json::from_str(&text).map_err(|_| malformed())?;
    let item = envelope.get("item").and_then(Value::as_object);
    let thinking = envelope.get("thinking").and_then(Value::as_str);
    match (item, thinking) {
        (Some(item), Some(thinking))
            if item.get("type").and_then(Value::as_str) == Some("reasoning")
                && item
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .is_some_and(|content| !content.is_empty()) =>
        {
            Ok(Some(Unpacked {
                item: item.clone(),
                thinking: thinking.to_string(),
            }))
        }
        _ => Err(malformed()),
    }
}
