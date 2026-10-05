//! Claude thinking blocks carried verbatim through a Responses reasoning item.
//!
//! Claude signs the thinking blocks of an assistant turn and requires them back
//! exactly as it produced them: the block kind, its text and its signature are
//! all part of what it verifies. A Responses item has one opaque slot for that,
//! so the whole block is packed into `encrypted_content` and unpacked on replay.
//!
//! Nothing is rebuilt from the readable `summary`, which a client may shorten or
//! drop: reconstructing a block from its signature is what upstream rejects as
//! "blocks must remain as they were in the original response".
//!
//! Each envelope records where its block sat in the turn, so a client that
//! reorders them or drops one from the middle is caught here rather than
//! upstream. A dropped trailing block still leaves ordinals reading 0..n-1 --
//! the total is not known until the turn ends -- so the reasoning cache's count
//! catches that one instead.
//!
//! Only a replayable block is packed [empirical]: the subscription edge signs
//! reasoning whose text it never streams, and a signature covering text that
//! never arrived cannot be sent back.
//!
//! The version tag is part of the payload contract. Any change to the shape
//! below must bump it, so an older build's blobs are reported as an unreadable
//! version rather than misread. Clients keep append-only histories, so a build
//! that rejects what an earlier one wrote strands every later turn of those
//! sessions.

use crate::json::{dumps, integer, Dumps, Object};
use base64::alphabet;
use base64::engine::{general_purpose, DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use base64::Engine;
use serde_json::Value;

/// Version tag, so a later shape change is detectable instead of misread.
/// v1 carried the bare block; v2 wraps it with its ordinal.
pub const ENVELOPE_PREFIX: &str = "llpv2-claude-thinking:";

/// The two block kinds Claude signs, and the fields each one must carry.
fn block_fields(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "thinking" => Some(&["thinking", "signature"]),
        "redacted_thinking" => Some(&["data"]),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Ok,
    /// No envelope of ours: another upstream's blob, or a pre-envelope session.
    Foreign,
    /// Ours by prefix, but a version this build cannot read.
    BadVersion,
    /// Ours by prefix and version, and damaged.
    Malformed,
    /// Ours and intact, carrying a block Claude will not accept back: it
    /// signed reasoning whose text it never streamed, and the signature
    /// covers that text rather than the empty string stored here.
    Withheld,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unpacked {
    pub outcome: Outcome,
    pub block: Option<Object>,
    /// Position of this block among the signed blocks of its assistant turn.
    pub ordinal: i64,
}

impl Unpacked {
    pub fn without_block(outcome: Outcome) -> Self {
        Unpacked {
            outcome,
            block: None,
            ordinal: 0,
        }
    }
}

/// The opaque `encrypted_content` carrying one Claude thinking block.
pub fn pack(block: &Object, ordinal: usize) -> String {
    let mut envelope = Object::new();
    envelope.insert("n".into(), Value::from(ordinal));
    envelope.insert("block".into(), Value::Object(block.clone()));
    let payload = dumps(&Value::Object(envelope), Dumps::CANONICAL);
    format!(
        "{ENVELOPE_PREFIX}{}",
        general_purpose::URL_SAFE.encode(payload)
    )
}

fn valid(block: &Value) -> bool {
    let Some(map) = block.as_object() else {
        return false;
    };
    let Some(required) = map
        .get("type")
        .and_then(Value::as_str)
        .and_then(block_fields)
    else {
        return false;
    };
    required
        .iter()
        .all(|field| map.get(*field).is_some_and(Value::is_string))
}

/// `base64.urlsafe_b64decode`, which skips characters outside the alphabet
/// and insists on padding only where the data needs it.
fn urlsafe_decode(text: &str) -> Option<Vec<u8>> {
    let data: String = text
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '+' | '/'))
        .map(|c| match c {
            '+' => '-',
            '/' => '_',
            other => other,
        })
        .collect();
    let padding = text.chars().filter(|c| *c == '=').count();
    match data.len() % 4 {
        1 => return None,
        0 => {}
        rest if padding < 4 - rest => return None,
        _ => {}
    }
    const LENIENT: GeneralPurpose = GeneralPurpose::new(
        &alphabet::URL_SAFE,
        GeneralPurposeConfig::new()
            .with_decode_padding_mode(DecodePaddingMode::Indifferent)
            .with_decode_allow_trailing_bits(true),
    );
    LENIENT.decode(data).ok()
}

/// Classify one `encrypted_content` blob and recover its block.
pub fn unpack(encrypted: &Value) -> Unpacked {
    let Some(encrypted) = encrypted.as_str().filter(|text| !text.is_empty()) else {
        return Unpacked::without_block(Outcome::Foreign);
    };
    let Some(body) = encrypted.strip_prefix(ENVELOPE_PREFIX) else {
        // A bare `llp` prefix we do not know is a version we cannot honour; any
        // other blob was simply written by something else.
        let head = encrypted.split(':').next().unwrap_or("");
        if head.starts_with("llp") && head.ends_with("-claude-thinking") {
            return Unpacked::without_block(Outcome::BadVersion);
        }
        return Unpacked::without_block(Outcome::Foreign);
    };
    let malformed = || Unpacked::without_block(Outcome::Malformed);
    let Some(payload) = urlsafe_decode(body) else {
        return malformed();
    };
    let Ok(payload) = String::from_utf8(payload) else {
        return malformed();
    };
    let Ok(Value::Object(mut envelope)) = serde_json::from_str::<Value>(&payload) else {
        return malformed();
    };
    if !envelope.get("block").is_some_and(valid) {
        return malformed();
    }
    let Some(ordinal) = envelope
        .get("n")
        .and_then(integer)
        .filter(|ordinal| *ordinal >= 0)
    else {
        return malformed();
    };
    let Some(Value::Object(block)) = envelope.remove("block") else {
        return malformed();
    };
    Unpacked {
        outcome: Outcome::Ok,
        block: Some(block),
        ordinal,
    }
}
