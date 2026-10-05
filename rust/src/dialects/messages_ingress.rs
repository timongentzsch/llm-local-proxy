//! Anthropic Messages request bodies -> ChatRequest.

use crate::error::Result;
use crate::ir::ChatRequest;
use crate::json::Object;

pub fn parse(_body: &Object, _session: &str) -> Result<ChatRequest> {
    todo!("port of dialects/anthropic/ingress.py")
}

/// Parse a count_tokens body, which has no max_tokens.
pub fn parse_count(_body: &Object, _session: &str) -> Result<ChatRequest> {
    todo!("port of dialects/anthropic/ingress.py")
}
