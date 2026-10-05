//! ChatRequest -> the Codex Responses request body.

use crate::error::Result;
use crate::ir::ChatRequest;
use crate::json::Object;
use crate::reasoning::ReasoningCache;

/// The upstream body and the prompt-cache key it was given.
pub fn build(
    _request: &ChatRequest,
    _cache: &ReasoningCache,
    _reasoning_efforts: Option<&[String]>,
) -> Result<(Object, String)> {
    todo!("port of providers/codex/request.py")
}
