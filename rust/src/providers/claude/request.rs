//! ChatRequest -> the Claude Messages request body.

use crate::error::Result;
use crate::ids::Ids;
use crate::ir::ChatRequest;
use crate::json::Object;
use crate::reasoning::ReasoningCache;

/// What the catalog knows about the model being asked.
#[derive(Default, Clone, Copy)]
pub struct Options<'a> {
    /// The model's maximum output, used when the client names no max_tokens.
    pub max_output: Option<i64>,
    /// How the model takes thinking ("adaptive", "enabled", ...), if known.
    pub thinking: Option<&'a str>,
    /// The effort tiers the catalog lists; None when it lists none at all.
    pub reasoning_efforts: Option<&'a [String]>,
    pub reasoning_cache: Option<&'a ReasoningCache>,
}

/// The upstream body and the beta features it needs.
pub fn build(
    _request: &ChatRequest,
    _model: &str,
    _options: Options<'_>,
    _ids: &dyn Ids,
) -> Result<(Object, Vec<String>)> {
    todo!("port of providers/claude/request.py")
}
