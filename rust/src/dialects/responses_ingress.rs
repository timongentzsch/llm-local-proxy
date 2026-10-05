//! Responses request bodies -> ChatRequest.

use crate::error::Result;
use crate::ir::ChatRequest;
use crate::json::Object;

pub fn parse(_body: &Object, _session: &str) -> Result<ChatRequest> {
    todo!("port of dialects/openai/responses_ingress.py")
}
