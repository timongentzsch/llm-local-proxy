//! Codex Responses API SSE events -> canonical stream events.

use crate::error::Result;
use crate::ir::{Decoder, StreamEvent};
use crate::reasoning::ReasoningCache;
use serde_json::Value;
use std::sync::Arc;

pub struct CodexDecoder {}

impl CodexDecoder {
    pub fn new(_cache: Arc<ReasoningCache>) -> Self {
        todo!("port of providers/codex/events.py")
    }
}

impl Decoder for CodexDecoder {
    fn decode(&mut self, _event: &Value) -> Result<Vec<StreamEvent>> {
        todo!()
    }
    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        todo!()
    }
}
