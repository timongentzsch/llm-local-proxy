//! Claude Messages SSE events -> canonical stream events.

use crate::error::Result;
use crate::ids::SharedIds;
use crate::ir::{Decoder, StreamEvent};
use crate::reasoning::ReasoningCache;
use crate::tools::Names;
use serde_json::Value;
use std::sync::Arc;

pub struct ClaudeDecoder {}

impl ClaudeDecoder {
    /// `names` restores flattened tool names, from `tools::flatten`.
    pub fn new(_cache: Option<Arc<ReasoningCache>>, _names: Names, _ids: SharedIds) -> Self {
        todo!("port of providers/claude/events.py")
    }
}

impl Decoder for ClaudeDecoder {
    fn decode(&mut self, _event: &Value) -> Result<Vec<StreamEvent>> {
        todo!()
    }
    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        todo!()
    }
}
