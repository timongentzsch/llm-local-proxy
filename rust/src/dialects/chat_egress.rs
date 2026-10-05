//! Canonical stream events -> Chat Completions chunks and completions.

use crate::dialects::base::{Driver, Encoder};
use crate::error::Result;
use crate::ids::SharedIds;
use crate::ir::{Decoder, StreamEvent};
use serde_json::Value;

pub struct ChunkEncoder {
    driver: Driver,
}

impl ChunkEncoder {
    /// `now` is the creation time in whole seconds since the epoch.
    pub fn new(_model: &str, decoder: Box<dyn Decoder>, _ids: &SharedIds, _now: i64) -> Self {
        ChunkEncoder {
            driver: Driver::new(decoder),
        }
    }
}

impl Encoder for ChunkEncoder {
    fn driver(&mut self) -> &mut Driver {
        &mut self.driver
    }
    fn one(&mut self, _event: StreamEvent) -> Result<Vec<Value>> {
        todo!("port of dialects/openai/egress.py")
    }
    fn start(&mut self) -> Value {
        todo!()
    }
    fn finish(&mut self) -> Result<Vec<Value>> {
        todo!()
    }
    fn result(&mut self) -> Result<Value> {
        todo!()
    }
    fn set_id(&mut self, _id: String) {
        todo!()
    }
}
