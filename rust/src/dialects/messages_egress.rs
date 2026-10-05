//! Canonical stream events -> Anthropic Messages frames and messages.

use crate::dialects::base::{Driver, Encoder};
use crate::error::Result;
use crate::ids::SharedIds;
use crate::ir::{Decoder, StreamEvent};
use serde_json::Value;

pub struct MessageEncoder {
    driver: Driver,
}

impl MessageEncoder {
    pub fn new(_model: &str, decoder: Box<dyn Decoder>, _ids: &SharedIds) -> Self {
        MessageEncoder {
            driver: Driver::new(decoder),
        }
    }
}

impl Encoder for MessageEncoder {
    fn driver(&mut self) -> &mut Driver {
        &mut self.driver
    }
    fn one(&mut self, _event: StreamEvent) -> Result<Vec<Value>> {
        todo!("port of dialects/anthropic/egress.py")
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
