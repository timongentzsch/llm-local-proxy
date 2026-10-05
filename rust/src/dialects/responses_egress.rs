//! Canonical stream events -> Responses events and response objects.

use crate::dialects::base::{Driver, Encoder};
use crate::error::Result;
use crate::ids::SharedIds;
use crate::ir::{ChatRequest, Decoder, StreamEvent};
use serde_json::Value;

pub struct ResponseEncoder {
    driver: Driver,
}

impl ResponseEncoder {
    /// `request` is what the client sent, echoed back in every response
    /// object; `now` is the creation time in whole seconds since the epoch.
    pub fn new(
        _model: &str,
        decoder: Box<dyn Decoder>,
        _request: Option<ChatRequest>,
        _ids: &SharedIds,
        _now: i64,
    ) -> Self {
        ResponseEncoder {
            driver: Driver::new(decoder),
        }
    }
}

impl Encoder for ResponseEncoder {
    fn driver(&mut self) -> &mut Driver {
        &mut self.driver
    }
    fn one(&mut self, _event: StreamEvent) -> Result<Vec<Value>> {
        todo!("port of dialects/openai/responses_egress.py")
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
