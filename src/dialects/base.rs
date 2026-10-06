//! What every dialect's encoder shares: driving a provider's decoder.

use crate::error::{Error, Result};
use crate::ir::{Decoder, StreamEvent};
use serde_json::Value;

/// Flatten text-only content without silently discarding other parts.
pub fn block_text(parts: &[Value]) -> Result<String> {
    let mut texts = Vec::with_capacity(parts.len());
    for part in parts {
        match (part.get("type").and_then(Value::as_str), part.get("text")) {
            (Some("text"), Some(Value::String(text))) if part.is_object() => {
                texts.push(text.as_str())
            }
            _ => return Err(Error::request("content must contain text blocks")),
        }
    }
    Ok(texts.join("\n"))
}

/// The decoder an encoder drives, and whether its end has been collected.
pub struct Driver {
    pub decoder: Box<dyn Decoder>,
    drained: bool,
}

impl Driver {
    pub fn new(decoder: Box<dyn Decoder>) -> Self {
        Driver {
            decoder,
            drained: false,
        }
    }
}

/// Drives one provider's decoder and shapes its events for one dialect.
///
/// An implementation supplies `one` (one event to frames) and the lifecycle
/// ends; `feed` and `drain` are the same for every dialect.
pub trait Encoder: Send {
    fn driver(&mut self) -> &mut Driver;

    /// One canonical event as this dialect's frames.
    fn one(&mut self, event: StreamEvent) -> Result<Vec<Value>>;

    /// The frame that opens a stream.
    fn start(&mut self) -> Value;

    /// The frames that close a stream.
    fn finish(&mut self) -> Result<Vec<Value>>;

    /// The whole response, for a client that did not ask to stream.
    fn result(&mut self) -> Result<Value>;

    /// A mid-stream failure frame; None sends the dialect's error body.
    fn error(&mut self, _message: &str) -> Option<Value> {
        None
    }

    /// The response id, and for formats that carry one its creation time.
    /// Settable so a replay can pin what a recorded case generated.
    fn set_id(&mut self, id: String);
    fn set_created(&mut self, _created: i64) {}

    fn feed(&mut self, event: &Value) -> Result<Vec<Value>> {
        let events = self.driver().decoder.decode(event)?;
        self.encode(events)
    }

    /// Collect whatever the decoder only knows once the stream ends.
    fn drain(&mut self) -> Result<Vec<Value>> {
        if self.driver().drained {
            return Ok(Vec::new());
        }
        self.driver().drained = true;
        let events = self.driver().decoder.finish()?;
        self.encode(events)
    }

    fn encode(&mut self, events: Vec<StreamEvent>) -> Result<Vec<Value>> {
        let mut frames = Vec::new();
        for event in events {
            frames.extend(self.one(event)?);
        }
        Ok(frames)
    }
}
