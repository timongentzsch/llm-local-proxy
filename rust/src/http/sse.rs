//! Server-sent event plumbing.
//!
//! Anthropic Messages and OpenAI Responses name every frame after its `type`
//! and end without a sentinel; Chat Completions sends anonymous frames and
//! ends with `data: [DONE]`. The route selects which; the keepalive belongs
//! to the dialect.

use crate::dialects::base::Encoder;
use crate::dialects::Dialect;
use crate::error::{Error, Result};
use crate::providers::EventStream;
use bytes::Bytes;
use futures_util::stream::{self, Stream, StreamExt};
use serde_json::Value;
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;

/// How long the upstream may stay silent before the client is sent a
/// keepalive, so idle connections are not closed in between.
const HEARTBEAT: Duration = Duration::from_secs(15);

/// One SSE frame; the event line is written only for named frames.
pub fn render(data: &Value, named: bool) -> Bytes {
    let mut frame = Vec::with_capacity(128);
    if named {
        if let Some(kind) = data.get("type").and_then(Value::as_str) {
            frame.extend_from_slice(b"event: ");
            frame.extend_from_slice(kind.as_bytes());
            frame.push(b'\n');
        }
    }
    frame.extend_from_slice(b"data: ");
    serde_json::to_writer(&mut frame, data).expect("a JSON value serialises");
    frame.extend_from_slice(b"\n\n");
    Bytes::from(frame)
}

/// A bug in a translator must cost one response, not the process.
pub fn guarded<T>(run: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(run)).unwrap_or_else(|panic| {
        let detail = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("panic");
        Err(Error::upstream(format!("internal error: {detail}")))
    })
}

struct Frames {
    /// None once the stream is over; dropping it closes the upstream.
    events: Option<EventStream>,
    encoder: Box<dyn Encoder>,
    dialect: &'static Dialect,
    named: bool,
    ready: VecDeque<Bytes>,
}

impl Frames {
    fn push(&mut self, frames: Vec<Value>) {
        let named = self.named;
        self.ready
            .extend(frames.iter().map(|frame| render(frame, named)));
    }

    /// Close the response; anonymous streams say so with a sentinel.
    fn end(&mut self) {
        self.events = None;
        if !self.named {
            self.ready
                .push_back(Bytes::from_static(b"data: [DONE]\n\n"));
        }
    }

    /// Headers are already sent, so a failure travels in-band.
    fn fail(&mut self, error: &Error) {
        let message = error.message();
        let frame = self
            .encoder
            .error(message)
            .unwrap_or_else(|| (self.dialect.error)(502, message));
        self.push(vec![frame]);
        self.end();
    }

    async fn advance(&mut self) {
        let Some(events) = self.events.as_mut() else {
            return;
        };
        match tokio::time::timeout(HEARTBEAT, events.next()).await {
            Err(_) => self
                .ready
                .push_back(Bytes::from_static(self.dialect.keepalive)),
            Ok(Some(Ok(event))) => match guarded(|| self.encoder.feed(&event)) {
                Ok(frames) => self.push(frames),
                Err(error) => self.fail(&error),
            },
            Ok(Some(Err(error))) => self.fail(&error),
            Ok(None) => match guarded(|| self.encoder.finish()) {
                Ok(frames) => {
                    self.push(frames);
                    self.end();
                }
                Err(error) => self.fail(&error),
            },
        }
    }
}

/// The bytes of one streamed response: the opening frame at once, then every
/// upstream event as this dialect's frames, with keepalives while it is quiet.
///
/// Dropping the stream -- which is what a client hanging up does -- drops the
/// upstream connection with it.
pub fn body(
    events: EventStream,
    mut encoder: Box<dyn Encoder>,
    dialect: &'static Dialect,
    named: bool,
) -> impl Stream<Item = Bytes> + Send {
    let start = render(&encoder.start(), named);
    let state = Frames {
        events: Some(events),
        encoder,
        dialect,
        named,
        ready: VecDeque::from([start]),
    };
    stream::unfold(state, |mut state| async move {
        loop {
            if let Some(frame) = state.ready.pop_front() {
                return Some((frame, state));
            }
            state.events.as_ref()?;
            state.advance().await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialects::base::Driver;
    use crate::dialects::{ANTHROPIC, OPENAI};
    use crate::ir::{Decoder, StreamEvent};
    use serde_json::json;

    #[test]
    fn frames_are_named_only_for_named_streams() {
        let frame = json!({"type": "response.completed", "n": "ß"});
        assert_eq!(
            render(&frame, true),
            "event: response.completed\ndata: {\"type\":\"response.completed\",\"n\":\"ß\"}\n\n"
        );
        assert_eq!(
            render(&json!({"id": "chatcmpl-1"}), false),
            "data: {\"id\":\"chatcmpl-1\"}\n\n"
        );
    }

    struct Text;

    impl Decoder for Text {
        fn decode(&mut self, event: &Value) -> Result<Vec<StreamEvent>> {
            match event["boom"].as_str() {
                Some("error") => Err(Error::upstream("decoder refused")),
                Some(_) => panic!("decoder bug"),
                None => Ok(Vec::new()),
            }
        }
        fn finish(&mut self) -> Result<Vec<StreamEvent>> {
            Ok(Vec::new())
        }
    }

    struct Echo(Driver);

    impl Encoder for Echo {
        fn driver(&mut self) -> &mut Driver {
            &mut self.0
        }
        fn one(&mut self, _event: StreamEvent) -> Result<Vec<Value>> {
            Ok(Vec::new())
        }
        fn start(&mut self) -> Value {
            json!({"type": "start"})
        }
        fn finish(&mut self) -> Result<Vec<Value>> {
            Ok(vec![json!({"type": "stop"})])
        }
        fn result(&mut self) -> Result<Value> {
            Ok(json!({}))
        }
        fn set_id(&mut self, _id: String) {}
        fn feed(&mut self, event: &Value) -> Result<Vec<Value>> {
            self.0.decoder.decode(event)?;
            Ok(vec![event.clone()])
        }
    }

    async fn run(events: Vec<Result<Value>>, dialect: &'static Dialect, named: bool) -> String {
        let encoder = Box::new(Echo(Driver::new(Box::new(Text))));
        let frames: Vec<Bytes> = body(Box::pin(stream::iter(events)), encoder, dialect, named)
            .collect()
            .await;
        String::from_utf8(frames.concat()).unwrap()
    }

    #[tokio::test]
    async fn a_stream_starts_at_once_and_ends_by_its_dialects_rule() {
        let named = run(vec![Ok(json!({"type": "delta"}))], &ANTHROPIC, true).await;
        assert_eq!(
            named,
            "event: start\ndata: {\"type\":\"start\"}\n\nevent: delta\ndata: {\"type\":\"delta\"}\n\nevent: stop\ndata: {\"type\":\"stop\"}\n\n"
        );
        let anonymous = run(vec![], &OPENAI, false).await;
        assert!(anonymous.ends_with("data: {\"type\":\"stop\"}\n\ndata: [DONE]\n\n"));
    }

    #[tokio::test]
    async fn failures_travel_in_band_and_end_the_stream() {
        let upstream = run(
            vec![Err(Error::provider(429, "limited")), Ok(json!({}))],
            &OPENAI,
            false,
        )
        .await;
        assert!(upstream.contains("\"message\":\"limited\""));
        assert!(upstream.ends_with("data: [DONE]\n\n"));
        assert!(!upstream.contains("stop"));

        let refused = run(vec![Ok(json!({"boom": "error"}))], &ANTHROPIC, true).await;
        assert!(refused.contains("\"type\":\"api_error\",\"message\":\"decoder refused\""));

        let bug = run(vec![Ok(json!({"boom": "panic"}))], &ANTHROPIC, true).await;
        assert!(bug.contains("internal error: decoder bug"));
    }

    #[tokio::test(start_paused = true)]
    async fn a_quiet_upstream_gets_keepalives() {
        let quiet = stream::once(async {
            tokio::time::sleep(Duration::from_secs(40)).await;
            Ok(json!({"type": "late"}))
        });
        let encoder = Box::new(Echo(Driver::new(Box::new(Text))));
        let frames: Vec<Bytes> = body(Box::pin(quiet), encoder, &ANTHROPIC, true)
            .collect()
            .await;
        let pings = frames
            .iter()
            .filter(|frame| frame.starts_with(b"event: ping"))
            .count();
        assert_eq!(pings, 2);
    }
}
