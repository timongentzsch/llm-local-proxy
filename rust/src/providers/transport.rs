//! HTTP plumbing shared by every provider transport.
//!
//! Only what is provably identical lives here. The 401-retry envelopes in the
//! two upstreams look alike but differ in URL, headers, token source and error
//! mapping, so they stay where they are.

use crate::error::{Error, Result};
use crate::providers::EventStream;
use futures_util::stream::{self, StreamExt};
use serde_json::Value;
use std::collections::VecDeque;
use std::time::Duration;

/// One client for every upstream call, so connections are reused.
///
/// A subscription endpoint that redirects is a failure, not a hop. `timeout`
/// bounds each read rather than the whole response: a generation may run for
/// many minutes, but an upstream that says nothing for this long has stalled.
pub fn client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(30).min(timeout))
        .read_timeout(timeout)
        .pool_idle_timeout(Duration::from_secs(90))
        .build()
        .expect("the TLS backend initialises")
}

/// An upstream URL, or its stand-in when `LLM_PROXY_TEST_UPSTREAM` names
/// one. That variable exists for the end-to-end tests, which serve canned
/// upstream answers from a local port; it replaces only the scheme and host.
pub fn endpoint(url: &str) -> String {
    match std::env::var("LLM_PROXY_TEST_UPSTREAM") {
        Ok(base) if !base.is_empty() => {
            let path = url.splitn(4, '/').nth(3).unwrap_or_default();
            format!("{}/{path}", base.trim_end_matches('/'))
        }
        _ => url.to_string(),
    }
}

/// `Retry-After` in whole seconds, when the upstream sent one.
pub fn retry_after(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let value = headers.get("retry-after")?.to_str().ok()?.trim();
    value
        .parse::<f64>()
        .ok()
        .filter(|s| *s >= 0.0)
        .map(|s| s.ceil() as u64)
}

/// Seconds from now until an epoch timestamp, at least one.
pub fn seconds_until(epoch: f64) -> Option<u64> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs_f64();
    (epoch > 0.0).then(|| (epoch - now).ceil().max(1.0) as u64)
}

/// An `application/x-www-form-urlencoded` body.
pub fn form(pairs: &[(&str, &str)]) -> String {
    let encode = |text: &str| -> String {
        text.bytes()
            .map(|byte| match byte {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'*' => {
                    (byte as char).to_string()
                }
                b' ' => "+".to_string(),
                other => format!("%{other:02X}"),
            })
            .collect()
    };
    pairs
        .iter()
        .map(|(key, value)| format!("{}={}", encode(key), encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

/// What went wrong reaching an upstream, in words a client can act on.
pub fn unreachable(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        "upstream timed out".into()
    } else if error.is_connect() {
        format!("upstream is unreachable: {error}")
    } else {
        format!("upstream connection failed: {error}")
    }
}

struct Frames {
    body: futures_util::stream::BoxStream<'static, reqwest::Result<bytes::Bytes>>,
    terminal_events: &'static [&'static str],
    buffer: Vec<u8>,
    data: Vec<String>,
    ready: VecDeque<Result<Value>>,
    terminal: bool,
    finished: bool,
}

impl Frames {
    /// One line of the stream; a blank line ends the frame collected so far.
    fn line(&mut self, raw: &[u8]) {
        let Ok(text) = std::str::from_utf8(raw) else {
            self.fail(Error::upstream("upstream stream is not valid UTF-8"));
            return;
        };
        let line = text.trim_end_matches(['\r', '\n']);
        if let Some(rest) = line.strip_prefix("data:") {
            self.data
                .push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        } else if line.is_empty() && !self.data.is_empty() {
            let payload = self.data.join("\n");
            self.data.clear();
            if payload == "[DONE]" {
                self.end();
            } else if !payload.is_empty() {
                self.payload(&payload);
            }
        }
    }

    fn payload(&mut self, payload: &str) {
        match serde_json::from_str::<Value>(payload) {
            Ok(event) if event.is_object() => {
                let kind = event
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                self.terminal |= self.terminal_events.contains(&kind);
                self.ready.push_back(Ok(event));
            }
            Ok(_) => self.fail(Error::upstream("upstream SSE payload must be an object")),
            Err(error) => self.fail(Error::upstream(format!(
                "upstream SSE payload is not valid JSON: {error}"
            ))),
        }
    }

    fn fail(&mut self, error: Error) {
        self.ready.push_back(Err(error));
        self.finished = true;
    }

    /// The upstream closed, or said it was done.
    fn end(&mut self) {
        if !self.finished && !self.terminal_events.is_empty() && !self.terminal {
            self.ready.push_back(Err(Error::upstream(
                "upstream stream ended before its terminal event",
            )));
        }
        self.finished = true;
    }

    fn feed(&mut self, chunk: &[u8]) {
        self.buffer.extend_from_slice(chunk);
        while !self.finished {
            let Some(end) = self.buffer.iter().position(|byte| *byte == b'\n') else {
                break;
            };
            let line: Vec<u8> = self.buffer.drain(..=end).collect();
            self.line(&line);
        }
    }
}

/// Decode complete SSE frames and reject an upstream that ends mid-response.
///
/// Multiple data lines form one JSON payload. A final unterminated frame is
/// incomplete, even when its bytes happen to form valid JSON. Dropping the
/// stream drops the response and closes the upstream connection.
pub fn read_events(
    response: reqwest::Response,
    terminal_events: &'static [&'static str],
) -> EventStream {
    let frames = Frames {
        body: response.bytes_stream().boxed(),
        terminal_events,
        buffer: Vec::new(),
        data: Vec::new(),
        ready: VecDeque::new(),
        terminal: false,
        finished: false,
    };
    Box::pin(stream::unfold(frames, |mut frames| async move {
        loop {
            if let Some(event) = frames.ready.pop_front() {
                return Some((event, frames));
            }
            if frames.finished {
                return None;
            }
            match frames.body.next().await {
                Some(Ok(chunk)) => frames.feed(&chunk),
                Some(Err(error)) => frames.fail(Error::upstream(unreachable(&error))),
                None => {
                    let rest = std::mem::take(&mut frames.buffer);
                    if !rest.is_empty() {
                        frames.line(&rest);
                    }
                    frames.end();
                }
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(terminal_events: &'static [&'static str]) -> Frames {
        Frames {
            body: stream::empty().boxed(),
            terminal_events,
            buffer: Vec::new(),
            data: Vec::new(),
            ready: VecDeque::new(),
            terminal: false,
            finished: false,
        }
    }

    fn drain(mut frames: Frames) -> Vec<Result<Value>> {
        frames.end();
        frames.ready.into_iter().collect()
    }

    #[test]
    fn multiline_frames_join_and_split_chunks_reassemble() {
        let mut state = frames(&["done"]);
        state.feed(b"event: x\ndata: {\"type\":\n");
        state.feed(b"data: \"done\"}\r\n\r\n");
        let events = drain(state);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].as_ref().unwrap()["type"], "done");
    }

    #[test]
    fn an_upstream_that_ends_early_is_an_error() {
        let mut state = frames(&["done"]);
        state.feed(b"data: {\"type\":\"delta\"}\n\n");
        // Valid JSON, but its frame was never terminated.
        state.feed(b"data: {\"type\":\"done\"}\n");
        let events = drain(state);
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[1].as_ref().unwrap_err().message(),
            "upstream stream ended before its terminal event"
        );
    }

    #[test]
    fn done_sentinel_and_non_objects() {
        let mut state = frames(&[]);
        state.feed(b"data: {\"a\":1}\n\ndata: [DONE]\n\ndata: {\"b\":2}\n\n");
        assert_eq!(drain(state).len(), 1);
        let mut state = frames(&[]);
        state.feed(b"data: [1]\n\n");
        assert!(drain(state)[0].is_err());
    }
}
