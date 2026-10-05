//! ChatGPT's private Codex transport.

use super::app_server::AppServer;
use super::usage::read_usage;
use crate::error::{Error, Result};
use crate::json::{dumps, py_str, Dumps, Object};
use crate::ledger::{track, TokenLedger, UsageTracker};
use crate::providers::transport::{endpoint, read_events, unreachable};
use crate::providers::EventStream;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
const USER_AGENT: &str = concat!("llm-local-proxy/", env!("CARGO_PKG_VERSION"));
/// The events that end a response; `error` ends the stream without one.
const TERMINAL_EVENTS: [&str; 3] = [
    "response.completed",
    "response.incomplete",
    "response.failed",
];
const STREAM_END: [&str; 4] = [
    "response.completed",
    "response.incomplete",
    "response.failed",
    "error",
];
/// The transport's effort enum changes with Codex releases, not by the
/// minute, and asking costs a refused request against the live backend.
const EFFORT_PROBE_TTL: Duration = Duration::from_secs(3600);

type Efforts = Option<BTreeSet<String>>;

/// The only module coupled to ChatGPT's private Codex transport.
pub struct Upstream {
    pub app: Arc<AppServer>,
    pub ledger: Arc<TokenLedger>,
    http: reqwest::Client,
    probed: Mutex<Option<(Instant, String, Efforts)>>,
}

impl Upstream {
    pub fn new(app: Arc<AppServer>, http: reqwest::Client, ledger: Arc<TokenLedger>) -> Self {
        Upstream {
            app,
            ledger,
            http,
            probed: Mutex::new(None),
        }
    }

    pub async fn events(&self, body: &Object, caller: &str) -> Result<EventStream> {
        let response = self.open(body).await?;
        let tracker = UsageTracker::new(
            self.ledger.clone(),
            |event: &Value| read_usage(event, 0),
            &TERMINAL_EVENTS,
            caller,
        );
        Ok(track(read_events(response, &STREAM_END), tracker))
    }

    /// Discover the transport enum without running a generation.
    pub async fn reasoning_efforts(&self, model: &str) -> Result<Efforts> {
        if let Some((at, probed, efforts)) = self.probed.lock().unwrap().as_ref() {
            if probed == model && at.elapsed() < EFFORT_PROBE_TTL {
                return Ok(efforts.clone());
            }
        }
        let body = crate::obj! {
            "model": model,
            "input": [],
            "store": false,
            "stream": true,
            "reasoning": {"effort": "__probe__"},
        };
        let efforts = match self.open(&body).await {
            Ok(_) => None,
            Err(error) if error.account_unavailable() => return Err(error),
            Err(error) if error.status() == 400 => effort_values(error.message()),
            // Nothing learned: ask again on the next discovery.
            Err(_) => return Ok(None),
        };
        *self.probed.lock().unwrap() = Some((Instant::now(), model.to_string(), efforts.clone()));
        Ok(efforts)
    }

    async fn open(&self, body: &Object) -> Result<reqwest::Response> {
        // Byte for byte what the reference sends: compact, ASCII-escaped.
        let payload = dumps(&Value::Object(body.clone()), Dumps::COMPACT);
        let mut refresh = false;
        loop {
            let (access, account) = self
                .app
                .token(refresh)
                .await
                .map_err(|error| Error::unavailable(401, error.message()))?;
            let response = self
                .http
                .post(endpoint(RESPONSES_URL))
                .header("Authorization", format!("Bearer {access}"))
                .header("ChatGPT-Account-ID", account)
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header("User-Agent", USER_AGENT)
                .header("originator", "llm_local_proxy")
                .body(payload.clone())
                .send()
                .await
                .map_err(|error| Error::provider(502, unreachable(&error)))?;
            let status = response.status();
            if status.is_success() {
                return Ok(response);
            }
            if status.as_u16() == 401 && !refresh {
                refresh = true;
                continue;
            }
            let raw = response.text().await.unwrap_or_default();
            return Err(Error::Provider {
                status: status.as_u16(),
                message: error_message(&raw, status),
                account_unavailable: status.as_u16() == 401,
            });
        }
    }
}

fn error_message(raw: &str, status: reqwest::StatusCode) -> String {
    match serde_json::from_str::<Value>(raw) {
        Ok(value) if value.is_object() => {
            let detail = value.get("error").unwrap_or(&value);
            match detail.get("message") {
                Some(message) if detail.is_object() => py_str(message),
                _ => py_str(detail),
            }
        }
        _ if raw.is_empty() => status
            .canonical_reason()
            .unwrap_or("request failed")
            .to_string(),
        _ => raw.to_string(),
    }
}

/// Parse the enum returned for an invalid reasoning effort probe.
fn effort_values(message: &str) -> Efforts {
    let (_, values) = message.split_once("Supported values are:")?;
    let quote = |c: char| c == '\'' || c == '"';
    let word = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    let mut found = BTreeSet::new();
    let chars: Vec<char> = values.chars().collect();
    let mut at = 0;
    while at < chars.len() {
        if quote(chars[at]) {
            let end = (at + 1..chars.len())
                .find(|i| !word(chars[*i]))
                .unwrap_or(chars.len());
            if end > at + 1 && end < chars.len() && quote(chars[end]) {
                found.insert(chars[at + 1..end].iter().collect());
                at = end + 1;
                continue;
            }
        }
        at += 1;
    }
    (!found.is_empty()).then_some(found)
}

/// The reasoning-effort probe's answer for the catalog: `None` when the
/// transport did not say.
pub fn accepted(efforts: &Efforts) -> Option<Vec<String>> {
    efforts.as_ref().map(|set| set.iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_refusal_lists_the_transport_enum() {
        let message = "Invalid value: '__probe__'. Supported values are: 'none', \"low\", 'medium', 'x-high'.";
        let efforts = effort_values(message).unwrap();
        assert_eq!(
            efforts.into_iter().collect::<Vec<_>>(),
            ["low", "medium", "none", "x-high"]
        );
        assert_eq!(effort_values("Invalid value"), None);
        assert_eq!(effort_values("Supported values are: none"), None);
    }

    #[test]
    fn error_bodies_are_reduced_to_their_message() {
        let ok = reqwest::StatusCode::BAD_REQUEST;
        assert_eq!(error_message(r#"{"error":{"message":"nope"}}"#, ok), "nope");
        assert_eq!(error_message(r#"{"detail":"x"}"#, ok), "{'detail': 'x'}");
        assert_eq!(error_message(r#"{"error":"plain"}"#, ok), "plain");
        assert_eq!(error_message("gateway down", ok), "gateway down");
        assert_eq!(error_message("", ok), "Bad Request");
    }
}
