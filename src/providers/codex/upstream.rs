//! ChatGPT's private Codex transport.

use super::auth::{CodexAuth, USER_AGENT};
use super::usage::read_usage;
use crate::error::{Error, Result};
use crate::json::{py_str, Object};
use crate::ledger::{track, TokenLedger, UsageTracker};
use crate::providers::transport::{endpoint, read_events, retry_after, seconds_until, unreachable};
use crate::providers::EventStream;
use reqwest::header::HeaderValue;
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RESPONSES_URL: &str = "https://chatgpt.com/backend-api/codex/responses";
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
const BURST_COOLDOWN_SECONDS: u64 = 60;

type Efforts = Option<BTreeSet<String>>;

/// The only module coupled to ChatGPT's private Codex transport.
pub struct Upstream {
    pub auth: Arc<CodexAuth>,
    pub ledger: Arc<TokenLedger>,
    http: reqwest::Client,
    probed: Mutex<Option<(Instant, String, Efforts)>>,
}

impl Upstream {
    pub fn new(auth: Arc<CodexAuth>, http: reqwest::Client, ledger: Arc<TokenLedger>) -> Self {
        Upstream {
            auth,
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
        // Serialised once, in the client's key order, and shared by a retry.
        let payload = bytes::Bytes::from(
            serde_json::to_vec(body).map_err(|error| Error::provider(502, error.to_string()))?,
        );
        // The backend reuses a cached prefix only for requests that name
        // their session here, as Codex CLI does; the body's key alone never
        // hits. A key that is not a header value goes without.
        let session = body
            .get("prompt_cache_key")
            .and_then(Value::as_str)
            .and_then(|key| HeaderValue::from_str(key).ok());
        let mut refresh = false;
        loop {
            let (access, account) = self.auth.token(refresh).await?;
            let mut request = self
                .http
                .post(endpoint(RESPONSES_URL))
                .header("Authorization", format!("Bearer {access}"))
                .header("ChatGPT-Account-ID", account)
                .header("Content-Type", "application/json")
                .header("Accept", "text/event-stream")
                .header("User-Agent", USER_AGENT)
                .header("originator", "llm_local_proxy")
                .body(payload.clone());
            if let Some(session) = &session {
                request = request.header("session-id", session);
            }
            let response = request
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
            let headers = response.headers().clone();
            let raw = response.text().await.unwrap_or_default();
            return Err(Error::Provider {
                status: status.as_u16(),
                message: error_message(&raw, status),
                account_unavailable: status.as_u16() == 401,
                cooldown: (status.as_u16() == 429).then(|| cooldown(&raw, &headers)),
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

/// How long a rate-limited account should rest.
///
/// An exhausted usage window names its own reset; any other 429 is a burst,
/// over in about a minute unless the response says otherwise.
fn cooldown(raw: &str, headers: &reqwest::header::HeaderMap) -> u64 {
    let body: Value = serde_json::from_str(raw).unwrap_or(Value::Null);
    let error = body.get("error").unwrap_or(&Value::Null);
    if error.get("type").and_then(Value::as_str) == Some("usage_limit_reached") {
        let reset = error
            .get("resets_in_seconds")
            .and_then(Value::as_f64)
            .map(|seconds| seconds.ceil().max(1.0) as u64)
            .or_else(|| seconds_until(error.get("resets_at")?.as_f64()?));
        if let Some(reset) = reset {
            return reset;
        }
    }
    retry_after(headers).unwrap_or(BURST_COOLDOWN_SECONDS)
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
    fn an_exhausted_window_rests_until_it_resets_and_a_burst_for_a_minute() {
        let mut headers = reqwest::header::HeaderMap::new();
        let exhausted = r#"{"error":{"type":"usage_limit_reached","resets_in_seconds":7200}}"#;
        assert_eq!(cooldown(exhausted, &headers), 7200);
        let at = format!(
            r#"{{"error":{{"type":"usage_limit_reached","resets_at":{}}}}}"#,
            crate::ledger::now() + 500
        );
        assert!((498..=501).contains(&cooldown(&at, &headers)));
        assert_eq!(
            cooldown(r#"{"error":{"message":"slow down"}}"#, &headers),
            60
        );
        headers.insert("retry-after", "7".parse().unwrap());
        assert_eq!(cooldown("not json", &headers), 7);
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
