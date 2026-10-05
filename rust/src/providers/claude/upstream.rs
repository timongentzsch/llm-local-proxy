//! Claude subscription transport: the Messages API behind the proxy.

use super::auth::{ClaudeAuth, OAUTH_BETA};
use super::usage::ClaudeUsage;
use crate::error::{Error, Result};
use crate::ids::{self, SharedIds};
use crate::json::{dumps, get, py_str, truthy, Dumps, Object};
use crate::ledger::{track, TokenLedger, UsageTracker};
use crate::providers::transport::{endpoint, read_events, unreachable};
use crate::providers::EventStream;
use crate::status::{window_label, Limit};
use futures_util::stream;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const COUNT_TOKENS_URL: &str = "https://api.anthropic.com/v1/messages/count_tokens";
const MODELS_URL: &str = "https://api.anthropic.com/v1/models";
// Subscription utilization, the endpoint Claude usage trackers read.
// Undocumented; needs the user:profile scope the login requests.
const USAGE_URL: &str = "https://api.anthropic.com/api/oauth/usage";
// A usage read feeds status and routing, so it must fail fast rather than hang.
const USAGE_TIMEOUT: Duration = Duration::from_secs(10);
const ANTHROPIC_VERSION: &str = "2023-06-01";
// Beta the subscription edge uses to recognize Claude Code traffic; requests
// without it (and the system marker) are billed against the API pool and 429.
const CLAUDE_CODE_BETA: &str = "claude-code-20250219";
// The subscription accepts the Claude Code client user agent.
const USER_AGENT: &str = "claude-cli/2.1.251 (external, sdk-cli)";

fn upstream(status: u16, message: impl Into<String>) -> Error {
    Error::provider(status, message)
}

/// The only module coupled to Claude's private subscription transport.
pub struct ClaudeUpstream {
    pub auth: Arc<ClaudeAuth>,
    pub ledger: Arc<TokenLedger>,
    http: reqwest::Client,
    ids: SharedIds,
}

impl ClaudeUpstream {
    pub fn new(
        auth: Arc<ClaudeAuth>,
        http: reqwest::Client,
        ledger: Arc<TokenLedger>,
        ids: SharedIds,
    ) -> Self {
        ClaudeUpstream {
            auth,
            ledger,
            http,
            ids,
        }
    }

    pub async fn models(&self) -> Result<Vec<Value>> {
        let value = self.get(&endpoint(MODELS_URL), "model list", None).await?;
        let items = value
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| upstream(502, "Claude model list is malformed"))?;
        Ok(items.iter().filter_map(normalize_model).collect())
    }

    /// Read-only metadata: it costs no tokens and cannot open a window, and
    /// it covers the whole subscription, other clients included.
    pub async fn limits(&self) -> Result<Vec<Limit>> {
        limits(
            &self
                .get(&endpoint(USAGE_URL), "usage", Some(USAGE_TIMEOUT))
                .await?,
        )
    }

    pub async fn events(
        &self,
        body: &Object,
        betas: &[String],
        caller: &str,
    ) -> Result<EventStream> {
        let betas_header = betas_header(betas);
        let prewarm = body.get("max_tokens") == Some(&json!(0));
        let mut outgoing = body.clone();
        if prewarm {
            outgoing.insert("stream".into(), json!(false));
        }
        let response = match self
            .open(&outgoing, &betas_header, &endpoint(MESSAGES_URL))
            .await
        {
            Ok(response) => response,
            // Reported only once the failure is final: the budget retry below
            // recovers on its own, and dumping the turn for it would name a
            // fault that never reached the caller.
            Err(error) if !thinking_rejected(&error, &outgoing) => {
                report_block_shape(&error, &outgoing);
                return Err(error);
            }
            Err(_) => {
                let mut adaptive = crate::obj! { "type": "adaptive" };
                if let Some(display) = outgoing["thinking"].get("display").filter(|d| !d.is_null())
                {
                    adaptive.insert("display".into(), display.clone());
                }
                outgoing.insert("thinking".into(), Value::Object(adaptive));
                match self
                    .open(&outgoing, &betas_header, &endpoint(MESSAGES_URL))
                    .await
                {
                    Ok(response) => response,
                    Err(retried) => {
                        report_block_shape(&retried, &outgoing);
                        return Err(retried);
                    }
                }
            }
        };
        let events = if prewarm {
            message_events(response).await?
        } else {
            read_events(response, &["message_stop", "error"])
        };
        let mut usage = ClaudeUsage::new();
        let tracker = UsageTracker::new(
            self.ledger.clone(),
            move |event: &Value| usage.read(event),
            &["message_stop"],
            caller,
        );
        Ok(track(events, tracker))
    }

    /// Ask the edge how many input tokens a request would cost.
    ///
    /// Generates nothing and is not billed, which is what makes it worth a
    /// round trip: only the server knows the exact tokenisation of tool
    /// schemas and system blocks.
    pub async fn count_tokens(&self, body: &Object, betas: &[String]) -> Result<Value> {
        let response = self
            .open(body, &betas_header(betas), &endpoint(COUNT_TOKENS_URL))
            .await?;
        let value: Value = response
            .text()
            .await
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .ok_or_else(|| upstream(502, "Claude token count is unreadable"))?;
        match crate::json::integer(get(&value, "input_tokens")) {
            Some(tokens) => Ok(json!({ "input_tokens": tokens })),
            None => Err(upstream(502, "Claude token count is malformed")),
        }
    }

    /// GET a JSON document with the subscription's OAuth credentials.
    async fn get(&self, url: &str, what: &str, timeout: Option<Duration>) -> Result<Value> {
        let mut refresh = false;
        loop {
            let mut request = self
                .http
                .get(url)
                .header(
                    "Authorization",
                    format!("Bearer {}", self.token(refresh).await?),
                )
                .header("Accept", "application/json")
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("anthropic-beta", OAUTH_BETA)
                .header("User-Agent", USER_AGENT);
            if let Some(timeout) = timeout {
                request = request.timeout(timeout);
            }
            let response = request
                .send()
                .await
                .map_err(|error| upstream(502, unreachable(&error)))?;
            let status = response.status();
            if status.as_u16() == 401 && !refresh {
                refresh = true;
                continue;
            }
            // A timeout or a cut connection while the body is still arriving.
            let raw = response
                .text()
                .await
                .map_err(|_| upstream(502, format!("Claude {what} is unreadable")))?;
            if !status.is_success() {
                return Err(upstream_error(status.as_u16(), &raw));
            }
            return serde_json::from_str(&raw)
                .map_err(|_| upstream(502, format!("Claude {what} is not valid JSON")));
        }
    }

    async fn token(&self, refresh: bool) -> Result<String> {
        self.auth.access_token(refresh).await.map_err(|error| {
            let status = error.status();
            Error::Provider {
                status,
                message: error.message().to_string(),
                account_unavailable: matches!(status, 400 | 401 | 403),
            }
        })
    }

    /// POST a body, refreshing the token once if the edge refuses it.
    async fn open(&self, body: &Object, betas: &str, url: &str) -> Result<reqwest::Response> {
        // Byte for byte what the reference sends: compact, ASCII-escaped.
        let payload = dumps(&Value::Object(body.clone()), Dumps::COMPACT);
        let mut refresh = false;
        loop {
            let response = self
                .http
                .post(url)
                .header(
                    "Authorization",
                    format!("Bearer {}", self.token(refresh).await?),
                )
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .header("anthropic-version", ANTHROPIC_VERSION)
                .header("anthropic-beta", betas)
                .header("User-Agent", USER_AGENT)
                .header("x-app", "cli")
                .header("x-client-request-id", ids::hex(self.ids.as_ref()))
                .body(payload.clone())
                .send()
                .await
                .map_err(|error| upstream(502, unreachable(&error)))?;
            let status = response.status();
            if status.is_success() {
                return Ok(response);
            }
            if status.as_u16() == 401 && !refresh {
                refresh = true;
                continue;
            }
            let raw = response.text().await.unwrap_or_default();
            return Err(upstream_error(status.as_u16(), &raw));
        }
    }
}

fn betas_header(betas: &[String]) -> String {
    [CLAUDE_CODE_BETA, OAUTH_BETA]
        .into_iter()
        .chain(betas.iter().map(String::as_str))
        .collect::<Vec<_>>()
        .join(",")
}

/// Turn a zero-token non-streaming Message into the normal event lifecycle.
async fn message_events(response: reqwest::Response) -> Result<EventStream> {
    let raw = response
        .text()
        .await
        .map_err(|error| upstream(502, unreachable(&error)))?;
    let message: Value = serde_json::from_str(&raw)
        .map_err(|_| upstream(502, "Claude prewarm response is not valid JSON"))?;
    if !message.is_object() || get(&message, "type") != "message" {
        return Err(upstream(502, "Claude prewarm response is malformed"));
    }
    let usage = match get(&message, "usage") {
        usage if usage.is_object() => usage.clone(),
        _ => json!({}),
    };
    let stop_reason = match get(&message, "stop_reason") {
        reason if truthy(reason) => reason.clone(),
        _ => json!("max_tokens"),
    };
    let events = vec![
        Ok(json!({"type": "message_start", "message": message})),
        Ok(json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": usage,
        })),
        Ok(json!({"type": "message_stop"})),
    ];
    Ok(Box::pin(stream::iter(events)))
}

fn number(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64().filter(|n| n.is_finite()),
        _ => None,
    }
}

/// Dashboard bars from a usage response; unknown fields are ignored.
pub fn limits(value: &Value) -> Result<Vec<Limit>> {
    if !value.is_object() {
        return Err(upstream(502, "Claude usage is malformed"));
    }
    let mut items = Vec::new();
    for (key, window) in [("five_hour", "5h"), ("seven_day", "7d")] {
        let entry = get(value, key);
        if let Some(used) = number(get(entry, "utilization")).filter(|_| entry.is_object()) {
            items.push(Limit {
                label: window_label(window).to_string(),
                used_percent: used,
                resets_at: get(entry, "resets_at").clone(),
                model: String::new(),
            });
        }
    }
    // Model-scoped weekly caps (e.g. one model's own allowance) appear only
    // in this list; the unscoped session and weekly entries repeat the above.
    for entry in get(value, "limits").as_array().into_iter().flatten() {
        if !entry.is_object() || get(entry, "kind") != "weekly_scoped" {
            continue;
        }
        let model = get(get(entry, "scope"), "model");
        let name = [get(model, "display_name"), get(model, "id")]
            .into_iter()
            .find(|name| truthy(name));
        if let (Some(name), Some(used)) = (name, number(get(entry, "percent"))) {
            items.push(Limit {
                label: format!("{} {}", py_str(name), window_label("7d")),
                used_percent: used,
                resets_at: get(entry, "resets_at").clone(),
                model: py_str(name),
            });
        }
    }
    Ok(items)
}

/// Log the block shape of every turn when upstream refuses a signed one.
///
/// The rejection names a message and a position; this says what the proxy put
/// there. Kinds and sizes are enough to place the fault, so the text -- which
/// is the conversation -- stays out of the log.
fn report_block_shape(error: &Error, body: &Object) {
    if error.status() != 400 || !error.message().to_lowercase().contains("thinking") {
        return;
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return;
    };
    let mut lines = vec![format!("claude: upstream rejected a signed turn: {error}")];
    for (index, message) in messages.iter().enumerate() {
        let Some(content) = get(message, "content").as_array() else {
            continue;
        };
        let length = |block: &Value, key: &str| match block.get(key) {
            Some(value) => py_str(value).chars().count(),
            None => 0,
        };
        let shapes: Vec<String> = content
            .iter()
            .map(|block| match block.get("type").and_then(Value::as_str) {
                Some("thinking") => format!(
                    "thinking(text={},sig={})",
                    length(block, "thinking"),
                    length(block, "signature")
                ),
                Some("redacted_thinking") => format!("redacted(data={})", length(block, "data")),
                Some(kind) => kind.to_string(),
                None if block.is_object() => py_str(get(block, "type")),
                None => "?".into(),
            })
            .collect();
        lines.push(format!(
            "  [{index}] {}: {}",
            py_str(get(message, "role")),
            shapes.join(", ")
        ));
    }
    eprintln!("{}", lines.join("\n"));
}

/// True when a request was refused solely for its explicit thinking budget.
fn thinking_rejected(error: &Error, body: &Object) -> bool {
    let enabled = body
        .get("thinking")
        .is_some_and(|thinking| thinking.is_object() && get(thinking, "type") == "enabled");
    enabled && error.status() == 400 && error.message().to_lowercase().contains("thinking")
}

fn supported(value: &Value) -> bool {
    value.is_object() && truthy(get(value, "supported"))
}

/// Seconds since the epoch for an RFC 3339 timestamp; a missing offset is UTC.
fn timestamp(text: &str) -> Option<i64> {
    let (date, time) = text.split_once(['T', ' '])?;
    let mut parts = date.split('-').map(str::parse::<i64>);
    let (year, month, day) = (
        parts.next()?.ok()?,
        parts.next()?.ok()?,
        parts.next()?.ok()?,
    );
    let (clock, offset) = match time.find(['Z', '+', '-']) {
        Some(at) => time.split_at(at),
        None => (time, "Z"),
    };
    let mut fields = clock.split(':');
    let hour: i64 = fields.next()?.parse().ok()?;
    let minute: i64 = fields.next()?.parse().ok()?;
    let second: f64 = fields.next().unwrap_or("0").parse().ok()?;
    let offset_seconds = match offset {
        "Z" => 0,
        other => {
            let sign = if other.starts_with('-') { -1 } else { 1 };
            let (hours, minutes) = other[1..].split_once(':')?;
            sign * (hours.parse::<i64>().ok()? * 3600 + minutes.parse::<i64>().ok()? * 60)
        }
    };
    // Days from civil, after Howard Hinnant.
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let doy = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + hour * 3600 + minute * 60 + second as i64 - offset_seconds)
}

fn normalize_model(item: &Value) -> Option<Value> {
    let model_id = item.get("id")?.as_str().filter(|id| !id.is_empty())?;
    let name = match get(item, "display_name") {
        name if truthy(name) => py_str(name),
        _ => model_id.to_string(),
    };
    let mut value = crate::obj! { "id": model_id, "name": name };
    if let Some(created) = get(item, "created_at").as_str().and_then(timestamp) {
        value.insert("created".into(), json!(created));
    }
    let positive = |key: &str| crate::json::integer(get(item, key)).filter(|n| *n > 0);
    if let Some(max_tokens) = positive("max_tokens") {
        value.insert("max_output_tokens".into(), json!(max_tokens));
    }
    if let Some(max_input) = positive("max_input_tokens") {
        value.insert("context_length".into(), json!(max_input));
    }
    let capabilities = get(item, "capabilities");
    if capabilities.is_object() {
        let mut modalities = vec!["text"];
        if supported(get(capabilities, "image_input")) {
            modalities.push("image");
        }
        value.insert("modalities".into(), json!(modalities));
        if let Some(efforts) = get(capabilities, "effort").as_object() {
            // Empty when the model declares no effort tiers, which is not the
            // same as a catalog entry that says nothing about effort.
            let tiers: Vec<&String> = efforts
                .iter()
                .filter(|(_, support)| supported(support))
                .map(|(name, _)| name)
                .collect();
            value.insert("reasoning_efforts".into(), json!(tiers));
        }
        let types = get(get(capabilities, "thinking"), "types");
        if types.is_object() {
            let (adaptive, enabled) = (
                supported(get(types, "adaptive")),
                supported(get(types, "enabled")),
            );
            if adaptive && !enabled {
                value.insert("thinking".into(), json!("adaptive"));
            } else if enabled {
                value.insert("thinking".into(), json!("enabled"));
            }
        }
    }
    Some(Value::Object(value))
}

fn upstream_error(status: u16, raw: &str) -> Error {
    let mut message = error_message(raw);
    if status == 429 && (message.is_empty() || message == "Error") {
        message = "Claude usage limit reached; the subscription is rate limited".into();
    }
    // A 403 naming a scope is the credential, not the request: the same body
    // succeeds on a login that holds inference access.
    let scope_denied = status == 403 && message.to_lowercase().contains("scope");
    Error::Provider {
        status,
        message,
        account_unavailable: status == 401 || scope_denied,
    }
}

fn error_message(raw: &str) -> String {
    let or_raw = |value: Option<&Value>| match value {
        Some(found) => py_str(found),
        None => raw.to_string(),
    };
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return if raw.is_empty() {
            "Claude response failed".into()
        } else {
            raw.to_string()
        };
    };
    if !value.is_object() {
        return raw.to_string();
    }
    let error = get(&value, "error");
    if error.is_object() {
        return or_raw(
            [get(error, "message"), get(error, "type")]
                .into_iter()
                .find(|found| truthy(found)),
        );
    }
    or_raw(Some(get(&value, "message")).filter(|found| truthy(found)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_becomes_bars_and_unknown_fields_are_ignored() {
        let bars = limits(&json!({
            "five_hour": {"utilization": 12.5, "resets_at": "2026-10-05T12:00:00Z"},
            "seven_day": {"utilization": 40, "resets_at": null},
            "seven_day_opus": {"utilization": 99},
            "limits": [
                {"kind": "weekly_scoped", "percent": 7, "scope": {"model": {"display_name": "Opus"}}},
                {"kind": "session", "percent": 12.5},
            ],
        }))
        .unwrap();
        let labels: Vec<&str> = bars.iter().map(|bar| bar.label.as_str()).collect();
        assert_eq!(labels, ["5 hour", "weekly", "Opus weekly"]);
        assert_eq!(bars[2].model, "Opus");
        assert_eq!(bars[0].used_percent, 12.5);
        assert!(limits(&json!({"five_hour": {"utilization": true}}))
            .unwrap()
            .is_empty());
        assert!(limits(&json!([])).is_err());
    }

    #[test]
    fn errors_keep_their_meaning() {
        let limited = upstream_error(429, "Error");
        assert!(limited.message().contains("usage limit"));
        assert!(!limited.account_unavailable());
        assert!(upstream_error(401, r#"{"error":{"message":"expired"}}"#).account_unavailable());
        let scope = r#"{"type":"error","error":{"type":"permission_error","message":"OAuth token does not meet scope requirement any_of(user:inference)"}}"#;
        assert!(upstream_error(403, scope).account_unavailable());
        assert!(!upstream_error(403, r#"{"error":{"message":"forbidden"}}"#).account_unavailable());
        assert_eq!(upstream_error(500, "not json").message(), "not json");
    }

    #[test]
    fn a_refused_thinking_budget_is_recognised() {
        let error = upstream(400, "thinking.budget_tokens: not supported");
        let body = crate::obj! { "thinking": {"type": "enabled", "budget_tokens": 1024} };
        assert!(thinking_rejected(&error, &body));
        assert!(!thinking_rejected(
            &error,
            &crate::obj! { "thinking": {"type": "adaptive"} }
        ));
        assert!(!thinking_rejected(&upstream(429, "thinking"), &body));
    }

    #[test]
    fn catalog_entries_are_normalised() {
        let model = normalize_model(&json!({
            "id": "claude-x",
            "display_name": "Claude X",
            "created_at": "2026-02-03T04:05:06Z",
            "max_tokens": 64000,
            "max_input_tokens": 200000,
            "capabilities": {
                "image_input": {"supported": true},
                "effort": {"low": {"supported": true}, "max": {"supported": false}},
                "thinking": {"types": {"adaptive": {"supported": true}, "enabled": {"supported": false}}},
            },
        }))
        .unwrap();
        assert_eq!(model["created"], 1_770_091_506);
        assert_eq!(model["max_output_tokens"], 64000);
        assert_eq!(model["modalities"], json!(["text", "image"]));
        assert_eq!(model["reasoning_efforts"], json!(["low"]));
        assert_eq!(model["thinking"], "adaptive");
        assert!(normalize_model(&json!({"display_name": "x"})).is_none());
    }
}
