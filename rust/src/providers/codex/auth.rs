//! ChatGPT subscription OAuth: the proxy keeps each slot's token pair itself.
//!
//! The same device-code flow, token endpoint and client id as the Codex CLI,
//! and the same `auth.json` in the slot's own `CODEX_HOME`, so a login made
//! by either is usable by the other. Each slot has its own file and its own
//! refresh token; nothing here touches `~/.codex` of a CLI on the machine.
//!
//! Established from the open-source CLI (`codex-rs/login`); none of it is a
//! published API.

use crate::atomic;
use crate::error::{Error, Result};
use crate::json::{get, py_str, truthy, Object};
use crate::providers::limits::LimitsStore;
use crate::providers::pool::Auth;
use crate::providers::transport::{endpoint, form, unreachable};
use crate::providers::BoxFuture;
use crate::status::{window_name, AccountStatus, Limit};
use base64::Engine;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const REVOKE_URL: &str = "https://auth.openai.com/oauth/revoke";
const USER_CODE_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/usercode";
const DEVICE_TOKEN_URL: &str = "https://auth.openai.com/api/accounts/deviceauth/token";
/// Where the user types the code. Opened in their browser, never fetched here.
const VERIFICATION_URL: &str = "https://auth.openai.com/codex/device";
const DEVICE_REDIRECT_URL: &str = "https://auth.openai.com/deviceauth/callback";
/// Subscription utilization, the endpoint the CLI's `/status` reads.
const USAGE_URL: &str = "https://chatgpt.com/backend-api/wham/usage";
const MODELS_URL: &str = "https://chatgpt.com/backend-api/codex/models";

pub const USER_AGENT: &str = concat!("llm-local-proxy/", env!("CARGO_PKG_VERSION"));
const REFRESH_SKEW_SECONDS: i64 = 120;
const TIMEOUT: Duration = Duration::from_secs(30);
/// A device code is good for fifteen minutes.
const LOGIN_WINDOW: Duration = Duration::from_secs(15 * 60);

const NOT_SIGNED_IN: &str = "not signed in; open the proxy status page";

/// The claims of a JWT, unverified: they only say what to show and send.
fn jwt_claims(token: &str) -> Object {
    let decode = || {
        let raw = token.split('.').nth(1)?;
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(raw.trim_end_matches('='))
            .ok()?;
        serde_json::from_slice::<Value>(&bytes)
            .ok()?
            .as_object()
            .cloned()
    };
    decode().unwrap_or_default()
}

/// The ChatGPT part of a token's claims.
fn chatgpt_claim(token: &str, key: &str) -> String {
    let claims = Value::Object(jwt_claims(token));
    let value = get(get(&claims, "https://api.openai.com/auth"), key);
    if truthy(value) {
        py_str(value)
    } else {
        String::new()
    }
}

/// `2026-10-05T19:04:11Z`, as the CLI stamps `last_refresh`.
fn rfc3339(epoch: i64) -> String {
    let (days, rest) = (epoch.div_euclid(86_400), epoch.rem_euclid(86_400));
    // Civil from days, after Howard Hinnant.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

fn field(tokens: &Object, key: &str) -> String {
    match tokens.get(key) {
        Some(value) if truthy(value) => py_str(value),
        _ => String::new(),
    }
}

/// One slot's ChatGPT login.
pub struct CodexAuth {
    path: PathBuf,
    http: reqwest::Client,
    this: Weak<CodexAuth>,
    /// Serialises refreshes: a refresh token is spent by its first use.
    refreshing: tokio::sync::Mutex<()>,
    /// The device-code login in progress, replaced by the next one started.
    polling: Mutex<Option<tokio::task::AbortHandle>>,
    limits: OnceLock<Arc<LimitsStore>>,
}

impl CodexAuth {
    /// `path` is the slot's `auth.json`.
    pub fn new(path: PathBuf, http: reqwest::Client) -> Arc<Self> {
        Arc::new_cyclic(|this| CodexAuth {
            path,
            http,
            this: this.clone(),
            refreshing: tokio::sync::Mutex::new(()),
            polling: Mutex::new(None),
            limits: OnceLock::new(),
        })
    }

    /// The bars to forget when this slot's login changes.
    pub fn watch(&self, limits: Arc<LimitsStore>) {
        let _ = self.limits.set(limits);
    }

    fn forget_limits(&self) {
        if let Some(limits) = self.limits.get() {
            limits.clear();
        }
    }

    // -- private store --------------------------------------------------------

    async fn read(&self) -> Object {
        // The CLI rewrites the file in place, so a read can land in between.
        for _ in 0..3 {
            match atomic::read_json(&self.path) {
                Ok(Some(Ok(Value::Object(data)))) => return data,
                Ok(None) | Ok(Some(Ok(_))) => return Object::new(),
                _ => tokio::time::sleep(Duration::from_millis(20)).await,
            }
        }
        Object::new()
    }

    fn tokens(auth: &Object) -> Object {
        auth.get("tokens")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default()
    }

    /// Store what the token endpoint returned, keeping what it left out.
    fn persist(&self, mut auth: Object, issued: &Value) -> Result<Object> {
        let mut tokens = Self::tokens(&auth);
        for key in ["id_token", "access_token", "refresh_token"] {
            if let Some(Value::String(value)) = issued.get(key).filter(|value| truthy(value)) {
                tokens.insert(key.into(), json!(value));
            }
        }
        if field(&tokens, "account_id").is_empty() {
            let account = [field(&tokens, "id_token"), field(&tokens, "access_token")]
                .iter()
                .map(|token| chatgpt_claim(token, "chatgpt_account_id"))
                .find(|account| !account.is_empty());
            if let Some(account) = account {
                tokens.insert("account_id".into(), json!(account));
            }
        }
        auth.entry("auth_mode").or_insert_with(|| json!("chatgpt"));
        auth.entry("OPENAI_API_KEY").or_insert(Value::Null);
        auth.insert("tokens".into(), Value::Object(tokens));
        auth.insert("last_refresh".into(), json!(rfc3339(crate::ledger::now())));
        atomic::write_json(&self.path, &Value::Object(auth.clone())).map_err(|error| {
            Error::upstream(format!("could not store the Codex login: {error}"))
        })?;
        Ok(auth)
    }

    // -- token access -----------------------------------------------------------

    /// The access token and ChatGPT account id for an upstream request.
    pub async fn token(&self, force_refresh: bool) -> Result<(String, String)> {
        let _guard = self.refreshing.lock().await;
        let mut auth = self.read().await;
        let mut tokens = Self::tokens(&auth);
        let access = field(&tokens, "access_token");
        let expiry = get(&Value::Object(jwt_claims(&access)), "exp")
            .as_f64()
            .unwrap_or(0.0) as i64;
        let now = crate::ledger::now();
        let refresh = field(&tokens, "refresh_token");
        // A token that does not say when it expires is used until refused.
        let stale = access.is_empty() || (expiry != 0 && expiry <= now + REFRESH_SKEW_SECONDS);
        if (force_refresh || stale) && !refresh.is_empty() {
            match self.refresh(&refresh).await {
                Ok(issued) => {
                    auth = self.persist(auth, &issued)?;
                    tokens = Self::tokens(&auth);
                }
                // The token in hand is still good; a later request retries.
                Err(error) if !error.account_unavailable() && !force_refresh && expiry > now => {}
                Err(error) => return Err(error),
            }
        }
        let access = field(&tokens, "access_token");
        if access.is_empty() {
            return Err(Error::unavailable(401, NOT_SIGNED_IN));
        }
        let mut account = field(&tokens, "account_id");
        if account.is_empty() {
            account = chatgpt_claim(&access, "chatgpt_account_id");
        }
        if account.is_empty() {
            return Err(Error::unavailable(401, "ChatGPT account id is missing"));
        }
        Ok((access, account))
    }

    async fn refresh(&self, refresh_token: &str) -> Result<Value> {
        let body = json!({
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "refresh_token": refresh_token,
        });
        let response = self
            .http
            .post(endpoint(TOKEN_URL))
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
            .timeout(TIMEOUT)
            .body(body.to_string())
            .send()
            .await
            .map_err(|error| Error::provider(502, unreachable(&error)))?;
        let status = response.status().as_u16();
        let raw = response.text().await.unwrap_or_default();
        if (200..300).contains(&status) {
            return serde_json::from_str(&raw).map_err(|_| {
                Error::provider(502, "Codex token refresh answered with invalid JSON")
            });
        }
        Err(refresh_failure(status, &raw))
    }

    /// GET a JSON document from the ChatGPT backend with this slot's login.
    async fn backend(&self, url: &str, what: &str) -> Result<Value> {
        let mut refresh = false;
        loop {
            let (access, account) = self.token(refresh).await?;
            let response = self
                .http
                .get(url)
                .header("Authorization", format!("Bearer {access}"))
                .header("ChatGPT-Account-Id", account)
                .header("Accept", "application/json")
                .header("User-Agent", USER_AGENT)
                .timeout(TIMEOUT)
                .send()
                .await
                .map_err(|error| Error::provider(502, unreachable(&error)))?;
            let status = response.status().as_u16();
            if status == 401 && !refresh {
                refresh = true;
                continue;
            }
            let raw = response.text().await.unwrap_or_default();
            if !(200..300).contains(&status) {
                return Err(Error::Provider {
                    status,
                    message: format!("Codex {what} failed: {}", error_detail(&raw).1),
                    account_unavailable: status == 401,
                });
            }
            return serde_json::from_str(&raw)
                .map_err(|_| Error::provider(502, format!("Codex {what} is not valid JSON")));
        }
    }

    /// Read-only metadata: it costs no tokens and covers the whole
    /// subscription, other clients included.
    pub async fn limits(&self) -> Result<Vec<Limit>> {
        Ok(limits(&self.backend(&endpoint(USAGE_URL), "usage").await?))
    }

    /// The models this login may use, as the backend lists them.
    pub async fn models(&self, client_version: &str) -> Result<Vec<Value>> {
        let url = format!("{}?client_version={client_version}", endpoint(MODELS_URL));
        let value = self.backend(&url, "model list").await?;
        match value.get("models").and_then(Value::as_array) {
            Some(models) => Ok(models.clone()),
            None => Err(Error::provider(502, "Codex model list is malformed")),
        }
    }

    // -- login flow -------------------------------------------------------------

    async fn start(&self) -> Result<Value> {
        let response = self
            .http
            .post(endpoint(USER_CODE_URL))
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
            .timeout(TIMEOUT)
            .body(json!({ "client_id": CLIENT_ID }).to_string())
            .send()
            .await
            .map_err(|error| Error::provider(502, unreachable(&error)))?;
        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::provider(
                502,
                format!("Codex device code request failed with status {status}"),
            ));
        }
        let value: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
        let device = py_str(get(&value, "device_auth_id"));
        let code = ["user_code", "usercode"]
            .iter()
            .map(|key| get(&value, key))
            .find(|code| truthy(code))
            .map(py_str)
            .unwrap_or_default();
        if !truthy(get(&value, "device_auth_id")) || code.is_empty() {
            return Err(Error::provider(
                502,
                "Codex device code response is malformed",
            ));
        }
        // Sent as a string of seconds.
        let interval = py_str(get(&value, "interval"))
            .trim()
            .parse()
            .unwrap_or(5)
            .max(1);

        if let Some(this) = self.this.upgrade() {
            let user_code = code.clone();
            let task = tokio::spawn(async move {
                if let Err(error) = this.complete(&device, &user_code, interval).await {
                    eprintln!("codex: device login did not complete: {error}");
                }
            });
            if let Some(previous) = self.polling.lock().unwrap().replace(task.abort_handle()) {
                previous.abort();
            }
        }
        Ok(json!({ "url": VERIFICATION_URL, "code": code }))
    }

    /// Wait for the user to approve the code, then store the token pair.
    async fn complete(&self, device: &str, user_code: &str, interval: u64) -> Result<()> {
        let started = Instant::now();
        let approved = loop {
            let response = self
                .http
                .post(endpoint(DEVICE_TOKEN_URL))
                .header("Content-Type", "application/json")
                .header("User-Agent", USER_AGENT)
                .timeout(TIMEOUT)
                .body(json!({"device_auth_id": device, "user_code": user_code}).to_string())
                .send()
                .await
                .map_err(|error| Error::provider(502, unreachable(&error)))?;
            let status = response.status().as_u16();
            if (200..300).contains(&status) {
                let raw = response.text().await.unwrap_or_default();
                break serde_json::from_str::<Value>(&raw).unwrap_or(Value::Null);
            }
            // Not approved yet.
            if status != 403 && status != 404 {
                return Err(Error::provider(
                    502,
                    format!("device auth failed with status {status}"),
                ));
            }
            if started.elapsed() >= LOGIN_WINDOW {
                return Err(Error::provider(
                    502,
                    "device auth timed out after 15 minutes",
                ));
            }
            tokio::time::sleep(Duration::from_secs(interval)).await;
        };
        let code = py_str(get(&approved, "authorization_code"));
        let verifier = py_str(get(&approved, "code_verifier"));
        // The one-time code must never be posted twice, so no retry here.
        let response = self
            .http
            .post(endpoint(TOKEN_URL))
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("User-Agent", USER_AGENT)
            .timeout(TIMEOUT)
            .body(form(&[
                ("grant_type", "authorization_code"),
                ("client_id", CLIENT_ID),
                ("code", &code),
                ("redirect_uri", DEVICE_REDIRECT_URL),
                ("code_verifier", &verifier),
            ]))
            .send()
            .await
            .map_err(|error| Error::provider(502, unreachable(&error)))?;
        let status = response.status().as_u16();
        let raw = response.text().await.unwrap_or_default();
        if !(200..300).contains(&status) {
            return Err(Error::provider(
                502,
                format!("device code exchange failed: {}", error_detail(&raw).1),
            ));
        }
        let issued: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
        if !truthy(get(&issued, "access_token")) || !truthy(get(&issued, "refresh_token")) {
            return Err(Error::provider(
                502,
                "Codex OAuth response is missing its tokens",
            ));
        }
        let _guard = self.refreshing.lock().await;
        // A new login replaces the slot's tokens and the account they name.
        self.persist(Object::new(), &issued)?;
        self.forget_limits();
        Ok(())
    }

    async fn sign_out(&self) {
        if let Some(polling) = self.polling.lock().unwrap().take() {
            polling.abort();
        }
        let _guard = self.refreshing.lock().await;
        let refresh = field(&Self::tokens(&self.read().await), "refresh_token");
        let _ = std::fs::remove_file(&self.path);
        self.forget_limits();
        if refresh.is_empty() {
            return;
        }
        // Best effort, as in the CLI: the local login is gone either way.
        let body = json!({
            "token": refresh,
            "token_type_hint": "refresh_token",
            "client_id": CLIENT_ID,
        });
        let _ = self
            .http
            .post(endpoint(REVOKE_URL))
            .header("Content-Type", "application/json")
            .header("User-Agent", USER_AGENT)
            .timeout(Duration::from_secs(10))
            .body(body.to_string())
            .send()
            .await;
    }

    async fn card(&self) -> AccountStatus {
        let tokens = Self::tokens(&self.read().await);
        if field(&tokens, "access_token").is_empty() {
            return AccountStatus::default();
        }
        let id_token = field(&tokens, "id_token");
        let email = match jwt_claims(&id_token).get("email") {
            Some(email) if truthy(email) => py_str(email),
            _ => "ChatGPT".to_string(),
        };
        let plan = [&id_token, &field(&tokens, "access_token")]
            .iter()
            .map(|token| chatgpt_claim(token, "chatgpt_plan_type"))
            .find(|plan| !plan.is_empty());
        AccountStatus {
            signed_in: true,
            account: match plan {
                Some(plan) => format!("{email} · {plan}"),
                None => email,
            },
            ..Default::default()
        }
    }
}

impl Auth for CodexAuth {
    fn login_start(&self) -> BoxFuture<'_, Result<Value>> {
        Box::pin(self.start())
    }

    fn logout(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async {
            self.sign_out().await;
            Ok(())
        })
    }

    fn signed_in(&self) -> BoxFuture<'_, Result<bool>> {
        Box::pin(async {
            let tokens = Self::tokens(&self.read().await);
            Ok(!field(&tokens, "access_token").is_empty())
        })
    }

    fn status(&self) -> BoxFuture<'_, Result<AccountStatus>> {
        Box::pin(async {
            let mut status = self.card().await;
            if let Some(limits) = self.limits.get().filter(|_| status.signed_in) {
                (status.limits, status.updated_at) = limits.current(true).await;
            }
            Ok(status)
        })
    }
}

/// The error code and a readable message from a token or backend error body.
fn error_detail(raw: &str) -> (String, String) {
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return (String::new(), raw.to_string());
    };
    let error = get(&value, "error");
    let text = |source: &Value, keys: &[&str]| {
        keys.iter()
            .map(|key| get(source, key))
            .find(|found| found.is_string() && truthy(found))
            .map(py_str)
    };
    if error.is_object() {
        let code = text(error, &["code", "type"]).unwrap_or_default();
        let message = text(error, &["message"]).unwrap_or_else(|| code.clone());
        return (
            code,
            if message.is_empty() {
                raw.to_string()
            } else {
                message
            },
        );
    }
    let code = text(&value, &["error", "code"]).unwrap_or_default();
    let message = text(&value, &["error_description", "message", "detail"]).unwrap_or_else(|| {
        if code.is_empty() {
            raw.to_string()
        } else {
            code.clone()
        }
    });
    (code, message)
}

/// A refused refresh: a dead login needs the user, anything else may pass.
fn refresh_failure(status: u16, raw: &str) -> Error {
    let (code, message) = error_detail(raw);
    let code = code.to_ascii_lowercase();
    let dead = status == 401
        || (status == 400 && code == "invalid_grant")
        || matches!(
            code.as_str(),
            "refresh_token_expired" | "refresh_token_reused" | "refresh_token_invalidated"
        );
    if dead {
        Error::unavailable(
            401,
            format!("Codex login is no longer valid; sign in again ({message})"),
        )
    } else {
        Error::provider(
            502,
            format!("Codex token refresh failed with status {status}: {message}"),
        )
    }
}

/// Dashboard bars from the usage document.
///
/// `rate_limit` is the account's own limit; each of `additional_rate_limits`
/// restricts one metered model only.
pub fn limits(value: &Value) -> Vec<Limit> {
    let mut items: Vec<(i64, String, Limit)> = Vec::new();
    let mut add = |name: &str, scoped: bool, details: &Value| {
        for window in [
            get(details, "primary_window"),
            get(details, "secondary_window"),
        ] {
            if !window.is_object() {
                continue;
            }
            let seconds = get(window, "limit_window_seconds").as_f64().unwrap_or(0.0) as i64;
            let minutes = if seconds > 0 { (seconds + 59) / 60 } else { 0 };
            items.push((
                minutes,
                name.to_string(),
                Limit {
                    label: format!("{name} · {}", window_name(minutes)),
                    used_percent: get(window, "used_percent").as_f64().unwrap_or(0.0),
                    resets_at: get(window, "reset_at").clone(),
                    model: if scoped {
                        name.to_string()
                    } else {
                        String::new()
                    },
                },
            ));
        }
    };
    add("codex", false, get(value, "rate_limit"));
    for extra in get(value, "additional_rate_limits")
        .as_array()
        .into_iter()
        .flatten()
    {
        let name = [get(extra, "limit_name"), get(extra, "metered_feature")]
            .into_iter()
            .find(|name| truthy(name))
            .map(py_str)
            .unwrap_or_else(|| "limit".into());
        add(&name, true, get(extra, "rate_limit"));
    }
    items.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    items.into_iter().map(|(_, _, limit)| limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt(claims: Value) -> String {
        let body = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.to_string());
        format!("header.{body}.signature")
    }

    #[test]
    fn claims_are_read_with_or_without_padding() {
        let token = jwt(json!({
            "exp": 123,
            "email": "a@b.c",
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct", "chatgpt_plan_type": "pro"},
        }));
        assert_eq!(jwt_claims(&token)["exp"], 123);
        assert_eq!(chatgpt_claim(&token, "chatgpt_account_id"), "acct");
        assert_eq!(chatgpt_claim(&token, "missing"), "");
        assert!(jwt_claims("not-a-jwt").is_empty());
        assert!(jwt_claims("a.!!!.c").is_empty());
    }

    #[test]
    fn timestamps_are_rfc3339() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_770_091_506), "2026-02-03T04:05:06Z");
        assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn usage_becomes_sorted_bars_and_scoped_limits_name_their_model() {
        let bars = limits(&json!({
            "plan_type": "pro",
            "rate_limit": {
                "allowed": true,
                "primary_window": {"used_percent": 16, "limit_window_seconds": 18000, "reset_at": 1787234107},
                "secondary_window": {"used_percent": 4, "limit_window_seconds": 604800, "reset_at": 1787820907},
            },
            "additional_rate_limits": [{
                "limit_name": "Spark",
                "metered_feature": "spark",
                "rate_limit": {"primary_window": {"used_percent": 1, "limit_window_seconds": 7200, "reset_at": 1}, "secondary_window": null},
            }],
        }));
        let labels: Vec<&str> = bars.iter().map(|bar| bar.label.as_str()).collect();
        assert_eq!(
            labels,
            ["Spark · 2 hour", "codex · 5 hour", "codex · weekly"]
        );
        assert_eq!(bars[0].model, "Spark");
        assert_eq!(bars[1].model, "");
        assert_eq!(bars[1].used_percent, 16.0);
        assert_eq!(bars[1].resets_at, 1787234107);
    }

    #[test]
    fn a_weekly_only_plan_and_an_empty_document_are_fine() {
        let weekly = limits(&json!({
            "rate_limit": {
                "primary_window": {"used_percent": 39, "limit_window_seconds": 604800, "reset_at": 1},
                "secondary_window": null,
            },
            "additional_rate_limits": null,
        }));
        assert_eq!(weekly.len(), 1);
        assert_eq!(weekly[0].label, "codex · weekly");
        assert!(limits(&json!({})).is_empty());
    }

    #[test]
    fn only_a_dead_login_asks_for_a_new_sign_in() {
        let dead = [
            (401, "{}"),
            (400, r#"{"error":"invalid_grant"}"#),
            (
                400,
                r#"{"error":{"code":"refresh_token_reused","message":"already used"}}"#,
            ),
            (403, r#"{"error":{"code":"refresh_token_expired"}}"#),
        ];
        for (status, body) in dead {
            assert!(
                refresh_failure(status, body).account_unavailable(),
                "{status} {body}"
            );
        }
        for (status, body) in [
            (429, "slow down"),
            (500, "{}"),
            (400, r#"{"error":"invalid_request"}"#),
        ] {
            let error = refresh_failure(status, body);
            assert!(!error.account_unavailable(), "{status} {body}");
            assert_eq!(error.status(), 502);
        }
    }

    #[tokio::test]
    async fn tokens_are_stored_in_the_cli_layout_and_read_back() {
        let dir = std::env::temp_dir().join(format!("llp-codex-auth-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let auth = CodexAuth::new(dir.join("auth.json"), reqwest::Client::new());
        assert!(!auth.signed_in().await.unwrap());
        assert!(auth.token(false).await.unwrap_err().account_unavailable());

        let exp = crate::ledger::now() + 3600;
        let access = jwt(json!({"exp": exp}));
        let id = jwt(json!({
            "email": "a@b.c",
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1", "chatgpt_plan_type": "pro"},
        }));
        let issued = json!({"id_token": id, "access_token": access, "refresh_token": "rt-1"});
        let stored = auth.persist(Object::new(), &issued).unwrap();
        assert_eq!(stored["auth_mode"], "chatgpt");
        assert_eq!(stored["tokens"]["account_id"], "acct-1");
        assert!(stored["OPENAI_API_KEY"].is_null());
        assert!(stored["last_refresh"].as_str().unwrap().ends_with('Z'));

        assert_eq!(
            auth.token(false).await.unwrap(),
            (access.clone(), "acct-1".to_string())
        );
        assert_eq!(auth.card().await.account, "a@b.c · pro");
        // A refresh that returns only an access token keeps the rest.
        let kept = auth
            .persist(stored, &json!({"access_token": "new"}))
            .unwrap();
        assert_eq!(kept["tokens"]["refresh_token"], "rt-1");
        assert_eq!(kept["tokens"]["access_token"], "new");
        let _ = std::fs::remove_dir_all(dir);
    }
}
