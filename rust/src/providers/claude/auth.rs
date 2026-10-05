//! Claude subscription OAuth: the proxy keeps its own token pair.
//!
//! Unlike the Codex upstream there is no binary that owns the login. The
//! proxy performs the same authorization-code + PKCE flow as the Claude Code
//! CLI (against the same client id) and stores the resulting token pair in a
//! private file next to the config. Because the proxy refreshes its own
//! refresh token, it does not contend with the token pairs held by Claude
//! Code on other machines of the same subscription.

use crate::atomic;
use crate::error::{Error, Result};
use crate::json::{py_str, truthy, Object};
use crate::providers::pool::Auth;
use crate::providers::transport::endpoint;
use crate::providers::BoxFuture;
use crate::status::AccountStatus;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
// The subscription (claude.ai) login. platform.claude.com/oauth/authorize is
// the Console login and grants only org:create_api_key user:profile, which
// cannot call Messages. Verified against the CLI 2.1.282 bundle.
const AUTHORIZE_URL: &str = "https://claude.com/cai/oauth/authorize";
const TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
const PROFILE_URL: &str = "https://api.anthropic.com/api/oauth/profile";
const MANUAL_REDIRECT_URL: &str = "https://platform.claude.com/oauth/code/callback";
const SCOPE: &str =
    "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers";
pub const OAUTH_BETA: &str = "oauth-2025-04-20";
/// Without it the token can sign in and read a profile but every Messages or
/// Models call answers 403, so such a grant is not a usable login.
const INFERENCE_SCOPE: &str = "user:inference";
const REFRESH_SKEW_SECONDS: i64 = 120;
const USER_AGENT: &str = concat!("llm-local-proxy/", env!("CARGO_PKG_VERSION"));
const TIMEOUT: Duration = Duration::from_secs(30);

const NOT_SIGNED_IN: &str = "not signed in to Claude; use the sign in button on the status page";

fn auth_error(message: impl Into<String>, status: u16) -> Error {
    Error::provider(status, message)
}

#[derive(Default)]
struct Login {
    verifier: String,
    state: String,
}

/// The proxy's own Claude subscription login, independent of Claude Code.
pub struct ClaudeAuth {
    path: PathBuf,
    http: reqwest::Client,
    /// Serialises the login flow and token refresh: a refresh token is spent
    /// by its first use, so two refreshes must never race.
    lock: tokio::sync::Mutex<Login>,
    profile_attempted: AtomicBool,
}

impl ClaudeAuth {
    pub fn new(path: PathBuf, http: reqwest::Client) -> Self {
        ClaudeAuth {
            path,
            http,
            lock: tokio::sync::Mutex::new(Login::default()),
            profile_attempted: AtomicBool::new(false),
        }
    }

    // -- private store --------------------------------------------------------

    fn read(&self) -> Option<Object> {
        match atomic::read_json(&self.path) {
            Ok(Some(Ok(Value::Object(value)))) => Some(value),
            _ => None,
        }
    }

    fn write(&self, value: &Object) -> Result<()> {
        atomic::write_json(&self.path, &Value::Object(value.clone()))
            .map_err(|error| Error::upstream(format!("could not store the Claude login: {error}")))
    }

    fn login_path(&self) -> PathBuf {
        let mut name = self.path.file_name().unwrap_or_default().to_os_string();
        name.push(".login");
        self.path.with_file_name(name)
    }

    fn read_login(&self) -> Object {
        match atomic::read_json(&self.login_path()) {
            Ok(Some(Ok(Value::Object(value)))) => value,
            _ => Object::new(),
        }
    }

    fn clear_login(&self) {
        let _ = std::fs::remove_file(self.login_path());
    }

    // -- login flow -------------------------------------------------------------

    async fn start(&self) -> Result<Value> {
        let mut random = [0u8; 32];
        getrandom::getrandom(&mut random).expect("the operating system has randomness");
        let verifier = URL_SAFE_NO_PAD.encode(random);
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
        let state = atomic::token_urlsafe(24);
        let query: Vec<String> = [
            // The CLI sends code=true for the manual (paste the code) flow.
            ("code", "true"),
            ("client_id", CLIENT_ID),
            ("response_type", "code"),
            ("redirect_uri", MANUAL_REDIRECT_URL),
            ("scope", SCOPE),
            ("state", &state),
            ("code_challenge", &challenge),
            ("code_challenge_method", "S256"),
        ]
        .iter()
        .map(|(key, value)| format!("{key}={}", quote_plus(value)))
        .collect();
        let mut login = self.lock.lock().await;
        atomic::write_json(
            &self.login_path(),
            &json!({ "verifier": verifier, "state": state }),
        )
        .map_err(|error| Error::upstream(error.to_string()))?;
        login.verifier = verifier;
        login.state = state;
        Ok(json!({ "url": format!("{AUTHORIZE_URL}?{}", query.join("&")) }))
    }

    /// Exchange the pasted code for a token pair and store it.
    pub async fn finish(&self, code: &str) -> Result<Value> {
        let value = extract_code(code)?;
        let mut login = self.lock.lock().await;
        let stored = self.read_login();
        let pick = |key: &str, memory: &str| match stored.get(key) {
            Some(found) if truthy(found) => py_str(found),
            _ => memory.to_string(),
        };
        let verifier = pick("verifier", &login.verifier);
        if verifier.is_empty() {
            return Err(auth_error("start a Claude login first", 400));
        }
        let expected_state = pick("state", &login.state);
        let pasted_state = extract_state(code);
        if !pasted_state.is_empty() && !expected_state.is_empty() && pasted_state != expected_state
        {
            return Err(auth_error("state does not match this login", 400));
        }
        // The token endpoint wants a JSON body, and the authorization_code
        // grant rejects one without `state` ("Invalid request format"); both
        // verified against the CLI 2.1.235 source and a live probe.
        let state = if expected_state.is_empty() {
            pasted_state
        } else {
            expected_state
        };
        let token = self
            .token_request(json!({
                "grant_type": "authorization_code",
                "code": value,
                "redirect_uri": MANUAL_REDIRECT_URL,
                "client_id": CLIENT_ID,
                "code_verifier": verifier,
                "state": state,
            }))
            .await?;
        if !token.get("access_token").is_some_and(truthy) {
            return Err(auth_error(
                "Claude OAuth response is missing access_token",
                502,
            ));
        }
        require_inference(&token)?;
        let token = self.with_profile(token).await;
        self.write(&token)?;
        *login = Login::default();
        self.clear_login();
        Ok(self.card().payload())
    }

    async fn sign_out(&self) {
        let mut login = self.lock.lock().await;
        *login = Login::default();
        self.clear_login();
        let _ = std::fs::remove_file(&self.path);
    }

    // -- token access -----------------------------------------------------------

    pub fn has_login(&self) -> bool {
        self.read()
            .is_some_and(|value| value.get("access_token").is_some_and(truthy))
    }

    fn card(&self) -> AccountStatus {
        let Some(value) = self
            .read()
            .filter(|v| v.get("access_token").is_some_and(truthy))
        else {
            return AccountStatus::default();
        };
        let field = |key: &str| match value.get(key) {
            Some(found) if truthy(found) => py_str(found),
            _ => String::new(),
        };
        let parts: Vec<String> = [field("email"), field("subscription_type")]
            .into_iter()
            .filter(|part| !part.is_empty())
            .collect();
        AccountStatus {
            signed_in: true,
            account: if parts.is_empty() {
                "claude".into()
            } else {
                parts.join(" · ")
            },
            ..Default::default()
        }
    }

    /// Populate identity metadata when a credential file lacks it.
    pub async fn hydrate_profile(&self) {
        let _guard = self.lock.lock().await;
        let Some(value) = self.read() else { return };
        let has = |key: &str| value.get(key).is_some_and(truthy);
        if !has("access_token")
            || (has("email") && has("subscription_type"))
            || self.profile_attempted.load(Ordering::SeqCst)
        {
            return;
        }
        let enriched = self.with_profile(value.clone()).await;
        if enriched != value {
            let _ = self.write(&enriched);
        }
    }

    /// A token good for at least the next two minutes.
    pub async fn access_token(&self, force_refresh: bool) -> Result<String> {
        let _guard = self.lock.lock().await;
        let mut value = self
            .read()
            .filter(|value| value.get("access_token").is_some_and(truthy))
            .ok_or_else(|| auth_error(NOT_SIGNED_IN, 401))?;
        // Before any refresh: it requests the granted scopes again, so it
        // cannot repair a grant without inference.
        require_inference(&value)?;
        let expires_at = value
            .get("expires_at")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as i64;
        let stale = expires_at <= crate::ledger::now() + REFRESH_SKEW_SECONDS;
        if (force_refresh || stale) && value.get("refresh_token").is_some_and(truthy) {
            value = self.refresh(value).await?;
        }
        Ok(py_str(&value["access_token"]))
    }

    async fn refresh(&self, value: Object) -> Result<Object> {
        let scope = match value.get("scopes").and_then(Value::as_array) {
            Some(scopes) if !scopes.is_empty() => {
                scopes.iter().map(py_str).collect::<Vec<_>>().join(" ")
            }
            _ => SCOPE.to_string(),
        };
        let token = self
            .token_request(json!({
                "grant_type": "refresh_token",
                "refresh_token": py_str(&value["refresh_token"]),
                "client_id": CLIENT_ID,
                "scope": scope,
            }))
            .await?;
        if !token.get("access_token").is_some_and(truthy) {
            return Err(auth_error(
                "Claude token refresh is missing access_token",
                502,
            ));
        }
        let mut merged = value;
        merged.extend(token);
        let merged = self.with_profile(merged).await;
        self.write(&merged)?;
        Ok(merged)
    }

    async fn with_profile(&self, value: Object) -> Object {
        let has = |key: &str| value.get(key).is_some_and(truthy);
        if has("email") && has("subscription_type") {
            return value;
        }
        self.profile_attempted.store(true, Ordering::SeqCst);
        let Some(access) = value.get("access_token").map(py_str) else {
            return value;
        };
        // Identity metadata is useful for identifying a pool slot, but a
        // temporary profile failure must not invalidate a working login.
        let Ok(profile) = self.profile_request(&access).await else {
            return value;
        };
        let mut enriched = value;
        let email = profile.get("account").and_then(|a| a.get("email"));
        if !enriched.get("email").is_some_and(truthy) {
            if let Some(email) = email.filter(|email| truthy(email)) {
                enriched.insert("email".into(), json!(py_str(email)));
            }
        }
        let plan = plan(profile.get("organization"));
        if !enriched.get("subscription_type").is_some_and(truthy) && !plan.is_empty() {
            enriched.insert("subscription_type".into(), json!(plan));
        }
        enriched
    }

    async fn profile_request(&self, access_token: &str) -> Result<Value> {
        let response = self
            .http
            .get(endpoint(PROFILE_URL))
            .header("Authorization", format!("Bearer {access_token}"))
            .header("anthropic-beta", OAUTH_BETA)
            .header("Accept", "application/json")
            .header("User-Agent", USER_AGENT)
            .timeout(TIMEOUT)
            .send()
            .await
            .map_err(|error| auth_error(format!("Claude profile unreachable: {error}"), 502))?;
        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(auth_error(
                format!("Claude profile failed: {}", error_message(&raw, status)),
                status.as_u16(),
            ));
        }
        Ok(serde_json::from_str::<Value>(&raw)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({})))
    }

    async fn token_request(&self, fields: Value) -> Result<Object> {
        // The CLI posts JSON (not form data) to the token endpoint; the
        // authorization_code grant answers form bodies with 400
        // "Invalid request format".
        let response = self
            .http
            .post(endpoint(TOKEN_URL))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .header("User-Agent", USER_AGENT)
            .timeout(TIMEOUT)
            .body(fields.to_string())
            .send()
            .await
            .map_err(|error| auth_error(format!("Claude OAuth unreachable: {error}"), 502))?;
        let status = response.status();
        let raw = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(auth_error(
                format!("Claude OAuth failed: {}", error_message(&raw, status)),
                400,
            ));
        }
        Ok(normalize(&raw))
    }
}

impl Auth for ClaudeAuth {
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
        Box::pin(async { Ok(self.has_login()) })
    }

    fn status(&self) -> BoxFuture<'_, Result<AccountStatus>> {
        Box::pin(async { Ok(self.card()) })
    }
}

/// `urllib.parse.quote_plus`: everything but unreserved characters escaped,
/// spaces as `+`.
fn quote_plus(value: &str) -> String {
    let mut out = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                out.push(byte as char)
            }
            b' ' => out.push('+'),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

fn query_of(value: &str) -> &str {
    value.split_once('?').map_or(value, |(_, query)| query)
}

/// The state a pasted `code#state` or callback URL carries, if any.
fn extract_state(value: &str) -> String {
    if value.contains("state=") {
        return query_of(value)
            .split('&')
            .find_map(|pair| pair.strip_prefix("state="))
            .map(|state| state.trim().to_string())
            .unwrap_or_default();
    }
    if value.contains('#') && !value.contains("code=") {
        return value
            .split_once('#')
            .map_or("", |(_, state)| state)
            .trim()
            .to_string();
    }
    String::new()
}

fn extract_code(value: &str) -> Result<String> {
    let mut code = value.trim();
    if code.contains("code=") {
        if let Some(found) = query_of(code)
            .split('&')
            .find_map(|pair| pair.strip_prefix("code="))
        {
            return Ok(found.trim().to_string());
        }
    }
    if let Some((before, _)) = code.split_once('#') {
        code = before.trim();
    }
    if code.is_empty() {
        return Err(auth_error("code is required", 400));
    }
    Ok(code.to_string())
}

/// Refuse a grant that names its scopes and lacks inference.
///
/// Credentials written before scopes were recorded carry none and are
/// accepted; the upstream's own 403 still catches them.
fn require_inference(token: &Object) -> Result<()> {
    let Some(scopes) = token.get("scopes").filter(|scopes| truthy(scopes)) else {
        return Ok(());
    };
    let granted = match scopes {
        Value::Array(items) => items.iter().any(|scope| scope == INFERENCE_SCOPE),
        Value::String(text) => text.contains(INFERENCE_SCOPE),
        _ => false,
    };
    if granted {
        return Ok(());
    }
    Err(auth_error(
        "this login lacks Claude inference access; sign out and sign in with a Claude subscription account",
        403,
    ))
}

fn error_message(raw: &str, status: reqwest::StatusCode) -> String {
    let fallback = || {
        if raw.is_empty() {
            status
                .canonical_reason()
                .unwrap_or("request failed")
                .to_string()
        } else {
            raw.to_string()
        }
    };
    let Ok(value) = serde_json::from_str::<Value>(raw) else {
        return fallback();
    };
    let first = |source: &Value, keys: &[&str]| {
        keys.iter()
            .filter_map(|key| source.get(*key))
            .find(|found| truthy(found))
            .map(py_str)
    };
    match value.get("error") {
        Some(error) if error.is_object() => {
            first(error, &["message", "error_description", "type"]).unwrap_or_else(fallback)
        }
        _ if value.is_object() => {
            first(&value, &["error_description", "message", "error"]).unwrap_or_else(fallback)
        }
        _ => raw.to_string(),
    }
}

/// A plan label from the profile, for logins whose token did not name one.
///
/// The rate-limit tier carries the multiplier (`default_claude_max_5x` is
/// "max 5x"); the organization type (`claude_max`) is the fallback.
fn plan(organization: Option<&Value>) -> String {
    for key in ["rate_limit_tier", "organization_type"] {
        if let Some(value) = organization
            .and_then(|o| o.get(key))
            .and_then(Value::as_str)
        {
            if !value.is_empty() {
                let name = value.strip_prefix("default_").unwrap_or(value);
                let name = name.strip_prefix("claude_").unwrap_or(name);
                return name.replace('_', " ");
            }
        }
    }
    String::new()
}

/// The token endpoint's answer as the stored credential fields.
fn normalize(raw: &str) -> Object {
    let mut token = Object::new();
    let Ok(Value::Object(value)) = serde_json::from_str::<Value>(raw) else {
        return token;
    };
    for key in ["access_token", "refresh_token"] {
        if let Some(found) = value.get(key).filter(|found| truthy(found)) {
            token.insert(key.into(), json!(py_str(found)));
        }
    }
    if let Some(expires_in) = value.get("expires_in").and_then(Value::as_f64) {
        if expires_in > 0.0 && !value["expires_in"].is_boolean() {
            let expires_at = crate::providers::limits::wall_clock() + expires_in;
            token.insert("expires_at".into(), json!(expires_at as i64));
        }
    }
    if let Some(scope) = value.get("scope").and_then(Value::as_str) {
        if !scope.is_empty() {
            let scopes: Vec<&str> = scope.split_whitespace().collect();
            token.insert("scopes".into(), json!(scopes));
        }
    }
    let subscription = ["subscriptionType", "subscription_type"]
        .iter()
        .filter_map(|key| value.get(*key))
        .find(|found| truthy(found));
    if let Some(subscription) = subscription {
        token.insert("subscription_type".into(), json!(py_str(subscription)));
    }
    token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pasted_codes_come_in_three_shapes() {
        assert_eq!(extract_code(" abc#xyz ").unwrap(), "abc");
        assert_eq!(extract_state("abc#xyz"), "xyz");
        let url = "https://platform.claude.com/oauth/code/callback?code=abc&state=xyz";
        assert_eq!(extract_code(url).unwrap(), "abc");
        assert_eq!(extract_state(url), "xyz");
        assert_eq!(extract_code("abc").unwrap(), "abc");
        assert_eq!(extract_state("abc"), "");
        assert!(extract_code("  ").is_err());
    }

    #[test]
    fn token_answers_are_normalised() {
        let token = normalize(
            r#"{"access_token":"a","refresh_token":"r","expires_in":3600,"scope":"user:profile user:inference","subscriptionType":"max"}"#,
        );
        assert_eq!(token["access_token"], "a");
        assert_eq!(token["scopes"], json!(["user:profile", "user:inference"]));
        assert_eq!(token["subscription_type"], "max");
        assert!(token["expires_at"].as_i64().unwrap() > crate::ledger::now() + 3500);
        assert!(require_inference(&token).is_ok());
        assert!(normalize("nope").is_empty());
    }

    #[test]
    fn a_grant_without_inference_is_refused_and_an_unscoped_one_is_not() {
        let narrow = crate::obj! { "scopes": ["user:profile"] };
        assert_eq!(require_inference(&narrow).unwrap_err().status(), 403);
        assert!(require_inference(&Object::new()).is_ok());
    }

    #[test]
    fn plans_read_like_the_dashboard_shows_them() {
        assert_eq!(
            plan(Some(&json!({"rate_limit_tier": "default_claude_max_5x"}))),
            "max 5x"
        );
        assert_eq!(
            plan(Some(&json!({"organization_type": "claude_max"}))),
            "max"
        );
        assert_eq!(plan(None), "");
    }

    #[test]
    fn the_authorize_query_is_encoded_like_urlencode() {
        assert_eq!(quote_plus("a b:c/d"), "a+b%3Ac%2Fd");
    }
}
