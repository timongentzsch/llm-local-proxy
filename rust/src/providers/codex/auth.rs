//! Codex/ChatGPT auth adapter.
//!
//! Codex owns its own OAuth: `codex app-server` holds the token pair and
//! refreshes it. This adapter only forwards the JSON-RPC calls the status
//! page and the HTTP handlers need, and reads back whether a session exists.

use super::app_server::AppServer;
use crate::error::Result;
use crate::json::{get, py_str, truthy, Object};
use crate::providers::limits::LimitsStore;
use crate::providers::pool::Auth;
use crate::providers::BoxFuture;
use crate::status::{window_name, AccountStatus, Limit};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How long an answer to "is this slot signed in" is reused. Routing asks for
/// every account on every request; the app-server is asked at most this
/// often. A sign-in completes inside the app-server (the device-code flow),
/// so a signed-out answer is kept only briefly.
const SIGNED_IN_TTL: Duration = Duration::from_secs(30);
const SIGNED_OUT_TTL: Duration = Duration::from_secs(2);

pub struct CodexAuth {
    app: Arc<AppServer>,
    pub limits: Arc<LimitsStore>,
    signed_in: Mutex<Option<(Instant, bool)>>,
}

impl CodexAuth {
    pub fn new(app: Arc<AppServer>) -> Self {
        let reader = app.clone();
        let limits = LimitsStore::new(
            "codex",
            Box::new(move || {
                let reader = reader.clone();
                Box::pin(async move {
                    let value = reader.call("account/rateLimits/read", json!({})).await?;
                    Ok(limits(&value))
                })
            }),
        );
        CodexAuth {
            app,
            limits,
            signed_in: Mutex::new(None),
        }
    }

    async fn account(&self) -> Result<Value> {
        let value = self
            .app
            .call("account/read", json!({"refreshToken": false}))
            .await?;
        let account = value.get("account").cloned().unwrap_or(Value::Null);
        *self.signed_in.lock().unwrap() = Some((Instant::now(), truthy(&account)));
        Ok(account)
    }

    /// Ask again next time: the login has just changed, or was refused.
    pub fn forget(&self) {
        *self.signed_in.lock().unwrap() = None;
    }
}

impl Auth for CodexAuth {
    fn login_start(&self) -> BoxFuture<'_, Result<Value>> {
        Box::pin(async {
            let value = self
                .app
                .call("account/login/start", json!({"type": "chatgptDeviceCode"}))
                .await?;
            self.forget();
            let field = |key: &str| value.get(key).cloned().unwrap_or_else(|| json!(""));
            Ok(json!({"url": field("verificationUrl"), "code": field("userCode")}))
        })
    }

    fn logout(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async {
            self.app.call("account/logout", json!({})).await?;
            self.limits.clear();
            self.forget();
            Ok(())
        })
    }

    fn signed_in(&self) -> BoxFuture<'_, Result<bool>> {
        Box::pin(async {
            let cached = *self.signed_in.lock().unwrap();
            if let Some((at, answer)) = cached {
                let ttl = if answer {
                    SIGNED_IN_TTL
                } else {
                    SIGNED_OUT_TTL
                };
                if at.elapsed() < ttl {
                    return Ok(answer);
                }
            }
            Ok(truthy(&self.account().await?))
        })
    }

    fn status(&self) -> BoxFuture<'_, Result<AccountStatus>> {
        Box::pin(async {
            let account = self.account().await?;
            if !truthy(&account) {
                return Ok(AccountStatus::default());
            }
            let (limits, updated_at) = self.limits.current(true).await;
            Ok(AccountStatus {
                signed_in: true,
                account: account_line(&account),
                limits,
                updated_at,
                ..Default::default()
            })
        })
    }
}

fn account_line(account: &Value) -> String {
    let first = |keys: &[&str], default: &str| {
        keys.iter()
            .map(|key| get(account, key))
            .find(|value| truthy(value))
            .map(py_str)
            .unwrap_or_else(|| default.to_string())
    };
    let plan = first(&["planType", "type"], "");
    let name = first(&["email"], "ChatGPT");
    if plan.is_empty() {
        name
    } else {
        format!("{name} · {plan}")
    }
}

/// Dashboard bars from `account/rateLimits/read`.
pub fn limits(value: &Object) -> Vec<Limit> {
    let mut items: Vec<(i64, String, Limit)> = Vec::new();
    // The top-level entry is the account's own limit; any other is one
    // model's (it restricts only that model, not the whole account).
    let default = value.get("rateLimits").filter(|entry| entry.is_object());
    let default_id = default
        .map(|entry| get(entry, "limitId"))
        .filter(|id| !id.is_null());
    // The protocol allows the map to be null; the default limit then stands
    // alone.
    let entries: Vec<&Value> = match value.get("rateLimitsByLimitId").and_then(Value::as_object) {
        Some(map) => map.values().collect(),
        None => default.into_iter().collect(),
    };
    for entry in entries.into_iter().filter(|entry| entry.is_object()) {
        let name = [get(entry, "limitName"), get(entry, "limitId")]
            .into_iter()
            .find(|name| truthy(name))
            .map(py_str)
            .unwrap_or_else(|| "limit".into());
        let scoped = default_id.is_some_and(|id| get(entry, "limitId") != id);
        for window in [get(entry, "primary"), get(entry, "secondary")] {
            if !window.is_object() {
                continue;
            }
            let minutes = get(window, "windowDurationMins").as_f64().unwrap_or(0.0) as i64;
            items.push((
                minutes,
                name.clone(),
                Limit {
                    label: format!("{name} · {}", window_name(minutes)),
                    used_percent: get(window, "usedPercent").as_f64().unwrap_or(0.0),
                    resets_at: get(window, "resetsAt").clone(),
                    model: if scoped { name.clone() } else { String::new() },
                },
            ));
        }
    }
    items.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    items.into_iter().map(|(_, _, limit)| limit).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_become_sorted_bars_and_scoped_limits_name_their_model() {
        let bars = limits(&crate::obj! {
            "rateLimits": {"limitId": "codex"},
            "rateLimitsByLimitId": {
                "codex": {
                    "limitId": "codex",
                    "limitName": "Codex",
                    "primary": {"usedPercent": 16, "windowDurationMins": 300, "resetsAt": 1787234107},
                    "secondary": {"usedPercent": 4, "windowDurationMins": 10080},
                },
                "spark": {
                    "limitId": "spark",
                    "limitName": null,
                    "primary": {"usedPercent": 1, "windowDurationMins": 120},
                    "secondary": null,
                },
            },
        });
        let labels: Vec<&str> = bars.iter().map(|bar| bar.label.as_str()).collect();
        assert_eq!(
            labels,
            ["spark · 2 hour", "Codex · 5 hour", "Codex · weekly"]
        );
        assert_eq!(bars[0].model, "spark");
        assert_eq!(bars[1].model, "");
        assert_eq!(bars[1].used_percent, 16.0);
    }

    #[test]
    fn a_null_limit_map_falls_back_to_the_default_limit() {
        let window = json!({"usedPercent": 12, "windowDurationMins": 300});
        let bars = limits(&crate::obj! {
            "rateLimits": {"limitId": "codex", "primary": window},
            "rateLimitsByLimitId": null,
        });
        assert_eq!(bars.len(), 1);
        assert_eq!(bars[0].used_percent, 12.0);
        assert_eq!(bars[0].model, "");
        assert!(limits(&crate::obj! { "rateLimitsByLimitId": null }).is_empty());
    }

    #[test]
    fn the_account_line_names_the_plan_when_there_is_one() {
        assert_eq!(
            account_line(&json!({"email": "a@b.c", "planType": "pro"})),
            "a@b.c · pro"
        );
        assert_eq!(
            account_line(&json!({"type": "chatgpt"})),
            "ChatGPT · chatgpt"
        );
        assert_eq!(account_line(&json!({"email": "a@b.c"})), "a@b.c");
    }
}
