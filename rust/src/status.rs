//! Uniform status shape every provider reports to the dashboard.
//!
//! The page renders one card per provider from these fields alone: sign-in
//! state, an account line, usage bars, proxy token counts and a freshness
//! stamp. Providers normalise their own upstream shapes into this, so adding
//! a provider needs no dashboard change.

use serde_json::{json, Value};

/// One subscription usage bar.
#[derive(Debug, Clone, PartialEq)]
pub struct Limit {
    pub label: String,
    pub used_percent: f64,
    /// Epoch seconds or an ISO timestamp; the page formats either.
    pub resets_at: Value,
    /// The model this bar is restricted to; empty when it limits the account.
    pub model: String,
}

/// One independently authenticated login within a provider pool.
#[derive(Debug, Clone, Default)]
pub struct AccountStatus {
    /// Internal slot id; filled in by the pool, empty from an Auth.
    pub id: String,
    pub signed_in: bool,
    /// One-line account description (e.g. "user@example.com · pro").
    pub account: String,
    pub limits: Vec<Limit>,
    /// Token ledger windows: `{"5h": {"input": .., "output": .., ...}, ...}`.
    pub tokens: Value,
    /// When the usage numbers were last observed, epoch seconds.
    pub updated_at: Option<f64>,
    /// Set when the login could not be read; the row degrades to this.
    pub error: String,
    /// Near a limit, so new sessions start on other accounts.
    pub draining: bool,
}

impl AccountStatus {
    pub fn payload(&self) -> Value {
        json!({
            "id": self.id,
            "signed_in": self.signed_in,
            "account": self.account,
            "limits": self.limits.iter().map(|limit| json!({
                "label": limit.label,
                "used_percent": limit.used_percent,
                "resets_at": limit.resets_at,
                "model": limit.model,
            })).collect::<Vec<_>>(),
            "tokens": if self.tokens.is_null() { json!({}) } else { self.tokens.clone() },
            "updated_at": self.updated_at,
            "error": self.error,
            "draining": self.draining,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProviderStatus {
    pub signed_in: bool,
    /// Set when the provider could not be reached; the card degrades to this.
    pub error: String,
    pub accounts: Vec<AccountStatus>,
}

impl ProviderStatus {
    pub fn payload(&self) -> Value {
        json!({
            "signed_in": self.signed_in,
            "error": self.error,
            "accounts": self.accounts.iter().map(AccountStatus::payload).collect::<Vec<_>>(),
        })
    }
}

/// Rolling windows named the same way across providers and the token ledger.
pub fn window_label(key: &str) -> &str {
    match key {
        "5h" => "5 hour",
        "7d" => "weekly",
        other => other,
    }
}

pub fn window_name(minutes: i64) -> String {
    match minutes {
        300 => "5 hour".into(),
        10080 => "weekly".into(),
        value if value % 1440 == 0 => format!("{} day", value / 1440),
        value if value % 60 == 0 => format!("{} hour", value / 60),
        value => format!("{value} minute"),
    }
}
