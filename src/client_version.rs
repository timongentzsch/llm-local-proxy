//! Follow published CLI versions without fetching on the request path.

use crate::atomic;
use crate::providers::transport::endpoint;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::watch;

const CHECK_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
const CHECK_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
pub(crate) struct ReleaseFeed {
    name: &'static str,
    url: &'static str,
    fallback: &'static str,
    parse: fn(&str) -> Option<String>,
}

pub(crate) const CODEX: ReleaseFeed = ReleaseFeed {
    name: "codex",
    url: "https://api.github.com/repos/openai/codex/releases/latest",
    fallback: "0.171.0",
    parse: |body| {
        let release: Value = serde_json::from_str(body).ok()?;
        Some(
            release
                .get("tag_name")?
                .as_str()?
                .strip_prefix("rust-v")?
                .into(),
        )
    },
};

pub(crate) const CLAUDE: ReleaseFeed = ReleaseFeed {
    name: "claude",
    url: "https://downloads.claude.ai/claude-code-releases/latest",
    fallback: "2.1.330",
    parse: |body| Some(body.trim().into()),
};

/// Shared reported version for all accounts of one provider.
#[derive(Clone)]
pub struct ClientVersion(watch::Receiver<String>);

impl From<&str> for ClientVersion {
    fn from(version: &str) -> Self {
        Self(watch::channel(version.to_owned()).1)
    }
}

impl ClientVersion {
    pub(crate) fn current(&self) -> String {
        self.0.borrow().clone()
    }

    pub(crate) async fn resolve(
        configured: &str,
        feed: ReleaseFeed,
        directory: &Path,
        http: reqwest::Client,
    ) -> Self {
        if configured != "auto" {
            return configured.into();
        }

        let cache = directory.join(format!("{}-client-version.json", feed.name));
        let (version, mut delay) = cached_version(&cache, feed.fallback);
        let (sender, receiver) = watch::channel(version);
        let refresh = Refresh { feed, http, cache };

        if delay.is_zero() {
            refresh.check(&sender).await;
            delay = CHECK_INTERVAL;
        }
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = sender.closed() => break,
                    _ = tokio::time::sleep(delay) => {}
                }
                refresh.check(&sender).await;
                delay = CHECK_INTERVAL;
            }
        });
        Self(receiver)
    }
}

struct Refresh {
    feed: ReleaseFeed,
    http: reqwest::Client,
    cache: PathBuf,
}

impl Refresh {
    async fn fetch(&self) -> Result<String, String> {
        let body = self
            .http
            .get(endpoint(self.feed.url))
            .header(
                "User-Agent",
                concat!("llm-local-proxy/", env!("CARGO_PKG_VERSION")),
            )
            .timeout(CHECK_TIMEOUT)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|error| error.to_string())?
            .text()
            .await
            .map_err(|error| error.to_string())?;
        (self.feed.parse)(&body)
            .filter(|version| is_release_version(version))
            .ok_or_else(|| "release feed did not return a stable version number".into())
    }

    async fn check(&self, sender: &watch::Sender<String>) {
        let name = self.feed.name;
        let version = match self.fetch().await {
            Ok(version) => version,
            Err(error) => {
                eprintln!(
                    "{name}: client version check failed, keeping {}: {error}",
                    sender.borrow().as_str()
                );
                return;
            }
        };
        if *sender.borrow() != version {
            println!("{name}: using CLI client version {version}");
        }
        let stored = json!({"version": version, "checked_at": unix_seconds()});
        sender.send_replace(version);
        if let Err(error) = atomic::write_json(&self.cache, &stored) {
            eprintln!("{name}: could not cache the CLI client version: {error}");
        }
    }
}

fn cached_version(path: &Path, fallback: &str) -> (String, Duration) {
    let Some(Ok(cached)) = atomic::read_json(path).ok().flatten() else {
        return (fallback.into(), Duration::ZERO);
    };
    let Some(version) = cached["version"].as_str().filter(|v| is_release_version(v)) else {
        return (fallback.into(), Duration::ZERO);
    };
    let delay = cached["checked_at"]
        .as_u64()
        .and_then(|checked_at| unix_seconds().checked_sub(checked_at))
        .map(|age| CHECK_INTERVAL.saturating_sub(Duration::from_secs(age)))
        .unwrap_or_default();
    (version.into(), delay)
}

fn is_release_version(version: &str) -> bool {
    let mut parts = version.split('.');
    parts.clone().count() == 3
        && parts.all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
