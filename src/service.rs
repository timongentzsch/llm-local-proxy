//! Composition root.
//!
//! Builds the provider registry, then aggregates it: the merged model
//! catalog, the status cards, health. It is deliberately transport-free --
//! nothing here knows about HTTP -- and provider-agnostic.

use crate::config::Config;
use crate::dialects::DIALECTS;
use crate::error::Result;
use crate::ids::{self, SharedIds};
use crate::keys::KeyStore;
use crate::providers::claude::provider::Claude;
use crate::providers::codex::provider::Codex;
use crate::providers::{transport, Provider};
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;

pub struct Service {
    pub config: Config,
    /// In match priority: the first live catalog to claim a model serves it.
    pub providers: Vec<Arc<dyn Provider>>,
    pub keys: KeyStore,
    pub ids: SharedIds,
}

impl Service {
    pub async fn new(config: Config) -> Result<Self> {
        let directory = config.directory().to_path_buf();
        let http = transport::client(Duration::from_secs(config.request_timeout));
        let ids = ids::random();
        let claude = Claude::new(&directory, http.clone(), ids.clone()).await?;
        let codex = Codex::new(
            &directory,
            &config.codex_home,
            &config.codex_client_version,
            http,
        )
        .await?;
        Ok(Service {
            keys: KeyStore::new(directory.join("keys.json")),
            providers: vec![Arc::new(claude), Arc::new(codex)],
            config,
            ids,
        })
    }

    pub fn provider(&self, name: &str) -> Option<&Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|provider| provider.name() == name)
    }

    /// First provider whose catalog claims the model, with its canonical id.
    pub async fn route(&self, model: &str) -> Option<(Arc<dyn Provider>, String)> {
        for provider in &self.providers {
            if let Some(canonical) = provider.match_model(model).await {
                return Some((provider.clone(), canonical));
            }
        }
        None
    }

    pub async fn healthy(&self) -> bool {
        for provider in &self.providers {
            if !provider.healthy().await {
                return false;
            }
        }
        true
    }

    /// The merged catalog; each provider caches its own slice.
    pub async fn models(&self, refresh: bool) -> Vec<Value> {
        let mut data = Vec::new();
        for provider in &self.providers {
            if refresh {
                provider.forget();
            }
            data.extend(provider.models().await);
        }
        data
    }

    pub async fn status(&self) -> Value {
        let mut cards = Vec::new();
        for provider in &self.providers {
            let mut card = crate::obj! {
                "name": provider.name(),
                "routes": provider.routes(),
            };
            if let Value::Object(fields) = provider.status().await.payload() {
                card.extend(fields);
            }
            cards.push(Value::Object(card));
        }
        json!({"dialects": base_urls(&self.config.origin()), "providers": cards})
    }

    /// Proxy token windows per calling key, then per provider.
    pub fn usage(&self) -> Value {
        let mut result = crate::json::Object::new();
        for provider in &self.providers {
            for (caller, windows) in provider.callers() {
                let entry = result.entry(caller).or_insert_with(|| json!({}));
                entry[provider.name()] = json!(windows);
            }
        }
        Value::Object(result)
    }

    pub async fn close(&self) {
        for provider in &self.providers {
            provider.close().await;
        }
    }
}

/// Where a client reaches each dialect, from one listener's origin.
pub fn base_urls(origin: &str) -> Value {
    DIALECTS
        .iter()
        .map(|dialect| {
            json!({
                "name": dialect.name,
                "base_url": format!("{origin}{}", dialect.base_path),
            })
        })
        .collect()
}
