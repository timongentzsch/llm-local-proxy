//! The Claude provider: an Anthropic subscription over the Claude Code edge.

use super::auth::ClaudeAuth;
use super::catalog::model_info;
use super::events::ClaudeDecoder;
use super::request::{build, Options};
use super::upstream::ClaudeUpstream;
use crate::error::{Error, Result};
use crate::ids::SharedIds;
use crate::ir::{ChatRequest, Decoder};
use crate::json::{get, integer, Object};
use crate::ledger::{TokenLedger, Windows};
use crate::providers::catalog::match_model;
use crate::providers::limits::LimitsStore;
use crate::providers::pool::{account_file, account_id, Account, AccountPool, Backend, Pooled};
use crate::providers::{BoxFuture, EventStream, Provider};
use crate::status::{AccountStatus, ProviderStatus};
use crate::tools::flatten;
use futures_util::TryFutureExt;
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The request fields /v1/messages/count_tokens accepts, per the pinned spec.
const COUNTED_FIELDS: [&str; 7] = [
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "cache_control",
];

pub struct ClaudeBackend {
    directory: PathBuf,
    http: reqwest::Client,
    ids: SharedIds,
}

type Client = Arc<ClaudeUpstream>;

impl Backend for ClaudeBackend {
    type Client = Client;

    const NAME: &'static str = "claude";

    fn new_account(&self, slot: &str) -> BoxFuture<'_, Result<Account<Client>>> {
        let file = |name: &str| account_file(&self.directory, Self::NAME, slot, name);
        let auth = Arc::new(ClaudeAuth::new(file("credentials"), self.http.clone()));
        let ledger = TokenLedger::new(Some(file("tokens")), false);
        let upstream = Arc::new(ClaudeUpstream::new(
            auth.clone(),
            self.http.clone(),
            ledger.clone(),
            self.ids.clone(),
        ));
        let reader = upstream.clone();
        let limits = LimitsStore::new(
            Self::NAME,
            Box::new(move || {
                let reader = reader.clone();
                Box::pin(async move { reader.limits().await })
            }),
        );
        let account = Account {
            id: slot.to_string(),
            auth,
            client: upstream,
            limits: Some(limits),
            ledger: Some(ledger),
        };
        Box::pin(async move { Ok(account) })
    }

    fn fetch_catalog(
        &self,
        account: Arc<Account<Client>>,
    ) -> BoxFuture<'static, Result<Vec<Value>>> {
        Box::pin(async move { account.client.models().await })
    }

    fn account_status<'a>(
        &'a self,
        account: &'a Account<Client>,
        pool: &'a AccountPool<Client>,
    ) -> BoxFuture<'a, Result<AccountStatus>> {
        Box::pin(async move {
            account.client.auth.hydrate_profile().await;
            let mut status = account.auth.status().await?;
            // A login awaiting reauthentication would only fail the read again.
            if !status.signed_in || !pool.account_error(&account.id).is_empty() {
                return Ok(status);
            }
            if let Some(limits) = &account.limits {
                (status.limits, status.updated_at) = limits.current(true).await;
            }
            status.tokens = json!(account.client.ledger.windows());
            Ok(status)
        })
    }

    fn no_account(&self) -> Error {
        Error::provider(
            401,
            "not signed in to Claude; use the sign in button on the status page",
        )
    }

    fn state_dirs(&self, slot: &str) -> Vec<PathBuf> {
        vec![self.directory.join("accounts").join(Self::NAME).join(slot)]
    }
}

pub struct Claude {
    pooled: Pooled<ClaudeBackend>,
    ids: SharedIds,
}

impl Claude {
    pub async fn new(directory: &Path, http: reqwest::Client, ids: SharedIds) -> Result<Self> {
        let backend = ClaudeBackend {
            directory: directory.to_path_buf(),
            http,
            ids: ids.clone(),
        };
        Ok(Claude {
            pooled: Pooled::new(backend, directory).await?,
            ids,
        })
    }

    /// What the live catalog says about one model, or Null.
    async fn capability(&self, model: &str, key: &str) -> Value {
        self.pooled
            .live_catalog()
            .await
            .iter()
            .find(|item| get(item, "id") == model)
            .map(|item| get(item, key).clone())
            .unwrap_or(Value::Null)
    }

    async fn request(
        &self,
        canonical: &str,
        request: &ChatRequest,
    ) -> Result<(Object, Vec<String>)> {
        if !self.pooled.signed_in().await {
            return Err(self.pooled.backend.no_account());
        }
        let efforts: Option<Vec<String>> = self
            .capability(canonical, "reasoning_efforts")
            .await
            .as_array()
            .map(|items| items.iter().map(crate::json::py_str).collect());
        let thinking = self.capability(canonical, "thinking").await;
        let options = Options {
            max_output: integer(&self.capability(canonical, "max_output_tokens").await),
            thinking: thinking.as_str(),
            reasoning_efforts: efforts.as_deref(),
            reasoning_cache: Some(&self.pooled.cache),
        };
        build(request, canonical, options, self.ids.as_ref())
    }

    async fn finish_login(&self, body: &Object) -> Result<Value> {
        let code = match body.get("code") {
            Some(Value::String(code)) if !code.trim().is_empty() => code,
            _ => return Err(Error::request("code is required")),
        };
        let slot = account_id(body)?;
        let account = self.pooled.pool.get(&slot)?;
        let result = account.client.auth.finish(code).await?;
        // The slot may now hold a different login; its old bars are not ours.
        if let Some(limits) = &account.limits {
            limits.clear();
        }
        self.pooled.pool.clear_account_error(&slot);
        self.pooled.forget();
        Ok(result)
    }
}

impl Provider for Claude {
    fn name(&self) -> &'static str {
        ClaudeBackend::NAME
    }

    fn match_model<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Option<String>> {
        Box::pin(async move {
            if !self.pooled.signed_in().await {
                return None;
            }
            match_model(model, &self.pooled.live_catalog().await)
        })
    }

    fn chat<'a>(
        &'a self,
        canonical: &'a str,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<(EventStream, Box<dyn Decoder>)>> {
        Box::pin(async move {
            let (body, betas) = self.request(canonical, request).await?;
            let names = flatten(&request.tools)?.1;
            let (body, betas) = (Arc::new(body), Arc::new(betas));
            let caller = request.caller.clone();
            let opening = self.pooled.pool.clone().stream(
                Some(request.session.clone()),
                move |account| {
                    let (body, betas, caller) = (body.clone(), betas.clone(), caller.clone());
                    Box::pin(async move { account.client.events(&body, &betas, &caller).await })
                },
                self.pooled.failover(request.starts_conversation()),
            );
            let events: EventStream = Box::pin(opening.try_flatten_stream());
            let decoder =
                ClaudeDecoder::new(Some(self.pooled.cache.clone()), names, self.ids.clone());
            Ok((events, Box::new(decoder) as Box<dyn Decoder>))
        })
    }

    fn models(&self) -> BoxFuture<'_, Vec<Value>> {
        Box::pin(async move {
            if !self.pooled.signed_in().await {
                return Vec::new();
            }
            self.pooled
                .live_catalog()
                .await
                .iter()
                .map(model_info)
                .collect()
        })
    }

    fn status(&self) -> BoxFuture<'_, ProviderStatus> {
        Box::pin(self.pooled.status())
    }

    fn routes(&self) -> Vec<&'static str> {
        vec!["accounts", "code", "login", "logout"]
    }

    fn route<'a>(
        &'a self,
        route: &'a str,
        body: &'a Object,
    ) -> BoxFuture<'a, Option<Result<Value>>> {
        Box::pin(async move {
            match route {
                "code" => Some(self.finish_login(body).await),
                other => self.pooled.route(other, body).await,
            }
        })
    }

    fn count_tokens<'a>(
        &'a self,
        canonical: &'a str,
        request: &'a ChatRequest,
    ) -> Option<BoxFuture<'a, Result<Value>>> {
        Some(Box::pin(async move {
            let (body, betas) = self.request(canonical, request).await?;
            // Its schema accepts only prompt fields; the rest are rejected.
            let counted: Object = COUNTED_FIELDS
                .iter()
                .filter_map(|key| body.get(*key).map(|value| (key.to_string(), value.clone())))
                .collect();
            let (counted, betas) = (Arc::new(counted), Arc::new(betas));
            self.pooled
                .pool
                .call(
                    Some(&request.session),
                    move |account| {
                        let (counted, betas) = (counted.clone(), betas.clone());
                        Box::pin(async move { account.client.count_tokens(&counted, &betas).await })
                    },
                    self.pooled.failover(request.starts_conversation()),
                )
                .await
        }))
    }

    fn callers(&self) -> IndexMap<String, Windows> {
        self.pooled.callers()
    }

    fn forget(&self) {
        self.pooled.forget()
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            for account in self.pooled.pool.accounts() {
                account.client.ledger.flush();
            }
        })
    }
}
