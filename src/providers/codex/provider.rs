//! The Codex provider: a ChatGPT subscription over the Codex backend.

use super::auth::CodexAuth;
use super::catalog::catalog;
use super::events::CodexDecoder;
use super::request::build;
use super::upstream::{accepted, Upstream};
use crate::error::{Error, Result};
use crate::ir::{ChatRequest, Decoder};
use crate::json::{get, py_str, truthy, Object};
use crate::ledger::{TokenLedger, Windows};
use crate::providers::catalog::match_model;
use crate::providers::limits::LimitsStore;
use crate::providers::pool::{account_file, Account, AccountPool, Backend, Pooled};
use crate::providers::{BoxFuture, EventStream, Provider};
use crate::status::{AccountStatus, ProviderStatus};
use futures_util::TryFutureExt;
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct CodexBackend {
    directory: PathBuf,
    codex_home: PathBuf,
    /// The CLI version the model list is asked for; the backend offers a
    /// client only the models that version can drive.
    client_version: String,
    http: reqwest::Client,
}

type Client = Arc<Upstream>;

impl CodexBackend {
    fn home(&self, slot: &str) -> PathBuf {
        self.codex_home.join("accounts").join(slot)
    }
}

impl Backend for CodexBackend {
    type Client = Client;

    const NAME: &'static str = "codex";

    fn new_account(&self, slot: &str) -> BoxFuture<'_, Result<Account<Client>>> {
        let auth = CodexAuth::new(self.home(slot).join("auth.json"), self.http.clone());
        let reader = auth.clone();
        let limits = LimitsStore::new(
            Self::NAME,
            Box::new(move || {
                let reader = reader.clone();
                Box::pin(async move { reader.limits().await })
            }),
        );
        auth.watch(limits.clone());
        let tokens = account_file(&self.directory, Self::NAME, slot, "tokens");
        let ledger = TokenLedger::new(Some(tokens), true);
        let account = Account {
            id: slot.to_string(),
            limits: Some(limits),
            client: Arc::new(Upstream::new(
                auth.clone(),
                self.http.clone(),
                ledger.clone(),
            )),
            auth,
            ledger: Some(ledger),
        };
        Box::pin(async move { Ok(account) })
    }

    fn fetch_catalog(
        &self,
        account: Arc<Account<Client>>,
    ) -> BoxFuture<'static, Result<Vec<Value>>> {
        let client_version = self.client_version.clone();
        Box::pin(async move {
            let upstream = &account.client;
            let models = upstream.auth.models(&client_version).await?;
            let first_model = models
                .iter()
                .filter(|item| get(item, "visibility") == "list")
                .min_by_key(|item| get(item, "priority").as_i64().unwrap_or(i64::MAX))
                .map(|item| get(item, "slug"))
                .filter(|slug| truthy(slug))
                .map(py_str);
            let transport_efforts = match first_model {
                Some(model) => accepted(&upstream.reasoning_efforts(&model).await?),
                None => None,
            };
            Ok(catalog(&models, transport_efforts.as_deref()))
        })
    }

    fn account_status<'a>(
        &'a self,
        account: &'a Account<Client>,
        _pool: &'a AccountPool<Client>,
    ) -> BoxFuture<'a, Result<AccountStatus>> {
        Box::pin(async move {
            let mut status = account.auth.status().await?;
            status.tokens = json!(account.client.ledger.windows());
            Ok(status)
        })
    }

    fn no_account(&self) -> Error {
        Error::provider(
            401,
            "not signed in to Codex; use the sign in button on the status page",
        )
    }

    fn state_dirs(&self, slot: &str) -> Vec<PathBuf> {
        vec![
            self.directory.join("accounts").join(Self::NAME).join(slot),
            self.home(slot),
        ]
    }
}

pub struct Codex {
    pooled: Pooled<CodexBackend>,
}

impl Codex {
    pub async fn new(
        directory: &Path,
        codex_home: &Path,
        client_version: &str,
        http: reqwest::Client,
    ) -> Result<Self> {
        let backend = CodexBackend {
            directory: directory.to_path_buf(),
            codex_home: codex_home.to_path_buf(),
            client_version: client_version.to_string(),
            http,
        };
        Ok(Codex {
            pooled: Pooled::new(backend, directory).await?,
        })
    }
}

impl Provider for Codex {
    fn name(&self) -> &'static str {
        CodexBackend::NAME
    }

    fn match_model<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Option<String>> {
        Box::pin(async move { match_model(model, &self.pooled.live_catalog().await) })
    }

    fn chat<'a>(
        &'a self,
        canonical: &'a str,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<(EventStream, Box<dyn Decoder>)>> {
        Box::pin(async move {
            let catalog = self.pooled.live_catalog().await;
            let efforts: Option<Vec<String>> = catalog
                .iter()
                .find(|item| get(item, "id") == canonical)
                .and_then(|item| get(item, "supported_reasoning_efforts").as_array())
                .map(|items| items.iter().map(py_str).collect());
            let (body, cache_key) = build(request, &self.pooled.cache, efforts.as_deref())?;
            // Without a session, the cache key, which is derived when the
            // client names none: each account has its own upstream cache, and
            // round-robin would hand every turn of one conversation a
            // different one.
            let session = if request.session.is_empty() {
                cache_key
            } else {
                request.session.clone()
            };
            let body = Arc::new(body);
            let caller = request.caller.clone();
            let opening = self.pooled.pool.clone().stream(
                Some(session),
                move |account| {
                    let (body, caller) = (body.clone(), caller.clone());
                    Box::pin(async move { account.client.events(&body, &caller).await })
                },
                self.pooled.failover(request.starts_conversation()),
            );
            let events: EventStream = Box::pin(opening.try_flatten_stream());
            let decoder = CodexDecoder::new(self.pooled.cache.clone());
            Ok((events, Box::new(decoder) as Box<dyn Decoder>))
        })
    }

    fn models(&self) -> BoxFuture<'_, Vec<Value>> {
        Box::pin(async move { self.pooled.live_catalog().await.as_ref().clone() })
    }

    fn status(&self) -> BoxFuture<'_, ProviderStatus> {
        Box::pin(self.pooled.status())
    }

    fn routes(&self) -> Vec<&'static str> {
        vec!["accounts", "login", "logout"]
    }

    fn route<'a>(
        &'a self,
        route: &'a str,
        body: &'a Object,
    ) -> BoxFuture<'a, Option<Result<Value>>> {
        Box::pin(self.pooled.route(route, body))
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
