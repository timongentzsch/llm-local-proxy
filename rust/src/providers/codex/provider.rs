//! The Codex provider: a ChatGPT subscription driven through codex app-server.

use super::app_server::AppServer;
use super::auth::CodexAuth;
use super::catalog::model_info;
use super::events::CodexDecoder;
use super::request::build;
use super::upstream::{accepted, Upstream};
use crate::error::{Error, Result};
use crate::ir::{ChatRequest, Decoder};
use crate::json::{get, py_str, truthy, Object};
use crate::ledger::{TokenLedger, Windows};
use crate::providers::catalog::match_model;
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
    binary: String,
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
        let slot = slot.to_string();
        Box::pin(async move {
            let app = Arc::new(AppServer::start(&self.binary, self.home(&slot)).await?);
            let tokens = account_file(&self.directory, Self::NAME, &slot, "tokens");
            let ledger = TokenLedger::new(Some(tokens), true);
            let auth = Arc::new(CodexAuth::new(app.clone()));
            Ok(Account {
                id: slot,
                limits: Some(auth.limits.clone()),
                auth,
                client: Arc::new(Upstream::new(app, self.http.clone(), ledger.clone())),
                ledger: Some(ledger),
            })
        })
    }

    fn fetch_catalog(
        &self,
        account: Arc<Account<Client>>,
    ) -> BoxFuture<'static, Result<Vec<Value>>> {
        Box::pin(async move {
            let upstream = &account.client;
            let listing = json!({"limit": 100, "includeHidden": false});
            let result = upstream.app.call("model/list", listing).await?;
            let contexts = upstream.app.model_contexts().await;
            let items: Vec<&Value> = result
                .get("data")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|item| item.is_object())
                .collect();
            let first_model = items
                .iter()
                .flat_map(|item| [get(item, "model"), get(item, "id")])
                .find(|name| truthy(name))
                .map(py_str);
            let transport_efforts = match first_model {
                Some(model) => accepted(&upstream.reasoning_efforts(&model).await?),
                None => None,
            };
            Ok(items
                .into_iter()
                .filter_map(|item| model_info(item, &contexts, transport_efforts.as_deref()))
                .collect())
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

    fn closed<'a>(&'a self, account: &'a Account<Client>) -> BoxFuture<'a, ()> {
        Box::pin(account.client.app.close())
    }

    fn retry_if(error: &Error) -> bool {
        matches!(error, Error::Rpc(_))
    }
}

pub struct Codex {
    pooled: Pooled<CodexBackend>,
}

impl Codex {
    pub async fn new(
        directory: &Path,
        codex_home: &Path,
        binary: &str,
        http: reqwest::Client,
    ) -> Result<Self> {
        let backend = CodexBackend {
            directory: directory.to_path_buf(),
            codex_home: codex_home.to_path_buf(),
            binary: binary.to_string(),
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

    fn healthy(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            for account in self.pooled.pool.accounts() {
                if !account.client.app.alive().await {
                    return false;
                }
            }
            true
        })
    }

    fn close(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            for account in self.pooled.pool.accounts() {
                account.client.app.close().await;
                account.client.ledger.flush();
            }
        })
    }
}
