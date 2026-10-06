//! Shared account selection and rate-limit failover for subscription providers.

use crate::atomic;
use crate::error::{Error, Result};
use crate::json::Object;
use crate::ledger::{merge, TokenLedger, Windows};
use crate::providers::limits::{fullest, wall_clock, LimitsStore};
use crate::providers::{BoxFuture, EventStream};
use crate::reasoning::ReasoningCache;
use crate::status::{AccountStatus, ProviderStatus};
use futures_util::stream::{self, StreamExt};
use indexmap::IndexMap;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// How long a rate-limited account rests when the upstream did not say, and
/// the longest it rests when it did: a reset days away is asked about again
/// hourly, so a wrong hint cannot bench an account for a week.
const RATE_LIMIT_COOLDOWN_SECONDS: u64 = 300;
const MAX_COOLDOWN_SECONDS: u64 = 3600;
const AUTH_FAILURE_COOLDOWN_SECONDS: f64 = 60.0;
const CATALOG_TTL_SECONDS: f64 = 60.0;
const CATALOG_RETRY_SECONDS: f64 = 5.0;
/// A session stays on the account that last served it for this long: past the
/// longest upstream prompt-cache lifetime (Anthropic's 1h) there is nothing to
/// keep. The table is bounded; the oldest entry goes first.
const SESSION_TTL_SECONDS: f64 = 3600.0;
const SESSION_LIMIT: usize = 4096;
/// An account this full on a window that limits it whole takes no new
/// sessions while another has room; the sessions it serves stay, keeping
/// their cache.
const SOFT_LIMIT_PERCENT: f64 = 90.0;

/// Login lifecycle that any provider can expose to the status page.
///
/// Code-paste flows (Claude) expose an extra `finish(code)` on the concrete
/// type; device-code flows (Codex) do not. Only the operations every provider
/// shares live here.
pub trait Auth: Send + Sync {
    /// Start a login: `{"url": ...}` plus `"code"` for device flows.
    fn login_start(&self) -> BoxFuture<'_, Result<Value>>;
    /// Sign out locally and revoke the stored session if supported.
    fn logout(&self) -> BoxFuture<'_, Result<()>>;
    /// Whether the provider currently has a usable session.
    fn signed_in(&self) -> BoxFuture<'_, Result<bool>>;
    /// Normalised card for `/api/status` (secrets excluded).
    fn status(&self) -> BoxFuture<'_, Result<AccountStatus>>;
}

pub struct Account<C> {
    pub id: String,
    pub auth: Arc<dyn Auth>,
    pub client: C,
    /// Where the account's usage bars are kept, when it reports any.
    pub limits: Option<Arc<LimitsStore>>,
    /// Where the account's proxy token counts are kept.
    pub ledger: Option<Arc<TokenLedger>>,
}

async fn signed_in(auth: &dyn Auth) -> bool {
    auth.signed_in().await.unwrap_or(false)
}

struct State<C> {
    accounts: Vec<Arc<Account<C>>>,
    cursor: usize,
    cooldown: HashMap<String, f64>,
    account_errors: HashMap<String, String>,
    /// session -> (account id, last served), oldest first.
    sessions: IndexMap<String, (String, f64)>,
}

/// Select signed-in accounts and retry a request before its first event.
///
/// A session stays on the account that last served it, for prompt-cache
/// locality; a new one starts on its rendezvous-hash account, and sessionless
/// requests round-robin. Both prefer accounts below `SOFT_LIMIT_PERCENT`.
/// Rate limits and unusable credentials cool an account and advance to the
/// next. Once an event has been yielded, errors pass through unchanged.
pub struct AccountPool<C> {
    state: Mutex<State<C>>,
}

/// How one request may be retried on another account.
#[derive(Clone)]
pub struct Failover {
    /// What to answer when no account is signed in.
    pub no_account: Error,
    /// Failures beyond 429 and unusable credentials that another account may
    /// not share.
    pub retry_if: fn(&Error) -> bool,
    /// Only a request starting a conversation avoids nearly-full accounts.
    pub starting: bool,
}

impl<C: Send + Sync + 'static> AccountPool<C> {
    pub fn new(accounts: Vec<Arc<Account<C>>>) -> Arc<Self> {
        Arc::new(AccountPool {
            state: Mutex::new(State {
                accounts,
                cursor: 0,
                cooldown: HashMap::new(),
                account_errors: HashMap::new(),
                sessions: IndexMap::new(),
            }),
        })
    }

    pub fn accounts(&self) -> Vec<Arc<Account<C>>> {
        self.state.lock().unwrap().accounts.clone()
    }

    pub fn add(&self, account: Arc<Account<C>>) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if state.accounts.iter().any(|item| item.id == account.id) {
            return Err(Error::request(format!(
                "account already exists: {}",
                account.id
            )));
        }
        state.accounts.push(account);
        Ok(())
    }

    pub fn remove(&self, account_id: &str) -> Result<Arc<Account<C>>> {
        let mut state = self.state.lock().unwrap();
        let index = state
            .accounts
            .iter()
            .position(|item| item.id == account_id)
            .ok_or_else(|| Error::request(format!("unknown account: {account_id}")))?;
        let account = state.accounts.remove(index);
        state.cooldown.remove(account_id);
        state.account_errors.remove(account_id);
        state.sessions.retain(|_, (pinned, _)| pinned != account_id);
        Ok(account)
    }

    pub fn get(&self, account_id: &str) -> Result<Arc<Account<C>>> {
        self.accounts()
            .into_iter()
            .find(|account| account.id == account_id)
            .ok_or_else(|| Error::request(format!("unknown account: {account_id}")))
    }

    /// Refuse another slot while an existing one still needs a login.
    pub async fn require_no_unsigned(&self) -> Result<()> {
        for account in self.accounts() {
            if !signed_in(account.auth.as_ref()).await {
                return Err(Error::request(
                    "sign in or remove the existing unsigned account first",
                ));
            }
        }
        Ok(())
    }

    /// Accounts to try in order.
    ///
    /// Only a request `starting` a conversation avoids nearly-full accounts:
    /// one that continues a conversation has a prompt cache on its account,
    /// remembered or not (e.g. after a restart), so it goes there regardless.
    pub async fn candidates(&self, session: Option<&str>, starting: bool) -> Vec<Arc<Account<C>>> {
        let mut available = Vec::new();
        for account in self.accounts() {
            if signed_in(account.auth.as_ref()).await {
                available.push(account);
            }
        }
        if available.is_empty() {
            return available;
        }
        // With one account there is nowhere else to go, so skip the read.
        let mut full = HashSet::new();
        if starting && available.len() > 1 {
            for account in &available {
                if self.draining(account).await {
                    full.insert(account.id.clone());
                }
            }
        }
        let session = session.filter(|session| !session.is_empty());
        let mut state = self.state.lock().unwrap();
        let now = wall_clock();
        let ready: Vec<_> = available
            .iter()
            .filter(|account| state.cooldown.get(&account.id).copied().unwrap_or(0.0) <= now)
            .cloned()
            .collect();
        let mut choices = if ready.is_empty() { available } else { ready };
        let mut kept = None;
        let order: HashMap<String, i128> = match session {
            Some(session) => {
                if let Some((pinned, served)) = state.sessions.get(session) {
                    if now - served < SESSION_TTL_SECONDS {
                        kept = Some(pinned.clone());
                    }
                }
                choices
                    .iter()
                    .map(|a| (a.id.clone(), -i128::from(affinity(session, &a.id))))
                    .collect()
            }
            None => {
                let count = choices.len();
                let start = state.cursor % count;
                state.cursor += 1;
                choices
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (a.id.clone(), ((i + count - start) % count) as i128))
                    .collect()
            }
        };
        drop(state);
        choices.sort_by_key(|account| {
            (
                kept.as_deref() != Some(account.id.as_str()),
                full.contains(&account.id),
                order[&account.id],
            )
        });
        choices
    }

    /// Near a limit that covers the whole account.
    pub async fn draining(&self, account: &Account<C>) -> bool {
        let Some(limits) = &account.limits else {
            return false;
        };
        let (bars, _) = limits.current(false).await;
        fullest(&bars).unwrap_or(0.0) >= SOFT_LIMIT_PERCENT
    }

    fn remember(&self, session: Option<&str>, account_id: &str) {
        let Some(session) = session.filter(|session| !session.is_empty()) else {
            return;
        };
        let mut state = self.state.lock().unwrap();
        state.sessions.shift_remove(session);
        state
            .sessions
            .insert(session.to_string(), (account_id.to_string(), wall_clock()));
        while state.sessions.len() > SESSION_LIMIT {
            state.sessions.shift_remove_index(0);
        }
    }

    /// Rest an account for as long as its upstream asked, within bounds.
    pub fn mark_rate_limited(&self, account_id: &str, asked: Option<u64>) {
        let seconds = asked
            .unwrap_or(RATE_LIMIT_COOLDOWN_SECONDS)
            .min(MAX_COOLDOWN_SECONDS);
        // Zero: the refusal was about the request, not the account.
        if seconds == 0 {
            return;
        }
        let mut state = self.state.lock().unwrap();
        state
            .cooldown
            .insert(account_id.to_string(), wall_clock() + seconds as f64);
    }

    /// Latest terminal authentication failure observed for one account.
    pub fn account_error(&self, account_id: &str) -> String {
        let state = self.state.lock().unwrap();
        state
            .account_errors
            .get(account_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn clear_account_error(&self, account_id: &str) {
        let mut state = self.state.lock().unwrap();
        state.account_errors.remove(account_id);
        state.cooldown.remove(account_id);
    }

    fn mark_account_error(&self, account_id: &str, error: &Error) {
        let message = match error.message() {
            "" => "authentication failed",
            message => message,
        };
        let mut state = self.state.lock().unwrap();
        state
            .account_errors
            .insert(account_id.to_string(), message.to_string());
        state.cooldown.insert(
            account_id.to_string(),
            wall_clock() + AUTH_FAILURE_COOLDOWN_SECONDS,
        );
    }

    /// Whether `error` moves the request to the next account, cooling this one.
    fn fails_over(&self, account_id: &str, error: &Error, failover: &Failover) -> bool {
        let unavailable = error.account_unavailable();
        let limited = error.status() == 429 && matches!(error, Error::Provider { .. });
        if !(limited || unavailable || (failover.retry_if)(error)) {
            return false;
        }
        if unavailable {
            self.mark_account_error(account_id, error);
        } else if limited {
            self.mark_rate_limited(account_id, error.cooldown());
        }
        true
    }

    /// Open a stream, failing over on rate limits or unusable auth before
    /// output begins. The first event is awaited here, since only then is the
    /// account known to have accepted the request.
    pub async fn stream<F>(
        self: Arc<Self>,
        session: Option<String>,
        create: F,
        failover: Failover,
    ) -> Result<EventStream>
    where
        F: Fn(Arc<Account<C>>) -> BoxFuture<'static, Result<EventStream>> + Send,
    {
        let session = session.as_deref();
        let candidates = self.candidates(session, failover.starting).await;
        let mut last = failover.no_account.clone();
        for account in candidates {
            let mut events = match create(account.clone()).await {
                Ok(events) => events,
                Err(error) if self.fails_over(&account.id, &error, &failover) => {
                    last = error;
                    continue;
                }
                Err(error) => return Err(error),
            };
            match events.next().await {
                Some(Ok(first)) => {
                    self.clear_account_error(&account.id);
                    self.remember(session, &account.id);
                    return Ok(Box::pin(
                        stream::once(async move { Ok(first) }).chain(events),
                    ));
                }
                Some(Err(error)) if self.fails_over(&account.id, &error, &failover) => {
                    last = error;
                }
                Some(Err(error)) => return Err(error),
                None => {
                    self.clear_account_error(&account.id);
                    return Ok(Box::pin(stream::empty()));
                }
            }
        }
        Err(last)
    }

    /// Non-streaming equivalent used by token counting and catalog discovery.
    pub async fn call<T, F>(
        &self,
        session: Option<&str>,
        invoke: F,
        failover: Failover,
    ) -> Result<T>
    where
        F: Fn(Arc<Account<C>>) -> BoxFuture<'static, Result<T>>,
    {
        let candidates = self.candidates(session, failover.starting).await;
        let mut last = failover.no_account.clone();
        for account in candidates {
            match invoke(account.clone()).await {
                Ok(value) => {
                    self.clear_account_error(&account.id);
                    self.remember(session, &account.id);
                    return Ok(value);
                }
                Err(error) if self.fails_over(&account.id, &error, &failover) => last = error,
                Err(error) => return Err(error),
            }
        }
        Err(last)
    }
}

/// Rendezvous score: a session starts on its highest-scoring account.
fn affinity(session: &str, account_id: &str) -> u64 {
    let digest = Sha256::digest(format!("{session}\0{account_id}").as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("a digest has 32 bytes"))
}

/// The canonical private state path for one provider account.
pub fn account_file(directory: &Path, provider: &str, account_id: &str, name: &str) -> PathBuf {
    directory
        .join("accounts")
        .join(provider)
        .join(account_id)
        .join(format!("{name}.json"))
}

/// Persistent, uncapped slot ids shared by every pooled provider.
pub struct AccountStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl AccountStore {
    pub fn new(directory: &Path, provider: &str) -> Self {
        AccountStore {
            path: directory.join("accounts").join(provider).join("slots.json"),
            lock: Mutex::new(()),
        }
    }

    pub fn ids(&self) -> Result<Vec<String>> {
        let _guard = self.lock.lock().unwrap();
        self.read()
    }

    pub fn add(&self) -> Result<String> {
        let _guard = self.lock.lock().unwrap();
        let mut ids = self.read()?;
        let used: HashSet<u64> = ids.iter().filter_map(|id| id.parse().ok()).collect();
        let free = (1..).find(|value| !used.contains(value)).unwrap_or(1);
        ids.push(free.to_string());
        self.write(&ids)?;
        Ok(free.to_string())
    }

    pub fn remove(&self, account_id: &str) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let mut ids = self.read()?;
        if !ids.iter().any(|id| id == account_id) {
            return Err(Error::request(format!("unknown account: {account_id}")));
        }
        ids.retain(|id| id != account_id);
        self.write(&ids)
    }

    fn read(&self) -> Result<Vec<String>> {
        let invalid =
            || Error::upstream(format!("invalid account registry: {}", self.path.display()));
        let value = match atomic::read_json(&self.path) {
            Ok(None) => return Ok(Vec::new()),
            Ok(Some(Ok(value))) => value,
            Ok(Some(Err(_))) => return Err(invalid()),
            Err(error) => return Err(Error::upstream(error.to_string())),
        };
        let items = value
            .get("accounts")
            .and_then(Value::as_array)
            .ok_or_else(invalid)?;
        let ids: Vec<String> = items.iter().map(crate::json::py_str).collect();
        let distinct: HashSet<&String> = ids.iter().collect();
        let numbered = |id: &String| {
            !id.is_empty()
                && id.bytes().all(|b| b.is_ascii_digit())
                && id.parse::<u64>().is_ok_and(|n| n >= 1)
        };
        if distinct.len() != ids.len() || !ids.iter().all(numbered) {
            return Err(invalid());
        }
        Ok(ids)
    }

    fn write(&self, ids: &[String]) -> Result<()> {
        atomic::write_json(&self.path, &json!({ "accounts": ids }))
            .map_err(|error| Error::upstream(error.to_string()))
    }
}

/// Delete one validated slot directory after it leaves the registry.
fn remove_account_state(path: &Path) {
    if path.exists() {
        let _ = std::fs::remove_dir_all(path);
    }
}

pub fn account_id(body: &Object) -> Result<String> {
    match body.get("account") {
        Some(Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        _ => Err(Error::request("account is required")),
    }
}

/// What one subscription supplies: how to build an account, read the catalog
/// through it, and describe it.
pub trait Backend: Send + Sync + 'static {
    type Client: Send + Sync + 'static;

    const NAME: &'static str;

    fn new_account(&self, slot: &str) -> BoxFuture<'_, Result<Account<Self::Client>>>;

    fn fetch_catalog(
        &self,
        account: Arc<Account<Self::Client>>,
    ) -> BoxFuture<'static, Result<Vec<Value>>>;

    fn account_status<'a>(
        &'a self,
        account: &'a Account<Self::Client>,
        pool: &'a AccountPool<Self::Client>,
    ) -> BoxFuture<'a, Result<AccountStatus>>;

    /// What a request is answered with while nobody is signed in.
    fn no_account(&self) -> Error;

    fn state_dirs(&self, slot: &str) -> Vec<PathBuf>;

    /// Release what a removed slot holds beyond its files.
    fn closed<'a>(&'a self, _account: &'a Account<Self::Client>) -> BoxFuture<'a, ()> {
        Box::pin(async {})
    }

    fn retry_if(_error: &Error) -> bool {
        false
    }
}

#[derive(Default)]
struct Catalog {
    fetched: Option<(f64, Arc<Vec<Value>>)>,
    retry_at: f64,
}

/// What every subscription provider shares: slots, logins, catalog, status.
///
/// The slot lifecycle, catalog cache and the "reauthentication required"
/// overlay are identical for every upstream.
pub struct Pooled<B: Backend> {
    pub backend: B,
    pub store: AccountStore,
    pub pool: Arc<AccountPool<B::Client>>,
    pub cache: Arc<ReasoningCache>,
    catalog: Mutex<Catalog>,
    /// One caller refreshes the catalog; the rest wait for its answer.
    refresh: tokio::sync::Mutex<()>,
    accounts_lock: tokio::sync::Mutex<()>,
}

impl<B: Backend> Pooled<B> {
    pub async fn new(backend: B, directory: &Path) -> Result<Self> {
        let store = AccountStore::new(directory, B::NAME);
        let mut accounts = Vec::new();
        for slot in store.ids()? {
            accounts.push(Arc::new(backend.new_account(&slot).await?));
        }
        Ok(Pooled {
            backend,
            store,
            pool: AccountPool::new(accounts),
            cache: Arc::new(ReasoningCache::default()),
            catalog: Mutex::new(Catalog::default()),
            refresh: tokio::sync::Mutex::new(()),
            accounts_lock: tokio::sync::Mutex::new(()),
        })
    }

    pub fn failover(&self, starting: bool) -> Failover {
        Failover {
            no_account: self.backend.no_account(),
            retry_if: B::retry_if,
            starting,
        }
    }

    /// Token windows per calling key, over every account of this provider.
    pub fn callers(&self) -> IndexMap<String, Windows> {
        let mut groups: IndexMap<String, Vec<Windows>> = IndexMap::new();
        for account in self.pool.accounts() {
            let Some(ledger) = &account.ledger else {
                continue;
            };
            for (caller, windows) in ledger.by_caller() {
                groups.entry(caller).or_default().push(windows);
            }
        }
        groups
            .into_iter()
            .map(|(caller, items)| (caller, merge(items)))
            .collect()
    }

    pub async fn signed_in(&self) -> bool {
        for account in self.pool.accounts() {
            if signed_in(account.auth.as_ref()).await {
                return true;
            }
        }
        false
    }

    /// The routes every pooled provider serves; None for one it does not.
    pub async fn route(&self, name: &str, body: &Object) -> Option<Result<Value>> {
        Some(match name {
            "login" => self.login(body).await,
            "logout" => self.logout(body).await,
            "accounts" => self.manage_accounts(body).await,
            _ => return None,
        })
    }

    async fn login(&self, body: &Object) -> Result<Value> {
        let slot = account_id(body)?;
        let mut started = self.pool.get(&slot)?.auth.login_start().await?;
        if let Some(map) = started.as_object_mut() {
            map.insert("account".into(), json!(slot));
        }
        Ok(started)
    }

    async fn logout(&self, body: &Object) -> Result<Value> {
        self.pool.get(&account_id(body)?)?.auth.logout().await?;
        self.forget();
        Ok(json!({ "ok": true }))
    }

    async fn manage_accounts(&self, body: &Object) -> Result<Value> {
        let slot = match body.get("action").and_then(Value::as_str) {
            Some("add") => {
                let _guard = self.accounts_lock.lock().await;
                self.pool.require_no_unsigned().await?;
                let slot = self.store.add()?;
                let added = match self.backend.new_account(&slot).await {
                    Ok(account) => self.pool.add(Arc::new(account)),
                    Err(error) => Err(error),
                };
                if let Err(error) = added {
                    let _ = self.store.remove(&slot);
                    for path in self.backend.state_dirs(&slot) {
                        remove_account_state(&path);
                    }
                    return Err(error);
                }
                slot
            }
            Some("remove") => {
                let slot = account_id(body)?;
                let _guard = self.accounts_lock.lock().await;
                let account = self.pool.get(&slot)?;
                if account.auth.signed_in().await? {
                    return Err(Error::request("sign out before removing this account"));
                }
                self.store.remove(&slot)?;
                self.pool.remove(&slot)?;
                self.backend.closed(&account).await;
                for path in self.backend.state_dirs(&slot) {
                    remove_account_state(&path);
                }
                slot
            }
            _ => return Err(Error::request("action must be add or remove")),
        };
        self.forget();
        Ok(json!({ "ok": true, "account": slot }))
    }

    /// Drop the cached catalog, e.g. after a login changes what is visible.
    pub fn forget(&self) {
        *self.catalog.lock().unwrap() = Catalog::default();
    }

    fn fresh_catalog(&self) -> Option<Arc<Vec<Value>>> {
        let now = wall_clock();
        let catalog = self.catalog.lock().unwrap();
        match &catalog.fetched {
            Some((at, items)) if now - at < CATALOG_TTL_SECONDS => Some(items.clone()),
            Some((_, items)) if now < catalog.retry_at => Some(items.clone()),
            None if now < catalog.retry_at => Some(Arc::new(Vec::new())),
            _ => None,
        }
    }

    pub async fn live_catalog(&self) -> Arc<Vec<Value>> {
        if let Some(cached) = self.fresh_catalog() {
            return cached;
        }
        // One caller refreshes; the rest wait for its answer instead of each
        // sending their own discovery upstream.
        let _guard = self.refresh.lock().await;
        if let Some(cached) = self.fresh_catalog() {
            return cached;
        }
        let fetched = self
            .pool
            .call(
                None,
                |account| self.backend.fetch_catalog(account),
                self.failover(true),
            )
            .await;
        let mut catalog = self.catalog.lock().unwrap();
        match fetched {
            Ok(items) => {
                let items = Arc::new(items);
                catalog.fetched = Some((wall_clock(), items.clone()));
                catalog.retry_at = 0.0;
                items
            }
            // A failed discovery says nothing about which models exist: keep
            // serving the last catalog and ask again shortly, rather than
            // unrouting every model for a whole cache lifetime.
            Err(_) => {
                catalog.retry_at = wall_clock() + CATALOG_RETRY_SECONDS;
                match &catalog.fetched {
                    Some((_, items)) => items.clone(),
                    None => Arc::new(Vec::new()),
                }
            }
        }
    }

    pub async fn status(&self) -> ProviderStatus {
        let mut accounts = Vec::new();
        for account in self.pool.accounts() {
            let mut value = match self.backend.account_status(&account, &self.pool).await {
                Ok(value) => value,
                Err(error) => AccountStatus {
                    error: match error.message() {
                        "" => "unavailable".into(),
                        message => message.to_string(),
                    },
                    ..Default::default()
                },
            };
            let observed = self.pool.account_error(&account.id);
            if !observed.is_empty() && value.signed_in {
                value.signed_in = false;
                value.error = format!("reauthentication required: {observed}");
            }
            value.draining = value.signed_in && self.pool.draining(&account).await;
            value.id = account.id.clone();
            accounts.push(value);
        }
        ProviderStatus {
            signed_in: accounts.iter().any(|account| account.signed_in),
            error: String::new(),
            accounts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::status::Limit;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct FakeAuth(AtomicBool);

    impl Auth for FakeAuth {
        fn login_start(&self) -> BoxFuture<'_, Result<Value>> {
            Box::pin(async { Ok(json!({})) })
        }
        fn logout(&self) -> BoxFuture<'_, Result<()>> {
            Box::pin(async { Ok(()) })
        }
        fn signed_in(&self) -> BoxFuture<'_, Result<bool>> {
            Box::pin(async { Ok(self.0.load(Ordering::SeqCst)) })
        }
        fn status(&self) -> BoxFuture<'_, Result<AccountStatus>> {
            Box::pin(async { Ok(AccountStatus::default()) })
        }
    }

    fn account(id: &str) -> Arc<Account<()>> {
        Arc::new(Account {
            id: id.to_string(),
            auth: Arc::new(FakeAuth(AtomicBool::new(true))),
            client: (),
            limits: None,
            ledger: None,
        })
    }

    fn failover() -> Failover {
        Failover {
            no_account: Error::provider(401, "not signed in"),
            retry_if: |_| false,
            starting: true,
        }
    }

    fn ids(accounts: &[Arc<Account<()>>]) -> Vec<&str> {
        accounts.iter().map(|a| a.id.as_str()).collect()
    }

    #[test]
    fn affinity_is_pinned() {
        // sha256("s\x001")[:8] big-endian. A session's account must not change
        // between versions, or every conversation loses its prompt cache.
        assert_eq!(affinity("s", "1"), 0xed65_7641_1d3b_bfd0);
    }

    #[tokio::test]
    async fn sessionless_requests_round_robin() {
        let pool = AccountPool::new(vec![account("1"), account("2"), account("3")]);
        let mut seen = Vec::new();
        for _ in 0..4 {
            seen.push(pool.candidates(None, true).await[0].id.clone());
        }
        assert_eq!(seen, ["1", "2", "3", "1"]);
    }

    #[tokio::test]
    async fn a_session_keeps_its_account_and_moves_only_when_it_leaves() {
        let pool = AccountPool::new(vec![account("1"), account("2"), account("3")]);
        let home = pool.candidates(Some("chat"), true).await[0].id.clone();
        for _ in 0..3 {
            assert_eq!(pool.candidates(Some("chat"), true).await[0].id, home);
        }
        // Served elsewhere once (a failover): it stays where its cache now is.
        let other = if home == "1" { "2" } else { "1" };
        pool.remember(Some("chat"), other);
        assert_eq!(pool.candidates(Some("chat"), true).await[0].id, other);
        pool.remove(other).unwrap();
        assert_eq!(pool.candidates(Some("chat"), true).await[0].id, home);
    }

    #[tokio::test]
    async fn a_rate_limited_account_is_cooled_and_the_next_one_serves() {
        let pool = AccountPool::new(vec![account("1"), account("2")]);
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let served = pool
            .call(
                None,
                move |account| {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async move {
                        if account.id == "1" {
                            Err(Error::provider(429, "limited"))
                        } else {
                            Ok(account.id.clone())
                        }
                    })
                },
                failover(),
            )
            .await;
        assert_eq!(served.unwrap(), "2");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        // Cooling: account 1 now sorts after the ready one.
        assert_eq!(ids(&pool.candidates(None, true).await), ["2"]);
    }

    #[tokio::test]
    async fn an_account_rests_as_long_as_its_upstream_asked() {
        let pool = AccountPool::new(vec![account("1"), account("2"), account("3")]);
        let rest = |id: &str| {
            let state = pool.state.lock().unwrap();
            state
                .cooldown
                .get(id)
                .map(|until| (until - wall_clock()).round() as i64)
        };
        pool.mark_rate_limited("1", Some(42));
        pool.mark_rate_limited("2", Some(86_400));
        pool.mark_rate_limited("3", Some(0));
        assert_eq!(rest("1"), Some(42));
        assert_eq!(rest("2"), Some(MAX_COOLDOWN_SECONDS as i64));
        assert_eq!(rest("3"), None);
        pool.mark_rate_limited("3", None);
        assert_eq!(rest("3"), Some(RATE_LIMIT_COOLDOWN_SECONDS as i64));
    }

    #[tokio::test]
    async fn unusable_credentials_are_remembered_and_other_errors_are_not_retried() {
        let pool = AccountPool::new(vec![account("1"), account("2")]);
        let result: Result<()> = pool
            .call(
                None,
                |_| Box::pin(async { Err(Error::unavailable(401, "token revoked")) }),
                failover(),
            )
            .await;
        assert_eq!(result.unwrap_err().message(), "token revoked");
        assert_eq!(pool.account_error("1"), "token revoked");
        assert_eq!(pool.account_error("2"), "token revoked");

        let pool = AccountPool::new(vec![account("1"), account("2")]);
        let calls = Arc::new(AtomicUsize::new(0));
        let counted = calls.clone();
        let result: Result<()> = pool
            .call(
                None,
                move |_| {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async { Err(Error::provider(500, "boom")) })
                },
                failover(),
            )
            .await;
        assert!(result.is_err());
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn nothing_fails_over_once_output_has_started() {
        let pool = AccountPool::new(vec![account("1"), account("2")]);
        let opened = Arc::new(AtomicUsize::new(0));
        let counted = opened.clone();
        let events = pool
            .clone()
            .stream(
                None,
                move |_| {
                    counted.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async {
                        let items = vec![Ok(json!({"n": 1})), Err(Error::provider(429, "late"))];
                        Ok(Box::pin(stream::iter(items)) as EventStream)
                    })
                },
                failover(),
            )
            .await
            .unwrap();
        let items: Vec<_> = events.collect().await;
        assert_eq!(items.len(), 2);
        assert!(items[1].is_err());
        assert_eq!(opened.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_new_conversation_avoids_a_nearly_full_account() {
        let bars = |percent: f64| {
            LimitsStore::new(
                "test",
                Box::new(move || {
                    Box::pin(async move {
                        Ok(vec![Limit {
                            label: "5 hour".into(),
                            used_percent: percent,
                            resets_at: Value::Null,
                            model: String::new(),
                        }])
                    })
                }),
            )
        };
        let limited = |id: &str, percent: f64| {
            Arc::new(Account {
                id: id.to_string(),
                auth: Arc::new(FakeAuth(AtomicBool::new(true))) as Arc<dyn Auth>,
                client: (),
                limits: Some(bars(percent)),
                ledger: None,
            })
        };
        let (full, roomy) = (limited("1", 95.0), limited("2", 20.0));
        for account in [&full, &roomy] {
            account.limits.as_ref().unwrap().current(true).await;
        }
        let pool = AccountPool::new(vec![full, roomy]);
        for _ in 0..3 {
            assert_eq!(pool.candidates(None, true).await[0].id, "2");
        }
        // A conversation already under way keeps its account regardless.
        let continuing: Vec<String> = {
            let mut firsts = Vec::new();
            for _ in 0..2 {
                firsts.push(pool.candidates(None, false).await[0].id.clone());
            }
            firsts
        };
        assert!(continuing.contains(&"1".to_string()));
    }

    #[test]
    fn slots_are_numbered_from_the_first_free_id() {
        let dir = std::env::temp_dir().join(format!("llp-slots-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = AccountStore::new(&dir, "claude");
        assert_eq!(store.add().unwrap(), "1");
        assert_eq!(store.add().unwrap(), "2");
        store.remove("1").unwrap();
        assert_eq!(store.add().unwrap(), "1");
        assert_eq!(store.ids().unwrap(), ["2", "1"]);
        assert!(store.remove("9").is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
