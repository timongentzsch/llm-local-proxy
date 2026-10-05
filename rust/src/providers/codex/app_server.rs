//! A small JSON-RPC client for `codex app-server`; Codex owns OAuth and
//! token refresh, and the proxy only drives it.

use crate::error::{Error, Result};
use crate::json::{get, py_str, truthy, Object};
use base64::Engine;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::oneshot;

const CALL_TIMEOUT: Duration = Duration::from_secs(30);
/// A binary that fails to start is not asked again for this long.
const RESPAWN_BACKOFF: Duration = Duration::from_secs(5);

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Value>>>>;

fn jwt_payload(token: &str) -> Object {
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

/// One running `codex app-server`.
struct Process {
    child: tokio::sync::Mutex<Child>,
    stdin: tokio::sync::Mutex<Option<ChildStdin>>,
    pending: Pending,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
}

impl Process {
    fn spawn(binary: &str, codex_home: &Path) -> Result<Arc<Process>> {
        let mut child = Command::new(binary)
            .args([
                "app-server",
                "-c",
                "cli_auth_credentials_store=\"file\"",
                "--stdio",
            ])
            .env("CODEX_HOME", codex_home)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| Error::rpc(format!("could not start {binary} app-server: {error}")))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("stdout is piped");
        let stderr = child.stderr.take().expect("stderr is piped");
        let process = Arc::new(Process {
            child: tokio::sync::Mutex::new(child),
            stdin: tokio::sync::Mutex::new(stdin),
            pending: Pending::default(),
            next_id: AtomicU64::new(1),
            alive: Arc::new(AtomicBool::new(true)),
        });

        // Its log is noise until it dies; then the last lines say why.
        let recent = Arc::new(Mutex::new(VecDeque::<String>::new()));
        let kept = recent.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut kept = kept.lock().unwrap();
                if kept.len() == 5 {
                    kept.pop_front();
                }
                kept.push_back(line);
            }
        });

        let reader = process.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(message) = serde_json::from_str::<Value>(&line) else {
                    continue;
                };
                reader.receive(message).await;
            }
            reader.alive.store(false, Ordering::SeqCst);
            let failure = json!({"error": {"message": "codex app-server stopped"}});
            for (_, waiting) in reader.pending.lock().unwrap().drain() {
                let _ = waiting.send(failure.clone());
            }
            let last = recent
                .lock()
                .unwrap()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(" | ");
            eprintln!(
                "codex: app-server stopped{}",
                if last.is_empty() {
                    String::new()
                } else {
                    format!(": {last}")
                }
            );
        });
        Ok(process)
    }

    async fn receive(&self, message: Value) {
        let id = get(&message, "id");
        let waiting = id
            .as_u64()
            .and_then(|id| self.pending.lock().unwrap().remove(&id));
        if let Some(waiting) = waiting {
            let _ = waiting.send(message);
        } else if !id.is_null() && truthy(get(&message, "method")) {
            // A request from the server: answer, or it may wait for ever.
            let refusal = json!({
                "id": id,
                "error": {"code": -32601, "message": "unsupported client method"},
            });
            let _ = self.send(&refusal).await;
        }
    }

    async fn send(&self, message: &Value) -> Result<()> {
        let stopped = || Error::rpc("codex app-server is not running");
        if !self.alive.load(Ordering::SeqCst) {
            return Err(stopped());
        }
        let mut stdin = self.stdin.lock().await;
        let pipe = stdin.as_mut().ok_or_else(stopped)?;
        let line = format!("{message}\n");
        pipe.write_all(line.as_bytes())
            .await
            .map_err(|_| stopped())?;
        pipe.flush().await.map_err(|_| stopped())
    }

    async fn call(&self, method: &str, params: Value) -> Result<Object> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (reply, answer) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, reply);
        let outcome = async {
            self.send(&json!({"method": method, "id": id, "params": params}))
                .await?;
            match tokio::time::timeout(CALL_TIMEOUT, answer).await {
                Ok(Ok(message)) => Ok(message),
                Ok(Err(_)) => Err(Error::rpc(format!("{method}: codex app-server stopped"))),
                Err(_) => Err(Error::rpc(format!("{method} timed out"))),
            }
        }
        .await;
        self.pending.lock().unwrap().remove(&id);
        let message = outcome?;
        if let Some(error) = message.get("error") {
            let detail = match error.get("message") {
                Some(detail) => py_str(detail),
                None => py_str(error),
            };
            return Err(Error::rpc(format!("{method}: {detail}")));
        }
        Ok(match message.get("result") {
            Some(Value::Object(result)) => result.clone(),
            Some(other) => crate::obj! { "value": other },
            None => Object::new(),
        })
    }

    /// Ask it to leave by closing its input; insist after three seconds.
    async fn close(&self) {
        self.stdin.lock().await.take();
        let mut child = self.child.lock().await;
        if tokio::time::timeout(Duration::from_secs(3), child.wait())
            .await
            .is_err()
        {
            let _ = child.kill().await;
        }
        self.alive.store(false, Ordering::SeqCst);
    }
}

pub struct AppServer {
    binary: String,
    codex_home: PathBuf,
    auth_path: PathBuf,
    process: tokio::sync::Mutex<Option<Arc<Process>>>,
    failed_at: Mutex<Option<Instant>>,
    closed: AtomicBool,
    model_contexts: tokio::sync::Mutex<Option<Arc<HashMap<String, i64>>>>,
}

impl AppServer {
    /// Start the app-server for one account's own `CODEX_HOME`.
    pub async fn start(binary: &str, codex_home: PathBuf) -> Result<AppServer> {
        crate::atomic::private_dir(&codex_home)
            .map_err(|error| Error::rpc(format!("{}: {error}", codex_home.display())))?;
        let server = AppServer {
            binary: binary.to_string(),
            auth_path: codex_home.join("auth.json"),
            codex_home,
            process: tokio::sync::Mutex::new(None),
            failed_at: Mutex::new(None),
            closed: AtomicBool::new(false),
            model_contexts: tokio::sync::Mutex::new(None),
        };
        server.process().await?;
        Ok(server)
    }

    /// The running process, started again if it has died.
    async fn process(&self) -> Result<Arc<Process>> {
        let mut slot = self.process.lock().await;
        if let Some(process) = slot.as_ref().filter(|p| p.alive.load(Ordering::SeqCst)) {
            return Ok(process.clone());
        }
        let stopped = || Error::rpc("codex app-server is not running");
        if self.closed.load(Ordering::SeqCst) {
            return Err(stopped());
        }
        if self
            .failed_at
            .lock()
            .unwrap()
            .is_some_and(|at| at.elapsed() < RESPAWN_BACKOFF)
        {
            return Err(stopped());
        }
        let started = async {
            let process = Process::spawn(&self.binary, &self.codex_home)?;
            let client = json!({
                "clientInfo": {
                    "name": "llm_local_proxy",
                    "title": "LLM Local Proxy",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            });
            process.call("initialize", client).await?;
            process
                .send(&json!({"method": "initialized", "params": {}}))
                .await?;
            Ok(process)
        }
        .await;
        match started {
            Ok(process) => {
                *self.failed_at.lock().unwrap() = None;
                *slot = Some(process.clone());
                Ok(process)
            }
            Err(error) => {
                *self.failed_at.lock().unwrap() = Some(Instant::now());
                *slot = None;
                Err(error)
            }
        }
    }

    pub async fn call(&self, method: &str, params: Value) -> Result<Object> {
        self.process().await?.call(method, params).await
    }

    async fn auth(&self) -> Object {
        // Codex truncates and rewrites the file in place, so a read can land
        // in between.
        for _ in 0..3 {
            match crate::atomic::read_json(&self.auth_path) {
                Ok(Some(Ok(Value::Object(data)))) => return data,
                Ok(Some(Ok(_))) => return Object::new(),
                _ => tokio::time::sleep(Duration::from_millis(20)).await,
            }
        }
        Object::new()
    }

    /// The access token and ChatGPT account id for an upstream request.
    pub async fn token(&self, force_refresh: bool) -> Result<(String, String)> {
        let read = |auth: &Object| -> (String, Object) {
            let tokens = auth
                .get("tokens")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let access = tokens.get("access_token").map(py_str).unwrap_or_default();
            (access, tokens)
        };
        let (mut access, mut tokens) = read(&self.auth().await);
        let expiry = get(&Value::Object(jwt_payload(&access)), "exp")
            .as_f64()
            .unwrap_or(0.0) as i64;
        if force_refresh || access.is_empty() || expiry <= crate::ledger::now() + 120 {
            self.call("account/read", json!({"refreshToken": true}))
                .await?;
            (access, tokens) = read(&self.auth().await);
        }
        if access.is_empty() {
            return Err(Error::rpc("not signed in; open the proxy status page"));
        }
        let mut account_id = tokens.get("account_id").map(py_str).unwrap_or_default();
        if account_id.is_empty() {
            let claims = jwt_payload(&access);
            if let Some(claim) = claims
                .get("https://api.openai.com/auth")
                .filter(|c| c.is_object())
            {
                let id = get(claim, "chatgpt_account_id");
                if truthy(id) {
                    account_id = py_str(id);
                }
            }
        }
        if account_id.is_empty() {
            return Err(Error::rpc("ChatGPT account id is missing"));
        }
        Ok((access, account_id))
    }

    /// Whether the process is running, starting it again if it has died.
    pub async fn alive(&self) -> bool {
        self.process().await.is_ok()
    }

    /// Effective context windows from the installed Codex catalog.
    pub async fn model_contexts(&self) -> Arc<HashMap<String, i64>> {
        let mut cached = self.model_contexts.lock().await;
        if let Some(contexts) = cached.as_ref() {
            return contexts.clone();
        }
        let contexts = Arc::new(self.read_model_contexts().await.unwrap_or_default());
        *cached = Some(contexts.clone());
        contexts
    }

    async fn read_model_contexts(&self) -> Option<HashMap<String, i64>> {
        let run = Command::new(&self.binary)
            .args(["debug", "models"])
            .env("CODEX_HOME", &self.codex_home)
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output();
        let output = tokio::time::timeout(CALL_TIMEOUT, run).await.ok()?.ok()?;
        if !output.status.success() {
            return None;
        }
        let value: Value = serde_json::from_slice(&output.stdout).ok()?;
        let models = get(&value, "models").as_array()?;
        Some(
            models
                .iter()
                .filter_map(|item| {
                    let slug = get(item, "slug");
                    let window =
                        crate::json::integer(get(item, "context_window")).filter(|n| *n > 0)?;
                    truthy(slug).then(|| (py_str(slug), window))
                })
                .collect(),
        )
    }

    pub async fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        if let Some(process) = self.process.lock().await.take() {
            process.close().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jwt(claims: Value) -> String {
        let body = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(claims.to_string());
        format!("header.{body}.signature")
    }

    #[test]
    fn jwt_claims_are_read_with_or_without_padding() {
        let token =
            jwt(json!({"exp": 123, "https://api.openai.com/auth": {"chatgpt_account_id": "acct"}}));
        let claims = jwt_payload(&token);
        assert_eq!(claims["exp"], 123);
        assert_eq!(
            claims["https://api.openai.com/auth"]["chatgpt_account_id"],
            "acct"
        );
        assert!(jwt_payload("not-a-jwt").is_empty());
        assert!(jwt_payload("a.!!!.c").is_empty());
    }

    /// A stand-in app-server: answers every request with its method name.
    fn fake(dir: &Path, script: &str) -> String {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join("codex");
        std::fs::write(&path, format!("#!/bin/sh\n{script}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().into_owned()
    }

    const ECHO: &str = r#"while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  method=$(printf '%s' "$line" | sed -n 's/.*"method":"\([^"]*\)".*/\1/p')
  [ -n "$id" ] && printf '{"id":%s,"result":{"method":"%s"}}\n' "$id" "$method"
  [ "$method" = "die" ] && exit 0
done"#;

    #[tokio::test]
    async fn calls_are_answered_and_a_dead_server_is_started_again() {
        let dir = std::env::temp_dir().join(format!("llp-app-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let server = AppServer::start(&fake(&dir, ECHO), dir.join("home"))
            .await
            .unwrap();
        let reply = server.call("account/read", json!({})).await.unwrap();
        assert_eq!(reply["method"], "account/read");

        let _ = server.call("die", json!({})).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        // The next call finds it gone and starts another.
        let reply = server.call("model/list", json!({})).await.unwrap();
        assert_eq!(reply["method"], "model/list");
        server.close().await;
        assert!(server.call("x", json!({})).await.is_err());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn a_missing_binary_is_an_rpc_error() {
        let dir = std::env::temp_dir().join(format!("llp-app-missing-{}", std::process::id()));
        let error = AppServer::start("/nonexistent/codex", dir.clone())
            .await
            .err()
            .unwrap();
        assert!(matches!(error, Error::Rpc(_)));
        let _ = std::fs::remove_dir_all(dir);
    }
}
