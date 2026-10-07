//! The proxy's configuration file.

use crate::atomic;
use std::net::IpAddr;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// The backend lists models per client version; raise this (or set
/// `codex_client_version`) when a newer model does not show up.
const CODEX_CLIENT_VERSION: &str = "0.160.0";
/// The subscription refuses newer models to older Claude Code versions;
/// raise this (or set `claude_client_version`) when it asks for an update.
const CLAUDE_CLIENT_VERSION: &str = "2.1.292";

#[derive(Debug, Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub api_key: String,
    /// The Claude Code version requests to the subscription claim.
    pub claude_client_version: String,
    /// Where each Codex slot keeps its `auth.json` (`accounts/<slot>/`).
    pub codex_home: PathBuf,
    /// The Codex CLI version the model list is requested for.
    pub codex_client_version: String,
    /// Seconds an upstream may stay silent before the request fails.
    pub request_timeout: u64,
    pub path: PathBuf,
    /// The optional listener for named keys only (0 = none). It may bind a
    /// network address, unlike `host`, but only when told to explicitly.
    pub public_host: String,
    pub public_port: u16,
    /// The address remote clients use (e.g. a Tailscale Serve URL), for the
    /// launch commands a named key is shown.
    pub public_url: String,
}

impl Config {
    pub fn origin(&self) -> String {
        let host = match self.host.as_str() {
            "0.0.0.0" | "::" => "127.0.0.1",
            host => host,
        };
        if host.contains(':') {
            format!("http://[{host}]:{}", self.port)
        } else {
            format!("http://{host}:{}", self.port)
        }
    }

    /// Where credentials, keys and token ledgers are kept.
    pub fn directory(&self) -> &Path {
        self.path.parent().unwrap_or(Path::new("."))
    }
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
}

fn expand_user(path: &str) -> PathBuf {
    match path.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None if path == "~" => home(),
        None => PathBuf::from(path),
    }
}

pub fn default_path() -> PathBuf {
    let root = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".config"));
    root.join("llm-local-proxy").join("config.toml")
}

fn is_loopback(host: &str) -> bool {
    host == "localhost" || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn container_mode() -> bool {
    std::env::var("LLM_PROXY_CONTAINER").as_deref() == Ok("1") && Path::new("/.dockerenv").exists()
}

fn write_default(path: &Path) -> Result<(), String> {
    let host = if container_mode() {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    };
    let codex_home = std::env::var("CODEX_HOME").unwrap_or_else(|_| "~/.codex".into());
    let text = format!(
        "host = \"{host}\"\nport = 8787\napi_key = \"{}\"\ncodex_home = \"{codex_home}\"\n\
         request_timeout = 600\n",
        atomic::token_urlsafe(32)
    );
    atomic::write_bytes(path, text.as_bytes()).map_err(|error| error.to_string())
}

fn string(data: &toml::Table, key: &str, default: &str) -> String {
    match data.get(key) {
        Some(toml::Value::String(text)) => text.clone(),
        Some(toml::Value::Integer(value)) => value.to_string(),
        Some(toml::Value::Float(value)) => value.to_string(),
        Some(toml::Value::Boolean(value)) => value.to_string(),
        _ => default.to_string(),
    }
}

fn integer(data: &toml::Table, key: &str, default: i64) -> Result<i64, String> {
    match data.get(key) {
        None => Ok(default),
        Some(toml::Value::Integer(value)) => Ok(*value),
        Some(toml::Value::String(text)) => text
            .trim()
            .parse()
            .map_err(|_| format!("{key} must be an integer")),
        Some(_) => Err(format!("{key} must be an integer")),
    }
}

fn port(value: i64, name: &str) -> Result<u16, String> {
    u16::try_from(value)
        .ok()
        .filter(|port| *port >= 1)
        .ok_or_else(|| format!("{name} must be between 1 and 65535"))
}

pub fn load(path: Option<&Path>) -> Result<Config, String> {
    let resolved = match path {
        Some(path) => expand_user(&path.to_string_lossy()),
        None => default_path(),
    };
    if !resolved.exists() {
        write_default(&resolved)?;
    }
    let metadata = std::fs::metadata(&resolved).map_err(|error| error.to_string())?;
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(format!(
            "config is not private; run: chmod 600 {}",
            resolved.display()
        ));
    }
    let text = std::fs::read_to_string(&resolved).map_err(|error| error.to_string())?;
    let data: toml::Table = text
        .parse()
        .map_err(|error: toml::de::Error| error.to_string())?;

    let host = string(&data, "host", "127.0.0.1");
    let api_key = string(&data, "api_key", "");
    if !is_loopback(&host) && !(container_mode() && matches!(host.as_str(), "0.0.0.0" | "::")) {
        return Err("host must be a loopback address".into());
    }
    let port_number = port(integer(&data, "port", 8787)?, "port")?;
    if !api_key.is_empty() && api_key.chars().count() < 24 {
        return Err("api_key must be empty or contain at least 24 characters".into());
    }
    let public_port = match integer(&data, "public_port", 0)? {
        0 => 0,
        value => port(value, "public_port")?,
    };
    if public_port != 0 && api_key.is_empty() {
        // A network port must never sit next to an admin side with auth off.
        return Err("public_port requires an api_key".into());
    }
    if public_port != 0 && public_port == port_number {
        return Err("public_port must differ from port".into());
    }
    Ok(Config {
        host,
        port: port_number,
        api_key,
        claude_client_version: string(&data, "claude_client_version", CLAUDE_CLIENT_VERSION),
        codex_home: expand_user(&string(&data, "codex_home", "~/.codex")),
        codex_client_version: string(&data, "codex_client_version", CODEX_CLIENT_VERSION),
        request_timeout: integer(&data, "request_timeout", 600)?.max(1) as u64,
        path: resolved,
        public_host: string(&data, "public_host", "127.0.0.1"),
        public_port,
        public_url: string(&data, "public_url", "")
            .trim_end_matches('/')
            .to_string(),
    })
}
