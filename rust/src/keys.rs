//! Named API keys: who is calling, for attribution and a reduced dashboard.
//!
//! The master key stays in `config.toml`. Named keys live next to it in
//! `keys.json`, private to its owner like the config, and are kept readable
//! so the dashboard can hand a key out again.

use crate::atomic;
use crate::error::{Error, Result};
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Mutex;
use subtle::ConstantTimeEq;

pub const MASTER: &str = "master";

/// `[a-z0-9][a-z0-9._-]{0,31}`
fn valid_name(name: &str) -> bool {
    let plain = |c: char| c.is_ascii_lowercase() || c.is_ascii_digit();
    let mut chars = name.chars();
    chars.next().is_some_and(plain)
        && name.len() <= 32
        && chars.all(|c| plain(c) || matches!(c, '.' | '_' | '-'))
}

pub fn constant_time_eq(a: &str, b: &str) -> bool {
    a.as_bytes().ct_eq(b.as_bytes()).into()
}

pub struct KeyStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl KeyStore {
    pub fn new(path: PathBuf) -> Self {
        KeyStore {
            path,
            lock: Mutex::new(()),
        }
    }

    /// Every name with its key.
    pub fn all(&self) -> Result<IndexMap<String, String>> {
        let _guard = self.lock.lock().unwrap();
        self.read()
    }

    pub fn add(&self, name: &str) -> Result<String> {
        if !valid_name(name) || name == MASTER {
            return Err(Error::request(format!(
                "key name must be 1-32 lowercase letters, digits, '.', '_' or '-' and not '{MASTER}'"
            )));
        }
        let _guard = self.lock.lock().unwrap();
        let mut keys = self.read()?;
        if keys.contains_key(name) {
            return Err(Error::request(format!("key already exists: {name}")));
        }
        let key = format!("llp_{}", atomic::token_urlsafe(32));
        keys.insert(name.to_string(), key.clone());
        self.write(&keys)?;
        Ok(key)
    }

    pub fn remove(&self, name: &str) -> Result<()> {
        let _guard = self.lock.lock().unwrap();
        let mut keys = self.read()?;
        if keys.shift_remove(name).is_none() {
            return Err(Error::request(format!("unknown key: {name}")));
        }
        self.write(&keys)
    }

    /// The name whose key this is; every key is compared in constant time.
    pub fn identify(&self, token: &str) -> Result<Option<String>> {
        let mut found = None;
        for (name, key) in self.all()? {
            if constant_time_eq(token, &key) {
                found = Some(name);
            }
        }
        Ok(found)
    }

    fn read(&self) -> Result<IndexMap<String, String>> {
        let invalid = || Error::upstream(format!("invalid key registry: {}", self.path.display()));
        let value = match atomic::read_json(&self.path) {
            Ok(None) => return Ok(IndexMap::new()),
            Ok(Some(Ok(value))) => value,
            Ok(Some(Err(_))) => return Err(invalid()),
            Err(error) => return Err(Error::upstream(error.to_string())),
        };
        let keys = value.get("keys").and_then(Value::as_object).ok_or_else(invalid)?;
        Ok(keys
            .iter()
            .map(|(name, key)| (name.clone(), crate::json::py_str(key)))
            .collect())
    }

    fn write(&self, keys: &IndexMap<String, String>) -> Result<()> {
        atomic::write_json(&self.path, &json!({ "keys": keys }))
            .map_err(|error| Error::upstream(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_documented_pattern() {
        for good in ["a", "ci-bot", "alice.b_c", "0x", &"a".repeat(32)] {
            assert!(valid_name(good), "{good}");
        }
        for bad in ["", "-a", ".a", "Alice", "a b", &"a".repeat(33), "ä"] {
            assert!(!valid_name(bad), "{bad}");
        }
    }

    #[test]
    fn keys_round_trip_and_identify() {
        let dir = std::env::temp_dir().join(format!("llp-keys-{}", std::process::id()));
        let store = KeyStore::new(dir.join("keys.json"));
        let key = store.add("alice").unwrap();
        assert!(key.starts_with("llp_"));
        assert!(store.add("alice").is_err());
        assert!(store.add("master").is_err());
        assert_eq!(store.identify(&key).unwrap().as_deref(), Some("alice"));
        assert_eq!(store.identify("nope").unwrap(), None);
        store.remove("alice").unwrap();
        assert!(store.remove("alice").is_err());
        let _ = std::fs::remove_dir_all(dir);
    }
}
