//! Rolling per-request token ledger shared by the provider upstreams.
//!
//! The subscription utilisation bars are weighted and opaque; the only hard
//! token numbers come from each request's usage block. This ledger sums those
//! over the same 5h/7d windows the bars use, so the dashboard can show how
//! many tokens the *proxy* consumed in each window. It reflects only proxy
//! traffic; other clients of the same subscription are not visible here.
//!
//! The file keeps the reference implementation's layout (a JSON list of
//! records) so an existing ledger carries over. It is written shortly after
//! a request rather than inside it: the list grows for a week, and rewriting
//! it must not sit between the model's last token and the client.

use crate::atomic;
use crate::ir::Usage;
use crate::keys::MASTER;
use indexmap::IndexMap;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// (label, seconds) windows mirroring the subscription utilisation buckets.
pub const WINDOWS: [(&str, i64); 2] = [("5h", 5 * 3600), ("7d", 7 * 86400)];
const MAX_AGE: i64 = 7 * 86400;
/// How long a finished request may wait before it is on disk.
const FLUSH_DELAY: Duration = Duration::from_secs(1);

pub fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Default, PartialEq)]
struct Record {
    ts: i64,
    input: i64,
    output: i64,
    cache_read: i64,
    cache_write: i64,
    partial: bool,
    caller: String,
}

impl Record {
    fn from_json(value: &Value) -> Option<Record> {
        let map = value.as_object()?;
        let count = |key: &str| map.get(key).and_then(Value::as_i64).unwrap_or(0);
        Some(Record {
            ts: map.get("ts").and_then(Value::as_f64).unwrap_or(0.0) as i64,
            input: count("input"),
            output: count("output"),
            cache_read: count("cache_read"),
            cache_write: count("cache_write"),
            partial: map.get("partial").is_some_and(crate::json::truthy),
            caller: map
                .get("caller")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        })
    }

    fn to_json(&self) -> Value {
        let mut record = crate::obj! {
            "ts": self.ts,
            "input": self.input,
            "output": self.output,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
        };
        if self.partial {
            record.insert("partial".into(), json!(true));
        }
        if !self.caller.is_empty() {
            record.insert("caller".into(), json!(self.caller));
        }
        Value::Object(record)
    }
}

/// Per-request token counts with sliding-window sums.
///
/// Persisted to `path` so totals survive restarts; records older than a week
/// are pruned on write and on read.
pub struct TokenLedger {
    path: Option<PathBuf>,
    /// OpenAI reports cached tokens as a subset of input_tokens, while
    /// Anthropic reports plain input, cache reads and cache writes apart.
    /// Records keep each provider's native shape; window totals are
    /// normalised to plain (uncached) input for the dashboard.
    input_includes_cache: bool,
    records: Mutex<Vec<Record>>,
    flush_pending: AtomicBool,
}

pub type Windows = IndexMap<String, IndexMap<String, i64>>;

impl TokenLedger {
    pub fn new(path: Option<PathBuf>, input_includes_cache: bool) -> Arc<Self> {
        let records = path.as_deref().map(load).unwrap_or_default();
        Arc::new(TokenLedger {
            path,
            input_includes_cache,
            records: Mutex::new(records),
            flush_pending: AtomicBool::new(false),
        })
    }

    /// Persist canonical usage in the provider-native layout.
    pub fn record(self: &Arc<Self>, usage: &Usage, partial: bool, caller: &str) {
        let input = if self.input_includes_cache {
            usage.prompt
        } else {
            (usage.prompt - usage.cache_read - usage.cache_write).max(0)
        };
        let record = Record {
            ts: now(),
            input,
            output: usage.completion,
            cache_read: usage.cache_read,
            cache_write: usage.cache_write,
            partial,
            caller: caller.to_string(),
        };
        let cutoff = now() - MAX_AGE;
        {
            let mut records = self.records.lock().unwrap();
            records.push(record);
            records.retain(|record| record.ts > cutoff);
        }
        self.schedule_flush();
    }

    fn schedule_flush(self: &Arc<Self>) {
        if self.path.is_none() {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            self.flush();
            return;
        };
        if self.flush_pending.swap(true, Ordering::SeqCst) {
            return;
        }
        let ledger = self.clone();
        runtime.spawn(async move {
            tokio::time::sleep(FLUSH_DELAY).await;
            let _ = tokio::task::spawn_blocking(move || ledger.flush()).await;
        });
    }

    /// Write the records out now; also called once at shutdown.
    pub fn flush(&self) {
        let Some(path) = &self.path else { return };
        self.flush_pending.store(false, Ordering::SeqCst);
        let records: Vec<Value> = self
            .records
            .lock()
            .unwrap()
            .iter()
            .map(Record::to_json)
            .collect();
        if let Err(error) = atomic::write_json(path, &Value::Array(records)) {
            eprintln!("ledger: {}: {error}", path.display());
        }
    }

    /// Summed tokens per window (`{"5h": {...}, "7d": {...}}`).
    pub fn windows(&self) -> Windows {
        let records = self.records.lock().unwrap();
        self.sum(records.iter())
    }

    /// The same windows per calling key's name.
    ///
    /// Records from before keys had names were all made with the master key.
    pub fn by_caller(&self) -> IndexMap<String, Windows> {
        let records = self.records.lock().unwrap();
        let mut groups: IndexMap<&str, Vec<&Record>> = IndexMap::new();
        for record in records.iter() {
            let caller = if record.caller.is_empty() {
                MASTER
            } else {
                &record.caller
            };
            groups.entry(caller).or_default().push(record);
        }
        groups
            .into_iter()
            .map(|(caller, items)| (caller.to_string(), self.sum(items.into_iter())))
            .collect()
    }

    fn sum<'a>(&self, records: impl Iterator<Item = &'a Record> + Clone) -> Windows {
        let now = now();
        let mut result = Windows::new();
        for (label, seconds) in WINDOWS {
            let since = now - seconds;
            let (mut input, mut output, mut read, mut write, mut partial) = (0, 0, 0, 0, 0);
            for record in records.clone().filter(|record| record.ts > since) {
                input += if self.input_includes_cache {
                    (record.input - record.cache_read - record.cache_write).max(0)
                } else {
                    record.input
                };
                output += record.output;
                read += record.cache_read;
                write += record.cache_write;
                partial += i64::from(record.partial);
            }
            let mut totals: IndexMap<String, i64> = IndexMap::from([
                ("input".to_string(), input),
                ("output".to_string(), output),
                ("cache_read".to_string(), read),
                ("cache_write".to_string(), write),
            ]);
            if partial > 0 {
                totals.insert("partial_requests".into(), partial);
            }
            result.insert(label.to_string(), totals);
        }
        result
    }
}

fn load(path: &std::path::Path) -> Vec<Record> {
    let Ok(Some(Ok(Value::Array(items)))) = atomic::read_json(path) else {
        return Vec::new();
    };
    let cutoff = now() - MAX_AGE;
    items
        .iter()
        .filter_map(Record::from_json)
        .filter(|record| record.ts > cutoff)
        .collect()
}

/// Sum several `windows()` results field by field.
pub fn merge(windows: impl IntoIterator<Item = Windows>) -> Windows {
    let mut total = Windows::new();
    for item in windows {
        for (label, counts) in item {
            let into = total.entry(label).or_default();
            for (field, value) in counts {
                *into.entry(field).or_insert(0) += value;
            }
        }
    }
    total
}

/// Records one request before its terminal event, or partial usage when the
/// stream is dropped first.
///
/// A terminal response can have incomplete output but authoritative usage.
/// Partial means the stream ended before that final accounting arrived --
/// including a client that hung up, which drops the stream and this with it.
pub struct UsageTracker<R: FnMut(&Value) -> Option<Usage>> {
    ledger: Arc<TokenLedger>,
    read: R,
    terminal_events: &'static [&'static str],
    caller: String,
    pending: Option<Usage>,
    terminal: bool,
}

impl<R: FnMut(&Value) -> Option<Usage>> UsageTracker<R> {
    pub fn new(
        ledger: Arc<TokenLedger>,
        read: R,
        terminal_events: &'static [&'static str],
        caller: &str,
    ) -> Self {
        UsageTracker {
            ledger,
            read,
            terminal_events,
            caller: caller.to_string(),
            pending: None,
            terminal: false,
        }
    }

    /// Call with each upstream event before it is passed on.
    pub fn observe(&mut self, event: &Value) {
        if self.terminal {
            return;
        }
        if let Some(usage) = (self.read)(event) {
            self.pending = Some(usage);
        }
        let kind = event.get("type").and_then(Value::as_str).unwrap_or_default();
        if self.terminal_events.contains(&kind) {
            self.terminal = true;
            if let Some(usage) = &self.pending {
                self.ledger.record(usage, false, &self.caller);
            }
        }
    }
}

impl<R: FnMut(&Value) -> Option<Usage>> Drop for UsageTracker<R> {
    fn drop(&mut self) {
        if !self.terminal {
            if let Some(usage) = &self.pending {
                self.ledger.record(usage, true, &self.caller);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(prompt: i64, completion: i64, cache_read: i64) -> Usage {
        Usage {
            prompt,
            completion,
            cache_read,
            ..Default::default()
        }
    }

    #[test]
    fn windows_report_uncached_input_in_both_layouts() {
        let apart = TokenLedger::new(None, false);
        apart.record(&usage(100, 5, 60), false, "");
        assert_eq!(apart.windows()["5h"]["input"], 40);
        let subset = TokenLedger::new(None, true);
        subset.record(&usage(100, 5, 60), false, "alice");
        assert_eq!(subset.windows()["7d"]["input"], 40);
        assert_eq!(subset.windows()["7d"]["cache_read"], 60);
        assert!(subset.by_caller().contains_key("alice"));
        assert!(apart.by_caller().contains_key(MASTER));
    }

    #[test]
    fn a_stream_dropped_before_its_terminal_event_is_partial() {
        let ledger = TokenLedger::new(None, false);
        let read = |event: &Value| event.get("n").and_then(Value::as_i64).map(|n| usage(n, 0, 0));
        {
            let mut tracker = UsageTracker::new(ledger.clone(), read, &["done"], "");
            tracker.observe(&json!({"type": "x", "n": 7}));
        }
        assert_eq!(ledger.windows()["5h"]["partial_requests"], 1);
        {
            let mut tracker = UsageTracker::new(ledger.clone(), read, &["done"], "");
            tracker.observe(&json!({"type": "x", "n": 3}));
            tracker.observe(&json!({"type": "done"}));
        }
        let windows = ledger.windows();
        assert_eq!(windows["5h"]["input"], 10);
        assert_eq!(windows["5h"]["partial_requests"], 1);
    }

    #[test]
    fn records_persist_in_the_reference_layout() {
        let path = std::env::temp_dir().join(format!("llp-ledger-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let ledger = TokenLedger::new(Some(path.clone()), true);
        ledger.record(&usage(9, 2, 4), true, "ci");
        let text = std::fs::read_to_string(&path).unwrap();
        let saved: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(saved[0]["input"], 9);
        assert_eq!(saved[0]["partial"], true);
        assert_eq!(saved[0]["caller"], "ci");
        let reloaded = TokenLedger::new(Some(path.clone()), true);
        assert_eq!(reloaded.windows()["5h"]["input"], 5);
        let _ = std::fs::remove_file(path);
    }
}
