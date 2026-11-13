//! Subscription utilization bars, read on demand and shared across callers.
//!
//! Each provider supplies how to read its bars; this spaces the reads so any
//! number of dashboards and routing decisions cost one upstream call, and
//! keeps the last bars, with the stamp of when they were observed, through a
//! failure. A kept window whose reset has since passed reads as unused: the
//! allowance it measured was renewed, whether or not a read has seen it yet.

use crate::error::{Error, Result};
use crate::providers::BoxFuture;
use crate::status::Limit;
use crate::sync::LockExt;
use serde_json::Value;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Time between reads; after a refusal (credentials or rate limit) the next
/// read waits for as long as upstream says, or the backoff when it does not.
const TTL: Duration = Duration::from_secs(30);
const BACKOFF: Duration = Duration::from_secs(300);
const LONGEST_BACKOFF: Duration = Duration::from_secs(3600);

/// One answer of a provider's usage endpoint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reading {
    pub bars: Vec<Limit>,
    /// Account facts that are not a window, e.g. limit resets still unspent.
    pub notes: Vec<String>,
}

impl From<Vec<Limit>> for Reading {
    fn from(bars: Vec<Limit>) -> Self {
        Reading {
            bars,
            notes: Vec::new(),
        }
    }
}

type Read = Box<dyn Fn() -> BoxFuture<'static, Result<Reading>> + Send + Sync>;

#[derive(Default)]
struct State {
    reading: Reading,
    updated_at: Option<f64>,
    next_read: Option<Instant>,
    busy: bool,
    /// Bumped by clear(), so a read begun for the previous login is dropped.
    generation: u64,
    last_error: String,
}

/// One read at a time, outside the lock: callers meanwhile get the last bars.
pub struct LimitsStore {
    name: &'static str,
    read: Read,
    state: Mutex<State>,
}

impl LimitsStore {
    pub fn new(name: &'static str, read: Read) -> Arc<Self> {
        Arc::new(LimitsStore {
            name,
            read,
            state: Mutex::new(State::default()),
        })
    }

    /// The latest bars and when they were observed, refreshed if due.
    ///
    /// Without `wait` a due read runs in the background and the call returns
    /// at once, so request routing never waits on the network.
    pub async fn current(self: &Arc<Self>, wait: bool) -> (Reading, Option<f64>) {
        let generation = {
            let mut state = self.state.locked();
            let due = state.next_read.is_none_or(|at| Instant::now() >= at);
            if state.busy || !due {
                return state.observed();
            }
            state.busy = true;
            state.generation
        };
        // Spawned even when waiting: a caller that leaves mid-read must not
        // take the read with it, or `busy` stays set and no read follows.
        let store = self.clone();
        let read = tokio::spawn(async move { store.refresh(generation).await });
        if wait {
            if let Ok(result) = read.await {
                return result;
            }
        }
        self.state.locked().observed()
    }

    /// The bars as last observed, without starting a read.
    pub fn last(&self) -> Vec<Limit> {
        self.state.locked().observed().0.bars
    }

    /// Forget the bars of a login this slot no longer holds.
    pub fn clear(&self) {
        let mut state = self.state.locked();
        state.reading = Reading::default();
        state.updated_at = None;
        state.next_read = None;
        state.generation += 1;
        state.last_error.clear();
    }

    async fn refresh(&self, generation: u64) -> (Reading, Option<f64>) {
        let outcome = (self.read)().await;
        let mut state = self.state.locked();
        state.busy = false;
        if generation != state.generation {
            return state.observed();
        }
        let mut delay = TTL;
        match outcome {
            Ok(reading) => {
                state.reading = reading;
                state.updated_at = Some(wall_clock());
                state.last_error.clear();
            }
            Err(error) => {
                if matches!(
                    error,
                    Error::Provider {
                        status: 401 | 403 | 429,
                        ..
                    }
                ) {
                    delay = error
                        .cooldown()
                        .map_or(BACKOFF, Duration::from_secs)
                        .clamp(TTL, LONGEST_BACKOFF);
                }
                // Logged once per distinct failure, not on every retry.
                if error.message() != state.last_error {
                    state.last_error = error.message().to_string();
                    eprintln!("{}: limits unavailable: {error}", self.name);
                }
            }
        }
        state.next_read = Some(Instant::now() + delay);
        state.observed()
    }
}

impl State {
    fn observed(&self) -> (Reading, Option<f64>) {
        let mut reading = self.reading.clone();
        if let Some(observed_at) = self.updated_at {
            renew(&mut reading.bars, observed_at, wall_clock());
        }
        (reading, self.updated_at)
    }
}

/// Empty each window whose reset fell between the observation and now.
///
/// A reset already past when it was observed is left alone: that is upstream
/// lagging or a clock apart from ours, not a renewal we can vouch for.
fn renew(bars: &mut [Limit], observed_at: f64, now: f64) {
    for bar in bars {
        if epoch(&bar.resets_at).is_some_and(|reset| observed_at < reset && reset <= now) {
            bar.used_percent = 0.0;
            bar.resets_at = Value::Null;
        }
    }
}

/// A reset time as epoch seconds: a number (seconds or milliseconds) or an
/// RFC 3339 timestamp, the two forms the providers answer with.
fn epoch(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => {
            let stamp = number.as_f64().filter(|stamp| *stamp > 0.0)?;
            Some(if stamp >= 1e12 { stamp / 1000.0 } else { stamp })
        }
        Value::String(text) => rfc3339(text.trim()),
        _ => None,
    }
}

fn rfc3339(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    if !text.is_ascii() || bytes.len() < 20 || bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    if !matches!(bytes[10], b'T' | b't' | b' ') || bytes[13] != b':' || bytes[16] != b':' {
        return None;
    }
    let field = |from: usize, to: usize| text.get(from..to)?.parse::<u32>().ok().map(i64::from);
    let (year, month, day) = (field(0, 4)?, field(5, 7)?, field(8, 10)?);
    let (hour, minute, second) = (field(11, 13)?, field(14, 16)?, field(17, 19)?);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) || hour > 23 || minute > 59 {
        return None;
    }
    let rest = &text[19..];
    let zone = rest.find(['Z', 'z', '+', '-'])?;
    let fraction = match &rest[..zone] {
        "" => 0.0,
        digits => format!("0{digits}").parse::<f64>().ok()?,
    };
    let offset = match &rest[zone..] {
        "Z" | "z" => 0,
        zone if zone.len() == 6 && zone.as_bytes()[3] == b':' => {
            let seconds =
                zone[1..3].parse::<i64>().ok()? * 3600 + zone[4..6].parse::<i64>().ok()? * 60;
            if zone.starts_with('-') {
                -seconds
            } else {
                seconds
            }
        }
        _ => return None,
    };
    // Days since the epoch of a proleptic Gregorian date.
    let (year, shifted) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = year.div_euclid(400);
    let of_era = year.rem_euclid(400);
    let of_year = (153 * shifted + 2) / 5 + day - 1;
    let days = era * 146_097 + of_era * 365 + of_era / 4 - of_era / 100 + of_year - 719_468;
    Some((days * 86_400 + hour * 3600 + minute * 60 + second - offset) as f64 + fraction)
}

pub fn wall_clock() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.0)
}

/// The fullest window that limits the whole account, if any is known.
pub fn fullest(bars: &[Limit]) -> Option<f64> {
    bars.iter()
        .filter(|bar| bar.model.is_empty())
        .map(|bar| bar.used_percent)
        .fold(None, |max, value| {
            Some(max.map_or(value, |max: f64| max.max(value)))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::sync::Notify;

    #[tokio::test]
    async fn a_read_outlives_the_caller_that_left() {
        let gate = Arc::new(Notify::new());
        let opened = gate.clone();
        let store = LimitsStore::new(
            "test",
            Box::new(move || {
                let gate = opened.clone();
                Box::pin(async move {
                    gate.notified().await;
                    Ok(vec![Limit {
                        label: "5 hour".into(),
                        used_percent: 1.0,
                        resets_at: Value::Null,
                        model: String::new(),
                    }]
                    .into())
                })
            }),
        );
        // A dashboard request dropped mid-read must not leave the store reading.
        let left = tokio::time::timeout(Duration::from_millis(10), store.current(true)).await;
        assert!(left.is_err());
        gate.notify_one();
        let landed = tokio::time::timeout(Duration::from_secs(1), async {
            while store.last().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await;
        assert!(landed.is_ok());
    }

    #[test]
    fn reset_times_are_read_in_either_form() {
        assert_eq!(epoch(&json!(1_794_816_757)), Some(1_794_816_757.0));
        assert_eq!(epoch(&json!(1_794_816_757_000_i64)), Some(1_794_816_757.0));
        assert_eq!(epoch(&json!("2026-02-03T04:05:06Z")), Some(1_770_091_506.0));
        assert_eq!(
            epoch(&json!("2026-02-03T06:05:06.5+02:00")),
            Some(1_770_091_506.5)
        );
        assert_eq!(
            epoch(&json!("2000-02-29T00:00:00-00:00")),
            Some(951_782_400.0)
        );
        for unreadable in [json!(null), json!(0), json!("soon"), json!("2026-02-03")] {
            assert_eq!(epoch(&unreadable), None);
        }
    }

    #[test]
    fn a_window_whose_reset_passed_since_it_was_seen_reads_as_unused() {
        let bar = |resets_at: Value| Limit {
            label: "weekly".into(),
            used_percent: 100.0,
            resets_at,
            model: String::new(),
        };
        let mut bars = vec![
            bar(json!(150)),
            bar(json!(250)),
            bar(json!(50)),
            bar(Value::Null),
        ];
        renew(&mut bars, 100.0, 200.0);
        let used: Vec<f64> = bars.iter().map(|bar| bar.used_percent).collect();
        // Renewed; still running; already past when observed; unknown.
        assert_eq!(used, [0.0, 100.0, 100.0, 100.0]);
        assert_eq!(bars[0].resets_at, Value::Null);
        assert_eq!(bars[1].resets_at, 250);
    }

    #[tokio::test]
    async fn a_rate_limited_read_waits_as_long_as_upstream_says() {
        let refused = |cooldown| {
            LimitsStore::new(
                "test",
                Box::new(move || {
                    Box::pin(async move {
                        Err(Error::Provider {
                            status: 429,
                            message: "slow down".into(),
                            account_unavailable: false,
                            cooldown,
                        })
                    })
                }),
            )
        };
        let wait = |store: Arc<LimitsStore>| async move {
            let before = Instant::now();
            store.current(true).await;
            let next = store.state.locked().next_read.unwrap();
            next.duration_since(before).as_secs()
        };
        assert!((119..=121).contains(&wait(refused(Some(120))).await));
        assert!((299..=301).contains(&wait(refused(None)).await));
        assert!((3599..=3601).contains(&wait(refused(Some(86_400))).await));
    }
}
