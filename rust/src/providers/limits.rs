//! Subscription utilization bars, read on demand and shared across callers.
//!
//! Each provider supplies how to read its bars; this spaces the reads so any
//! number of dashboards and routing decisions cost one upstream call, and
//! keeps the last bars, with the stamp of when they were observed, through a
//! failure.

use crate::error::{Error, Result};
use crate::providers::BoxFuture;
use crate::status::Limit;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Time between reads; after a refusal (credentials or rate limit) the next
/// read waits for the backoff instead.
const TTL: Duration = Duration::from_secs(30);
const BACKOFF: Duration = Duration::from_secs(300);

pub type Bars = Vec<Limit>;
type Read = Box<dyn Fn() -> BoxFuture<'static, Result<Bars>> + Send + Sync>;

#[derive(Default)]
struct State {
    bars: Bars,
    updated_at: Option<f64>,
    next_read: Option<Instant>,
    reading: bool,
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
    pub async fn current(self: &Arc<Self>, wait: bool) -> (Bars, Option<f64>) {
        let generation = {
            let mut state = self.state.lock().unwrap();
            let due = state.next_read.is_none_or(|at| Instant::now() >= at);
            if state.reading || !due {
                return (state.bars.clone(), state.updated_at);
            }
            state.reading = true;
            state.generation
        };
        if wait {
            return self.refresh(generation).await;
        }
        let store = self.clone();
        tokio::spawn(async move { store.refresh(generation).await });
        let state = self.state.lock().unwrap();
        (state.bars.clone(), state.updated_at)
    }

    /// The bars as last observed, without starting a read.
    pub fn last(&self) -> Bars {
        self.state.lock().unwrap().bars.clone()
    }

    /// Forget the bars of a login this slot no longer holds.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap();
        state.bars.clear();
        state.updated_at = None;
        state.next_read = None;
        state.generation += 1;
        state.last_error.clear();
    }

    async fn refresh(&self, generation: u64) -> (Bars, Option<f64>) {
        let outcome = (self.read)().await;
        let mut state = self.state.lock().unwrap();
        state.reading = false;
        if generation != state.generation {
            return (state.bars.clone(), state.updated_at);
        }
        let mut delay = TTL;
        match outcome {
            Ok(bars) => {
                state.bars = bars;
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
                    delay = BACKOFF;
                }
                // Logged once per distinct failure, not on every retry.
                if error.message() != state.last_error {
                    state.last_error = error.message().to_string();
                    eprintln!("{}: limits unavailable: {error}", self.name);
                }
            }
        }
        state.next_read = Some(Instant::now() + delay);
        (state.bars.clone(), state.updated_at)
    }
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
