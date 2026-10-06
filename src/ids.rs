//! Where ids come from, so tests can hand back the ones a case recorded.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub trait Ids: Send + Sync {
    /// 128 random bits, as `uuid.uuid4().int` would be.
    fn uuid(&self) -> u128;
}

pub type SharedIds = Arc<dyn Ids>;

/// The 32 lowercase hex digits of a uuid (`uuid4().hex`).
pub fn hex(ids: &dyn Ids) -> String {
    format!("{:032x}", ids.uuid())
}

/// The first 24 hex digits (`uuid4().hex[:24]`).
pub fn hex24(ids: &dyn Ids) -> String {
    hex(ids)[..24].to_string()
}

pub struct RandomIds;

impl Ids for RandomIds {
    fn uuid(&self) -> u128 {
        let mut bytes = [0u8; 16];
        getrandom::getrandom(&mut bytes).expect("the operating system has randomness");
        // Version 4, variant 1, as uuid4 sets them.
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        u128::from_be_bytes(bytes)
    }
}

pub fn random() -> SharedIds {
    Arc::new(RandomIds)
}

/// Ids handed out in a fixed order; drawing past the end is a test failure.
#[derive(Default)]
pub struct QueuedIds(Mutex<VecDeque<u128>>);

impl QueuedIds {
    pub fn push(&self, values: impl IntoIterator<Item = u128>) {
        self.0.lock().unwrap().extend(values);
    }

    pub fn remaining(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

impl Ids for QueuedIds {
    fn uuid(&self) -> u128 {
        self.0
            .lock()
            .unwrap()
            .pop_front()
            .expect("drew more ids than the case recorded")
    }
}
