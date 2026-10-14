//! Locking that survives a panic elsewhere.

use std::sync::{Mutex, MutexGuard, PoisonError};

/// A mutex that is still usable after a thread panicked holding it.
///
/// Every mutex here guards a cache, a counter table or a file-access turn,
/// updated in short steps with no await in between. A panic leaves at worst a
/// stale entry, which the next read or refresh replaces; refusing every later
/// request over it would be worse.
pub trait LockExt<T> {
    fn locked(&self) -> MutexGuard<'_, T>;
}

impl<T> LockExt<T> for Mutex<T> {
    fn locked(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
