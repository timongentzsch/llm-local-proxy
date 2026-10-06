//! Bounded replay cache for clients that cannot carry signed reasoning.
//!
//! Responses and Anthropic clients replay their native opaque blocks
//! directly. Chat Completions cannot, so providers retain those blocks by
//! tool-call id.

use indexmap::IndexMap;
use serde_json::Value;
use std::sync::{Arc, Mutex};

pub struct ReasoningCache {
    items: Mutex<IndexMap<String, Arc<Vec<Value>>>>,
    limit: usize,
}

impl Default for ReasoningCache {
    fn default() -> Self {
        Self::new(128)
    }
}

impl ReasoningCache {
    pub fn new(limit: usize) -> Self {
        Self {
            items: Mutex::new(IndexMap::new()),
            limit,
        }
    }

    /// The items kept for the first of `call_ids` the cache knows.
    pub fn get(&self, call_ids: &[String]) -> Vec<Value> {
        let mut items = self.items.lock().unwrap();
        for call_id in call_ids {
            if let Some(found) = items.shift_remove(call_id) {
                items.insert(call_id.clone(), found.clone());
                return found.as_ref().clone();
            }
        }
        Vec::new()
    }

    pub fn put(&self, call_ids: &[String], values: Vec<Value>) {
        if values.is_empty() {
            return;
        }
        let values = Arc::new(values);
        let mut items = self.items.lock().unwrap();
        for call_id in call_ids {
            items.shift_remove(call_id);
            items.insert(call_id.clone(), values.clone());
        }
        while items.len() > self.limit {
            items.shift_remove_index(0);
        }
    }
}
