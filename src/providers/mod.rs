//! Upstream subscriptions. Everything reverse-engineered lives here.
//!
//! A [`Provider`] wires one upstream's model matching, chat handling, model
//! catalog, status and HTTP routes into a single object the server can
//! iterate. Order in the registry is match priority: each provider is offered
//! a model name in turn and the first live catalog to claim it serves the
//! request.

pub mod catalog;
pub mod claude;
pub mod codex;
pub mod limits;
pub mod pool;
pub mod transport;

use crate::error::Result;
use crate::ir::{ChatRequest, Decoder};
use crate::json::Object;
use crate::ledger::Windows;
use crate::status::ProviderStatus;
use futures_util::Stream;
use indexmap::IndexMap;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Upstream events as they arrive. Dropping it closes the upstream.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<Value>> + Send>>;

pub trait Provider: Send + Sync {
    /// Route prefix and card name on the status page (e.g. "codex").
    fn name(&self) -> &'static str;

    /// Maps a requested model id to a canonical name for this provider, or
    /// None when the model does not belong to it (used to route requests).
    fn match_model<'a>(&'a self, model: &'a str) -> BoxFuture<'a, Option<String>>;

    /// (canonical model, parsed request) -> (upstream events, decoder).
    ///
    /// The request is rendered here, so one the upstream cannot express fails
    /// now; the stream itself opens on first poll.
    fn chat<'a>(
        &'a self,
        canonical: &'a str,
        request: &'a ChatRequest,
    ) -> BoxFuture<'a, Result<(EventStream, Box<dyn Decoder>)>>;

    /// Model catalog entries to merge into the /v1/models listing.
    fn models(&self) -> BoxFuture<'_, Vec<Value>>;

    /// The provider's card for /api/status, normalised so every provider
    /// renders through the same dashboard component.
    fn status(&self) -> BoxFuture<'_, ProviderStatus>;

    /// The names of its POST handlers at /api/<name>/<route>.
    fn routes(&self) -> Vec<&'static str>;

    /// Serve one of those; None when it has no such route.
    fn route<'a>(
        &'a self,
        route: &'a str,
        body: &'a Object,
    ) -> BoxFuture<'a, Option<Result<Value>>>;

    /// None when the upstream cannot count exactly; callers then get a 404
    /// rather than an estimate they would wrongly trust.
    fn count_tokens<'a>(
        &'a self,
        _canonical: &'a str,
        _request: &'a ChatRequest,
    ) -> Option<BoxFuture<'a, Result<Value>>> {
        None
    }

    /// Proxy token windows per calling key name, summed over accounts.
    fn callers(&self) -> IndexMap<String, Windows>;

    /// Drops the cached catalog, e.g. after a login changes what is visible.
    fn forget(&self);

    fn healthy(&self) -> BoxFuture<'_, bool> {
        Box::pin(async { true })
    }

    /// Release anything long-lived. Called once at shutdown.
    fn close(&self) -> BoxFuture<'_, ()>;
}
