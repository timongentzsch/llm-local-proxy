//! A local gateway that exposes OpenAI- and Anthropic-compatible APIs on top
//! of Codex and Claude subscriptions.
//!
//! ```text
//! request   dialects/<d> ingress -> ChatRequest -> providers/<p>/request -> account pool -> upstream
//! response  providers/<p>/events -> StreamEvent -> dialects/<d> egress   -> client
//! ```
//!
//! Everything under [`dialects`] and the `request`/`events` halves of
//! [`providers`] is pure: JSON in, JSON out, no clock, no network. That is
//! the part `tests/conformance.rs` replays against the Python reference.

pub mod dialects;
pub mod error;
pub mod ids;
pub mod ir;
pub mod ir_json;
pub mod json;
pub mod providers;
pub mod reasoning;
pub mod tools;
