//! The HTTP surface: listeners, request routing, SSE framing and the
//! loopback hardening in front of them.

pub mod handler;
pub mod security;
pub mod server;
pub mod sse;
