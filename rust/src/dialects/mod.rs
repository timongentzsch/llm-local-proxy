//! Downstream wire formats: the public APIs the proxy speaks *to clients*.
//!
//! Everything here describes a published specification, so a claim in this
//! module is checkable. Undocumented, reverse-engineered behaviour belongs
//! in a provider instead.

pub mod base;
pub mod chat_egress;
pub mod chat_ingress;
pub mod messages_egress;
pub mod messages_ingress;
pub mod openai_output;
pub mod openai_reasoning;
pub mod responses_egress;
pub mod responses_ingress;
