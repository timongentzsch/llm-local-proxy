//! Downstream wire formats: the public APIs the proxy speaks *to clients*.
//!
//! A [`Dialect`] is one such API, as opposed to a provider, which is one
//! upstream subscription the proxy speaks to. The two axes are independent:
//! any dialect can be served by any provider.
//!
//! Everything here describes a published specification, so a claim in this
//! module is checkable. Undocumented, reverse-engineered behaviour belongs
//! in a provider instead.
//!
//! Every dialect is mounted under its own prefix, so adding one can never
//! change what an existing route means. The default additionally answers on
//! the bare, unprefixed paths: those predate the prefixes and stay valid.

pub mod base;
pub mod chat_egress;
pub mod chat_ingress;
pub mod messages_egress;
pub mod messages_ingress;
pub mod openai_output;
pub mod openai_reasoning;
pub mod responses_egress;
pub mod responses_ingress;

use crate::error::Result;
use crate::ids::SharedIds;
use crate::ir::{ChatRequest, Decoder};
use crate::json::{get, truthy, Object};
use base::Encoder;
use serde_json::{json, Value};

/// A path below a dialect's prefix that accepts a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    ChatCompletions,
    Responses,
    Messages,
    /// Counts input tokens; has no encoder.
    CountTokens,
}

impl Route {
    /// (body, session) -> the dialect-neutral request every provider reads.
    pub fn parse(self, body: &Object, session: &str) -> Result<ChatRequest> {
        match self {
            Route::ChatCompletions => chat_ingress::parse(body, session),
            Route::Responses => responses_ingress::parse(body, session),
            Route::Messages => messages_ingress::parse(body, session),
            Route::CountTokens => messages_ingress::parse_count(body, session),
        }
    }

    /// (model, provider decoder, request) -> encoder. Pairing here keeps
    /// neither side naming the other. None: the route counts input tokens.
    pub fn encoder(
        self,
        model: &str,
        decoder: Box<dyn Decoder>,
        request: &ChatRequest,
        ids: &SharedIds,
        now: i64,
    ) -> Option<Box<dyn Encoder>> {
        Some(match self {
            Route::ChatCompletions => {
                Box::new(chat_egress::ChunkEncoder::new(model, decoder, ids, now))
            }
            Route::Responses => Box::new(responses_egress::ResponseEncoder::new(
                model,
                decoder,
                Some(request.clone()),
                ids,
                now,
            )),
            Route::Messages => Box::new(messages_egress::MessageEncoder::new(model, decoder, ids)),
            Route::CountTokens => return None,
        })
    }

    /// SSE frames are named after their `type` and need no end sentinel;
    /// otherwise they are anonymous and end with `data: [DONE]`.
    pub fn named(self) -> bool {
        !matches!(self, Route::ChatCompletions)
    }
}

pub struct Dialect {
    /// Registry key and mount name (e.g. "openai", "anthropic").
    pub name: &'static str,
    /// Mount point; every dialect has one so routes cannot collide.
    pub prefix: &'static str,
    /// What a client is configured with. Not always prefix + "/v1": clients
    /// differ in how much of the path they append themselves.
    pub base_path: &'static str,
    /// Written while the upstream is silent, so idle connections stay open.
    pub keepalive: &'static [u8],
    /// Headers its clients name a conversation with, after X-Session-Id.
    pub session_headers: &'static [&'static str],
    routes: &'static [(&'static str, Route)],
    /// (status, message) -> the dialect's error body.
    pub error: fn(u16, &str) -> Value,
    /// Merged provider catalogs -> this dialect's model listing.
    pub catalog: fn(&[Value]) -> Value,
}

impl Dialect {
    pub fn route(&self, path: &str) -> Option<Route> {
        self.routes
            .iter()
            .find(|(route, _)| *route == path)
            .map(|(_, route)| *route)
    }
}

/// Chat Completions and Responses. The proxy's default dialect, which owns
/// the bare `/v1` paths.
pub static OPENAI: Dialect = Dialect {
    name: "openai",
    prefix: "/openai",
    base_path: "/openai/v1",
    // A comment frame: ignored by every SSE client, costs no tokens.
    keepalive: b": keepalive\n\n",
    session_headers: &[],
    routes: &[
        ("/v1/chat/completions", Route::ChatCompletions),
        ("/v1/responses", Route::Responses),
    ],
    // The status travels in the HTTP status line; Chat Completions carries
    // only a message and a type in the body.
    error: |_, message| json!({"error": {"message": message, "type": "proxy_error"}}),
    catalog: |models| json!({"object": "list", "data": models}),
};

/// Anthropic Messages. Mounted under a prefix because /v1/models means
/// something different here than in Chat Completions, and the two shapes
/// must not collide. The named SSE events, the ping keepalive and the error
/// frame are defined in the streaming documentation, not the schema.
pub static ANTHROPIC: Dialect = Dialect {
    name: "anthropic",
    prefix: "/anthropic",
    // A client appends /v1/messages to this itself.
    base_path: "/anthropic",
    keepalive: b"event: ping\ndata: {\"type\":\"ping\"}\n\n",
    session_headers: &["x-claude-code-session-id"],
    routes: &[
        ("/v1/messages", Route::Messages),
        ("/v1/messages/count_tokens", Route::CountTokens),
    ],
    error: anthropic_error,
    catalog: anthropic_catalog,
};

pub static DIALECTS: [&Dialect; 2] = [&OPENAI, &ANTHROPIC];

fn anthropic_error(status: u16, message: &str) -> Value {
    // From the ErrorType enum in the spec; anything unmapped is api_error.
    let kind = match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        402 => "billing_error",
        403 => "permission_error",
        404 => "not_found_error",
        408 | 504 => "timeout_error",
        429 => "rate_limit_error",
        529 => "overloaded_error",
        _ => "api_error",
    };
    json!({
        "type": "error",
        // Required by the schema even when the proxy has no id of its own.
        "request_id": null,
        "error": {"type": kind, "message": message},
    })
}

fn anthropic_catalog(models: &[Value]) -> Value {
    let data: Vec<Value> = models
        .iter()
        .map(|model| {
            let id = get(model, "id");
            let name = get(model, "name");
            let mut entry = crate::obj! {
                "type": "model",
                "id": id,
                "display_name": if truthy(name) { name } else { id },
                "created_at": "1970-01-01T00:00:00Z",
            };
            // Anthropic's own field for the window. A client that sizes its
            // context against the catalog gets the same number on both mounts.
            let context = get(model, "context_length");
            if truthy(context) {
                entry.insert("max_input_tokens".into(), context.clone());
            }
            Value::Object(entry)
        })
        .collect();
    let id_of = |entry: Option<&Value>| entry.map(|e| e["id"].clone()).unwrap_or(Value::Null);
    let (first_id, last_id) = (id_of(data.first()), id_of(data.last()));
    json!({
        "data": data,
        "has_more": false,
        "first_id": first_id,
        "last_id": last_id,
    })
}

/// Split a request path into the dialect serving it and the rest.
///
/// A dialect claims both "/openai/v1/models" and, for clients that normalise
/// away the trailing slash, "/openai" itself.
pub fn resolve(path: &str) -> (&'static Dialect, &str) {
    for dialect in DIALECTS {
        if path == dialect.prefix {
            return (dialect, "/");
        }
        if let Some(rest) = path.strip_prefix(dialect.prefix) {
            if rest.starts_with('/') {
                return (dialect, rest);
            }
        }
    }
    (&OPENAI, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_select_the_dialect_and_bare_paths_stay_openai() {
        let name = |path: &str| {
            let (dialect, rest) = resolve(path);
            (dialect.name, rest.to_string())
        };
        assert_eq!(name("/openai/v1/models"), ("openai", "/v1/models".into()));
        assert_eq!(
            name("/anthropic/v1/messages"),
            ("anthropic", "/v1/messages".into())
        );
        assert_eq!(name("/anthropic"), ("anthropic", "/".into()));
        assert_eq!(
            name("/v1/chat/completions"),
            ("openai", "/v1/chat/completions".into())
        );
        assert_eq!(name("/anthropicx/v1"), ("openai", "/anthropicx/v1".into()));
        assert_eq!(
            ANTHROPIC.route("/v1/messages/count_tokens"),
            Some(Route::CountTokens)
        );
        assert_eq!(OPENAI.route("/v1/messages"), None);
    }

    #[test]
    fn each_dialect_has_its_own_error_and_catalog_shape() {
        assert_eq!(
            (OPENAI.error)(429, "slow down"),
            json!({"error": {"message": "slow down", "type": "proxy_error"}})
        );
        let error = (ANTHROPIC.error)(429, "slow down");
        assert_eq!(error["error"]["type"], "rate_limit_error");
        assert!(error["request_id"].is_null());
        assert_eq!((ANTHROPIC.error)(502, "x")["error"]["type"], "api_error");

        let models = [
            json!({"id": "m", "name": "M", "context_length": 1000}),
            json!({"id": "n"}),
        ];
        let listing = (ANTHROPIC.catalog)(&models);
        assert_eq!(listing["data"][0]["max_input_tokens"], 1000);
        assert_eq!(listing["data"][1]["display_name"], "n");
        assert_eq!(listing["first_id"], "m");
        assert_eq!(listing["last_id"], "n");
        assert_eq!((ANTHROPIC.catalog)(&[])["first_id"], Value::Null);
    }
}
