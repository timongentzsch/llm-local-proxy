//! The HTTP request handler.
//!
//! Routes are keyed by (dialect, path): the dialect is resolved from the
//! mount prefix first, and every dialect-shaped thing the response needs --
//! the error envelope, the stream framing -- comes from that object rather
//! than from a constant here.

use super::security;
use super::sse::{self, guarded};
use crate::dialects::{resolve, Dialect, Route};
use crate::error::{Error, Result};
use crate::json::Object;
use crate::keys::MASTER;
use crate::service::{base_urls, Service};
use bytes::Bytes;
use futures_util::StreamExt;
use http_body_util::combinators::UnsyncBoxBody;
use http_body_util::{BodyExt, Full, Limited, StreamBody};
use hyper::body::{Frame, Incoming};
use hyper::header::{HeaderMap, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub type Body = UnsyncBoxBody<Bytes, Infallible>;

const MAX_BODY: usize = 32 * 1024 * 1024;
/// A client that stops sending its body does not hold the connection for good.
const BODY_TIMEOUT: Duration = Duration::from_secs(60);
const PAGE: &str = include_str!("../../../src/llm_local_proxy/static/index.html");
const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; style-src 'unsafe-inline'; \
     script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'";

/// One listening socket's view of the service.
///
/// The public listener may face a network, so trust comes from the socket:
/// it serves only the model API and a key's own reduced dashboard, refuses
/// the master key, and never reaches account, key or status routes.
pub struct Listener {
    pub service: Arc<Service>,
    pub public: bool,
}

fn reply(status: StatusCode, data: Bytes, content_type: &'static str) -> Response<Body> {
    let mut response = Response::new(Full::new(data).boxed_unsync());
    *response.status_mut() = status;
    let headers = response.headers_mut();
    headers.insert("content-type", HeaderValue::from_static(content_type));
    headers.insert("cache-control", HeaderValue::from_static("no-store"));
    headers.insert(
        "x-content-type-options",
        HeaderValue::from_static("nosniff"),
    );
    headers.insert("x-frame-options", HeaderValue::from_static("DENY"));
    headers.insert(
        "content-security-policy",
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    response
}

fn json_reply(status: u16, value: &Value) -> Response<Body> {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY);
    let data = serde_json::to_vec(value).expect("a JSON value serialises");
    reply(status, data.into(), "application/json")
}

fn plain_error(status: u16, message: &str) -> Response<Body> {
    json_reply(status, &json!({ "error": message }))
}

fn api_error(dialect: &Dialect, status: u16, message: &str) -> Response<Body> {
    json_reply(status, &(dialect.error)(status, message))
}

fn failure(dialect: &Dialect, error: &Error) -> Response<Body> {
    api_error(dialect, error.status(), error.message())
}

/// `urllib.parse.parse_qs` for the one or two parameters the catalog takes.
fn query_value(query: &str, name: &str) -> String {
    let decode = |text: &str| {
        let bytes = text.as_bytes();
        let mut out = Vec::with_capacity(bytes.len());
        let mut at = 0;
        while at < bytes.len() {
            let hex = |i: usize| bytes.get(i).and_then(|b| (*b as char).to_digit(16));
            match bytes[at] {
                b'+' => out.push(b' '),
                b'%' => match (hex(at + 1), hex(at + 2)) {
                    (Some(high), Some(low)) => {
                        out.push((high * 16 + low) as u8);
                        at += 2;
                    }
                    _ => out.push(b'%'),
                },
                other => out.push(other),
            }
            at += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    };
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(key, value)| decode(key) == name && !value.is_empty())
        .map(|(_, value)| decode(value))
        .unwrap_or_default()
}

impl Listener {
    fn config(&self) -> &crate::config::Config {
        &self.service.config
    }

    fn named(&self, token: &str) -> Option<String> {
        match self.service.keys.identify(token) {
            Ok(name) => name,
            // A damaged keys.json refuses every named key; the master still works.
            Err(error) => {
                eprintln!("keys: {error}");
                None
            }
        }
    }

    /// The name of the request's key; the master key only locally.
    fn caller(&self, headers: &HeaderMap) -> Option<String> {
        let caller =
            security::identify(headers, &self.config().api_key, |token| self.named(token))?;
        (!(self.public && caller == MASTER)).then_some(caller)
    }

    fn valid_host(&self, headers: &HeaderMap) -> bool {
        self.public || security::valid_host(headers, &self.config().host)
    }

    fn refuse_admin(&self) -> Response<Body> {
        // The public listener does not have these routes at all.
        if self.public {
            plain_error(404, "not found")
        } else {
            plain_error(403, "requires the master key")
        }
    }

    fn unauthorized(&self, dialect: &Dialect) -> Response<Body> {
        api_error(dialect, 401, "invalid local API key")
    }

    /// What a key's own dashboard shows: its base URLs and its usage.
    fn me(&self, caller: &str, headers: &HeaderMap) -> Value {
        let origin = if !self.public {
            self.config().origin()
        } else if !self.config().public_url.is_empty() {
            self.config().public_url.clone()
        } else {
            let host = headers
                .get("host")
                .and_then(|h| h.to_str().ok())
                .unwrap_or_default();
            format!("http://{host}")
        };
        let master = caller == MASTER;
        // The master's dashboard reads every key's usage from /api/keys.
        let usage = match self.service.usage().get(caller) {
            Some(usage) if !master => usage.clone(),
            _ => json!({}),
        };
        json!({
            "name": caller,
            "role": if master { "master" } else { "user" },
            "dialects": base_urls(&origin),
            "usage": usage,
        })
    }

    fn keys(&self) -> Result<Value> {
        let keys: Vec<Value> = self
            .service
            .keys
            .all()?
            .into_iter()
            .map(|(name, key)| json!({"name": name, "key": key}))
            .collect();
        let public_url = &self.config().public_url;
        Ok(json!({
            "keys": keys,
            "usage": self.service.usage(),
            "public_url": public_url,
            // Where a named key's launch commands point, when it is set.
            "public_dialects": if public_url.is_empty() { json!([]) } else { base_urls(public_url) },
        }))
    }

    fn manage_keys(&self, body: &Object) -> Result<Value> {
        let Some(Value::String(name)) = body.get("name") else {
            return Err(Error::request("name is required"));
        };
        match body.get("action").and_then(Value::as_str) {
            Some("add") => Ok(json!({"name": name, "key": self.service.keys.add(name)?})),
            Some("remove") => {
                self.service.keys.remove(name)?;
                Ok(json!({ "ok": true }))
            }
            _ => Err(Error::request("action must be add or remove")),
        }
    }

    async fn get(&self, request: &Request<Incoming>) -> Response<Body> {
        let headers = request.headers();
        if !self.valid_host(headers) {
            return plain_error(421, "bad host");
        }
        let (dialect, path) = resolve(request.uri().path());
        match path {
            "/" => {
                let auth = if self.config().api_key.is_empty() {
                    "false"
                } else {
                    "true"
                };
                let page = PAGE.replace("__AUTH_REQUIRED__", auth);
                return reply(StatusCode::OK, page.into(), "text/html; charset=utf-8");
            }
            // Browsers ask for it unprompted; a 401 here is console noise.
            "/favicon.ico" => return reply(StatusCode::NO_CONTENT, Bytes::new(), "image/x-icon"),
            "/healthz" if !self.public => {
                return if self.service.healthy().await {
                    json_reply(200, &json!({"status": "ok"}))
                } else {
                    json_reply(503, &json!({"status": "unhealthy"}))
                };
            }
            _ => {}
        }
        let Some(caller) = self.caller(headers) else {
            return self.unauthorized(dialect);
        };
        if path == "/api/me" {
            return json_reply(200, &self.me(&caller, headers));
        }
        if path.starts_with("/api/") && caller != MASTER {
            return self.refuse_admin();
        }
        match path {
            "/api/status" => json_reply(200, &self.service.status().await),
            "/api/keys" => match self.keys() {
                Ok(keys) => json_reply(200, &keys),
                Err(error) => failure(dialect, &error),
            },
            "/v1/models" => {
                let query = request.uri().query().unwrap_or_default();
                let refresh = matches!(query_value(query, "refresh").as_str(), "1" | "true");
                let mut models = self.service.models(refresh).await;
                let wanted = query_value(query, "q").to_lowercase();
                if !wanted.is_empty() {
                    models.retain(|model| {
                        let text =
                            |key: &str| model[key].as_str().unwrap_or_default().to_lowercase();
                        format!("{} {}", text("id"), text("name")).contains(&wanted)
                    });
                }
                json_reply(200, &(dialect.catalog)(&models))
            }
            "/v1/models/count" => {
                let count = self.service.models(false).await.len();
                json_reply(200, &json!({"data": {"count": count}}))
            }
            _ => plain_error(404, "not found"),
        }
    }

    async fn post(&self, request: Request<Incoming>) -> Response<Body> {
        let headers = request.headers().clone();
        if !self.valid_host(&headers) {
            return plain_error(421, "bad host");
        }
        let path = request.uri().path().to_string();
        let (dialect, path) = resolve(&path);
        let Some(caller) = self.caller(&headers) else {
            return self.unauthorized(dialect);
        };
        if !security::same_origin(&headers) {
            return plain_error(403, "bad origin");
        }
        let body = match read_body(request).await {
            Ok(body) => body,
            Err(error) => return failure(dialect, &error),
        };
        if path.starts_with("/api/") && caller != MASTER {
            return self.refuse_admin();
        }
        let answer = |outcome: Result<Value>| match outcome {
            Ok(value) => json_reply(200, &value),
            Err(error) => failure(dialect, &error),
        };
        if path == "/api/keys" {
            return answer(self.manage_keys(&body));
        }
        // /api/<provider>/<route>
        if let ["", "api", provider, route] = path.split('/').collect::<Vec<_>>()[..] {
            if let Some(provider) = self
                .service
                .provider(provider)
                .filter(|_| !route.is_empty())
            {
                return match provider.route(route, &body).await {
                    Some(outcome) => answer(outcome),
                    None => plain_error(404, "not found"),
                };
            }
        }
        let Some(route) = dialect.route(path) else {
            return plain_error(404, "not found");
        };
        match self.serve(dialect, route, &body, &headers, caller).await {
            Ok(response) => response,
            Err(error) => failure(dialect, &error),
        }
    }

    fn session_id(&self, dialect: &Dialect, headers: &HeaderMap) -> String {
        std::iter::once(&"x-session-id")
            .chain(dialect.session_headers)
            .filter_map(|name| headers.get(*name)?.to_str().ok())
            .find(|value| !value.is_empty())
            .unwrap_or_default()
            .to_string()
    }

    async fn serve(
        &self,
        dialect: &'static Dialect,
        route: Route,
        body: &Object,
        headers: &HeaderMap,
        caller: String,
    ) -> Result<Response<Body>> {
        let mut request = route.parse(body, &self.session_id(dialect, headers))?;
        request.caller = caller;
        let (provider, canonical) = self.service.route(&request.model).await.ok_or_else(|| {
            Error::request(format!("no provider handles model: {}", request.model))
        })?;
        if route == Route::CountTokens {
            // Truthful for a provider whose upstream cannot count: the client
            // falls back to its own estimate knowing it is one.
            let Some(counting) = provider.count_tokens(&canonical, &request) else {
                let message = format!("{} cannot count tokens for {canonical}", provider.name());
                return Ok(api_error(dialect, 404, &message));
            };
            return Ok(json_reply(200, &counting.await?));
        }
        let (mut events, decoder) = provider.chat(&canonical, &request).await?;
        let mut encoder = route
            .encoder(
                &canonical,
                decoder,
                &request,
                &self.service.ids,
                crate::ledger::now(),
            )
            .expect("every route but token counting has an encoder");
        if !request.stream {
            while let Some(event) = events.next().await {
                let event = event?;
                guarded(|| encoder.feed(&event))?;
            }
            return Ok(json_reply(200, &guarded(|| encoder.result())?));
        }
        let frames = sse::body(events, encoder, dialect, route.named())
            .map(|frame| Ok::<_, Infallible>(Frame::data(frame)));
        let mut response = Response::new(BodyExt::boxed_unsync(StreamBody::new(frames)));
        let headers = response.headers_mut();
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("cache-control", HeaderValue::from_static("no-cache"));
        // Tell a buffering reverse proxy in front of the public listener to
        // pass frames through as they are written.
        headers.insert("x-accel-buffering", HeaderValue::from_static("no"));
        Ok(response)
    }
}

async fn read_body(request: Request<Incoming>) -> Result<Object> {
    let headers = request.headers();
    if headers.contains_key("transfer-encoding") {
        return Err(Error::request(
            "chunked request bodies are not supported; send Content-Length",
        ));
    }
    let size: i64 = match headers.get("content-length") {
        None => 0,
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .ok_or_else(|| Error::request("invalid Content-Length"))?,
    };
    if size <= 0 || size > MAX_BODY as i64 {
        return Err(Error::request(
            "request body must be between 1 byte and 32 MiB",
        ));
    }
    let collecting = Limited::new(request.into_body(), MAX_BODY).collect();
    let data = match tokio::time::timeout(BODY_TIMEOUT, collecting).await {
        Ok(Ok(collected)) => collected.to_bytes(),
        Ok(Err(_)) => return Err(Error::request("request body could not be read")),
        Err(_) => return Err(Error::request("request body timed out")),
    };
    match serde_json::from_slice::<Value>(&data) {
        Ok(Value::Object(body)) => Ok(body),
        Ok(_) => Err(Error::request("request body must be an object")),
        Err(_) => Err(Error::request("request body is not valid JSON")),
    }
}

/// Answer one request; never fails, so the connection always gets a reply.
pub async fn handle(
    listener: Arc<Listener>,
    peer: SocketAddr,
    request: Request<Incoming>,
) -> Response<Body> {
    let line = format!("{} {}", request.method(), request.uri());
    let response = match *request.method() {
        Method::GET => listener.get(&request).await,
        Method::POST => listener.post(request).await,
        _ => plain_error(405, "method not allowed"),
    };
    eprintln!("{} \"{line}\" {}", peer.ip(), response.status().as_u16());
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_values_are_decoded() {
        assert_eq!(query_value("q=gpt%205&refresh=1", "q"), "gpt 5");
        assert_eq!(query_value("q=a+b", "q"), "a b");
        assert_eq!(query_value("refresh=1", "refresh"), "1");
        assert_eq!(query_value("q=", "q"), "");
        assert_eq!(query_value("", "q"), "");
        assert_eq!(query_value("q=100%", "q"), "100%");
    }
}
