//! Replays `tests/conformance/*.jsonl` against the Rust translation core.
//!
//! The cases are recorded from the Python reference by
//! `scripts/record-conformance.py`: every request body its test suite parsed
//! with what both providers rendered from it, and every stream a decoder read
//! or an encoder shaped, step by step. Output must match exactly, key order
//! included -- an upstream may key its prompt cache on the bytes it is sent.
//!
//! Each layer is checked on its own, with the recorded IR as the hand-off, and
//! then end to end:
//!
//! ```text
//! ingress            body -> ChatRequest
//! codex_build        ChatRequest -> Codex request body
//! claude_build       ChatRequest -> Claude request body
//! codex_decoder      Codex events -> stream events
//! claude_decoder     Claude events -> stream events
//! chat_encoder       stream events -> Chat Completions
//! messages_encoder   stream events -> Messages
//! responses_encoder  stream events -> Responses
//! requests, builds, streams      the same cases through every layer
//! ```
//!
//! Narrowing a run:
//!
//! ```text
//! CONF=ingress,codex_build cargo test --test conformance
//! CONF_CASE=ingress:17 cargo test --test conformance -- --nocapture
//! CONF_SHOW=20 ...                                # more failures printed
//! CONF_ONLY=web_search ...                        # cases mentioning a word
//! ```

use llm_local_proxy::dialects::base::Encoder;
use llm_local_proxy::dialects::{
    chat_egress::ChunkEncoder, chat_ingress, messages_egress::MessageEncoder, messages_ingress,
    responses_egress::ResponseEncoder, responses_ingress,
};
use llm_local_proxy::error::{Error, Result};
use llm_local_proxy::ids::{Ids, QueuedIds, SharedIds};
use llm_local_proxy::ir::{ChatRequest, Decoder, StreamEvent};
use llm_local_proxy::ir_json::{event_from, event_json, request_from, request_json};
use llm_local_proxy::json::{integer, Object};
use llm_local_proxy::providers::claude::events::ClaudeDecoder;
use llm_local_proxy::providers::claude::request as claude_request;
use llm_local_proxy::providers::codex::events::CodexDecoder;
use llm_local_proxy::providers::codex::request as codex_request;
use llm_local_proxy::reasoning::ReasoningCache;
use llm_local_proxy::tools::{flatten, Names};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn corpus(name: &str) -> Vec<Value> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tests/conformance")
        .join(name);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .lines()
        .map(|line| serde_json::from_str(line).expect("a case is one JSON line"))
        .collect()
}

fn parse(dialect: &str, body: &Value, session: &str) -> Result<ChatRequest> {
    let body = body.as_object().expect("a recorded body is an object");
    match dialect {
        "chat" => chat_ingress::parse(body, session),
        "responses" => responses_ingress::parse(body, session),
        "messages" => messages_ingress::parse(body, session),
        "messages_count" => messages_ingress::parse_count(body, session),
        other => panic!("unknown dialect {other}"),
    }
}

/// An error as the recorder describes one.
fn failure(error: &Error) -> Value {
    match error {
        Error::Request(message) => json!({"kind": "request", "message": message}),
        Error::Provider {
            status, message, ..
        } => json!({"kind": "provider", "status": status, "message": message}),
        Error::Upstream(message) | Error::Rpc(message) => {
            json!({"kind": "upstream", "message": message})
        }
    }
}

/// Compare an outcome with `{"ok": ...}` or `{"error": ...}` as recorded.
fn check<T>(what: &str, expected: &Value, got: Result<T>, render: impl Fn(T) -> Value) -> Check {
    if let Some(error) = expected.get("error") {
        return match got {
            // A crash in the reference has no wording to match; failing is enough.
            Err(_) if error["kind"] == "crash" => Ok(()),
            Err(got) if &failure(&got) == error => Ok(()),
            Err(got) => Err(mismatch(what, error, &failure(&got))),
            Ok(value) => Err(mismatch(what, error, &json!({"ok": render(value)}))),
        };
    }
    let want = expected
        .get("ok")
        .or(expected.get("output"))
        .unwrap_or(expected);
    match got {
        Ok(value) => same(what, want, &render(value)),
        Err(error) => Err(mismatch(what, want, &json!({"error": failure(&error)}))),
    }
}

type Check = std::result::Result<(), String>;

/// The case belongs to another layer of the same file.
const SKIP: &str = "skip";

fn skip() -> Check {
    Err(SKIP.into())
}

/// Equal as JSON text: the same values in the same key order.
fn same(what: &str, want: &Value, got: &Value) -> Check {
    if want.to_string() == got.to_string() {
        Ok(())
    } else {
        Err(mismatch(what, want, got))
    }
}

fn mismatch(what: &str, want: &Value, got: &Value) -> String {
    let (want, got) = (want.to_string(), got.to_string());
    let order_only = {
        let a: Value = serde_json::from_str(&want).unwrap();
        let b: Value = serde_json::from_str(&got).unwrap();
        a == b
    };
    // Show the neighbourhood of the first difference, not two walls of JSON.
    let limit = std::env::var("CONF_WIDTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(700usize);
    let common = want
        .chars()
        .zip(got.chars())
        .take_while(|(a, b)| a == b)
        .count();
    let from = common.saturating_sub(160);
    let cut = |text: &str| -> String { text.chars().skip(from).take(limit).collect() };
    format!(
        "{what}{}\n    want: …{}\n    got:  …{}",
        if order_only { " (key order only)" } else { "" },
        cut(&want),
        cut(&got)
    )
}

fn uuids(value: &Value) -> Vec<u128> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|v| v.as_u64().map(u128::from))
                .collect()
        })
        .unwrap_or_default()
}

fn efforts(value: &Value) -> Option<Vec<String>> {
    value.as_array().map(|items| {
        items
            .iter()
            .map(|item| item.as_str().unwrap_or_default().to_string())
            .collect()
    })
}

fn cache_from(value: &Value) -> ReasoningCache {
    let cache = ReasoningCache::default();
    for entry in value.as_array().into_iter().flatten() {
        let key = entry[0].as_str().unwrap_or_default().to_string();
        let items = entry[1].as_array().cloned().unwrap_or_default();
        cache.put(&[key], items);
    }
    cache
}

fn codex_build(
    request: &ChatRequest,
    cache: &ReasoningCache,
    efforts: Option<&[String]>,
) -> Result<Value> {
    codex_request::build(request, cache, efforts)
        .map(|(body, session)| json!([Value::Object(body), session]))
}

fn claude_build(
    request: &ChatRequest,
    model: &str,
    options: claude_request::Options<'_>,
    ids: &dyn Ids,
) -> Result<Value> {
    claude_request::build(request, model, options, ids)
        .map(|(body, betas)| json!([Value::Object(body), betas]))
}

fn names_json(names: &Names) -> Value {
    Value::Object(
        names
            .iter()
            .map(|(key, (namespace, name))| (key.clone(), json!([namespace, name])))
            .collect::<Object>(),
    )
}

/// One recorded body through its ingress and both providers' defaults.
fn request_case(case: &Value) -> Check {
    let dialect = case["dialect"].as_str().unwrap();
    let session = case["session"].as_str().unwrap();
    let parsed = parse(dialect, &case["body"], session);
    if case["parse"].get("error").is_some() {
        return check("parse", &case["parse"], parsed, |_| json!("parsed"));
    }
    let request = match parsed {
        Ok(request) => request,
        Err(error) => return Err(mismatch("parse", &json!("ok"), &failure(&error))),
    };
    let model = case["body"]["model"].as_str().unwrap_or_default();

    let cache = ReasoningCache::default();
    check(
        "codex",
        &case["codex"],
        codex_build(&request, &cache, None),
        |v| v,
    )?;

    for (name, max_output) in [("claude", None), ("claude_max_output", Some(4096))] {
        let ids = QueuedIds::default();
        ids.push(uuids(&case[name]["uuids"]));
        let cache = ReasoningCache::default();
        let options = claude_request::Options {
            max_output,
            reasoning_cache: Some(&cache),
            ..Default::default()
        };
        check(
            name,
            &case[name],
            claude_build(&request, model, options, &ids),
            |v| v,
        )?;
    }

    let names = flatten(&request.tools).map(|(_, names)| names_json(&names));
    match case["names"].get("error") {
        Some(_) => check("names", &case["names"], names, |v| v),
        None => check("names", &json!({"ok": case["names"]}), names, |v| v),
    }
}

/// A builder call the reference's tests made with explicit options.
fn build_case(case: &Value) -> Check {
    let request = parse(
        case["dialect"].as_str().unwrap(),
        &case["body"],
        case["session"].as_str().unwrap(),
    )
    .map_err(|error| format!("parse failed: {error}"))?;
    let options = &case["options"];
    let efforts = efforts(&options["reasoning_efforts"]);
    let ids = QueuedIds::default();
    ids.push(uuids(&case["uuids"]));
    let got = match case["provider"].as_str().unwrap() {
        "codex" => codex_build(&request, &cache_from(&case["cache"]), efforts.as_deref()),
        _ => {
            let cache = (!case["cache"].is_null()).then(|| cache_from(&case["cache"]));
            let built = claude_request::Options {
                max_output: integer(&options["max_output"]),
                thinking: options["thinking"].as_str(),
                reasoning_efforts: efforts.as_deref(),
                reasoning_cache: cache.as_ref(),
            };
            claude_build(
                &request,
                options["model"].as_str().unwrap_or_default(),
                built,
                &ids,
            )
        }
    };
    check("build", &case["expect"], got, |v| v)?;
    match ids.remaining() {
        0 => Ok(()),
        left => Err(format!("{left} recorded ids were never drawn")),
    }
}

/// One encoder over one decoder, replayed step by step.
fn stream_case(case: &Value) -> Check {
    if case["decoder"].is_null() {
        return skip();
    }
    let queue = Arc::new(QueuedIds::default());
    let ids: SharedIds = queue.clone();
    queue.push(uuids(&case["uuids"]));
    let cache = Arc::new(ReasoningCache::default());
    let decoder: Box<dyn Decoder> = match case["decoder"]["kind"].as_str().unwrap() {
        "codex" => Box::new(CodexDecoder::new(cache)),
        _ => Box::new(ClaudeDecoder::new(
            Some(cache),
            names_from(&case["decoder"]["names"]),
            ids.clone(),
        )),
    };
    let request = match &case["request"] {
        Value::Null => None,
        origin => Some(
            parse(
                origin["dialect"].as_str().unwrap(),
                &origin["body"],
                origin["session"].as_str().unwrap(),
            )
            .map_err(|error| format!("the echoed request failed to parse: {error}"))?,
        ),
    };
    let mut encoder = new_encoder(case, decoder, request, &ids);
    if queue.remaining() != 0 {
        return Err("construction drew fewer ids than the reference".into());
    }
    run_steps(case, encoder.as_mut(), &queue, None)
}

// -- one layer at a time ------------------------------------------------------

/// body -> ChatRequest, compared as the IR the reference parsed.
fn ingress_case(case: &Value) -> Check {
    let parsed = parse(
        case["dialect"].as_str().unwrap(),
        &case["body"],
        case["session"].as_str().unwrap(),
    );
    if case["parse"].get("error").is_some() {
        return check("parse", &case["parse"], parsed, |_| json!("parsed"));
    }
    check("ir", &json!({"ok": case["ir"]}), parsed, |request| {
        request_json(&request)
    })
}

/// The defaults half of a request case, from the recorded IR.
fn codex_build_case(case: &Value) -> Check {
    if case.get("provider").is_some() {
        if case["provider"] != "codex" {
            return skip();
        }
        let efforts = efforts(&case["options"]["reasoning_efforts"]);
        let request = request_from(&case["ir"]);
        let got = codex_build(&request, &cache_from(&case["cache"]), efforts.as_deref());
        return check("build", &case["expect"], got, |v| v);
    }
    if case.get("ir").is_none() {
        return skip();
    }
    let request = request_from(&case["ir"]);
    let cache = ReasoningCache::default();
    check(
        "codex",
        &case["codex"],
        codex_build(&request, &cache, None),
        |v| v,
    )
}

fn claude_build_case(case: &Value) -> Check {
    if case.get("provider").is_some() {
        if case["provider"] != "claude" {
            return skip();
        }
        let options = &case["options"];
        let efforts = efforts(&options["reasoning_efforts"]);
        let ids = QueuedIds::default();
        ids.push(uuids(&case["uuids"]));
        let cache = (!case["cache"].is_null()).then(|| cache_from(&case["cache"]));
        let built = claude_request::Options {
            max_output: integer(&options["max_output"]),
            thinking: options["thinking"].as_str(),
            reasoning_efforts: efforts.as_deref(),
            reasoning_cache: cache.as_ref(),
        };
        let request = request_from(&case["ir"]);
        let model = options["model"].as_str().unwrap_or_default();
        let got = claude_build(&request, model, built, &ids);
        return check("build", &case["expect"], got, |v| v);
    }
    if case.get("ir").is_none() {
        return skip();
    }
    let request = request_from(&case["ir"]);
    let model = case["body"]["model"].as_str().unwrap_or_default();
    for (name, max_output) in [("claude", None), ("claude_max_output", Some(4096))] {
        let ids = QueuedIds::default();
        ids.push(uuids(&case[name]["uuids"]));
        let cache = ReasoningCache::default();
        let options = claude_request::Options {
            max_output,
            reasoning_cache: Some(&cache),
            ..Default::default()
        };
        check(
            name,
            &case[name],
            claude_build(&request, model, options, &ids),
            |v| v,
        )?;
    }
    Ok(())
}

fn names_from(value: &Value) -> Names {
    value
        .as_object()
        .into_iter()
        .flatten()
        .map(|(key, pair)| {
            let part = |i: usize| pair[i].as_str().unwrap_or_default().to_string();
            (key.clone(), (part(0), part(1)))
        })
        .collect()
}

/// Upstream events -> stream events, for one decoder.
fn decoder_case(kind: &'static str) -> impl Fn(&Value) -> Check {
    move |case: &Value| {
        if case["kind"] != kind {
            return skip();
        }
        let queue = Arc::new(QueuedIds::default());
        let ids: SharedIds = queue.clone();
        queue.push(uuids(&case["uuids"]));
        let cache = Arc::new(ReasoningCache::default());
        let mut decoder: Box<dyn Decoder> = match kind {
            "codex" => Box::new(CodexDecoder::new(cache)),
            _ => Box::new(ClaudeDecoder::new(
                Some(cache),
                names_from(&case["names"]),
                ids,
            )),
        };
        for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
            let op = step["op"].as_str().unwrap();
            let what = format!("step {index} ({op})");
            queue.push(uuids(&step["uuids"]));
            let got = match op {
                "decode" => decoder.decode(&step["input"]),
                _ => decoder.finish(),
            };
            check(&what, step, got, |events| {
                Value::Array(events.iter().map(event_json).collect())
            })?;
            if queue.remaining() != 0 {
                return Err(format!("{what}: drew fewer ids than the reference"));
            }
        }
        Ok(())
    }
}

/// Hands an encoder the events the reference's decoder returned.
struct Replay(Arc<Mutex<Script>>);

#[derive(Default)]
struct Script {
    answers: VecDeque<Vec<StreamEvent>>,
    /// What the decoder raised once its recorded answers ran out.
    error: Option<Error>,
}

impl Replay {
    fn next(&self) -> Result<Vec<StreamEvent>> {
        let mut script = self.0.lock().unwrap();
        match script.answers.pop_front() {
            Some(events) => Ok(events),
            None => Err(script
                .error
                .clone()
                .unwrap_or_else(|| Error::upstream("the decoder was asked more than recorded"))),
        }
    }
}

impl Decoder for Replay {
    fn decode(&mut self, _event: &Value) -> Result<Vec<StreamEvent>> {
        self.next()
    }
    fn finish(&mut self) -> Result<Vec<StreamEvent>> {
        self.next()
    }
}

fn error_from(value: &Value) -> Error {
    let message = value["message"].as_str().unwrap_or_default().to_string();
    match value["kind"].as_str() {
        Some("request") => Error::Request(message),
        Some("provider") => {
            Error::provider(value["status"].as_u64().unwrap_or(502) as u16, message)
        }
        _ => Error::Upstream(message),
    }
}

fn run_steps(
    case: &Value,
    encoder: &mut dyn Encoder,
    queue: &QueuedIds,
    script: Option<&Arc<Mutex<Script>>>,
) -> Check {
    for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
        let op = step["op"].as_str().unwrap();
        let what = format!("step {index} ({op})");
        encoder.set_id(step["id"].as_str().unwrap().to_string());
        if let Some(created) = step["created"].as_i64() {
            encoder.set_created(created);
        }
        let mut drawn = uuids(&step["uuids"]);
        if let Some(script) = script {
            // The decoder is not running, so neither are its draws.
            let theirs = uuids(&step["decoder_uuids"]);
            drawn.retain(|id| !theirs.contains(id));
            let mut script = script.lock().unwrap();
            script.answers = step["events"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|answer| {
                    answer["events"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(event_from)
                        .collect()
                })
                .collect();
            script.error = step.get("error").map(error_from);
        }
        queue.push(drawn);
        let got: Result<Value> = match op {
            "start" => Ok(encoder.start()),
            "feed" => encoder.feed(&step["input"]).map(Value::Array),
            "finish" => encoder.finish().map(Value::Array),
            "result" => encoder.result(),
            "error" => Ok(encoder
                .error(step["input"].as_str().unwrap_or_default())
                .unwrap_or(Value::Null)),
            other => panic!("unknown op {other}"),
        };
        check(&what, step, got, |v| v)?;
        if queue.remaining() != 0 {
            return Err(format!("{what}: drew fewer ids than the reference"));
        }
    }
    Ok(())
}

fn new_encoder(
    case: &Value,
    decoder: Box<dyn Decoder>,
    request: Option<ChatRequest>,
    ids: &SharedIds,
) -> Box<dyn Encoder> {
    let model = case["model"].as_str().unwrap();
    let created = case["steps"][0]["created"].as_i64().unwrap_or(0);
    match case["encoder"].as_str().unwrap() {
        "chat" => Box::new(ChunkEncoder::new(model, decoder, ids, created)),
        "messages" => Box::new(MessageEncoder::new(model, decoder, ids)),
        _ => Box::new(ResponseEncoder::new(model, decoder, request, ids, created)),
    }
}

/// Stream events -> one dialect's frames, without a real decoder.
fn encoder_case(kind: &'static str) -> impl Fn(&Value) -> Check {
    move |case: &Value| {
        if case["encoder"] != kind {
            return skip();
        }
        let queue = Arc::new(QueuedIds::default());
        let ids: SharedIds = queue.clone();
        queue.push(uuids(&case["uuids"]));
        let script = Arc::new(Mutex::new(Script::default()));
        let request = case["request"].get("ir").map(request_from);
        let mut encoder = new_encoder(case, Box::new(Replay(script.clone())), request, &ids);
        if queue.remaining() != 0 {
            return Err("construction drew fewer ids than the reference".into());
        }
        run_steps(case, encoder.as_mut(), &queue, Some(&script))
    }
}

#[test]
fn conformance() {
    let only_file = std::env::var("CONF").ok();
    let only_word = std::env::var("CONF_ONLY").ok();
    let single = std::env::var("CONF_CASE").ok();
    let show: usize = std::env::var("CONF_SHOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    // A stub that panics is a failed case, not a wall of backtraces.
    std::panic::set_hook(Box::new(|_| {}));

    type Layer = (&'static str, &'static str, Box<dyn Fn(&Value) -> Check>);
    let layers: Vec<Layer> = vec![
        ("ingress", "requests", Box::new(ingress_case)),
        ("codex_build", "requests", Box::new(codex_build_case)),
        ("codex_build", "builds", Box::new(codex_build_case)),
        ("claude_build", "requests", Box::new(claude_build_case)),
        ("claude_build", "builds", Box::new(claude_build_case)),
        ("codex_decoder", "decoders", Box::new(decoder_case("codex"))),
        (
            "claude_decoder",
            "decoders",
            Box::new(decoder_case("claude")),
        ),
        ("chat_encoder", "streams", Box::new(encoder_case("chat"))),
        (
            "messages_encoder",
            "streams",
            Box::new(encoder_case("messages")),
        ),
        (
            "responses_encoder",
            "streams",
            Box::new(encoder_case("responses")),
        ),
        ("requests", "requests", Box::new(request_case)),
        ("builds", "builds", Box::new(build_case)),
        ("streams", "streams", Box::new(stream_case)),
    ];
    let mut failed = 0;
    for (name, file, run) in layers {
        if only_file
            .as_deref()
            .is_some_and(|only| !only.split(',').any(|one| one == name))
        {
            continue;
        }
        let cases = corpus(&format!("{file}.jsonl"));
        let (mut passed, mut total, mut shown) = (0, 0, 0);
        for (index, case) in cases.iter().enumerate() {
            let label = format!("{name}:{index}");
            if single.as_deref().is_some_and(|one| one != label) {
                continue;
            }
            if only_word
                .as_deref()
                .is_some_and(|word| !case.to_string().contains(word))
            {
                continue;
            }
            total += 1;
            let outcome = catch_unwind(AssertUnwindSafe(|| run(case))).unwrap_or_else(|panic| {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "panic".into());
                Err(format!("panicked: {message}"))
            });
            match outcome {
                Ok(()) => passed += 1,
                Err(why) if why == SKIP => total -= 1,
                Err(why) => {
                    if shown < show {
                        shown += 1;
                        eprintln!("FAIL {label}: {why}");
                        if single.is_some() {
                            eprintln!("case: {case}");
                        }
                    }
                }
            }
        }
        eprintln!("{name} ({file}): {passed}/{total} pass");
        failed += total - passed;
    }
    assert_eq!(
        failed, 0,
        "{failed} conformance cases differ from the reference"
    );
}
