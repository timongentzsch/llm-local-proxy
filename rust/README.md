# The Rust port

The same proxy as the Python package in `../src`, as one static binary: the
same endpoints, config file, key registry, credential files, token ledgers and
dashboard, so either can serve an existing setup.

```sh
cargo build --release
./target/release/llm-local-proxy            # same config as the Python one
docker compose -f rust/compose.yaml up --build   # from the repository root
```

Run only one implementation against a config directory at a time. Both refresh
the same logins, and a refresh token is spent by its first use.

## Layout

```
src/
  ir.rs  tools.rs  json.rs  error.rs  ids.rs  reasoning.rs
  dialects/      ingress and egress for Chat Completions, Responses, Messages
  providers/
    pool.rs  limits.rs  catalog.rs  transport.rs
    claude/      OAuth, transport, request, events, catalog
    codex/       app-server client, auth, request, events, catalog
  http/          listeners, routing, SSE framing, loopback hardening
  config.rs  keys.rs  ledger.rs  status.rs  service.rs  atomic.rs
```

Everything under `dialects/` and the `request`/`events` halves of the
providers is pure: JSON in, JSON out, no clock, no network. Ids come from an
injected source (`ids.rs`) so a test can hand back the ones a case recorded.

## How it is kept equal to the reference

- `cargo test` replays `../tests/conformance/*.jsonl`: every request body the
  Python suite parses and what both providers render from it, every stream a
  decoder reads or an encoder shapes, step by step. Each layer is checked on
  its own and then end to end, key order included. Record the cases again
  with `python3 scripts/record-conformance.py` whenever the reference changes;
  see the header of `tests/conformance.rs` for narrowing a run.
- `python3 tests/e2e/compare.py --rust target/release/llm-local-proxy` starts
  both servers against the same canned upstreams and diffs what clients
  receive, what is sent upstream and what is left on disk.
- `python3 tests/e2e/bench.py python|<binary>` measures either one.

## Where it differs on purpose

- The token ledger is written a second after a request instead of inside it.
- Whether a Codex slot is signed in is asked of the app-server at most every
  30 seconds (2 while signed out) instead of per account on every request.
- The Codex effort probe is sent once an hour, not at every catalog refresh.
- A dead `codex app-server` is started again on its next use.
- Upstream connections are pooled; a client that hangs up closes its upstream
  request at once.
- Connections per listener are capped, and request headers must arrive within
  30 seconds.
- Streams are sent with chunked transfer encoding on a reusable connection,
  and non-ASCII text as UTF-8 rather than `\u` escapes.
