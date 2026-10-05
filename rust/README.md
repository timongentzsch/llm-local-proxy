# The Rust port

The same proxy as the Python package in `../src`, as one static binary with
nothing beside it: the same endpoints, config file, key registry, credential
files, token ledgers and dashboard, so either can serve an existing setup. The
image is that binary on an empty base.

It does not need the Codex CLI. Where the Python version drives
`codex app-server`, this one performs the ChatGPT device-code login, token
refresh, usage read and model discovery itself, and keeps each slot's login in
the CLI's own `auth.json` (`<codex_home>/accounts/<slot>/`), so a login made
by either implementation is usable by the other.

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
    codex/       OAuth, transport, request, events, catalog
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
- `python3 tests/e2e/native_codex.py <binary>` checks what only the port
  does: refreshing, storing and revoking a ChatGPT login.
- `python3 tests/e2e/bench.py python|<binary>` measures either one.

## Where it differs on purpose

- The token ledger is written a second after a request instead of inside it.
- No `codex` binary: ChatGPT login, refresh, usage and the model list are
  native. `codex_binary` in an existing config is ignored;
  `codex_client_version` (default 0.160.0) is the CLI version the model list
  is requested for, to raise when a newer model does not show up.
- The Codex effort probe is sent once an hour, not at every catalog refresh.
- A Codex token refresh that fails for a passing reason keeps using the token
  in hand while it is still valid; only a refused login asks for a new sign-in.
- A rate-limited account rests for as long as its upstream says (`Retry-After`,
  the Claude unified reset, the Codex usage window's reset), at most an hour
  and five minutes when it does not say; a Claude 429 asking for extra usage
  rests nothing, since it is about the model and not the account.
- A Claude token refresh refused for a passing reason (429, 5xx, unreachable)
  no longer marks the login as needing a new sign-in; only a rejected grant
  does.
- Logins are held in memory and re-read when their file changes, and a
  refreshed token survives a failed write to disk.
- Request bodies go upstream as UTF-8, as the first-party clients send them,
  not with `\u` escapes.
- Upstream connections are pooled; a client that hangs up closes its upstream
  request at once.
- Connections per listener are capped, and request headers must arrive within
  30 seconds.
- Streams are sent with chunked transfer encoding on a reusable connection,
  and non-ASCII text as UTF-8 rather than `\u` escapes.
