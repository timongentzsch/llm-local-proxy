# LLM Local Proxy

A local gateway that exposes OpenAI- and Anthropic-compatible APIs on top of
your own Codex (ChatGPT) and Claude subscriptions. Clients keep their agent and
tool loop; the proxy only translates the protocol.

![Dashboard with placeholder data](docs/dashboard.png)

## Quick start

```sh
docker compose up --build   # published on 127.0.0.1:8787 only
```

or, with a [Rust toolchain](https://rustup.rs):

```sh
cargo install --path rust
llm-local-proxy
```

It is one static binary with nothing beside it; the image is that binary on an
empty base (about 6 MB).

Open the URL printed at startup; its fragment carries the generated API key.
Sign in to one or more accounts per subscription, then copy a base URL or a
ready-made Codex CLI, Claude Code, OpenCode or OMP launch command from the
dashboard.

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic ANTHROPIC_AUTH_TOKEN=$KEY \
  claude --model claude-sonnet-5
```

## Endpoints

| Mount | Routes |
| --- | --- |
| `/openai/v1` | `chat/completions`, `responses` (stateless), `models`, `models/count` |
| `/anthropic` | `v1/messages`, `v1/messages/count_tokens`, `v1/models` |
| `/v1` | alias of `/openai/v1` |

Every format reaches every subscription; the request's `model` selects the
upstream. `models` accepts `?q=` to filter and `?refresh=1` to bypass the
60-second catalog cache. The proxy key is accepted as `Authorization: Bearer`
or `x-api-key` on every mount. `GET /healthz` is unauthenticated.

## Behaviour

**Translation.** Streaming, images, function tools, parallel calls, web
search, reasoning effort, structured outputs and token usage work in every
pairing. Anything without a faithful mapping is rejected with a 400 rather
than dropped: unsupported sampling parameters, cross-format tool options,
malformed tool-call arguments, a bare `json_object` format on Claude, or
Anthropic deferred tool loading. Tool definitions keep their order.
Anthropic-only content (documents, search results, file images) reaches Claude
verbatim, with its citations returned to Anthropic clients; Codex refuses it.

**Prompt caching.** Caching never changes output, so cache controls are hints
and never a reason to refuse a request. Claude receives Anthropic
`cache_control` breakpoints with their TTL wherever the client placed them;
when a request carries none (every OpenAI-format request), the proxy asks
Claude to cache the prompt automatically. Codex caches prefixes implicitly and
refuses explicit breakpoints, so they are dropped; `prompt_cache_key` reaches
it verbatim. `prompt_cache_retention` and `prompt_cache_options` are accepted
and ignored.

**Codex CLI.** Its Responses traffic works against every model. Namespaced
tools (how Codex CLI groups MCP and app tools), `text.verbosity` and
`reasoning.context` reach Codex unchanged. For Claude, namespaced tools are
flattened to qualified names and mapped back on each call; options Claude has
no equivalent for are refused.

**Models.** Model ids, context and output limits, modalities, thinking support
and effort tiers come from the authenticated upstream catalogs at runtime; only
catalogued models are routable. Codex effort tiers are intersected with what
its transport accepts, using a validation-only probe.

**Reasoning.** Signed reasoning round-trips statelessly. Responses clients
resend opaque reasoning items; Anthropic clients resend thinking blocks; Chat
Completions clients rely on a bounded in-memory cache keyed by tool-call id.
Claude thinking defaults to `display: "summarized"` so its text and signature
survive tool loops. A reasoning effort sent to a model whose catalog lists no
effort tiers (e.g. Haiku 4.5, which takes only a thinking budget) is a
preference it cannot express, like a cache hint, and is not sent. Codex streams reasoning summaries whenever a client asks
for reasoning or to see it, in the summary mode it named. The Responses
endpoint rejects `store: true`, `previous_response_id`, `conversation` and
`background`.

**Web search.** `web_search_options` (Chat Completions), the `web_search` tool
(Responses) and `web_search_20250305` (Messages) map to the serving upstream's
own search tool, which runs inside the subscription. Allowed domains and the
user's approximate location carry across formats; search caps and context
sizes are hints. Options with no equivalent are refused: blocked domains on
Codex, cache-only search on Claude. Results arrive as citations. A finished
search replays to the upstream that ran it and is left out elsewhere, so
conversations continue across formats. Function calls always return to the
client.

**Limits.** Codex ignores `max_tokens`. Claude requires one, so it receives the
requested value or the model's maximum, which also bounds any thinking budget.
`count_tokens` is exact for Claude models and returns 404 for Codex models,
whose upstream cannot count, so clients fall back to their own estimate.

**Accounts.** `X-Session-Id` (or Claude Code's `X-Claude-Code-Session-Id`, or
the request's `prompt_cache_key`) pins a conversation to one account for
prompt-cache locality. Codex requests without any are pinned by a key derived
from their instructions and first user turn; remaining sessionless traffic
round-robins. A session stays on the account that served it for an hour, so
another account's cooldown never moves it. A request that starts a
conversation (no assistant turn yet) skips an account at 90% of a window that
limits it whole while another has room; one that continues a conversation
keeps its account, even after a restart. Before any output is
streamed, a 429 rests the account for as long as its upstream says
(`Retry-After`, Claude's unified reset, the reset of an exhausted Codex
window), at most an hour and five minutes when it does not say; a rejected
credential (expired login or missing inference scope) marks the account for
reauthentication and rests it for one minute. Either way the request moves to
the next account. Nothing is retried after output starts.

**Logins.** Both are the proxy's own, made from the dashboard, and neither
needs a vendor CLI installed. Claude uses the authorization-code flow of
Claude Code; Codex uses the device-code flow of the Codex CLI and keeps each
account in that CLI's own `auth.json` layout. Tokens are refreshed shortly
before they expire. Only a refused grant asks for a new sign-in: a busy or
unreachable token endpoint does not, and the token in hand is used while it is
still valid.

**Usage.** The dashboard's utilization bars come from each subscription and
include its other clients: Claude's from its OAuth usage endpoint, read at most
every 30 seconds, and Codex's from its usage endpoint likewise. Neither read
sends a message, so watching the dashboard never costs tokens or starts a
window; a window with no activity yet, such as Codex's 5-hour one, appears once
it opens. The proxy token counts cover proxy traffic only: they come from
upstream usage and are recorded once per request over rolling 5-hour and
7-day windows. A stream that ends before final accounting keeps its last
reported counts and is marked as partial; missing counts are never estimated.
A stream that ends before its terminal event fails instead of looking
complete.

## Keys and remote access

The `api_key` in the config is the master key: it alone opens the full
dashboard, signs accounts in and out, and manages keys, and it is accepted only
on the admin listener (`host`/`port`). From the dashboard's **keys** panel the
master adds named keys (`alice`, `ci-bot`, ...), stored readable in
`keys.json` next to the config and private to its owner. A named key calls
every model endpoint, and its **copy link** opens that key's own reduced
dashboard: the model catalogue, launch commands with the key filled in, and
its own usage. Proxy token usage is attributed per key and provider; traffic
sent with the master key shows as `master`. Revoking a key takes effect on the
next request.

Other machines reach the proxy through the public listener, which serves only
the model API and the reduced dashboard, and refuses the master key. Trust
comes from the socket, so no header can promote a remote request to master.
Two ways to expose it:

- **Tailscale Serve (recommended).** Keep `public_host = "127.0.0.1"`, run
  `tailscale serve --bg 8788`, and set `public_url` to the served
  `https://<machine>.<tailnet>.ts.net`. Traffic is encrypted and only tailnet
  members can connect.
- **Direct bind.** `public_host = "0.0.0.0"` (under Docker, also publish the
  port in `compose.yaml`). This is plain HTTP, so keys and prompts cross the
  network unencrypted; put your own TLS proxy in front of it.

## Configuration

`~/.config/llm-local-proxy/config.toml`, or `--config PATH`
(`--show-config` prints the path). The file is created on first run and must
be readable only by its owner.

```toml
host = "127.0.0.1"
port = 8787
api_key = "long-random-local-secret"
codex_home = "~/.codex"
request_timeout = 600
```

`request_timeout` is how long an upstream may stay silent, in seconds.
`codex_client_version` (default `0.160.0`) is the Codex CLI version the model
list is requested for; raise it when a newer model does not show up.

An empty `api_key` disables authentication; otherwise it needs at least 24
characters. Native installs bind only to loopback addresses. Under Docker keep
the port published to `127.0.0.1` as supplied.

The optional public listener is off unless `public_port` is set:

```toml
public_host = "127.0.0.1"          # "0.0.0.0" for a direct network bind
public_port = 8788                 # must differ from port; needs api_key
public_url = "https://mac.example.ts.net"   # how remote clients reach it
```

Account slots are added and removed from the dashboard; each provider allows
one unsigned slot at a time, and a slot must be signed out before removal.
Codex logins live in `codex_home/accounts/<slot>/auth.json`; Claude
credentials and both token ledgers live in `accounts/<provider>/<slot>` next
to the config.

## Development

```sh
cd rust
cargo fmt --check && cargo clippy --all-targets -- -D warnings
cargo test                      # unit tests and the conformance replay
cargo build --release
cd ..
python3 tests/e2e/compare.py --rust rust/target/release/llm-local-proxy
python3 tests/e2e/native_codex.py rust/target/release/llm-local-proxy
python3 tests/e2e/bench.py rust/target/release/llm-local-proxy
```

The proxy was first written in Python, and that implementation stays in `src/`
as the reference the Rust one is held to:

- `cargo test` replays `tests/conformance/*.jsonl`: every request body the
  Python suite parses with what both providers render from it, and every
  stream a decoder reads or an encoder shapes, step by step. Each layer is
  checked on its own and then end to end, key order included.
- `tests/e2e/compare.py` starts both against the same canned upstreams and
  diffs what clients receive, what is sent upstream and what is left on disk.
- `tests/e2e/native_codex.py` covers what only the Rust one does: refreshing,
  storing and revoking a ChatGPT login.

To change translation behaviour, change the reference and its tests, record
the cases again, and make the port agree:

```sh
uv sync --locked
PYTHONPATH=src uv run python -m unittest discover -s tests
uv run ruff check src tests && uv run ruff format --check src tests
python3 scripts/record-conformance.py
```

The reference drives `codex app-server` where the Rust implementation talks to
ChatGPT itself, and rests a rate-limited account for a flat five minutes;
`tests/e2e/compare.py` names the few places the two are allowed to differ.

See [docs/architecture.md](docs/architecture.md) for the design and wire
contracts and [docs/specs.md](docs/specs.md) for the specifications they are
tested against.

## Disclaimer

An unofficial, independent project, not affiliated with or endorsed by OpenAI
or Anthropic. It runs on your machine against your own accounts; credentials
are stored locally and sent only to their provider. Both logins use the OAuth
flow and client id of the vendor's own first-party CLI.

The proxy speaks undocumented interfaces: the ChatGPT login, Codex backend and
usage endpoints, and the Claude subscription Messages transport and OAuth
usage endpoint, including the client identifiers that mark first-party traffic. They may change without notice, and
their use may fall outside your subscription's terms; review those terms and
use the paid APIs where a supported integration is required. No warranty.
