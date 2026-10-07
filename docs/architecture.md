# Architecture

The proxy serves three downstream formats (OpenAI Chat Completions, OpenAI
Responses, Anthropic Messages) over two upstream subscriptions (Codex, Claude).
Any format can reach any subscription; the request's `model` decides.

```
request   dialects/<d>_ingress ─► ChatRequest ─► providers/<p>/request ─► account pool ─► upstream
response  providers/<p>/events ─► StreamEvent ─► dialects/<d>_egress   ─► client
```

Paths below are in `src/`. What a request goes through is described for users
in [behaviour.md](behaviour.md).

Parsing each format once into an intermediate representation (IR) and
rendering each upstream from it keeps the cost of a new format or provider at
N + M instead of N × M.

## Intermediate representation

`ir.rs` defines both directions.

- **`ChatRequest`** carries shared semantics in typed fields: system blocks,
  turns of content blocks, function tools with strictness and parallel-call
  control, tool choice, token limits, reasoning effort and summary, thinking
  mode and display, cache hints and key, session, sampling parameters and
  output format. Wire-specific tool
  options keep their source format; `tools.rs` preserves them on compatible
  targets and rejects them elsewhere.
- **Opaque escape hatches** (`Block::Reasoning`, `Block::NativeResponseItem`,
  `Block::NativeAnthropicBlock`, `Tool::Native`) hold content without a lossless mapping,
  such as signed reasoning, rich tool results, documents and search results.
  They are forwarded verbatim on compatible routes and rejected on the rest,
  never interpreted.
- **`ToolNamespace`** keeps a Responses namespace both verbatim and as parsed
  function tools. Targets without namespaces use `tools::flatten`, which gives
  each member a qualified name of at most 64 characters and the map to restore
  calls; tool calls carry their `namespace` back to the client.
- **`StreamEvent`** is the response vocabulary: `TextDelta`, `ThinkingDelta`,
  `ThinkingSignature`, `RedactedThinkingDelta`, `ReasoningItem`, `NativeItem`,
  `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd`, `HostedToolEvent`, `Citation`,
  `Usage` and `Finish`. Stop reasons use Anthropic's seven-value enum; Chat
  Completions narrows them to four.

Each provider supplies a `Decoder` (upstream events to `StreamEvent`s); each
dialect supplies an `Encoder` that shapes those events into its own frames.
The HTTP layer pairs the two per request and knows neither provider.

Everything in this part is pure: JSON in, JSON out, no clock, no network. Ids
come from an injected source (`ids.rs`), so a test can hand back the ones a
recorded case drew.

## Layout

```
src/
  ir.rs  tools.rs  json.rs  error.rs  ids.rs  reasoning.rs
  service.rs         provider registry, merged catalog, status
  config.rs  atomic.rs  ledger.rs  status.rs  keys.rs
  http/              listeners, request routing, SSE framing, loopback security,
                     dashboard.html
  dialects/
    mod.rs           Dialect, Route, the registry
    base.rs          the Encoder trait
    chat_*  responses_*  messages_*    ingress and egress per format
  providers/
    mod.rs           the Provider trait
    pool.rs          AccountPool, AccountStore, Pooled, the Auth trait
    limits.rs        LimitsStore: usage bars read without blocking
    catalog.rs  transport.rs
    claude/          OAuth, transport, request, events, catalog
    codex/           OAuth, transport, request, events, catalog
tests/
  conformance.rs     replays tests/conformance/*.jsonl against the pure core
  e2e/               the binary against stand-ins for both upstreams
```

A dialect is one `Dialect` value with a mount prefix and a route table; a
provider is one implementation of `Provider`. Registering either is one line
in `dialects/mod.rs` or `service.rs`.

`json.rs` exists because the wire formats were first pinned by a Python
implementation: where its truthiness, `str()` or `json.dumps` reach the output
(an id, an envelope a client carries between turns), the helpers there
reproduce them byte for byte, so conversations begun on older versions still
replay.

## Adding a provider

1. Implement `Backend` (`providers/pool.rs`): `new_account`, `fetch_catalog`,
   `account_status`, `no_account` and `state_dirs`. Wrapping it in `Pooled`
   brings slots, logins, failover, catalog caching and status.
2. Render the upstream request from `ChatRequest`, reusing `tools.rs`
   (`render_function`, `responses_tool`, `flatten`, `arguments`) and rejecting
   anything the upstream cannot represent. Cache hints are the exception:
   honour the ones the upstream can express and ignore the rest.
3. Write a `Decoder` from upstream events to `StreamEvent`s; read the stream
   with `transport::read_events` and pass it through `ledger::track`.
4. Implement `Provider` over the `Pooled` value and add it to `Service::new`.

Every dialect then reaches the new provider without further changes.

## Endpoints

Each dialect has its own mount because the formats disagree about what
`/v1/models` returns.

| Mount | Routes |
| --- | --- |
| `/openai/v1` | `chat/completions`, `responses`, `models`, `models/count` |
| `/anthropic` | `v1/messages`, `v1/messages/count_tokens`, `v1/models` |
| `/v1` | alias of `/openai/v1` |

Streams follow each API: Messages and Responses name every frame after its
`type` and end without a sentinel, while Chat Completions sends anonymous
frames ending with `data: [DONE]`.

The proxy's own key is accepted in `Authorization: Bearer` or `x-api-key` on
every mount. `count_tokens` answers exactly where the upstream can count and
returns 404 otherwise, so clients never trust an invented number.

## Accounts

Each provider is a `Pooled` backend: an `AccountStore` of slot ids
(`slots.json`), an `AccountPool` of live accounts, and a shared catalog cache.
Routing sees one provider per subscription, so model ids carry no account
suffix.

- **Selection.** A session stays on the account that last served it for an
  hour (the longest prompt-cache lifetime); otherwise it starts on its
  rendezvous-hash account, so a cooldown moves only the sessions of the
  account that left. Without a session the starting account advances
  round-robin. A request that starts a conversation (no assistant turn)
  prefers accounts below `SOFT_LIMIT_PERCENT` of every whole-account window,
  read from `providers/limits.rs` without waiting; one that continues a
  conversation keeps its account, as its history already has a cache there.
  The session is
  `X-Session-Id`, else a header the dialect names (`Dialect::session_headers`,
  e.g. Claude Code's), else the request's `prompt_cache_key`.
- **Failover.** Before the first upstream event, a 429 rests the account for
  as long as the upstream names (`Retry-After`, Claude's unified reset, the
  reset of an exhausted Codex window), bounded to an hour, and for five
  minutes when it names nothing; a Claude 429 that asks for extra usage is
  about the model and rests nothing. A terminal authentication failure
  (rejected credentials or a missing inference scope) marks the account for
  reauthentication and rests it for one minute. The untouched request then
  moves to the next account. Other errors keep their status, and nothing
  switches accounts once output has started.
- **Logins.** `providers/claude/auth.rs` and `providers/codex/auth.rs` each run
  their vendor CLI's OAuth flow and refresh their own tokens, one refresh at a
  time per account. A login is held parsed in memory (`atomic::JsonFile`) and
  re-read when its file changes; a refreshed pair is kept even if the disk
  refuses it, since the old refresh token is spent. Only a rejected grant
  ends a login: a busy or unreachable token endpoint does not.
- **Catalog.** Discovery uses the same pool without affinity and accepts the
  first live catalog, so one stale login cannot hide a provider's models. It
  is cached for a minute; a failed discovery keeps the last catalog and is
  retried after five seconds, and concurrent refreshes share one call.
- **Keys.** `security::identify` names the caller: `master` for the configured
  key, or a named key from `KeyStore`. The name rides on `ChatRequest.caller`
  into each provider's token ledger, which keeps windows per caller
  (`TokenLedger::by_caller`, merged across accounts by `Pooled::callers`). The
  admin listener serves everything, gating `/api/*` except `/api/me` to the
  master key; the optional public listener (`Listener { public: true }`)
  serves only the model API, `/` and `/api/me`, and refuses the master key, so
  trust comes from the socket rather than from headers or peer addresses.
- **Ledger.** Token counts are recorded once per request, before its terminal
  event, and written to disk a second later; a stream dropped first, including
  by a client hanging up, is recorded as partial, with no counts when the
  upstream had reported none yet.
- **Slots.** Added and removed live from the dashboard. Only one unsigned slot
  may exist, and a slot must be signed out before removal. Codex state lives
  in `codex_home/accounts/<slot>`, proxy state in `accounts/<provider>/<slot>`.

## Evidence

Wire claims are labelled by how they can be checked:

- **[spec]**: the published OpenAPI documents (Anthropic's, and
  [openai/openai-openapi](https://github.com/openai/openai-openapi)).
- **[docs]**: published prose that no schema covers, chiefly SSE framing
  (`ping` and `error` events are defined only in the streaming docs).
- **[empirical]**: observed against a subscription edge or read from a
  vendor's open-source client, with no specification.

[spec] and [empirical] code never share a module. Everything reverse-engineered
lives under `providers/`: the Claude subscription marker, beta headers and
OAuth flow, the ChatGPT login, usage and model endpoints, and the Codex
transport and its effort probe. Nothing in `dialects/` is empirical.

## Tests

- **Conformance.** `tests/conformance.rs` replays `tests/conformance/*.jsonl`:
  request bodies with the IR they parse to and what both providers render,
  and every decoder and encoder step with the ids it drew. Each layer is
  checked alone, with the recorded IR as the hand-off, and then end to end.
  Output must match as JSON text, key order included. The cases were recorded
  from the Python implementation this one replaced; a deliberate change in
  behaviour means editing the cases it affects.
- **End to end.** `tests/e2e/run.py` starts the binary against
  `fake_upstream.py`, drives a scripted session (every format on both
  subscriptions, refusals, keys, account management) and compares client
  responses, upstream requests and files on disk with `expected.json`.
  `codex_login.py` covers refreshing, storing and revoking a ChatGPT login.
  `bench.py` measures memory, CPU and added latency.
- **Unit tests** sit next to the code they cover.

`LLM_PROXY_TEST_UPSTREAM` points every upstream URL at a local stand-in; it
exists only for these tests.

## Anthropic Messages contract

- **Request.** `model`, `messages` and `max_tokens` are required;
  `max_tokens: 0` is legal and pre-warms the prompt cache. A trailing
  assistant message is a prefill, and a `system` role inside `messages` is
  distinct from the top-level system prompt.
- **Response.** `Message` must include `stop_reason`, `stop_sequence`,
  `stop_details`, `container` and `usage`, with nullable fields present as
  `null`.
- **Usage.** Total input is `input_tokens + cache_creation_input_tokens +
  cache_read_input_tokens`. The IR carries the total, and the encoder splits
  it back out.
- **Streaming.** Named frames and no `[DONE]` sentinel. One content block is
  open at a time under rising indices. `message_start` reports zero input
  tokens because Codex reports usage only at the end; `message_delta` carries
  the final totals.

| Anthropic stop reason | Chat Completions |
| --- | --- |
| `end_turn`, `stop_sequence`, `pause_turn` | `stop` |
| `tool_use` | `tool_calls` |
| `max_tokens`, `model_context_window_exceeded` | `length` |
| `refusal` | `content_filter` |

## Sharp edges

1. **Subscription marker [empirical].** Claude requests must start with the
   Claude Code system block or they are billed against the API pool and
   rate-limited. A client that already sends it is not given a second one.
2. **Signed reasoning.** Claude verifies thinking blocks byte for byte.
   Anthropic clients resend them natively; Responses clients carry them in a
   versioned `encrypted_content` envelope; Chat Completions clients rely on the
   cache. Codex reasoning reaches Anthropic clients inside the thinking
   signature. A block whose text the upstream never streamed (omitted display)
   cannot be replayed, so its turn continues without thinking. Codex receives
   the requested summary mode verbatim, and `auto` whenever the client asked
   for an effort, summarized display or adaptive thinking; summary `none` and
   display `omitted` ask for none. Claude derives its display from the same
   request: `omitted` for either, `summarized` otherwise.
3. **Hosted search.** Web search runs upstream, so it is a `HostedToolEvent`,
   never a tool call the client would have to execute. Responses clients get a
   `web_search_call` item held open for the duration of the search; Anthropic
   clients get `server_tool_use` with its `web_search_tool_result`. Only
   forward lifecycle steps are emitted. When the client echoes these records,
   ingress parses them as `HostedSearch`: they replay verbatim to an upstream
   of the same format (Anthropic requires them to continue a `pause_turn`) and
   are omitted elsewhere, since the search has run and its answer is in the
   transcript. Cited text keeps its citations for Claude and its text for
   Codex.
4. **Prompt caching.** Cache controls are hints in the IR (`cache` on blocks,
   tools and the request: None, or a TTL). Claude renders them as
   `cache_control`, and adds its automatic top-level breakpoint only when the
   client placed none, since the upstream accepts at most four. Codex drops
   them: its backend refuses `prompt_cache_breakpoint`,
   `prompt_cache_options` and `prompt_cache_retention` on every model
   [empirical] and caches by `prompt_cache_key`, which is the client's own key
   when it sent one (`ChatRequest.cache_key`), else the session, else derived.
   The key is also sent as the `session-id` header: the backend reuses a
   cached prefix only for requests that carry it [empirical].
5. **Betas [empirical].** The Claude transport sends its subscription betas
   plus feature betas for web search and structured outputs as requested.

## Deliberately not shared

- The two upstream retry envelopes differ in URL, headers, token source and
  error mapping; a shared helper would need several callbacks to save a few
  lines.
- Usage parsing is specific to each upstream (Codex reports terminal totals,
  Claude cumulative snapshots); persistence and stream cleanup share
  `ledger::UsageTracker`.
- Decoders share a target type, not an implementation. Encoders share only
  the decoder-driving methods of the `Encoder` trait.
- The registries are a static array and a `Vec`; there is no plugin loader or
  code generation.

## Known gaps

- Chat Completions has no hosted-tool lifecycle: its clients get citations and
  the `web_search_requests` count, but no live search status.
- `pause_turn` is forwarded to Anthropic clients and narrowed to `stop` for
  Chat Completions; the proxy does not continue a paused turn itself.
- Native Responses output items require the Responses endpoint; other encoders
  fail explicitly.
- There is no automated test against a live subscription; live checks are run
  manually.
- The Codex device-code sign-in and token refresh are tested only against a
  stand-in for ChatGPT's login service.
