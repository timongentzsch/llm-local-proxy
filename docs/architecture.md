# Architecture

The proxy serves three downstream formats (OpenAI Chat Completions, OpenAI
Responses, Anthropic Messages) over two upstream subscriptions (Codex, Claude).
Any format can reach any subscription; the request's `model` decides.

```
request   dialects/<d>/ingress ─► ChatRequest ─► providers/<p>/request ─► account pool ─► upstream
response  providers/<p>/events ─► StreamEvent ─► dialects/<d>/egress   ─► client
```

Parsing each format once into an intermediate representation (IR) and
rendering each upstream from it keeps the cost of a new format or provider at
N + M instead of N × M.

## Intermediate representation

`ir.py` defines both directions.

- **`ChatRequest`** carries shared semantics in typed fields: system blocks,
  turns of content blocks, function tools with strictness and parallel-call
  control, tool choice, token limits, reasoning effort and summary, thinking
  mode and display, cache hints and key, session, sampling parameters and
  output format. Wire-specific tool
  options keep their source format; `tools.py` preserves them on compatible
  targets and rejects them elsewhere.
- **Opaque escape hatches** (`Reasoning`, `NativeResponseItem`,
  `NativeAnthropicBlock`, `NativeTool`) hold content without a lossless mapping,
  such as signed reasoning and rich tool results. They are forwarded verbatim
  on compatible routes and rejected on the rest, never interpreted.
- **`ToolNamespace`** keeps a Responses namespace both verbatim and as parsed
  function tools. Targets without namespaces use `tools.flatten`, which gives
  each member a qualified name of at most 64 characters and the map to restore
  calls; tool calls carry their `namespace` back to the client.
- **`StreamEvent`** is the response vocabulary: `TextDelta`, `ThinkingDelta`,
  `ThinkingSignature`, `RedactedThinkingDelta`, `ReasoningItem`, `NativeItem`,
  `ToolCallStart`/`ToolCallArgs`/`ToolCallEnd`, `HostedToolEvent`, `Citation`,
  `Usage` and `Finish`. Stop reasons use Anthropic's seven-value enum; Chat
  Completions narrows them to four.

Each provider supplies a `Decoder` (upstream events to `StreamEvent`s); each
dialect supplies an `Encoder` subclass that shapes those events into its own
frames. The HTTP layer pairs the two per request and handles `ProviderError`
without importing provider code.

## Layout

```
src/llm_local_proxy/
  ir.py  tools.py  errors.py  streaming.py
  service.py         provider registry, merged catalog, status
  config.py  atomic.py  ledger.py  status.py
  http/              server, request routing, SSE framing, loopback security
  dialects/
    base.py          Dialect, Route, Encoder base
    openai/          Chat Completions and Responses ingress/egress
    anthropic/       Messages ingress/egress
  providers/
    base.py          Provider, ProviderContext
    pool.py          AccountPool, AccountStore, PooledProvider
    auth.py          Auth: one login's lifecycle
    catalog.py  reasoning.py  transport.py
    codex/           app-server client, auth, request, events, catalog
    claude/          OAuth, transport, request, events, catalog
  static/index.html  dashboard
```

A dialect is one `Dialect` value with a mount prefix and a route table; a
provider is `create(ProviderContext) -> Provider`. Registering either is one
line in `dialects/__init__.py` or `providers/__init__.py`.

## Adding a provider

1. Subclass `PooledProvider` (`providers/pool.py`) and implement
   `new_account`, `fetch_catalog`, `account_status` and `no_account`. Slots,
   logins, failover, catalog caching and status come with it.
2. Render the upstream request from `ChatRequest`, reusing `tools.py`
   (`render_function`, `responses_tool`, `flatten`, `arguments`) and rejecting
   anything the upstream cannot represent. Cache hints are the exception:
   honour the ones the upstream can express and ignore the rest.
3. Write a `Decoder` from upstream events to `StreamEvent`s; wrap the stream in
   `ledger.track_usage` and read it with `transport.read_events`.
4. Expose `create(ProviderContext) -> Provider` via `PooledProvider.provider`
   and add it to `REGISTRY` in `providers/__init__.py`.

Every dialect then reaches the new provider without further changes; add its
lanes to `tests/test_golden.py` and `tests/test_protocol_matrix.py`.

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

Each provider is a `PooledProvider`: an `AccountStore` of slot ids
(`slots.json`), an `AccountPool` of live accounts, and a shared catalog cache.
Routing sees one provider per subscription, so model ids carry no account
suffix.

- **Selection.** A session id hashes to a stable signed-in account; without
  one, the starting account advances round-robin. The session is
  `X-Session-Id`, else a header the dialect names (`Dialect.session_headers`,
  e.g. Claude Code's), else the request's `prompt_cache_key`.
- **Failover.** Before the first upstream event, a 429 cools the account for
  five minutes; a terminal authentication failure (rejected credentials or a
  missing inference scope) marks it for reauthentication and cools it for one
  minute. The untouched request then moves to the next account. Other errors
  keep their status, and nothing switches accounts once output has started.
- **Catalog.** Discovery uses the same pool without affinity and accepts the
  first live catalog, so one stale login cannot hide a provider's models.
- **Slots.** Added and removed live from the dashboard. Only one unsigned slot
  may exist, and a slot must be signed out before removal. Codex state lives
  in `codex_home/accounts/<slot>`, proxy state in `accounts/<provider>/<slot>`.

## Evidence

Wire claims are labelled by how they can be checked:

- **[spec]**: the pinned Anthropic OpenAPI snapshot (see [specs.md](specs.md));
  `tests/test_conformance.py` fails when a refresh changes the contract.
- **[docs]**: published prose that no schema covers, chiefly SSE framing
  (`ping` and `error` events are defined only in the streaming docs).
- **[empirical]**: observed against a subscription edge, with no specification.

[spec] and [empirical] code never share a module. Everything reverse-engineered
(subscription marker, beta headers, OAuth flow, transport probes) lives under
`providers/`; nothing in `dialects/` is empirical.

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
5. **Betas [empirical].** The Claude transport sends its subscription betas
   plus feature betas for web search and structured outputs as requested.

## Deliberately not shared

- The two upstream retry envelopes differ in URL, headers, token source and
  error mapping; a shared helper would need several callbacks to save a few
  lines.
- Usage parsing is specific to each upstream (Codex reports terminal totals,
  Claude cumulative snapshots); persistence and stream cleanup share
  `ledger.track_usage`.
- Decoders share a target type, not an implementation. Encoders share only
  the decoder-driving loop in `dialects/base.Encoder`.
- Registries are plain tuples; there is no plugin loader or code generation.

## Known gaps

- Chat Completions has no hosted-tool lifecycle: its clients get citations and
  the `web_search_requests` count, but no live search status.
- `pause_turn` is forwarded to Anthropic clients and narrowed to `stop` for
  Chat Completions; the proxy does not continue a paused turn itself.
- Native Responses output items require the Responses endpoint; other encoders
  fail explicitly.
- There is no automated test against a live subscription; live checks are run
  manually.
