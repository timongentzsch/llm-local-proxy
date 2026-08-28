# Architecture

The proxy serves two downstream wire formats over two upstream subscriptions.
Any format can reach any subscription: the request's `model` decides.

```
  request    dialects/<d>/ingress.py  ──► ChatRequest ──► providers/<p>/request.py  ──► upstream
  response   providers/<p>/events.py  ──► StreamEvent ──► dialects/<d>/egress.py    ──► client
```

Everything else follows from keeping those two axes independent.

## The three axes

| Axis | Members | Adding one costs |
| --- | --- | --- |
| **Dialect** — downstream wire format | OpenAI Chat Completions, Anthropic Messages | one package, one registry line |
| **Provider** — upstream subscription | Codex, Claude | one package, one registry line |
| **Auth** — credential lifecycle | per provider, behind the `Auth` ABC | nothing outside that provider |

Without the intermediate representations this would be a grid: every dialect
would need a converter for every provider. With them it is N + M.

## The intermediate representations

**IR** is the compiler term *intermediate representation*: parse once into a
neutral in-memory shape, emit each target from it.

`ChatRequest` (`ir.py`) is **Anthropic-shaped**, not a lowest common
denominator. Anthropic's message model is the superset — typed content blocks,
tool results inside a user turn, signed thinking, cache breakpoints — so an IR
built from the intersection would have to discard exactly the things a proxy
must not lose. Shaping it after the superset costs the Chat Completions path
nothing, because that path never had those fields.

`StreamEvent` is the response vocabulary: `TextDelta`, `ThinkingDelta`,
`ThinkingSignature`, `ToolCallStart`/`Args`/`End`, `Citation`, `Usage`,
`Finish`. Finish reasons use Anthropic's seven-value enum, the richer of the
two; the Chat Completions encoder narrows them to four.

## Layout

```
src/llm_local_proxy/
  ir.py            # ChatRequest, content blocks, StreamEvent
  errors.py        # RequestError
  service.py       # registry, merged catalog, status
  config.py  atomic.py  ledger.py  status.py

  http/
    server.py      # socket, main()
    handler.py     # (dialect, path) route table
    sse.py         # framing driven by Dialect
    security.py    # loopback hardening, credential headers

  dialects/
    base.py        # Dialect: parse, encode, catalog, framing, errors
    __init__.py    # DIALECTS registry, prefix resolution
    openai/        # __init__.py  ingress.py  egress.py
    anthropic/     # __init__.py  ingress.py  egress.py

  providers/
    base.py        # Provider, ProviderContext
    __init__.py    # REGISTRY, in match-priority order
    auth.py        # Auth ABC
    catalog.py     # the OpenRouter model shape both providers report
    reasoning.py   # ReasoningCache
    transport.py   # no-redirect handler + SSE reader
    codex/         # app_server auth upstream request events catalog
    claude/        # auth upstream request events catalog subscription
  static/index.html
```

Each dialect's `__init__.py` *is* the `Dialect` value; each provider's is
`create(ProviderContext) -> Provider`. That plus the registry line is the whole
public surface of either.

## Endpoints

Every dialect is mounted under its own prefix, because the two formats disagree
about what `/v1/models` returns and neither can own it outright.

| Mount | Routes |
| --- | --- |
| `/openai/v1` | `chat/completions`, `models`, `models/count` |
| `/anthropic` | `v1/messages`, `v1/messages/count_tokens`, `v1/models` |
| `/v1` | alias of `/openai/v1`, for configs written before the prefixes |

The local API key is accepted in either `Authorization: Bearer` or `x-api-key`
on every mount. It is the proxy's own key, not a vendor's, so refusing the
header a client happens to send would only produce a confusing 401.

`count_tokens` is answered exactly by providers whose upstream can count, and
`404` by those that cannot. A client that gets the 404 falls back to its own
estimate knowing it is one; a client handed a guessed integer would trust it
and manage its context wrongly.

## Evidence rules

Wire claims are labelled, because they are not equally reliable:

- **[spec]** — verifiable against the OpenAPI documents. See
  [specs.md](specs.md); they are fetched, not committed.
- **[docs]** — authoritative prose, not machine-checkable. Chiefly the SSE
  framing: `MessageStreamEvent` unions six members and includes neither `ping`
  nor `error`, both of which the streaming docs define.
- **[empirical]** — observed against a subscription edge. No specification
  exists and it may break without notice.

The structural rule: **[spec] and [empirical] code do not share a module.**
Everything reverse-engineered lives in `providers/claude/subscription.py` and
the auth modules — the Claude Code system marker, the beta headers, the pinned
user agent. Nothing in `dialects/` is empirical.

`tests/test_conformance.py` turns the first category into a test, so a spec
refresh that changes the contract fails there rather than in a client. It
skips when `specs/` is absent.

## Wire contract

Details that cost real debugging, all spec-checked.

**Request.** Required: `model`, `messages`, `max_tokens`. `max_tokens: 0` is
legal — it pre-warms the prompt cache without generating. A trailing
`assistant` message is a prefill the response continues from, and consecutive
same-role turns merge server-side; both must survive the IR. `output_config`
carries the same effort tiers as `reasoning_effort`. A `system` role is valid
*inside* `messages`, distinct from the top-level system prompt.

**Response.** `Message` requires `stop_reason`, `stop_sequence`,
`stop_details`, `container` and `usage` to be **present**, nullable ones as
`null`. `stop_reason` has seven values: `end_turn`, `max_tokens`,
`stop_sequence`, `tool_use`, `pause_turn`, `refusal`,
`model_context_window_exceeded`.

**Usage.** Total input is `input_tokens + cache_creation_input_tokens +
cache_read_input_tokens`. Chat Completions reports the sum, so the Anthropic
encoder splits it back out.

**Streaming.** Named frames, no `[DONE]` sentinel. One content block open at a
time under monotonic indices, so the Anthropic encoder is a state machine that
closes on kind change. `message_start.usage.input_tokens` is non-nullable while
a Codex stream has no input count until the end, so the opening frame claims
zero and `message_delta` carries the authoritative totals.

| Anthropic | Chat Completions |
| --- | --- |
| `end_turn`, `stop_sequence`, `pause_turn` | `stop` |
| `tool_use` | `tool_calls` |
| `max_tokens`, `model_context_window_exceeded` | `length` |
| `refusal` | `content_filter` |

## Sharp edges

1. **The subscription marker [empirical].** Must be the first system block or
   the request bills against the API pool and 429s. A real Claude Code client
   already sends it, so it is deduplicated rather than prepended blindly.
2. **Cache breakpoints.** `system` is a block list, not a joined string.
   Flattening it drops `cache_control` and every turn re-pays full input.
3. **Signatures.** On anthropic→claude, thinking blocks are forwarded verbatim
   and the `ReasoningCache` is bypassed: the client holds them. On
   anthropic→codex they are dropped and the encrypted-reasoning cache is keyed
   by tool call id. On the openai→claude routes a block is only replayable
   when its text arrived [empirical]: the subscription edge signs reasoning it
   never streams, and that signature covers what Claude wrote rather than the
   empty string left here, so such a block is neither packed nor replayed and
   its turn goes up without thinking.
4. **`tool_result` images** are representable to Claude but not to Codex.
5. **Betas [empirical].** The client's `anthropic-beta` header is unioned with
   the required Claude Code betas through an allowlist.

## What is deliberately not shared

Modularity is measured as diff size, not as aesthetics, and some code only
looks duplicated:

- **The two `_open` retry envelopes** differ in URL, headers, token source and
  error mapping, and only Claude records rate-limit headers. A shared helper
  would need four callbacks to save ten lines.
- **The two usage tallies** are the trap: the most similar-looking code here
  and the least mergeable. Codex records once at `response.completed`; Claude
  accumulates across events, merges with `max()`, and commits at
  `message_stop`.
- **The two `_tools` parsers** share five guard lines, then diverge in tool
  shape, server-tool detection and error text.
- **No `Translator` base class.** The provider decoders share a target type,
  not an implementation; a common ABC would assert a similarity that is not
  there.
- **No plugin loader.** `REGISTRY` and `DIALECTS` are Python tuples.
- **No codegen** from the specs: three endpoints implemented against 96 paths.

`providers/transport.py` holds what *is* provably identical — the no-redirect
handler and the SSE reader — and nothing else.

## Known gaps

- Streaming, tool calls and `count_tokens` are covered by tests and by a smoke
  test against the real `claude` CLI, but the proxy has no automated test
  against a live subscription; those runs are manual.
- `pause_turn` is forwarded to Anthropic clients and narrowed to `stop` for
  Chat Completions ones. The proxy does not itself continue a paused
  server-tool turn.
- The Codex→Anthropic encoder is exact but numbers content blocks in Codex's
  ordering rather than a model's natural one.
