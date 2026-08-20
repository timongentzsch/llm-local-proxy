# Target architecture

This document describes the architecture the repository is moving to, and the
migration that gets there. It replaces the current layout rather than extending
it. The Anthropic Messages endpoint is the feature that motivated the work, but
it is a consequence of the design here, not the subject.

The current code is working and well tested (85 tests). That is an asset to be
preserved, not a legacy to be swept away: the migration below is a sequence of
behaviour-preserving steps, and the test suite passing unmodified — imports
aside — is the invariant that proves each one.

## Why replace rather than extend

*Historical: this section describes the layout before the migration, and the
line references point into commits before `4c9af23`.*

The system varied along one axis. `Provider` (`providers.py:21-45`) modelled
the upstream subscription, and modelled it well. Everything else assumed
there was exactly one downstream wire format, Chat Completions, and that
assumption was spread across seven sites:

| Layer | Site | Assumption |
| --- | --- | --- |
| SSE framing | `server.py:482`, `:519`, `:529` | unnamed `data:` frames, `[DONE]` terminator, comment keepalive |
| Errors | `server.py:594`, `:515` | `{"error":{message,type}}` |
| Auth | `server.py:552` | `Authorization: Bearer` only |
| Routing | `server.py:39`, `:413`, `:458` | one path space, OpenRouter model shape |
| Provider contract | `providers.py:34` | `chat()` takes a Chat Completions body, returns a chunk-emitting translator |
| Request builders | `protocol.py:142`, `claude_protocol.py:72` | both parse a Chat Completions body |
| Translators | `protocol.py:291`, `claude_protocol.py:407` | both emit `chat.completion.chunk` |

Adding a second dialect additively would leave a 2×2 grid of hand-written
converters that becomes N×M. The target makes dialect and provider genuinely
orthogonal, so each varies at N+M cost.

## The three axes

| Axis | Members today | Future members |
| --- | --- | --- |
| **Dialect** — downstream wire format | Chat Completions | Anthropic Messages, later Gemini or Ollama |
| **Provider** — upstream subscription | Codex, Claude | Copilot, Gemini CLI, Qwen Code |
| **Auth** — credential lifecycle | `Auth` ABC (`auth.py:20-43`) | per provider |

`Auth` is already correct and is carried across unchanged.

## Shape

Two intermediate representations sit between the axes. **IR** is the compiler
term *intermediate representation*: rather than translating every input format
directly to every output format, parse once into a neutral in-memory shape and
emit each target from it.

```
  request path
    dialects/<d>/ingress.py   wire body ──► ChatRequest ──► providers/<p>/request.py  ──► upstream body
  response path
    providers/<p>/events.py   upstream SSE ──► StreamEvent ──► dialects/<d>/egress.py ──► wire frames
```

`ChatRequest` is **Anthropic-shaped**, not a lowest common denominator. A true
intersection of the two formats would discard everything Anthropic has and Chat
Completions lacks — signed `thinking`, `tool_result` images, assistant prefill,
`cache_control`. Shaping it after the superset costs the OpenAI path nothing,
because that path never had those fields to lose.

`StreamEvent` is the canonical response vocabulary, derived from what the two
existing translators already compute internally:

```
MessageStart(model, id)            ToolCallStart(index, id, name)
TextDelta(text)                    ToolCallArgsDelta(index, json_fragment)
ThinkingDelta(text)                ToolCallStop(index)
ThinkingSignature(signature)       Citation(url, title, cited_text, span)
RedactedThinking(data)             ServerToolUse(kind)
Usage(input, output, cache_read, cache_write, thinking_tokens, server_tool_use)
Finish(reason)                     Error(status, message)
```

`Finish.reason` uses the Anthropic enum, which is the richer of the two — seven
values including `pause_turn`, `refusal` and `model_context_window_exceeded`
[spec] — and narrows to Chat Completions' four on the way out.

### A reversal, stated plainly

An earlier draft of this plan deliberately kept both translators intact and
adapted their Chat Completions output into Anthropic frames, on the grounds
that rewriting the best-tested code in the repo was not worth one feature. That
reasoning was correct for an additive change and is wrong for a replacement:
the adapter approach preserves the N×M grid, which is precisely what a
repo-wide refactor exists to remove. The response IR is therefore in scope —
and it is the single highest-risk item in this document, which is why R0 exists
before any of it.

## Layout

```
src/llm_local_proxy/
  __init__.py  __main__.py
  config.py  atomic.py  ledger.py  status.py     # carried across unchanged
  ir.py                                          # ChatRequest, blocks, StreamEvent

  errors.py                                      # RequestError
  service.py                                     # registry, catalog, status

  http/
    server.py        # HTTPServer wiring, main()
    handler.py       # (dialect, path) route table
    sse.py           # framing driven by Dialect
    security.py      # auth guard, host and origin checks

  dialects/
    base.py          # Dialect: parse, encode, catalog, framing, auth, errors
    __init__.py      # DIALECTS registry, prefix resolution
    openai/          # __init__.py  ingress.py  egress.py
    anthropic/       # __init__.py  ingress.py  egress.py

  providers/
    base.py          # Provider, ProviderContext
    __init__.py      # REGISTRY, in match-priority order
    auth.py          # Auth ABC
    reasoning.py     # ReasoningCache
    transport.py     # no-redirect handler + SSE reader, nothing else
    codex/           # __init__.py  app_server.py  auth.py  upstream.py
                     # request.py  events.py  catalog.py
    claude/          # __init__.py  auth.py  upstream.py
                     # request.py  events.py  catalog.py
                     # subscription.py  [empirical] marker, betas, user agent
  static/index.html
```

Each dialect's `__init__.py` is the `Dialect` value itself, so the package
name and the registry entry are the whole of its public surface. Each
provider's `__init__.py` is `create(ProviderContext) -> Provider`.

Tests mirror `src/`: `test_openai.py`-shaped units per layer, `test_golden.py`
for recorded transcripts, `test_endpoint.py` for real HTTP over both dialects,
and `test_conformance.py` for the pinned specs.

### Carried across unchanged

`config.py`, `atomic.py`, `ledger.py`, `status.py`, `providers/auth.py` and
`codex/app_server.py` were already single-purpose and dialect-neutral. They
moved by path at most. Roughly a third of `src/` was untouched, and
`test_config.py` and `test_ledger.py` never changed.

## Evidence rules

Every wire claim in this document is labelled:

- **[spec]** — verified against `specs/anthropic-openapi.json` or
  `specs/openai-openapi.yaml`, pinned by hash in `specs/PINNED.md`.
- **[docs]** — verified against `docs.anthropic.com/en/api/<page>.md`.
  Authoritative prose, not machine-checkable. Used where the schema is silent,
  chiefly SSE framing.
- **[empirical]** — established by observation against a subscription edge. No
  specification exists. May break without notice.

The structural rule this imposes: **[spec] code and [empirical] code must not
share a module.** Today they do — `claude_protocol.py` mixes clean block
translation with the Claude Code system marker (`claude_protocol.py:14-20`,
`:157-170`). In the target, everything undocumented is confined to
`providers/claude/subscription.py` and the auth modules.

## Verified contract

Recall was wrong about several of these; each is now spec-checked.

### Request [spec]

Required: `model`, `messages`, `max_tokens`. Optional: `system` (string *or*
block array), `stream`, `temperature`, `top_p`, `top_k`, `stop_sequences`,
`tools`, `tool_choice`, `thinking`, `metadata`, `cache_control`,
`service_tier`, `container`, `inference_geo`, `output_config`.

- `max_tokens: 0` is **legal** — it pre-warms the prompt cache without
  generating [docs]. Do not validate as a positive integer.
- `tool_choice` is `auto` / `any` / `tool` / `none`; all but `none` carry
  `disable_parallel_tool_use` [spec].
- `thinking` is `enabled{budget_tokens, display}` / `disabled` /
  `adaptive{display}` [spec] — matching what `claude_protocol.py:206-228`
  already emits.
- A custom tool requires `name` and `input_schema` [spec]. The rest of the
  `tools` union is server tools (`bash_*`, `code_execution_*`, `web_search_*`,
  `web_fetch_*`, `text_editor_*`, `tool_search_*`).
- A trailing `assistant` message is a **prefill** the response continues from,
  and consecutive same-role turns are merged server-side [docs]. Neither has a
  Chat Completions analogue; `ChatRequest` must preserve both. Limit of 100,000
  messages per request [docs].

### Response [spec]

`Message` requires `id`, `type`, `role`, `content`, `model`, `stop_reason`,
`stop_sequence`, `stop_details`, `usage`, `container`. The nullable ones must
still be **present as `null`** — an encoder omitting `stop_details` or
`container` is out of contract even though most clients will not notice today.
`stop_details` is `RefusalStopDetails{category, explanation}` or null.

`stop_reason` has seven values: `end_turn`, `max_tokens`, `stop_sequence`,
`tool_use`, `pause_turn`, `refusal`, `model_context_window_exceeded`.

`Usage` requires `input_tokens`, `output_tokens`, `cache_creation`,
`cache_creation_input_tokens`, `cache_read_input_tokens`,
`output_tokens_details`, `server_tool_use`, `service_tier`, `inference_geo`.
Total input is `input_tokens + cache_creation_input_tokens +
cache_read_input_tokens` [docs] — the sum `claude_protocol.py:613-635` already
performs in the other direction, and which must be split back out here.

### Streaming [docs, partly spec]

`MessageStreamEvent` unions exactly six members [spec]: `message_start`,
`content_block_start`, `content_block_delta`, `content_block_stop`,
`message_delta`, `message_stop`. `ping` and `error` are documented in prose and
absent from the schema [docs] — emit `ping` as keepalive and `error` for
mid-stream failure, but treat their shape as docs-level evidence.

Frames are named: `event: <type>\ndata: <json>\n\n`. There is **no `[DONE]`
sentinel**.

Delta types [spec]: `text_delta`, `input_json_delta`, `thinking_delta`,
`signature_delta`, `citations_delta`. `redacted_thinking_delta`, which
`claude_protocol.py:541` handles, is **not** in the union — keep the defensive
branch, do not advertise it as specified.

Content blocks openable in `content_block_start` [spec]: `text`, `thinking`,
`redacted_thinking`, `tool_use`, `server_tool_use`, `web_search_tool_result`,
`web_fetch_tool_result`, `code_execution_tool_result`,
`bash_code_execution_tool_result`, `text_editor_code_execution_tool_result`,
`tool_search_tool_result`, `container_upload`.

Two constraints on `dialects/anthropic/egress.py`:

1. `message_start` carries a `Message` whose `usage.input_tokens` is a required
   non-nullable integer, while `MessageDeltaUsage.input_tokens` is nullable
   [spec]. A Codex stream has no input count until the end, so emit best-known
   values at `message_start` and authoritative totals in `message_delta`.
2. Anthropic allows one open content block at a time with monotonic indices.
   The encoder needs an open-block state machine that closes on kind change and
   offsets tool-call indices past preceding text and thinking blocks.

### Errors [spec]

`{"type":"error","request_id":...,"error":{"type":...,"message":...}}` —
`request_id` is required. Types: `invalid_request_error`,
`authentication_error`, `permission_error`, `not_found_error`, `billing_error`,
`rate_limit_error`, `api_error`, `overloaded_error`, `gateway_timeout_error`.

### Auxiliary endpoints [spec]

- `POST /v1/messages/count_tokens` — requires only `messages` and `model`
  (**not** `max_tokens`); response is `{"input_tokens": int}`.
- `GET /v1/models` — `{data, has_more, first_id, last_id}`; each `ModelInfo` is
  `{type:"model", id, display_name, created_at, max_tokens, max_input_tokens,
  capabilities}`.

## Mappings

| Anthropic | Chat Completions | Note |
| --- | --- | --- |
| `end_turn` | `stop` | |
| `tool_use` | `tool_calls` | |
| `max_tokens` | `length` | |
| `stop_sequence` | `stop` | collapsed today at `claude_protocol.py:501-506`; `StreamEvent` keeps it |
| `refusal` | `content_filter` | plus `stop_details` |
| `model_context_window_exceeded` | `length` | |
| `pause_turn` | — | no analogue; resolved by the proxy, never forwarded to an OpenAI client |
| `input_tokens` + cache fields | `prompt_tokens` | sum in, split out |
| `output_tokens_details.thinking_tokens` | `completion_tokens_details.reasoning_tokens` | |

## Endpoints

Every dialect is mounted under its own prefix, because the two formats
disagree about what `/v1/models` returns and neither can own it outright.
`DEFAULT` (Chat Completions) additionally answers on the bare paths, which
predate the prefixes and stay valid.

| Mount | Routes |
| --- | --- |
| `/openai/v1` | `chat/completions`, `models`, `models/count` |
| `/anthropic` | `v1/messages`, `v1/messages/count_tokens`, `v1/models` |
| `/v1` | legacy alias of `/openai/v1`, byte-identical |

`Dialect.base_path` is what a client is configured with, and is not always
`prefix + "/v1"`: an OpenAI client wants `/v1` inside its base url and appends
`/chat/completions`, while an Anthropic client appends `/v1/messages` itself.
The dashboard renders one row per registered dialect from that field, so a new
dialect appears there without the dashboard or `Service` changing.

Model routing is unchanged: the registry order in `providers/__init__.py` is the
match priority, Claude claims `claude-*`, Codex is the fallback. Auth accepts
`x-api-key` alongside `Authorization: Bearer`, against the same local key.

## Migration

Each phase is one PR, independently revertible, and leaves the suite green.
No externally visible behaviour changes before R5.

| Phase | Scope | Risk | Est. | Status |
| --- | --- | --- | --- | --- |
| **R0** | **Safety net.** Capture golden transcripts: upstream SSE in, wire bytes out, for both providers × current dialect, plus a replay harness. No src changes | — | 0.5d | **done** |
| **R1** | Mechanical moves only: create the tree, relocate modules, fix imports. Zero logic edits. Tests change by import line only | low | 0.5d | **done** |
| **R2** | `Dialect` seam in `http/`: framing, errors, auth, route table. OpenAI output byte-identical against R0 goldens | low | 1d | **done** |
| **R3** | `ChatRequest` + ingress split; both `request.py` builders read IR; `subscription.py` split out of `claude_protocol.py` | medium | 1.5d | **done** |
| **R4** | `StreamEvent` + both `events.py` rewritten; `openai/egress.py` reconstructs chunks. **Goldens must be byte-identical** | **high** | 2–3d | **done** |
| **R5** | Anthropic dialect: ingress, egress, catalog. Genuinely additive by now | medium | 2d | **done** |
| **R6** | `REGISTRY` + `ProviderContext`; `Service` reduces to caching and status aggregation | low | 1d | **done** |
| **R7** | `transport.py` dedup, conformance tests, dashboard, README | low | 1d | **done** |

Roughly ten working days. R4 is the phase to slow down on; if goldens diverge
there and the cause is not understood within a day, revert it and fall back to
adapting Chat Completions output into Anthropic frames — a worse architecture
that still ships R5.

R0 is not optional. It is what converts "the tests still pass" into "the bytes
on the wire are identical", and it is the only thing that makes R4 safe.

## Where the migration stands

Complete, on `refactor/dialect-provider-architecture`, one commit per phase.
141 tests pass. The Chat Completions goldens pass **untouched** through both
rewrites, so every upstream request body and every downstream chunk is
byte-identical to what the pre-refactor code produced.

Both axes are done. A request is parsed once into `ChatRequest` by the
ingress of whichever dialect received it, and rendered by whichever provider
claims the model. The response comes back as `StreamEvent` from the provider's
decoder and is written by the dialect's encoder. Neither side names the other.

The modularity contract holds as stated below: `dialects/anthropic/` is three
files and one registry line, and touches no provider; `providers/*/` build
themselves from a `ProviderContext` and touch no dialect and no line of
`http/`.

### What is deliberately not implemented

- `POST /anthropic/v1/messages/count_tokens`. The upstream endpoint is not
  wired, and a client that trusts a guessed token count will manage its
  context wrongly. Better absent than approximate.
- A native Codex→Anthropic encoder. The existing one is exact, but the block
  indices it produces follow Codex's ordering rather than a model's natural
  one. No client has complained; revisit if one does.
- `pause_turn` resolution. It narrows to `stop` for Chat Completions clients
  and passes through intact to Anthropic ones, which is correct for both, but
  the proxy does not itself continue a paused server-tool turn.

## Transport: extract only what is provably identical

`upstream.py` and `claude_upstream.py` look heavily duplicated. Checking rather
than assuming, only part of that is real:

| Code | Verdict |
| --- | --- |
| `_NoRedirect` (`upstream.py:22-24`, `claude_upstream.py:38-40`) | **byte-identical** → extract |
| `_events` SSE reader (`upstream.py:56-70`, `claude_upstream.py:257-270`) | **byte-identical**, including the `[DONE]` skip → extract |
| `_open` 401-retry envelope | same *shape*; different URL, headers, token source, error mapping, and Claude alone updates `UsageStore` on both paths → **leave duplicated** until a third transport exists; a shared helper would need four injected callbacks to save ten lines |
| `_tracked` usage tallying | **genuinely different policy** — Codex records once at `response.completed` (`upstream.py:40-54`); Claude accumulates across `message_start` and repeated `message_delta`, merges with `max()`, commits only at `message_stop` (`claude_upstream.py:213-239`) → **never merge** |

`transport.py` is therefore about 20 lines of zero-policy plumbing. The
`_tracked` pair is the trap: the most similar-looking code in the repo and the
least mergeable.

## The modularity contract

Modularity is testable as diff size, not aesthetics. After R7 these hold, and
belong in each PR description as acceptance criteria:

- **Adding a dialect** touches: `dialects/<name>/`, one line in `DIALECTS`, one
  route-table line. **Zero** provider files.
- **Adding a provider** touches: `providers/<name>/`, one line in `REGISTRY`.
  **Zero** dialect files, **zero** lines in `http/`.
- **A spec refresh** touches `specs/` only. Behaviour changes it implies surface
  as failing conformance tests, not user bug reports.

## What not to abstract

Part of the plan, because each is a tempting no-op:

- **No `Translator` base class.** After R4 the per-provider `events.py` modules
  share a *target* type, not an implementation. A common ABC over them would be
  an empty interface asserting a similarity that is not there.
- **No middleware or hook chain.** There is one cross-cutting concern, usage
  accounting, and it is correctly per-transport — see `_tracked` above.
- **No config-driven provider loading.** `REGISTRY` is a Python list. A plugin
  system for two entries is cost without benefit.
- **No codegen from the pinned specs.** Three endpoints implemented against 96
  paths and 1100 schemas; the specs are a test oracle, not a build input.
- **No repackaging of `config.py`, `ledger.py`, `status.py`, `atomic.py`.**
  They are already right.

## Tests

- `tests/golden/` — recorded upstream streams and expected wire bytes. The
  regression gate for R1–R4 and the reason R0 comes first.
- `tests/test_conformance.py` — load `specs/`, extract the schemas we touch
  (`MessageParam`, `InputContentBlock`, `Usage`, `StopReason`,
  `MessageStreamEvent`, `ErrorResponse`, `Tool`, `ToolChoice`), assert the
  validator accepts every documented shape and rejects out-of-enum values.
- Mirrored unit tests per package; `test_config.py` and `test_ledger.py`
  unchanged.
- One 2×2 matrix test: every (dialect, provider) pair emits a well-formed
  stream and non-stream body.
- Live smoke: the real `claude` CLI against the proxy, for a `claude-*` model
  and a Codex model.

## Sharp edges

1. **Double marker [empirical].** `claude_protocol.py:162-170` unconditionally
   prepends `CLAUDE_CODE_SYSTEM_MARKER`. A real Claude Code client already
   sends it as system block 0. Dedupe — but never drop it, or the request bills
   against the API pool and 429s (`claude_upstream.py:24-27`).
2. **Signatures.** On anthropic→claude, forward `thinking`/`redacted_thinking`
   verbatim and bypass `ReasoningCache` — the client holds the signed blocks.
   On anthropic→codex, drop client thinking blocks and keep the
   encrypted-reasoning cache keyed by `tool_use` id, as today.
3. **`cache_control`.** Forward on the Claude lane, where explicit breakpoints
   are the only ones that do anything; strip on the Codex lane, which relies on
   implicit prefix caching.
4. **`tool_result` with image content** is representable to Claude but not to
   Codex (`protocol.py:80-101`). Lift into a following user block, or reject.
5. **Reject loudly**, per existing policy: `container`, `mcp_servers`,
   `service_tier`, `output_config`, and every server tool but `web_search`.
   Map `disable_parallel_tool_use` → `parallel_tool_calls:false`. Ignore
   `metadata.user_id`.
6. **Betas [empirical].** `claude_upstream.py:29` pins one beta set and user
   agent. Forward the union of the client's `anthropic-beta` header with the
   required Claude Code betas, through an allowlist.
7. **Rate-limit headers [empirical].** Re-emit the
   `anthropic-ratelimit-unified-*` values held in `UsageStore`
   (`claude_upstream.py:43`) so client-side backoff behaves.

## Open question

No specification describes what the real Claude Code client *sends*: which
`anthropic-beta` values, whether it calls `count_tokens` before each turn, how
it reacts to `pause_turn`. That is client behaviour, not API surface. Capture
it during R0 against a logging stub and keep the transcripts as fixtures.
