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

Today the system varies along one axis. `Provider` (`providers.py:21-45`)
models the upstream subscription, and it models it well. Everything else
assumes there is exactly one downstream wire format, Chat Completions, and that
assumption is spread across seven sites:

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

## Target tree

```
src/llm_local_proxy/
  __init__.py  __main__.py
  config.py  atomic.py  ledger.py  status.py     # carried across unchanged
  ir.py                                          # ChatRequest, blocks, StreamEvent

  http/
    server.py        # HTTPServer wiring, main()          ← server.py:630-665
    handler.py       # (dialect, path) route table         ← server.py:379-533
    sse.py           # framing driven by Dialect           ← server.py:496-531
    security.py      # auth guard, host and origin checks  ← server.py:552-590

  dialects/
    base.py          # Dialect: parse, encode, errors, framing, auth, catalog
    __init__.py      # DIALECTS registry
    openai/
      ingress.py     # body → ChatRequest                  ← protocol.py:66-140
      egress.py      # StreamEvent → chat.completion.chunk ← protocol.py:246-435
      catalog.py     # /v1/models shape                    ← server.py:44-95
    anthropic/
      ingress.py  egress.py  catalog.py                    # new

  providers/
    base.py          # Provider, ProviderContext, ReasoningCache ← providers.py, protocol.py:39
    auth.py          # Auth ABC                             ← auth.py
    transport.py     # _NoRedirect + SSE reader             ← proven-identical only
    __init__.py      # REGISTRY = [claude.build, codex.build]
    codex/
      app_server.py  ← app_server.py          auth.py     ← codex_auth.py
      upstream.py    ← upstream.py            request.py  ← protocol.py:142-244
      events.py      # Responses SSE → StreamEvent         ← protocol.py:291-409
      catalog.py     ← server.py:256-267
    claude/
      auth.py        ← claude_auth.py         upstream.py ← claude_upstream.py
      request.py     ← claude_protocol.py:72-228           events.py ← :407-652
      catalog.py     ← server.py:329-355
      subscription.py  # [empirical] marker, betas, user agent ← claude_protocol.py:14-20
  static/index.html
```

Tests mirror `src/`, plus `tests/golden/` for transcripts and
`tests/test_conformance.py` for the pinned specs.

### Carried across unchanged

`config.py`, `atomic.py`, `ledger.py`, `status.py`, `auth.py` and
`app_server.py` are already single-purpose and dialect-neutral. They move at
most by path. Roughly a third of `src/` is not touched by this refactor, and
`tests/test_config.py` and `tests/test_ledger.py` should not change at all.

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

Mount the new dialect under a prefix so the two `/v1/models` shapes cannot
collide: `ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic`.

- `POST /anthropic/v1/messages`, plus bare `/v1/messages` as an alias
- `POST /anthropic/v1/messages/count_tokens`
- `GET /anthropic/v1/models`
- `/v1/chat/completions` and `/v1/models` keep their current shapes exactly

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
| **R4** | `StreamEvent` + both `events.py` rewritten; `openai/egress.py` reconstructs chunks. **Goldens must be byte-identical** | **high** | 2–3d | next |
| **R5** | Anthropic dialect: ingress, egress, catalog. Genuinely additive by now | medium | 2d | |
| **R6** | `REGISTRY` + `ProviderContext`; `Service` reduces to caching and status aggregation | low | 1d | |
| **R7** | `transport.py` dedup, conformance tests, dashboard, README | low | 1d | |

Roughly ten working days. R4 is the phase to slow down on; if goldens diverge
there and the cause is not understood within a day, revert it and fall back to
adapting Chat Completions output into Anthropic frames — a worse architecture
that still ships R5.

R0 is not optional. It is what converts "the tests still pass" into "the bytes
on the wire are identical", and it is the only thing that makes R4 safe.

## Where the migration stands

R0–R3 are on `refactor/dialect-provider-architecture`, one commit per phase,
each green. 100 tests pass, and the 29 goldens pass **untouched** — so every
upstream request body is byte-identical to what the pre-refactor code sent.

The request axis is finished: a Chat Completions body is parsed once by
`dialects/openai/ingress.py` and rendered by `providers/codex/request.py` or
`providers/claude/request.py`, neither of which knows which dialect asked.
Adding the Anthropic *request* path is now one new ingress module.

The response axis is untouched: `providers/codex/events.py` and
`providers/claude/events.py` still emit `chat.completion.chunk` directly, so
the 2×2 grid survives on that side until R4.

### Starting R4

The decoder/encoder split is the shape to build:

- `providers/<p>/events.py` exposes a stateful decoder, `decode(event) ->
  list[StreamEvent]`, owning the reasoning-cache writes (encrypted content for
  Codex, signed thinking for Claude) since those are upstream state.
- `dialects/<d>/egress.py` exposes an encoder built around a decoder and
  holding the wire shaping: chunk envelopes, citation and usage mapping,
  finish-reason narrowing. `Dialect` gains an `encode(model, decoder)` factory
  so the handler wires the pair without either side naming the other.

Two pieces of genuine duplication collapse here, and they are the argument for
doing R4 at all rather than adapting chunks: `protocol._usage`/`_annotation`
and their Claude twins compute the same Chat Completions shapes from different
inputs. Under the split, both providers emit canonical `Usage` and `Citation`
events and a single encoder shapes them once.

The goldens are the specification for that encoder — each file is an exact
chunk sequence to reproduce. Work from them rather than from the old
translators, and keep the abort criterion in mind: if they diverge and the
cause is not understood within a day, revert and fall back to adapting Chat
Completions output into Anthropic frames.

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
