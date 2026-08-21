# Wire specifications

The downstream dialects this proxy speaks are public, specified APIs. These two
files are the ground truth for request validation, response shapes and the
conformance tests; nothing about the downstream wire format should be asserted
from memory or from a blog post when it can be read here.

The files are **not** committed: they are third-party artifacts of several
megabytes, and the Anthropic one carries no licence. Fetch them with
`scripts/refresh-specs.sh`, which writes `specs/` and reports any hash change.
A changed hash is a wire contract change, so review it.

`tests/test_conformance.py` checks the proxy against whatever is in `specs/`
and skips when the directory is absent, so a clone without them still passes.

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `openai-openapi.yaml` | 2968497 | `5de4ab93fa24e33be716983c03e09d9f5b43e8b73168363e7c7d887b6ac66e3c` |
| `anthropic-openapi.json` | 2017030 | `0042750ea1ee2eab1751ab0e8affb44e1d0a5d6bb39bb4ea914e9a90873dfcdc` |

Fetched 2026-08-20.

## openai-openapi.yaml

- Source: <https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml>
- OpenAPI 3.1, spec version 2.3.0, MIT licensed, published by OpenAI.
- Officially maintained and directly downloadable. `POST /v1/chat/completions`
  is `createChatCompletion`.

## anthropic-openapi.json

- Source: the `openapi_spec_url` pinned in
  <https://raw.githubusercontent.com/anthropics/anthropic-sdk-typescript/main/.stats.yml>
  (currently `storage.googleapis.com/stainless-sdk-openapi-specs/anthropic/anthropic-893a61e9c1cd6c69a70d1043c626ed02d12d6a492eb6ca6ef7a84c64cfb15393.yml`).
- OpenAPI 3.1, 96 paths, 1100 schemas. Served with a `.yml` extension but the
  body is JSON, hence the local name.

Anthropic publishes **no** official OpenAPI document. This is the Stainless
input that generates their first-party SDKs: authoritative in practice,
unannounced as a product, and reachable only through a content-addressed URL
that changes whenever the spec changes. Treat it as strong evidence, not as a
contract Anthropic offers.

Two things the schema does **not** cover, for which the prose docs are the
better source (append `.md` to any docs page for clean markdown; the HTML
reference is client-rendered and yields only navigation):

- **SSE framing.** The `200` response is typed `application/json` only. The
  named-event stream, the `ping` event and the `error` event are described in
  <https://docs.anthropic.com/en/api/messages-streaming.md>, not in the schema.
  `MessageStreamEvent` unions exactly six members and includes neither `ping`
  nor `error`.
- **Semantics of fields**, e.g. that `max_tokens: 0` pre-warms the prompt cache,
  or that a trailing `assistant` message is a prefill the response continues
  from. See <https://docs.anthropic.com/en/api/messages.md>.

## Not covered by any specification

The Claude *subscription* transport has no public specification of any kind:
the Claude Code beta headers and pinned user agent in
`src/llm_local_proxy/providers/claude/upstream.py`, the mandatory first system
block in `providers/claude/subscription.py`, and the OAuth flow in
`providers/claude/auth.py`. All of it was established empirically and may break
without notice. Never cite these files as spec-backed.
