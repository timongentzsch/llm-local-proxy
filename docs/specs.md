# Wire specifications

The downstream formats are public APIs, and their contracts should be checked
against published specifications rather than memory.

## Anthropic Messages

`scripts/refresh-specs.sh` downloads a reviewed, immutable snapshot of the
Anthropic OpenAPI document into `specs/` (not committed) and verifies its
checksum. CI runs it before the tests; locally, `tests/test_conformance.py`
skips when the file is absent.

| File | Snapshot | SHA-256 |
| --- | --- | --- |
| `anthropic-openapi.json` | Stainless `319861ef873b46e22d6feb51442e743643815093bdd2b3324df52ed202d7ab93` | `717ab2a5efd6263fc76a03b1b361c03d34fe3a0987c2b8f445b6537ede0c991a` |

Verified 2026-09-07. The snapshot is the Stainless generator input formerly
linked from Anthropic's TypeScript SDK and remains available at its
content-addressed URL. It documents the implemented contract, not current API
coverage. To upgrade, change the URL and checksum in the script together, run
the tests, and update this table.

The conformance tests check selected enums, required fields and request
structure; they are not a full schema validator. Behaviour outside the schema
follows the
[streaming](https://platform.claude.com/docs/en/api/streaming),
[stop reason](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)
and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
documentation and the
[SSE standard](https://html.spec.whatwg.org/multipage/server-sent-events.html).

## OpenAI Chat Completions and Responses

The reference is [openai/openai-openapi](https://github.com/openai/openai-openapi).
No snapshot is fetched; golden transcripts and endpoint tests pin the emitted
requests, streams and results.

## Not covered by any specification

The Claude subscription transport is undocumented: the beta headers and user
agent in `providers/claude/upstream.py`, the mandatory first system block in
`providers/claude/subscription.py`, and the OAuth flow in
`providers/claude/auth.py`. The Codex app-server JSON-RPC surface is likewise
private. All of it was established empirically and may change without notice.
