# Wire specifications

The downstream dialects this proxy speaks are public, specified APIs. These two
files are the ground truth for request validation, response shapes and the
conformance tests; nothing about the downstream wire format should be asserted
from memory or from a blog post when it can be read here.

The files are not committed. `scripts/refresh-specs.sh` downloads reviewed,
immutable snapshots and verifies their SHA-256 checksums before replacing local
copies. CI requires this step; local conformance tests skip only when the files
are absent. To review an upgrade, update the script's URL and checksum together,
run the tests, and record the new provenance here.

| File | Snapshot | SHA-256 |
| --- | --- | --- |
| `openai-openapi.yaml` | OpenAI commit `b61ced96515cb6e73794ff459e9e12ca57596c72` | `77a517da92356a777eb9be7ecc978c15adcc2f17ee387c282090e5a890170cf5` |
| `anthropic-openapi.json` | Stainless snapshot `319861ef873b46e22d6feb51442e743643815093bdd2b3324df52ed202d7ab93` | `717ab2a5efd6263fc76a03b1b361c03d34fe3a0987c2b8f445b6537ede0c991a` |

Verified 2026-09-07. Download URLs are pinned in the script.

OpenAI publishes its MIT-licensed specification in
[openai/openai-openapi](https://github.com/openai/openai-openapi). Anthropic's
snapshot is the Stainless generator input previously linked from its first-party
TypeScript SDK. The SDK no longer exposes `openapi_spec_url` in `.stats.yml`;
the last reviewed snapshot remains available at its content-addressed URL.
It is evidence for the implemented contract, not a claim of current API coverage.

The conformance tests check selected Anthropic enums, required fields, and
request structure. They are not a complete schema validator. Golden transcripts
and endpoint tests cover emitted streams and cross-format behavior.

For behavior outside the schemas, consult the
[SSE standard](https://html.spec.whatwg.org/multipage/server-sent-events.html),
[Anthropic streaming documentation](https://platform.claude.com/docs/en/api/streaming),
[stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons),
and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## Not covered by any specification

The Claude *subscription* transport has no public specification of any kind:
the Claude Code beta headers and pinned user agent in
`src/llm_local_proxy/providers/claude/upstream.py`, the mandatory first system
block in `providers/claude/subscription.py`, and the OAuth flow in
`providers/claude/auth.py`. All of it was established empirically and may break
without notice. Never cite these files as spec-backed.
