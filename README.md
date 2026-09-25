# LLM Local Proxy

A local gateway that exposes OpenAI- and Anthropic-compatible APIs on top of
your own Codex (ChatGPT) and Claude subscriptions. Clients keep their agent and
tool loop; the proxy only translates the protocol.

![Dashboard with placeholder data](docs/dashboard.png)

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and the
[Codex CLI](https://github.com/openai/codex), whose `app-server` the proxy
drives. The Docker image includes both.

```sh
uv tool install .
llm-local-proxy
```

```sh
docker compose up --build   # published on 127.0.0.1:8787 only
```

Open the URL printed at startup; its fragment carries the generated API key.
Sign in to one or more accounts per subscription, then copy a base URL or a
ready-made Codex CLI, Claude Code or OpenCode launch command from the dashboard.

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
survive tool loops. Codex streams reasoning summaries whenever a client asks
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
round-robins. Before any output is
streamed, a 429 cools the account for five minutes and a rejected credential
(expired login or missing inference scope) marks it for reauthentication and
cools it for one minute; either way the request moves to the next account.
Nothing is retried after output starts.

**Usage.** Token counts come from upstream usage and are recorded once per
request, over rolling 5-hour and 7-day windows of proxy traffic only. A stream
that ends before final accounting keeps its last reported counts and is marked
as partial; missing counts are never estimated. A stream that ends before its
terminal event fails instead of looking complete.

## Configuration

`~/.config/llm-local-proxy/config.toml`, or `--config PATH`
(`--show-config` prints the path). The file is created on first run and must
be readable only by its owner.

```toml
host = "127.0.0.1"
port = 8787
api_key = "long-random-local-secret"
codex_home = "~/.codex"
codex_binary = "codex"
request_timeout = 600
```

An empty `api_key` disables authentication; otherwise it needs at least 24
characters. Native installs bind only to loopback addresses. Under Docker keep
the port published to `127.0.0.1` as supplied.

Account slots are added and removed from the dashboard; each provider allows
one unsigned slot at a time, and a slot must be signed out before removal.
Codex logins live in `codex_home/accounts/<slot>`; credentials, usage and token
ledgers live in `accounts/<provider>/<slot>` next to the config.

## Development

```sh
uv sync --locked
./scripts/refresh-specs.sh
PYTHONPATH=src uv run python -m unittest discover -s tests
uv run ruff check src tests && uv run ruff format --check src tests
```

`tests/test_golden.py` pins the byte-level output of every request and response
lane, and `tests/test_protocol_matrix.py` replays a tool turn with populated
arguments and signed reasoning through all six format/subscription pairs.
Regenerate goldens only deliberately (`LLM_PROXY_RECORD=1`) and review the
diff. CI runs the suite on Python 3.11–3.14.

See [docs/architecture.md](docs/architecture.md) for the design and wire
contracts and [docs/specs.md](docs/specs.md) for the specifications they are
tested against.

## Disclaimer

An unofficial, independent project, not affiliated with or endorsed by OpenAI
or Anthropic. It runs on your machine against your own accounts; credentials
are stored locally and sent only to their provider. Codex login is delegated to
the official binary, and Claude tokens come from the OAuth flow of its
first-party client.

The proxy speaks two undocumented interfaces, the Codex app-server JSON-RPC
surface and the Claude subscription Messages transport, including the client
identifiers that mark first-party traffic. They may change without notice, and
their use may fall outside your subscription's terms; review those terms and
use the paid APIs where a supported integration is required. No warranty.
