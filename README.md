# LLM Local Proxy

A small local gateway that puts an OpenAI- or Anthropic-shaped API in front of
your own Codex (ChatGPT) and Claude subscriptions. Your client keeps the agent
and tool loop; this process only translates the protocol.

![Dashboard with example data](docs/dashboard.png)

_Account and key above are placeholders; the dashboard reads your own accounts
at runtime._

## Run

Needs [uv](https://docs.astral.sh/uv/) and the
[Codex binary](https://github.com/openai/codex), whose `app-server` the proxy
starts. Docker includes both.

```sh
uv tool install .
llm-local-proxy
```

Open the URL printed at startup — it carries the generated API key in the
fragment. Sign in to one or more accounts for either subscription from that
page, then point a client at one of the base URLs it shows. The dashboard also
generates copyable Codex CLI, Claude Code and OpenCode launch commands for the
selected live model.

```sh
docker compose up --build   # publishes 127.0.0.1:8787 only
```

## Use it

Both formats reach both subscriptions. The request's `model` decides which:
ask an Anthropic endpoint for a Codex model and it is translated both ways.

```sh
# Claude Code, or any Anthropic SDK
ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic ANTHROPIC_AUTH_TOKEN=$KEY \
  claude --model claude-opus-4-6

# The dashboard supplies the complete one-line Codex and OpenCode configs,
# including their custom provider declarations and selected model.
```

| Mount | Routes |
| --- | --- |
| `/openai/v1` | `responses` (stateless), `chat/completions`, `models` (`?q=` search), `models/count` |
| `/anthropic` | `v1/messages`, `v1/messages/count_tokens`, `v1/models` |
| `/v1` | alias of `/openai/v1` |

Streaming, images, function tools, web search, parallel calls, reasoning effort
and token usage work on both. The Responses route carries reasoning as opaque
items, so a client that resends its complete `input` continues tool loops
without proxy state. It rejects `store: true` and `previous_response_id`; use
`store: false`. The key is accepted as `Authorization: Bearer` or `x-api-key`
on every mount — it is the proxy's own key, not a vendor's.

Claude thinking uses `display: "summarized"` by default, so readable deltas and
their replay signature survive tool loops. A client can explicitly request
`display: "omitted"`; if the subscription edge withholds the signed text, that
turn continues without its thinking rather than replaying a signature the proxy
cannot verify against the missing text.

Model ids, context and output limits, input modalities, thinking support and
effort names come from the authenticated upstream catalogs at runtime. Codex's
transport enum is intersected with its app-server catalog using a validation-only
probe, so newly added tiers appear automatically and catalog-only tiers do not.
Only currently catalogued models are routable; the proxy does not guess a
provider from a model-name prefix or keep a stale built-in model list.

The dashboard also serves `GET /api/status` and the `POST /api/<provider>/login`
family it uses to manage sign-ins. `POST /api/<provider>/accounts` adds or
removes slots; a provider can have only one unsigned slot, and removal requires
signing out first. Login, pasted-code and logout bodies carry an internal slot
id such as `{"account":"2"}`. Missing Claude usage bars are warmed once when
the dashboard opens; later probes refresh them every minute.

### Example

```sh
curl -N http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.6-sol",
    "stream": true,
    "messages": [{"role": "user", "content": "What changed in Codex this week?"}],
    "tools": [{"type": "openrouter:web_search",
               "parameters": {"search_context_size": "low"}}]
  }'
```

`openrouter:web_search` is only OpenRouter's wire name for server-side search;
the proxy maps it to the upstream's own tool. No OpenRouter account is involved
and the search runs inside the subscription serving the request.

## Behaviour worth knowing

Function calls come back to your client to execute; only web search runs
upstream, with `url_citation` annotations and `server_tool_use` usage in the
stream.

`max_tokens` is a hint on Codex, which controls its own output length, but the
Messages API requires a limit — so Claude receives the requested value or the
model's maximum, and that limit also bounds any thinking budget. A small limit
therefore caps reasoning too.

`count_tokens` is exact for `claude-*` models and returns `404` for Codex ones,
whose upstream cannot count. A client that gets the 404 falls back to its own
estimate knowing it is one, rather than trusting a number the proxy invented.

Structured outputs cross every pairing: Chat Completions `response_format`,
Responses `text.format` and Messages `output_config.format` all carry one JSON
schema to either upstream. Only the wrapper differs, so a schema written for
one client constrains the other's model too; Messages names no schema, so the
schema's own `title` becomes the label Responses requires. Formats an upstream
cannot enforce — a bare `json_object` against Claude — and unsupported sampling
parameters fail explicitly instead of being silently ignored. Pricing,
user-configurable routing and spend history are absent on purpose: neither
subscription exposes anything equivalent.

Each account keeps its own credentials, usage windows and token ledger. A
downstream `X-Session-Id` stays on a stable account for prompt-cache locality;
Claude Code's native `X-Claude-Code-Session-Id` works the same way. Requests
without either one round-robin. If an account returns 429 before emitting any
output, it is cooled locally for five minutes and the same request tries the
next signed-in account. A terminal authentication failure follows the same safe
pre-output failover path, marks that slot as requiring reauthentication and
cools it for one minute. Catalog refreshes rotate between accounts and skip
cooled stale slots, so one bad login cannot hide a provider's models; the short
cooldown periodically retries them so a recovered login rejoins automatically.
The proxy never retries after output starts, because that could duplicate a
partial answer. Claude tokens are refreshed by the proxy itself, so they do not
contend with Claude Code logins on other machines.

## Configuration

`~/.config/llm-local-proxy/config.toml`, or `--config PATH`. Must be readable
only by its owner (`chmod 600`).

```toml
host = "127.0.0.1"
port = 8787
api_key = "long-random-local-secret"
codex_home = "~/.codex"
codex_binary = "codex"
request_timeout = 600
```

Account slots are added and removed live from the dashboard and have no
configured count or artificial cap, but each provider permits only one unsigned
slot at a time. Every Codex login has an isolated home at
`codex_home/accounts/<n>`. Provider credentials, usage, the slot registry and
token ledgers use the single layout `accounts/<provider>` inside the config
directory. The dashboard identifies signed-in slots by the email returned by
Codex or Claude; numeric slot IDs remain internal. A slot must be signed out
before removal. On first startup, canonical credential files are indexed into
the registry so existing logins survive this breaking configuration change.
There is no old-path fallback. Before upgrading an older single-account
install, stop the proxy and move the account-1 files as follows (using your
configured roots):

| Previous path | Current path |
| --- | --- |
| `<codex_home>/auth.json` | `<codex_home>/accounts/1/auth.json` |
| `<config>/claude-credentials.json` | `<config>/accounts/claude/1/credentials.json` |
| `<config>/claude-usage.json` | `<config>/accounts/claude/1/usage.json` |
| `<config>/claude-tokens.json` | `<config>/accounts/claude/1/tokens.json` |
| `<config>/codex-tokens.json` | `<config>/accounts/codex/1/tokens.json` |

An empty `api_key` disables authentication; otherwise it must be at least 24
characters. Native installs refuse non-loopback bind addresses. Under Docker,
no-auth mode is safe only while the port stays published to `127.0.0.1` as
supplied — never publish it on all interfaces.

## Development

```sh
uv sync
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv run ruff format --check src tests
```

[docs/architecture.md](docs/architecture.md) describes the dialect, provider
and auth axes and the wire contract each endpoint honours.
[docs/specs.md](docs/specs.md) covers the OpenAI and Anthropic specifications
those contracts are tested against; `scripts/refresh-specs.sh` fetches them.

Codex app-server owns login, refresh, models and account limits. Keeping the
binary avoids reimplementing undocumented ChatGPT authentication, at the cost
of a private transport that may change with Codex.

## Disclaimer

An unofficial, independent wrapper around the publicly distributed
[Codex CLI](https://github.com/openai/codex) and the subscription endpoints
used by Anthropic's Claude Code. Not affiliated with, endorsed by, or supported
by OpenAI or Anthropic.

It runs entirely on your machine against your own authenticated accounts. No
keys are intercepted, no authentication is bypassed and no credentials leave
your device: Codex login is delegated to the official binary, and Claude tokens
come from the same OAuth flow the first-party client uses.

It does speak two undocumented interfaces — the Codex app-server JSON-RPC
surface and the Claude subscription Messages transport, including the client
identifiers and beta headers that mark first-party traffic. These have no public
specification, may break without notice, and their use may fall outside what
your subscription permits. Review the OpenAI and Anthropic terms yourself, and
use the paid APIs when a supported integration is required. No warranty; use at
your own risk.
