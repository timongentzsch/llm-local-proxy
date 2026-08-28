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
fragment. Sign in to either subscription from that page, then point a client at
one of the base URLs it shows.

```sh
docker compose up --build   # publishes 127.0.0.1:8787 only
```

## Use it

Both formats reach both subscriptions. The request's `model` decides which:
ask an Anthropic endpoint for a Codex model and it is translated both ways.

```sh
# Claude Code, or any Anthropic SDK
ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic ANTHROPIC_API_KEY=$KEY claude

# Any OpenAI-compatible client
OPENAI_BASE_URL=http://127.0.0.1:8787/openai/v1 OPENAI_API_KEY=$KEY
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

One limit is worth knowing: the Claude subscription edge signs thinking blocks
whose text it never streams. A signature covering text the proxy never saw
cannot be replayed, so those turns continue without their thinking rather than
being rejected upstream.

The dashboard also serves `GET /api/status` and the `POST /api/<provider>/login`
family it uses to sign in and out.

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

Unsupported sampling and structured-output parameters fail explicitly instead
of being silently ignored. Pricing, provider routing and spend history are
absent on purpose: neither subscription exposes anything equivalent.

Each subscription keeps its own credentials, usage windows and rate limits.
Claude tokens are refreshed by the proxy itself, so it does not contend with a
Claude Code login on another machine.

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
