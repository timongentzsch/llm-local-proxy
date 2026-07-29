# Codex Local Proxy

A small OpenAI-compatible gateway for a Codex ChatGPT session. Your client keeps
the agent and tool loop; this process only translates the protocol.

![Dashboard with example data](docs/dashboard.png)

_Example data; the dashboard reads your local Codex account at runtime._

## Run

Native installs require [uv](https://docs.astral.sh/uv/) and the
[Codex binary](https://github.com/openai/codex). The proxy starts its
`app-server`; Docker includes both.

```sh
uv tool install .
codex-local-proxy
```

Open the URL printed at startup. The first run creates a private config and
random API key; the printed URL carries it in the browser fragment. Sign in,
then copy the base URL and key into your client.

Use `http://127.0.0.1:8787/api/v1` as an OpenRouter-compatible base URL,
the generated key as a bearer token, and any model listed on the status page.

## Example request

This streams a selected Codex model and lets it search when needed:

```sh
curl -N http://127.0.0.1:8787/api/v1/chat/completions \
  -H "Authorization: Bearer $CODEX_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.6-sol",
    "stream": true,
    "messages": [
      {"role": "user", "content": "What changed in Codex this week? Cite sources."}
    ],
    "tools": [
      {
        "type": "openrouter:web_search",
        "parameters": {"search_context_size": "low"}
      }
    ]
  }'
```

The stream uses Chat Completions chunks, OpenRouter-style `url_citation`
annotations, and `server_tool_use.web_search_requests` in the final usage chunk.

## Docker

```sh
docker compose up --build
```

Compose publishes only `127.0.0.1:8787`; config and Codex login state live in
named volumes and survive container restarts. `docker compose down -v` deletes
both volumes.

## API

- `GET /v1/models` with Codex capabilities and `?q=` search
- `GET /v1/models/count`
- `POST /v1/chat/completions`
- streaming, images, function tools, web search, parallel calls, reasoning
  effort, usage

The request's `model` selects the Codex model, as on OpenRouter. Responses
include prompt, completion, cached, and reasoning token counts. The proxy does
not invent per-token pricing, cost, provider routing, or generation history:
ChatGPT subscriptions do not expose truthful equivalents.

## OpenRouter compatibility

Both OpenAI's `/v1/...` and OpenRouter's `/api/v1/...` prefixes work. Supported
OpenRouter-style behavior includes model selection, model metadata/search/count,
streaming Chat Completions, tools, images, reasoning effort, and token usage.
Function calls are returned to the client for execution; only web search runs
upstream, with source annotations and search usage returned in the stream.
Codex controls output length, so `max_tokens` and `max_completion_tokens` are
accepted as compatibility hints but cannot be enforced. Unsupported sampling
and structured-output parameters fail explicitly rather than being ignored.
Pricing, provider preferences/fallbacks, key spend, and generation-cost history
are intentionally absent because Codex does not expose equivalent data.

API routes require the generated bearer key. Set `api_key = ""` to disable
authentication. Native installs always reject non-loopback bind addresses.
With Docker, no-auth mode is safe only while the port remains published to
`127.0.0.1` as supplied. Never publish it on all interfaces. Change the port or
key in:

```toml
host = "127.0.0.1"
port = 8787
api_key = "long-random-local-secret"
codex_home = "~/.codex"
codex_binary = "codex"
request_timeout = 600
```

The default path is `~/.config/codex-local-proxy/config.toml`; pass
`--config PATH` to use another. Config files must be readable only by their
owner (`chmod 600`).

## Development

```sh
uv sync
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv run ruff format --check src tests
```

Codex app-server owns login, refresh, models, and account limits. Keeping the
binary avoids reimplementing undocumented ChatGPT authentication. The small
private transport and credential adapters may change with Codex. Use the public
OpenAI API when a supported third-party integration is required.
