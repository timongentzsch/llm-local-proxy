from __future__ import annotations

import argparse
import hmac
import json
import queue
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, quote, urlparse

from ..config import Config, load
from ..protocol import ReasoningCache, RequestError, Translator, build_request
from ..providers import Provider
from ..providers.claude.auth import ClaudeAuth, ClaudeAuthError
from ..providers.claude.protocol import (
    CLAUDE_MODELS,
    ClaudeTranslator,
    build_messages_request,
    claude_model_name,
)
from ..providers.claude.upstream import ClaudeUpstream, ClaudeUpstreamError
from ..providers.codex.app_server import AppServer, RpcError
from ..providers.codex.auth import CodexAuth
from ..providers.codex.upstream import Upstream, UpstreamError
from ..status import ProviderStatus

SSE_HEARTBEAT_SECONDS = 15
_DONE = object()


def _api_path(path: str) -> str:
    prefix = "/api/v1/"
    return f"/v1/{path[len(prefix) :]}" if path.startswith(prefix) else path


def _model_info(
    item: dict[str, Any], context_windows: dict[str, int] | None = None
) -> dict[str, Any] | None:
    model = item.get("model") or item.get("id")
    if not model:
        return None
    raw_modalities = item.get("inputModalities")
    modalities = (
        [str(modality) for modality in raw_modalities]
        if isinstance(raw_modalities, list) and raw_modalities
        else ["text", "image"]
    )
    efforts = [
        effort.get("reasoningEffort")
        for effort in item.get("supportedReasoningEfforts", [])
        if isinstance(effort, dict) and effort.get("reasoningEffort")
    ]
    value = {
        "id": model,
        "canonical_slug": model,
        "object": "model",
        "created": 0,
        "owned_by": "openai",
        "name": item.get("displayName") or model,
        "architecture": {
            "modality": f"{'+'.join(modalities)}->text",
            "input_modalities": modalities,
            "output_modalities": ["text"],
        },
        "supported_parameters": [
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "reasoning_effort",
            "web_search",
        ],
        "default_parameters": (
            {"reasoning_effort": item["defaultReasoningEffort"]}
            if item.get("defaultReasoningEffort")
            else None
        ),
        "per_request_limits": None,
        "is_default": bool(item.get("isDefault")),
        "supported_reasoning_efforts": efforts,
    }
    context = (context_windows or {}).get(str(model), 0)
    if context > 0:
        value["context_length"] = context
    return value


def _claude_model_info(item: dict[str, Any]) -> dict[str, Any]:
    value = {
        "id": item["id"],
        "canonical_slug": item["id"],
        "object": "model",
        "created": int(item.get("created") or 0),
        "owned_by": "anthropic",
        "name": item["name"],
        "architecture": {
            "modality": "text+image->text",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        "supported_parameters": [
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "reasoning_effort",
            "web_search",
            "temperature",
            "top_p",
        ],
        "default_parameters": (
            {"max_tokens": item["max_output_tokens"]}
            if item.get("max_output_tokens")
            else None
        ),
        "per_request_limits": None,
        "is_default": False,
        "supported_reasoning_efforts": item.get("reasoning_efforts")
        or ["low", "medium", "high"],
    }
    context = item.get("context_length")
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        context = next(
            (
                model.get("context_length")
                for model in CLAUDE_MODELS
                if model["id"] == item["id"]
            ),
            None,
        )
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        value["context_length"] = context
    return value


def _with_heartbeats(
    events: Iterator[dict[str, Any]], interval: float = SSE_HEARTBEAT_SECONDS
) -> Iterator[dict[str, Any] | None]:
    items: queue.Queue[dict[str, Any] | Exception | object] = queue.Queue()
    stopped = threading.Event()

    def read() -> None:
        try:
            for event in events:
                if stopped.is_set():
                    break
                items.put(event)
        except Exception as error:  # noqa: BLE001 - cross the thread boundary
            items.put(error)
        finally:
            items.put(_DONE)

    threading.Thread(target=read, daemon=True).start()
    try:
        while True:
            try:
                item = items.get(timeout=interval)
            except queue.Empty:
                yield None
                continue
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield cast(dict[str, Any], item)
    finally:
        stopped.set()


class Service:
    def __init__(self, config: Config):
        self.config = config
        self.app = AppServer(config.codex_binary, config.codex_home)
        config_dir = config.path.parent
        self.upstream = Upstream(
            self.app,
            config.request_timeout,
            tokens_path=config_dir / "codex-tokens.json",
        )
        self.claude_auth = ClaudeAuth(config_dir / "claude-credentials.json")
        self.claude = ClaudeUpstream(
            self.claude_auth,
            config.request_timeout,
            usage_path=config_dir / "claude-usage.json",
            tokens_path=config_dir / "claude-tokens.json",
        )
        self.codex_auth = CodexAuth(self.app)
        self.cache = ReasoningCache()
        self.claude_reasoning = ReasoningCache()
        self.providers = self._providers()
        self._models: tuple[float, dict[str, Any]] | None = None
        self._claude_catalog: tuple[float, list[dict[str, Any]]] | None = None
        self._models_lock = threading.Lock()

    def _providers(self) -> list[Provider]:
        """The wired provider registry; later entries may fall back on any model."""

        def codex_match(model: str) -> str | None:
            # Codex is the fallback transport: it serves everything the more
            # specific providers did not claim.
            return None if claude_model_name(model) else model

        return [
            Provider(
                name="claude",
                auth=self.claude_auth,
                login_flow="paste_code",
                match=claude_model_name,
                chat=self._claude_chat,
                models=self._claude_items,
                status=self._claude_status,
                routes={
                    "code": self._claude_code,
                    "usage": self._claude_usage,
                },
            ),
            Provider(
                name="codex",
                auth=self.codex_auth,
                login_flow="device_code",
                match=codex_match,
                chat=self._codex_chat,
                models=self._codex_models,
                status=self._codex_status,
                routes={},
            ),
        ]

    def provider(self, name: str) -> Provider | None:
        return next((item for item in self.providers if item.name == name), None)

    def route(self, model: str) -> tuple[Provider, str] | None:
        """First provider whose ``match`` claims the model, or None."""
        for provider in self.providers:
            canonical = provider.match(model)
            if canonical is not None:
                return provider, canonical
        return None

    # -- per-provider handlers -------------------------------------------

    def _codex_chat(
        self, canonical: str, body: dict[str, Any], session: str
    ) -> tuple[Iterator[dict[str, Any]], Translator]:
        request, _ = build_request(body, self.cache, session)
        return self.upstream.events(request), Translator(canonical, self.cache)

    def _codex_models(self) -> list[dict[str, Any]]:
        result = self.app.call("model/list", {"limit": 100, "includeHidden": False})
        context_windows = self.app.model_contexts()
        models = []
        for item in result.get("data", []):
            if not isinstance(item, dict):
                continue
            model = _model_info(item, context_windows)
            if model:
                models.append(model)
        return models

    def _claude_chat(
        self, canonical: str, body: dict[str, Any], session: str
    ) -> tuple[Iterator[dict[str, Any]], ClaudeTranslator]:
        if not self.claude_auth.signed_in():
            raise ClaudeAuthError(
                "not signed in to Claude; use the sign in button on the status page"
            )
        request, betas = build_messages_request(
            body,
            canonical,
            max_output=self._claude_capability(canonical, "max_output_tokens"),
            thinking=self._claude_capability(canonical, "thinking"),
            reasoning_cache=self.claude_reasoning,
        )
        events = self.claude.events(request, tuple(betas))
        return events, ClaudeTranslator(canonical, self.claude_reasoning)

    def _claude_code(self, body: dict[str, Any]) -> dict[str, Any]:
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RequestError("code is required")
        result = self.claude_auth.finish(code)
        self.invalidate_models()
        return result

    def _claude_usage(self, body: dict[str, Any]) -> dict[str, Any]:
        return {"usage": self.claude.ping_usage()}

    def _claude_status(self) -> ProviderStatus:
        return replace(
            self.claude_auth.status(),
            limits=self.claude.usage.limits(),
            tokens=self.claude.ledger.windows(),
            updated_at=self.claude.usage.updated_at(),
        )

    def _codex_status(self) -> ProviderStatus:
        return replace(self.codex_auth.status(), tokens=self.upstream.ledger.windows())

    def models(self) -> dict[str, Any]:
        with self._models_lock:
            if self._models and time.time() - self._models[0] < 60:
                return self._models[1]
        data: list[dict[str, Any]] = []
        for provider in self.providers:
            try:
                data.extend(provider.models())
            except (RpcError, ClaudeUpstreamError, OSError, ValueError):
                # One provider being down must not take the whole catalog
                # down with it; degrade just that provider's slice.
                continue
        value = {"object": "list", "data": data}
        with self._models_lock:
            self._models = (time.time(), value)
        return value

    def invalidate_models(self) -> None:
        with self._models_lock:
            self._models = None
            self._claude_catalog = None

    def _load_claude_catalog(self) -> list[dict[str, Any]]:
        with self._models_lock:
            if self._claude_catalog and time.time() - self._claude_catalog[0] < 60:
                return self._claude_catalog[1]
        try:
            items = self.claude.models()
        except ClaudeUpstreamError:
            items = []
        with self._models_lock:
            self._claude_catalog = (time.time(), items)
        return items

    def _claude_items(self) -> list[dict[str, Any]]:
        if self.claude_auth.signed_in():
            live = self._load_claude_catalog()
            if live:
                return [_claude_model_info(item) for item in live]
        return [_claude_model_info(item) for item in CLAUDE_MODELS]

    def _claude_capability(self, model: str, key: str) -> Any:
        if not self.claude_auth.signed_in():
            return None
        for item in self._load_claude_catalog():
            if item.get("id") == model:
                return item.get(key)
        return None

    def status(self) -> dict[str, Any]:
        cards = []
        for provider in self.providers:
            try:
                value = provider.status()
            except (RpcError, ClaudeUpstreamError, OSError, ValueError) as error:
                # A provider that is unreachable degrades to its own card
                # instead of blanking out the healthy providers.
                value = ProviderStatus(error=str(error) or "unavailable")
            cards.append(
                {
                    "name": provider.name,
                    "login_flow": provider.login_flow,
                    "routes": sorted(provider.routes),
                    **value.payload(),
                }
            )
        return {"base_url": self.config.base_url, "providers": cards}

    def close(self) -> None:
        self.app.close()


def _handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llm-local-proxy/0.1"

        def do_GET(self) -> None:
            if not self._valid_host():
                return self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "bad host"})
            parsed = urlparse(self.path)
            path = _api_path(parsed.path)
            if path == "/":
                page = (
                    files("llm_local_proxy")
                    .joinpath("static/index.html")
                    .read_text()
                    .replace(
                        "__AUTH_REQUIRED__",
                        "true" if service.config.api_key else "false",
                    )
                )
                return self._reply(
                    HTTPStatus.OK, page.encode(), "text/html; charset=utf-8"
                )
            if path == "/healthz":
                healthy = service.app.alive()
                return self._json(
                    HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "ok" if healthy else "unhealthy"},
                )
            if not self._authorized():
                return self._unauthorized()
            try:
                if path == "/api/status":
                    return self._json(HTTPStatus.OK, service.status())
                if path == "/v1/models":
                    models = service.models()
                    query = parse_qs(parsed.query).get("q", [""])[0].casefold()
                    if query:
                        models = {
                            **models,
                            "data": [
                                model
                                for model in models["data"]
                                if query in f"{model['id']} {model['name']}".casefold()
                            ],
                        }
                    return self._json(HTTPStatus.OK, models)
                if path == "/v1/models/count":
                    return self._json(
                        HTTPStatus.OK,
                        {"data": {"count": len(service.models()["data"])}},
                    )
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except RpcError as error:
                self._api_error(HTTPStatus.BAD_GATEWAY, str(error))

        def do_POST(self) -> None:
            if not self._valid_host():
                return self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "bad host"})
            if not self._authorized():
                return self._unauthorized()
            path = _api_path(urlparse(self.path).path)
            if not self._same_origin():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "bad origin"})
            try:
                body = self._body()
                provider_route = self._provider_route(path)
                if provider_route:
                    provider, route = provider_route
                    if route == "login":
                        return self._json(HTTPStatus.OK, provider.auth.login_start())
                    if route == "logout":
                        provider.auth.logout()
                        service.invalidate_models()
                        return self._json(HTTPStatus.OK, {"ok": True})
                    handler = provider.routes.get(route)
                    if handler is None:
                        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return self._json(HTTPStatus.OK, handler(body))
                if path == "/v1/chat/completions":
                    return self._chat(body)
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except RequestError as error:
                self._api_error(HTTPStatus.BAD_REQUEST, str(error))
            except RpcError as error:
                self._api_error(HTTPStatus.BAD_GATEWAY, str(error))
            except UpstreamError as error:
                self._api_error(error.status, str(error))
            except ClaudeAuthError as error:
                self._api_error(error.status, str(error))
            except ClaudeUpstreamError as error:
                self._api_error(error.status, str(error))
            except (RuntimeError, OSError, ValueError) as error:
                self._api_error(HTTPStatus.BAD_GATEWAY, str(error))

        def _provider_route(self, path: str) -> tuple[Provider, str] | None:
            parts = path.split("/")
            if len(parts) == 4 and parts[0] == "" and parts[1] == "api":
                provider = service.provider(parts[2])
                if provider and parts[3]:
                    return provider, parts[3]
            return None

        def _chat(self, body: dict[str, Any]) -> None:
            session = self.headers.get("X-Session-Id", "")
            requested = str(body.get("model", ""))
            routed = service.route(requested)
            if routed is None:
                raise RequestError(f"no provider handles model: {requested}")
            canonical = routed[1]
            events, translator = routed[0].chat(canonical, body, session)
            if not body.get("stream", False):
                for event in events:
                    translator.feed(event)
                return self._json(HTTPStatus.OK, translator.result())

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self._sse(translator.start())
            try:
                for event in _with_heartbeats(events):
                    if event is None:
                        self._sse_comment("keepalive")
                        continue
                    for chunk in translator.feed(event):
                        self._sse(chunk)
                for chunk in translator.finish():
                    self._sse(chunk)
            except (BrokenPipeError, ConnectionResetError):
                return
            except (RuntimeError, OSError, ValueError) as error:
                try:
                    self._sse({"error": {"message": str(error), "type": "proxy_error"}})
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _sse(self, value: dict[str, Any]) -> None:
            data = json.dumps(value, separators=(",", ":")).encode()
            self.wfile.write(b"data: " + data + b"\n\n")
            self.wfile.flush()

        def _sse_comment(self, value: str) -> None:
            self.wfile.write(f": {value}\n\n".encode())
            self.wfile.flush()

        def _body(self) -> dict[str, Any]:
            if self.headers.get("Transfer-Encoding"):
                raise RequestError(
                    "chunked request bodies are not supported; send Content-Length"
                )
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise RequestError("invalid Content-Length") from error
            if size <= 0 or size > 32 * 1024 * 1024:
                raise RequestError("request body must be between 1 byte and 32 MiB")
            try:
                value = json.loads(self.rfile.read(size))
            except json.JSONDecodeError as error:
                raise RequestError("request body is not valid JSON") from error
            if not isinstance(value, dict):
                raise RequestError("request body must be an object")
            return value

        def _authorized(self) -> bool:
            if not service.config.api_key:
                return True
            scheme, _, token = self.headers.get("Authorization", "").partition(" ")
            return scheme.lower() == "bearer" and hmac.compare_digest(
                token.encode("utf-8"), service.config.api_key.encode("utf-8")
            )

        def _valid_host(self) -> bool:
            host, _ = self._request_host()
            host = host.casefold()
            return host in {"127.0.0.1", "::1", "localhost", service.config.host}

        def _request_host(self) -> tuple[str, int]:
            value = self.headers.get("Host", "")
            if value.startswith("["):
                host = value[1 : value.find("]")] if "]" in value else ""
                suffix = value[value.find("]") + 1 :] if "]" in value else ""
                port = suffix[1:] if suffix.startswith(":") else ""
            else:
                host, separator, port = value.rpartition(":")
                if not separator:
                    host, port = value, ""
            try:
                return host, int(port) if port else 80
            except ValueError:
                return "", 0

        def _same_origin(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True
            parsed = urlparse(origin)
            _, request_port = self._request_host()
            return (
                parsed.hostname in {"127.0.0.1", "::1", "localhost"}
                and (parsed.port or 80) == request_port
            )

        def _unauthorized(self) -> None:
            self._api_error(HTTPStatus.UNAUTHORIZED, "invalid local API key")

        def _api_error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": "proxy_error"}})

        def _json(self, status: int, value: Any) -> None:
            self._reply(
                status,
                json.dumps(value, separators=(",", ":")).encode(),
                "application/json",
            )

        def _reply(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            sys.stderr.write(f"{self.address_string()} {format % args}\n")

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local proxy for Codex and Claude subscriptions"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--show-config", action="store_true", help="print the config path and exit"
    )
    args = parser.parse_args()
    service = None
    try:
        config = load(args.config)
        if args.show_config:
            print(config.path)
            return
        service = Service(config)
        server = Server((config.host, config.port), _handler(service))
    except (OSError, ValueError, RpcError) as error:
        if service:
            service.close()
        raise SystemExit(f"llm-local-proxy: {error}") from error
    display_host = "127.0.0.1" if config.host in {"0.0.0.0", "::"} else config.host
    fragment = f"#key={quote(config.api_key)}" if config.api_key else ""
    print(f"LLM Local Proxy: http://{display_host}:{config.port}/{fragment}")
    print(f"Config: {config.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
