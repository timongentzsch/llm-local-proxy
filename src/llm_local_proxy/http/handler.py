"""The HTTP request handler.

Routes are keyed by (dialect, path): the dialect is resolved from the mount
prefix first, and every dialect-shaped thing the response needs — the error
envelope, the stream framing, the header the client authenticates with — comes
from that object rather than from a constant here.
"""

from __future__ import annotations

import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from importlib.resources import files
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..dialects import Dialect, resolve
from ..errors import ProviderError, RequestError
from ..providers import Provider
from ..service import Service
from . import security
from .sse import SseStream, with_heartbeats


def api_path(path: str) -> str:
    """Map the dashboard's /api/v1/... alias onto the real /v1/... route."""
    prefix = "/api/v1/"
    return f"/v1/{path[len(prefix) :]}" if path.startswith(prefix) else path


def _account(body: dict[str, Any]) -> str:
    account = body.get("account")
    if not isinstance(account, str) or not account:
        raise RequestError("account is required")
    return account


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llm-local-proxy/0.1"

        def do_GET(self) -> None:
            if not self._valid_host():
                return self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "bad host"})
            parsed = urlparse(self.path)
            dialect, path = resolve(api_path(parsed.path))
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
                healthy = service.healthy()
                return self._json(
                    HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "ok" if healthy else "unhealthy"},
                )
            if not self._authorized():
                return self._unauthorized(dialect)
            try:
                if path == "/api/status":
                    return self._json(HTTPStatus.OK, service.status())
                if path == "/v1/models":
                    models = service.models()["data"]
                    query = parse_qs(parsed.query).get("q", [""])[0].casefold()
                    if query:
                        models = [
                            model
                            for model in models
                            if query in f"{model['id']} {model['name']}".casefold()
                        ]
                    return self._json(HTTPStatus.OK, dialect.catalog(models))
                if path == "/v1/models/count":
                    return self._json(
                        HTTPStatus.OK,
                        {"data": {"count": len(service.models()["data"])}},
                    )
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except ProviderError as error:
                self._api_error(dialect, error.status, str(error))

        def do_POST(self) -> None:
            if not self._valid_host():
                return self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "bad host"})
            dialect, path = resolve(api_path(urlparse(self.path).path))
            if not self._authorized():
                return self._unauthorized(dialect)
            if not self._same_origin():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "bad origin"})
            try:
                body = self._body()
                provider_route = self._provider_route(path)
                if provider_route:
                    provider, route = provider_route
                    if route == "login":
                        account = _account(body)
                        return self._json(
                            HTTPStatus.OK, provider.auth.login_start(account)
                        )
                    if route == "logout":
                        account = _account(body)
                        provider.auth.logout(account)
                        service.invalidate_models()
                        return self._json(HTTPStatus.OK, {"ok": True})
                    handler = provider.routes.get(route)
                    if handler is None:
                        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return self._json(HTTPStatus.OK, handler(body))
                if path == dialect.chat_route:
                    return self._chat(dialect, body)
                if dialect.responses_route and path == dialect.responses_route:
                    return self._responses(dialect, body)
                if dialect.count_route and path == dialect.count_route:
                    return self._count_tokens(dialect, body)
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except RequestError as error:
                self._api_error(dialect, HTTPStatus.BAD_REQUEST, str(error))
            except ProviderError as error:
                self._api_error(dialect, error.status, str(error))
            except (RuntimeError, OSError, ValueError) as error:
                self._api_error(dialect, HTTPStatus.BAD_GATEWAY, str(error))

        def _provider_route(self, path: str) -> tuple[Provider, str] | None:
            parts = path.split("/")
            if len(parts) == 4 and parts[0] == "" and parts[1] == "api":
                provider = service.provider(parts[2])
                if provider and parts[3]:
                    return provider, parts[3]
            return None

        def _count_tokens(self, dialect: Dialect, body: dict[str, Any]) -> None:
            request = dialect.parse_count(body, self._session_id())
            provider, canonical = self._route(request.model)
            if provider.count_tokens is None:
                # Truthful for a provider whose upstream cannot count: the
                # client falls back to its own estimate knowing it is one.
                return self._api_error(
                    dialect,
                    HTTPStatus.NOT_FOUND,
                    f"{provider.name} cannot count tokens for {canonical}",
                )
            self._json(HTTPStatus.OK, provider.count_tokens(canonical, request))

        def _route(self, model: str) -> tuple[Provider, str]:
            routed = service.route(model)
            if routed is None:
                raise RequestError(f"no provider handles model: {model}")
            return routed

        def _responses(self, dialect: Dialect, body: dict[str, Any]) -> None:
            if dialect.parse_responses is None or dialect.encode_responses is None:
                raise RequestError("Responses API is not supported by this dialect")
            request = dialect.parse_responses(body, self._session_id())
            return self._generate(
                dialect,
                request,
                lambda model, decoder: dialect.encode_responses(
                    model, decoder, request
                ),
            )

        def _chat(self, dialect: Dialect, body: dict[str, Any]) -> None:
            request = dialect.parse(body, self._session_id())
            return self._generate(dialect, request, dialect.encode)

        def _session_id(self) -> str:
            return self.headers.get("X-Session-Id", "") or self.headers.get(
                "X-Claude-Code-Session-Id", ""
            )

        def _generate(self, dialect: Dialect, request: Any, encode: Any) -> None:
            provider, canonical = self._route(request.model)
            events, decoder = provider.chat(canonical, request)
            stream = encode(canonical, decoder)
            if not request.stream:
                for event in events:
                    stream.feed(event)
                return self._json(HTTPStatus.OK, stream.result())

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            sse = SseStream(self.wfile, dialect)
            start = stream.start()
            sse.send(start, dialect.event_name(start))
            try:
                for event in with_heartbeats(events):
                    if event is None:
                        sse.keepalive()
                        continue
                    for chunk in stream.feed(event):
                        sse.send(chunk, dialect.event_name(chunk))
                for chunk in stream.finish():
                    sse.send(chunk, dialect.event_name(chunk))
            except (BrokenPipeError, ConnectionResetError):
                return
            except (RuntimeError, OSError, ValueError) as error:
                try:
                    if hasattr(stream, "error"):
                        failure = stream.error(str(error))
                    else:
                        failure = dialect.error(HTTPStatus.BAD_GATEWAY, str(error))
                    sse.send(failure, dialect.event_name(failure))
                except (BrokenPipeError, ConnectionResetError):
                    return
            try:
                sse.end()
            except (BrokenPipeError, ConnectionResetError):
                pass

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
            return security.authorized(self.headers, service.config.api_key or "")

        def _valid_host(self) -> bool:
            return security.valid_host(self.headers, service.config.host)

        def _same_origin(self) -> bool:
            return security.same_origin(self.headers)

        def _unauthorized(self, dialect: Dialect) -> None:
            self._api_error(dialect, HTTPStatus.UNAUTHORIZED, "invalid local API key")

        def _api_error(self, dialect: Dialect, status: int, message: str) -> None:
            self._json(status, dialect.error(status, message))

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
