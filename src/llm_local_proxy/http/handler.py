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
from ..dialects.base import Route
from ..errors import ProviderError, RequestError
from ..keys import MASTER
from ..providers import Provider
from ..service import Service, base_urls
from ..streaming import closing_iterator
from . import security
from .sse import SseStream, with_heartbeats


def _describe(error: Exception) -> str:
    """A message for the client; a bug's bare KeyError text says too little."""
    if isinstance(error, (RuntimeError, OSError, ValueError)):
        return str(error)
    return f"internal error: {type(error).__name__}: {error}"


def make_handler(service: Service, public: bool = False):
    """Handlers for the admin listener, or with ``public`` for named keys only.

    The public listener may face a network, so trust comes from the socket:
    it serves only the model API and a key's own reduced dashboard, refuses
    the master key, and never reaches account, key or status routes.
    """

    def _named(token: str) -> str | None:
        try:
            return service.keys.identify(token)
        except (ValueError, TypeError) as error:
            # A damaged keys.json refuses every named key; the master still works.
            sys.stderr.write(f"keys: {error}\n")
            return None

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "llm-local-proxy/0.1"
        # Per socket operation, so a stalled client cannot hold a thread for
        # good; a quiet upstream still streams keepalives well within this.
        timeout = 60

        def do_GET(self) -> None:
            if not public and not self._valid_host():
                return self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "bad host"})
            parsed = urlparse(self.path)
            dialect, path = resolve(parsed.path)
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
            if path == "/favicon.ico":
                # Browsers ask for it unprompted; a 401 here is console noise.
                return self._reply(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            if path == "/healthz" and not public:
                healthy = service.healthy()
                return self._json(
                    HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "ok" if healthy else "unhealthy"},
                )
            caller = self._caller()
            if caller is None:
                return self._unauthorized(dialect)
            try:
                if path == "/api/me":
                    return self._json(HTTPStatus.OK, self._me(caller))
                if path.startswith("/api/") and caller != MASTER:
                    return self._refuse_admin()
                if path == "/api/status":
                    return self._json(HTTPStatus.OK, service.status())
                if path == "/api/keys":
                    return self._json(HTTPStatus.OK, self._keys())
                if path == "/v1/models":
                    params = parse_qs(parsed.query)
                    refresh = params.get("refresh", [""])[0] in {"1", "true"}
                    models = service.models(refresh=refresh)["data"]
                    query = params.get("q", [""])[0].casefold()
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
            # A refusal leaves the body unread, so the connection must not be
            # reused: its bytes would be parsed as the next request.
            self.close_connection = True
            if not public and not self._valid_host():
                return self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "bad host"})
            dialect, path = resolve(urlparse(self.path).path)
            caller = self._caller()
            if caller is None:
                return self._unauthorized(dialect)
            if not self._same_origin():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "bad origin"})
            try:
                body = self._body()
                if path.startswith("/api/") and caller != MASTER:
                    return self._refuse_admin()
                if path == "/api/keys":
                    return self._json(HTTPStatus.OK, self._manage_keys(body))
                provider_route = self._provider_route(path)
                if provider_route:
                    provider, route = provider_route
                    handler = provider.routes.get(route)
                    if handler is None:
                        return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                    return self._json(HTTPStatus.OK, handler(body))
                route = dialect.routes.get(path)
                if route is None:
                    return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                self._serve(dialect, route, body, caller)
            except RequestError as error:
                self._api_error(dialect, HTTPStatus.BAD_REQUEST, str(error))
            except ProviderError as error:
                self._api_error(dialect, error.status, str(error))
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                return
            except Exception as error:  # noqa: BLE001 - always answer the client
                try:
                    self._api_error(dialect, HTTPStatus.BAD_GATEWAY, _describe(error))
                except OSError:
                    return

        def _provider_route(self, path: str) -> tuple[Provider, str] | None:
            parts = path.split("/")
            if len(parts) == 4 and parts[0] == "" and parts[1] == "api":
                provider = service.provider(parts[2])
                if provider and parts[3]:
                    return provider, parts[3]
            return None

        def _route(self, model: str) -> tuple[Provider, str]:
            routed = service.route(model)
            if routed is None:
                raise RequestError(f"no provider handles model: {model}")
            return routed

        def _session_id(self, dialect: Dialect) -> str:
            for name in ("X-Session-Id", *dialect.session_headers):
                if self.headers.get(name):
                    return self.headers[name]
            return ""

        def _serve(
            self, dialect: Dialect, route: Route, body: dict[str, Any], caller: str
        ) -> None:
            request = route.parse(body, self._session_id(dialect))
            request.caller = caller
            provider, canonical = self._route(request.model)
            if route.encode is None:
                if provider.count_tokens is None:
                    # Truthful for a provider whose upstream cannot count: the
                    # client falls back to its own estimate knowing it is one.
                    return self._api_error(
                        dialect,
                        HTTPStatus.NOT_FOUND,
                        f"{provider.name} cannot count tokens for {canonical}",
                    )
                return self._json(
                    HTTPStatus.OK, provider.count_tokens(canonical, request)
                )
            events, decoder = provider.chat(canonical, request)
            stream = route.encode(canonical, decoder, request)
            if not request.stream:
                with closing_iterator(events):
                    for event in events:
                        stream.feed(event)
                return self._json(HTTPStatus.OK, stream.result())

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            sse = SseStream(self.wfile, dialect.keepalive, route.named)
            sse.send(stream.start())
            try:
                with closing_iterator(with_heartbeats(events)) as heartbeat:
                    for event in heartbeat:
                        if event is None:
                            sse.keepalive()
                            continue
                        for chunk in stream.feed(event):
                            sse.send(chunk)
                for chunk in stream.finish():
                    sse.send(chunk)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                # The client left or stopped reading; nothing can reach it.
                return
            except Exception as error:  # noqa: BLE001 - headers are sent; fail in-band
                message = _describe(error)
                try:
                    sse.send(
                        stream.error(message)
                        or dialect.error(HTTPStatus.BAD_GATEWAY, message)
                    )
                except OSError:
                    return
            try:
                sse.end()
            except OSError:
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

        def _caller(self) -> str | None:
            """The name of the request's key; the master key only locally."""
            caller = security.identify(self.headers, service.config.api_key, _named)
            return None if public and caller == MASTER else caller

        def _refuse_admin(self) -> None:
            # The public listener does not have these routes at all.
            if public:
                return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return self._json(
                HTTPStatus.FORBIDDEN, {"error": "requires the master key"}
            )

        def _me(self, caller: str) -> dict[str, Any]:
            """What a key's own dashboard shows: its base URLs and its usage."""
            if public:
                host = self.headers.get("Host", "")
                origin = service.config.public_url or f"http://{host}"
            else:
                origin = service.config.origin
            return {
                "name": caller,
                "role": "master" if caller == MASTER else "user",
                "dialects": base_urls(origin),
                # The master's dashboard reads every key's usage from /api/keys.
                "usage": {} if caller == MASTER else service.usage().get(caller, {}),
            }

        def _keys(self) -> dict[str, Any]:
            return {
                "keys": [
                    {"name": name, "key": key}
                    for name, key in service.keys.all().items()
                ],
                "usage": service.usage(),
                "public_url": service.config.public_url,
                # Where a named key's launch commands point, when it is set.
                "public_dialects": base_urls(service.config.public_url)
                if service.config.public_url
                else [],
            }

        def _manage_keys(self, body: dict[str, Any]) -> dict[str, Any]:
            name = body.get("name")
            if not isinstance(name, str):
                raise RequestError("name is required")
            action = body.get("action")
            if action == "add":
                return {"name": name, "key": service.keys.add(name)}
            if action == "remove":
                service.keys.remove(name)
                return {"ok": True}
            raise RequestError("action must be add or remove")

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
            if self.close_connection:
                self.send_header("Connection", "close")
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
