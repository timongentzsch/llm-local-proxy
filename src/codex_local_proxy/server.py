from __future__ import annotations

import argparse
import hmac
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from .app_server import AppServer, RpcError
from .config import Config, load
from .protocol import ReasoningCache, RequestError, Translator, build_request
from .upstream import Upstream, UpstreamError


def _api_path(path: str) -> str:
    prefix = "/api/v1/"
    return f"/v1/{path[len(prefix) :]}" if path.startswith(prefix) else path


def _model_info(item: dict[str, Any]) -> dict[str, Any] | None:
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
    return value


class Service:
    def __init__(self, config: Config):
        self.config = config
        self.app = AppServer(config.codex_binary, config.codex_home)
        self.upstream = Upstream(self.app, config.request_timeout)
        self.cache = ReasoningCache()
        self._models: tuple[float, dict[str, Any]] | None = None
        self._models_lock = threading.Lock()

    def models(self) -> dict[str, Any]:
        with self._models_lock:
            if self._models and time.time() - self._models[0] < 60:
                return self._models[1]
            result = self.app.call("model/list", {"limit": 100, "includeHidden": False})
            models = []
            for item in result.get("data", []):
                if not isinstance(item, dict):
                    continue
                model = _model_info(item)
                if model:
                    models.append(model)
            value = {"object": "list", "data": models}
            self._models = (time.time(), value)
            return value

    def status(self) -> dict[str, Any]:
        account = self.app.call("account/read", {"refreshToken": False})
        signed_in = bool(account.get("account"))

        def optional(method: str) -> dict[str, Any] | None:
            if not signed_in:
                return None
            try:
                return self.app.call(method)
            except RpcError:
                return None

        return {
            "account": account.get("account"),
            "rate_limits": optional("account/rateLimits/read"),
            "base_url": self.config.base_url,
        }

    def close(self) -> None:
        self.app.close()


def _handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "codex-local-proxy/0.1"

        def do_GET(self) -> None:
            if not self._valid_host():
                return self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "bad host"})
            parsed = urlparse(self.path)
            path = _api_path(parsed.path)
            if path == "/":
                page = (
                    files("codex_local_proxy")
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
                if path == "/api/login":
                    result = service.app.call(
                        "account/login/start", {"type": "chatgptDeviceCode"}
                    )
                    return self._json(HTTPStatus.OK, result)
                if path == "/api/logout":
                    service.app.call("account/logout")
                    return self._json(HTTPStatus.OK, {"ok": True})
                if path == "/v1/chat/completions":
                    return self._chat(body)
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except RequestError as error:
                self._api_error(HTTPStatus.BAD_REQUEST, str(error))
            except RpcError as error:
                self._api_error(HTTPStatus.BAD_GATEWAY, str(error))
            except UpstreamError as error:
                self._api_error(error.status, str(error))
            except (RuntimeError, OSError, ValueError) as error:
                self._api_error(HTTPStatus.BAD_GATEWAY, str(error))

        def _chat(self, body: dict[str, Any]) -> None:
            session = self.headers.get("X-Session-Id", "")
            request, _ = build_request(body, service.cache, session)
            events = service.upstream.events(request)
            translator = Translator(str(body["model"]), service.cache)
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
                for event in events:
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
    parser = argparse.ArgumentParser(description="Local Codex ChatGPT proxy")
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
        raise SystemExit(f"codex-local-proxy: {error}") from error
    display_host = "127.0.0.1" if config.host in {"0.0.0.0", "::"} else config.host
    fragment = f"#key={quote(config.api_key)}" if config.api_key else ""
    print(f"Codex Local Proxy: http://{display_host}:{config.port}/{fragment}")
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
