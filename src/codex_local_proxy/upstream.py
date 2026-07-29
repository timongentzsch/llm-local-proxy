from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .app_server import AppServer, RpcError

RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


class UpstreamError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Upstream:
    """The only module coupled to ChatGPT's private Codex transport."""

    def __init__(self, app: AppServer, timeout: int):
        self.app = app
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)

    def events(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        response = self._open(body, refresh=False)
        return self._events(response)

    @staticmethod
    def _events(response) -> Iterator[dict[str, Any]]:
        try:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                if isinstance(event, dict):
                    yield event
        finally:
            response.close()

    def _open(self, body: dict[str, Any], refresh: bool):
        try:
            access, account = self.app.token(force_refresh=refresh)
        except RpcError as error:
            raise UpstreamError(401, str(error)) from error
        request = urllib.request.Request(
            RESPONSES_URL,
            data=json.dumps(body, separators=(",", ":")).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {access}",
                "ChatGPT-Account-ID": account,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": "codex-local-proxy/0.1.0",
                "originator": "codex_local_proxy",
            },
        )
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            if error.code == 401 and not refresh:
                error.close()
                return self._open(body, refresh=True)
            raw = error.read().decode("utf-8", "replace")
            try:
                value = json.loads(raw)
                detail = value.get("error", value)
                message = (
                    detail.get("message", str(detail))
                    if isinstance(detail, dict)
                    else str(detail)
                )
            except json.JSONDecodeError:
                message = raw or error.reason
            raise UpstreamError(error.code, message) from error
        except urllib.error.URLError as error:
            raise UpstreamError(502, str(error.reason)) from error
