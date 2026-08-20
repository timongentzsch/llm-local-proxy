from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...ledger import TokenLedger
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

    def __init__(self, app: AppServer, timeout: int, tokens_path: Path | None = None):
        self.app = app
        self.timeout = timeout
        self.ledger = TokenLedger(tokens_path)
        self._opener = urllib.request.build_opener(_NoRedirect)

    def events(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        response = self._open(body, refresh=False)
        return self._tracked(self._events(response))

    def _tracked(self, events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield events unchanged, tallying the final usage of each request."""
        for event in events:
            if event.get("type") == "response.completed":
                usage = (event.get("response") or {}).get("usage") or {}
                details = usage.get("input_tokens_details") or {}
                record = {
                    "input_tokens": usage.get("input_tokens") or 0,
                    "output_tokens": usage.get("output_tokens") or 0,
                    "cache_read": details.get("cached_tokens") or 0,
                    "cache_write": details.get("cache_write_tokens") or 0,
                }
                if any(record.values()):
                    self.ledger.add(**record)
            yield event

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
                "User-Agent": "llm-local-proxy/0.1.0",
                "originator": "llm_local_proxy",
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
