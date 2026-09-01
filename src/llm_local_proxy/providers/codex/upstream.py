from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ...errors import ProviderError
from ...ledger import TokenLedger
from .. import transport
from .app_server import AppServer, RpcError

RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


class UpstreamError(ProviderError):
    def __init__(self, status: int, message: str, *, account_unavailable: bool = False):
        super().__init__(message)
        self.status = status
        self.account_unavailable = account_unavailable


class Upstream:
    """The only module coupled to ChatGPT's private Codex transport."""

    def __init__(self, app: AppServer, timeout: int, tokens_path: Path | None = None):
        self.app = app
        self.timeout = timeout
        self.ledger = TokenLedger(tokens_path, input_includes_cache=True)
        self._opener = transport.opener()

    def events(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        response = self._open(body, refresh=False)
        return self._tracked(transport.read_events(response))

    def reasoning_efforts(self, model: str) -> set[str] | None:
        """Discover the transport enum without running a generation."""
        body = {
            "model": model,
            "input": [],
            "store": False,
            "stream": True,
            "reasoning": {"effort": "__probe__"},
        }
        try:
            response = self._open(body, refresh=False)
        except UpstreamError as error:
            if error.account_unavailable:
                raise
            return _effort_values(str(error)) if error.status == 400 else None
        response.close()
        return None

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

    def _open(self, body: dict[str, Any], refresh: bool):
        try:
            access, account = self.app.token(force_refresh=refresh)
        except RpcError as error:
            raise UpstreamError(401, str(error), account_unavailable=True) from error
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
            raise UpstreamError(
                error.code,
                message,
                account_unavailable=error.code == 401,
            ) from error
        except urllib.error.URLError as error:
            raise UpstreamError(502, str(error.reason)) from error


def _effort_values(message: str) -> set[str] | None:
    """Parse the enum returned for an invalid reasoning effort probe."""
    _, marker, values = message.partition("Supported values are:")
    if not marker:
        return None
    found = set(re.findall(r"['\"]([a-zA-Z0-9_-]+)['\"]", values))
    return found or None
