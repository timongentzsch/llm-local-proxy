"""Claude subscription transport: the Messages API behind the proxy."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from .claude_auth import OAUTH_BETA, ClaudeAuth, ClaudeAuthError

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
# The subscription accepts the Claude Code client user agent.
USER_AGENT = "claude-cli/2.1.235 (external, cli)"


class ClaudeUpstreamError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ClaudeUpstream:
    """The only module coupled to Claude's private subscription transport."""

    def __init__(self, auth: ClaudeAuth, timeout: int):
        self.auth = auth
        self.timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirect)

    def models(self) -> list[dict[str, Any]]:
        return self._models(refresh=False)

    def _models(self, refresh: bool) -> list[dict[str, Any]]:
        try:
            token = self.auth.access_token(force_refresh=refresh)
        except ClaudeAuthError as error:
            raise ClaudeUpstreamError(error.status, str(error)) from error
        request = urllib.request.Request(
            MODELS_URL,
            method="GET",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": OAUTH_BETA,
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            if error.code == 401 and not refresh:
                error.close()
                return self._models(refresh=True)
            body = error.read().decode("utf-8", "replace")
            raise ClaudeUpstreamError(error.code, _error_message(body)) from error
        except urllib.error.URLError as error:
            raise ClaudeUpstreamError(502, str(error.reason)) from error
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ClaudeUpstreamError(502, "Claude model list is not valid JSON") from error
        items = value.get("data") if isinstance(value, dict) else None
        if not isinstance(items, list):
            raise ClaudeUpstreamError(502, "Claude model list is malformed")
        return [model for model in map(_normalize_model, items) if model]

    def events(self, body: dict[str, Any], betas: tuple[str, ...] = ()) -> Iterator[dict[str, Any]]:
        betas_header = ",".join((OAUTH_BETA, *betas))
        response = self._open(body, betas_header, refresh=False)
        yield from self._events(response)

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

    def _open(self, body: dict[str, Any], betas: str, refresh: bool):
        try:
            token = self.auth.access_token(force_refresh=refresh)
        except ClaudeAuthError as error:
            raise ClaudeUpstreamError(error.status, str(error)) from error
        request = urllib.request.Request(
            MESSAGES_URL,
            data=json.dumps(body, separators=(",", ":")).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": betas,
                "User-Agent": USER_AGENT,
            },
        )
        try:
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
            if error.code == 401 and not refresh:
                error.close()
                return self._open(body, betas, refresh=True)
            raw = error.read().decode("utf-8", "replace")
            raise ClaudeUpstreamError(error.code, _error_message(raw)) from error
        except urllib.error.URLError as error:
            raise ClaudeUpstreamError(502, str(error.reason)) from error


def _supported(value: Any) -> bool:
    return isinstance(value, dict) and bool(value.get("supported"))


def _normalize_model(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    model_id = item.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    value: dict[str, Any] = {
        "id": model_id,
        "name": str(item.get("display_name") or model_id),
    }
    created = item.get("created_at")
    if isinstance(created, str):
        try:
            value["created"] = int(datetime.fromisoformat(created).timestamp())
        except ValueError:
            pass
    max_tokens = item.get("max_tokens")
    if (
        isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens > 0
    ):
        value["max_output_tokens"] = max_tokens
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        efforts = capabilities.get("effort")
        if isinstance(efforts, dict):
            supported = [
                name
                for name in ("low", "medium", "high", "xhigh", "max")
                if _supported(efforts.get(name))
            ]
            if supported:
                value["reasoning_efforts"] = supported
        thinking = capabilities.get("thinking")
        types = thinking.get("types") if isinstance(thinking, dict) else None
        if isinstance(types, dict):
            if _supported(types.get("adaptive")) and not _supported(types.get("enabled")):
                value["thinking"] = "adaptive"
            elif _supported(types.get("enabled")):
                value["thinking"] = "enabled"
    return value


def _error_message(raw: str) -> str:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw or "Claude response failed"
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("type") or raw)
        return str(value.get("message") or raw)
    return raw
