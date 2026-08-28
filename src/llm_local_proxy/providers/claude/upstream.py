"""Claude subscription transport: the Messages API behind the proxy."""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ...atomic import atomic_write_json
from ...ledger import TokenLedger
from ...status import Limit, window_label
from .. import transport
from .auth import OAUTH_BETA, ClaudeAuth, ClaudeAuthError
from .subscription import CLAUDE_CODE_SYSTEM_MARKER

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"
MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
# Beta the subscription edge uses to recognize Claude Code traffic; requests
# without it (and the system marker) are billed against the API pool and 429.
CLAUDE_CODE_BETA = "claude-code-20250219"
# The subscription accepts the Claude Code client user agent.
USER_AGENT = "claude-cli/2.1.235 (external, sdk-cli)"


class ClaudeUpstreamError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class UsageStore:
    """Latest unified rate-limit state reported by the subscription edge.

    Every /v1/messages response carries anthropic-ratelimit-unified-* headers
    (5h/7d utilization, resets, overage status); /v1/usage is 404, so these
    are the only live usage numbers. Kept in memory and mirrored to disk so
    the dashboard survives a proxy restart.
    """

    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._value = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path:
            return {}
        try:
            value = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def capture(headers: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not headers:
            return None
        captured = {
            key: value
            for key, value in headers.items()
            if key.lower().startswith("anthropic-ratelimit-")
        }
        if not captured:
            return None
        return {"updated_at": int(time.time()), **captured}

    def update(self, headers: Mapping[str, Any] | None) -> None:
        value = self.capture(headers)
        if value is None:
            return
        with self._lock:
            self._value = value
            if self.path:
                atomic_write_json(self.path, value)

    def get(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._value) if self._value else None

    def limits(self) -> tuple[Limit, ...]:
        """The unified utilization headers as dashboard bars."""
        value = self.get() or {}
        items: list[Limit] = []
        for key, raw in value.items():
            if not key.endswith("-utilization"):
                continue
            try:
                used = float(raw)
            except (TypeError, ValueError):
                continue
            prefix = key[: -len("-utilization")]
            name = prefix.removeprefix("anthropic-ratelimit-unified-")
            items.append(
                Limit(
                    label=window_label(name),
                    used_percent=used * 100,
                    resets_at=value.get(f"{prefix}-reset"),
                )
            )
        return tuple(items)

    def updated_at(self) -> float | None:
        value = self.get() or {}
        stamp = value.get("updated_at")
        return float(stamp) if isinstance(stamp, (int, float)) else None


class ClaudeUpstream:
    """The only module coupled to Claude's private subscription transport."""

    def __init__(
        self,
        auth: ClaudeAuth,
        timeout: int,
        usage_path: Path | None = None,
        tokens_path: Path | None = None,
    ):
        self.auth = auth
        self.timeout = timeout
        self.usage = UsageStore(usage_path)
        self.ledger = TokenLedger(tokens_path)
        self._opener = transport.opener()

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
                self.usage.update(response.headers)
        except urllib.error.HTTPError as error:
            self.usage.update(error.headers)
            if error.code == 401 and not refresh:
                error.close()
                return self._models(refresh=True)
            raise _upstream_error(error) from error
        except urllib.error.URLError as error:
            raise ClaudeUpstreamError(502, str(error.reason)) from error
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ClaudeUpstreamError(
                502, "Claude model list is not valid JSON"
            ) from error
        items = value.get("data") if isinstance(value, dict) else None
        if not isinstance(items, list):
            raise ClaudeUpstreamError(502, "Claude model list is malformed")
        return [model for model in map(_normalize_model, items) if model]

    def events(
        self, body: dict[str, Any], betas: tuple[str, ...] = ()
    ) -> Iterator[dict[str, Any]]:
        betas_header = ",".join((CLAUDE_CODE_BETA, OAUTH_BETA, *betas))
        try:
            response = self._open(body, betas_header, refresh=False)
        except ClaudeUpstreamError as error:
            # Reported only once the failure is final: the budget retry below
            # recovers on its own, and dumping the turn for it would name a
            # fault that never reached the caller.
            if not _thinking_rejected(error, body):
                _report_block_shape(error, body)
                raise
            body = {**body, "thinking": {"type": "adaptive"}}
            try:
                response = self._open(body, betas_header, refresh=False)
            except ClaudeUpstreamError as retried:
                _report_block_shape(retried, body)
                raise
        yield from self._tracked(transport.read_events(response))

    @staticmethod
    def _token_usage(event: dict[str, Any]) -> dict[str, int] | None:
        """Hard per-request token numbers carried by the SSE events."""
        kind = event.get("type")
        if kind == "message_start":
            usage = (event.get("message") or {}).get("usage") or {}
            return {
                "input_tokens": usage.get("input_tokens") or 0,
                "output_tokens": 0,
                "cache_read": usage.get("cache_read_input_tokens") or 0,
                "cache_write": usage.get("cache_creation_input_tokens") or 0,
            }
        if kind == "message_delta":
            usage = event.get("usage") or {}
            return {
                "input_tokens": 0,
                "output_tokens": usage.get("output_tokens") or 0,
                "cache_read": usage.get("cache_read_input_tokens") or 0,
                "cache_write": usage.get("cache_creation_input_tokens") or 0,
            }
        return None

    def _tracked(self, events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield events unchanged while accumulating one request's usage.

        Usage arrives in message_start (input + cache) and possibly several
        message_delta events (cumulative output + later cache updates), so the
        record is only finalised at message_stop. A stream that is interrupted
        before message_stop (client disconnect, error) is not recorded.
        """
        pending: dict[str, int] | None = None
        for event in events:
            yield event
            if event.get("type") == "message_stop" and pending is not None:
                self.ledger.add(**pending)
                pending = None
                continue
            usage = self._token_usage(event)
            if usage is None:
                continue
            if pending is None:
                pending = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                }
            for key, value in usage.items():
                pending[key] = max(pending[key], value)

    def ping_usage(self) -> dict[str, Any] | None:
        """A 1-token call whose only job is to make the edge report usage."""
        body = {
            "model": "claude-haiku-4-5",
            "max_tokens": 1,
            "system": [{"type": "text", "text": CLAUDE_CODE_SYSTEM_MARKER}],
            "messages": [{"role": "user", "content": "usage check"}],
        }
        betas_header = f"{CLAUDE_CODE_BETA},{OAUTH_BETA}"
        response = self._open(body, betas_header, refresh=False)
        try:
            response.read()
        finally:
            response.close()
        return self.usage.get()

    def count_tokens(
        self, body: dict[str, Any], betas: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        """Ask the edge how many input tokens a request would cost.

        Generates nothing and is not billed, which is what makes it worth a
        round trip: only the server knows the exact tokenisation of tool
        schemas and system blocks.
        """
        betas_header = ",".join((CLAUDE_CODE_BETA, OAUTH_BETA, *betas))
        response = self._open(body, betas_header, refresh=False, url=COUNT_TOKENS_URL)
        try:
            value = json.loads(response.read())
        except (json.JSONDecodeError, OSError) as error:
            raise ClaudeUpstreamError(
                502, "Claude token count is unreadable"
            ) from error
        finally:
            response.close()
        tokens = value.get("input_tokens") if isinstance(value, dict) else None
        if not isinstance(tokens, int) or isinstance(tokens, bool):
            raise ClaudeUpstreamError(502, "Claude token count is malformed")
        return {"input_tokens": tokens}

    def _open(
        self,
        body: dict[str, Any],
        betas: str,
        refresh: bool,
        url: str = MESSAGES_URL,
    ):
        try:
            token = self.auth.access_token(force_refresh=refresh)
        except ClaudeAuthError as error:
            raise ClaudeUpstreamError(error.status, str(error)) from error
        request = urllib.request.Request(
            url,
            data=json.dumps(body, separators=(",", ":")).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": betas,
                "User-Agent": USER_AGENT,
                "x-app": "cli",
                "x-client-request-id": uuid.uuid4().hex,
            },
        )
        try:
            response = self._opener.open(request, timeout=self.timeout)
            self.usage.update(response.headers)
            return response
        except urllib.error.HTTPError as error:
            self.usage.update(error.headers)
            if error.code == 401 and not refresh:
                error.close()
                return self._open(body, betas, refresh=True, url=url)
            raise _upstream_error(error) from error
        except urllib.error.URLError as error:
            raise ClaudeUpstreamError(502, str(error.reason)) from error


def _report_block_shape(error: ClaudeUpstreamError, body: dict[str, Any]) -> None:
    """Log the block shape of every turn when upstream refuses a signed one.

    The rejection names a message and a position; this says what the proxy put
    there. Kinds and sizes are enough to place the fault, so the text -- which
    is the conversation -- stays out of the log.
    """
    if error.status != 400 or "thinking" not in str(error).casefold():
        return
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    lines = [f"claude: upstream rejected a signed turn: {error}"]
    for index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        shapes = []
        for block in content:
            kind = block.get("type") if isinstance(block, dict) else "?"
            if kind == "thinking":
                shapes.append(
                    f"thinking(text={len(str(block.get('thinking', '')))},"
                    f"sig={len(str(block.get('signature', '')))})"
                )
            elif kind == "redacted_thinking":
                shapes.append(f"redacted(data={len(str(block.get('data', '')))})")
            else:
                shapes.append(str(kind))
        lines.append(f"  [{index}] {message.get('role')}: {', '.join(shapes)}")
    sys.stderr.write("\n".join(lines) + "\n")


def _thinking_rejected(error: ClaudeUpstreamError, body: dict[str, Any]) -> bool:
    """True when a request was refused solely for its explicit thinking budget."""
    thinking = body.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return False
    return error.status == 400 and "thinking" in str(error).casefold()


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
            if _supported(types.get("adaptive")) and not _supported(
                types.get("enabled")
            ):
                value["thinking"] = "adaptive"
            elif _supported(types.get("enabled")):
                value["thinking"] = "enabled"
    return value


def _upstream_error(error: urllib.error.HTTPError) -> ClaudeUpstreamError:
    message = _error_message(error.read().decode("utf-8", "replace"))
    if error.code == 429 and message in {"", "Error"}:
        message = "Claude usage limit reached; the subscription is rate limited"
    return ClaudeUpstreamError(error.code, message)


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
