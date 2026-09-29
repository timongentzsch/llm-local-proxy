"""Claude subscription transport: the Messages API behind the proxy."""

from __future__ import annotations

import http.client
import json
import math
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ...errors import UpstreamError
from ...ledger import TokenLedger, track_usage
from ...status import Limit, window_label
from .. import transport
from ..limits import LimitsStore
from .auth import OAUTH_BETA, ClaudeAuth, ClaudeAuthError
from .usage import ClaudeUsage

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"
MODELS_URL = "https://api.anthropic.com/v1/models"
# Subscription utilization, the endpoint Claude usage trackers read.
# Undocumented; needs the user:profile scope the login requests.
USAGE_URL = "https://api.anthropic.com/api/oauth/usage"
# A usage read feeds status and routing, so it must fail fast rather than hang.
USAGE_TIMEOUT = 10
ANTHROPIC_VERSION = "2023-06-01"
# Beta the subscription edge uses to recognize Claude Code traffic; requests
# without it (and the system marker) are billed against the API pool and 429.
CLAUDE_CODE_BETA = "claude-code-20250219"
# The subscription accepts the Claude Code client user agent.
USER_AGENT = "claude-cli/2.1.251 (external, sdk-cli)"


class ClaudeUpstreamError(UpstreamError):
    pass


def _message_events(response: Any) -> Iterator[dict[str, Any]]:
    """Turn a zero-token non-streaming Message into the normal event lifecycle."""
    try:
        raw = response.read().decode("utf-8", "replace")
    finally:
        response.close()
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ClaudeUpstreamError(
            502, "Claude prewarm response is not valid JSON"
        ) from error
    if not isinstance(message, dict) or message.get("type") != "message":
        raise ClaudeUpstreamError(502, "Claude prewarm response is malformed")
    usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
    yield {"type": "message_start", "message": message}
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": message.get("stop_reason") or "max_tokens"},
        "usage": usage,
    }
    yield {"type": "message_stop"}


def _limits(value: Any) -> tuple[Limit, ...]:
    """Dashboard bars from a usage response; unknown fields are ignored."""
    if not isinstance(value, dict):
        raise ClaudeUpstreamError(502, "Claude usage is malformed")
    items: list[Limit] = []
    for key, window in (("five_hour", "5h"), ("seven_day", "7d")):
        entry = value.get(key)
        if isinstance(entry, dict) and _is_number(entry.get("utilization")):
            items.append(
                Limit(
                    label=window_label(window),
                    used_percent=float(entry["utilization"]),
                    resets_at=entry.get("resets_at"),
                )
            )
    # Model-scoped weekly caps (e.g. one model's own allowance) appear only
    # in this list; the unscoped session and weekly entries repeat the above.
    entries = value.get("limits")
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, dict) or entry.get("kind") != "weekly_scoped":
            continue
        scope = entry.get("scope") if isinstance(entry.get("scope"), dict) else {}
        model = scope.get("model") if isinstance(scope.get("model"), dict) else {}
        name = model.get("display_name") or model.get("id")
        if name and _is_number(entry.get("percent")):
            items.append(
                Limit(
                    label=f"{name} {window_label('7d')}",
                    used_percent=float(entry["percent"]),
                    resets_at=entry.get("resets_at"),
                    model=str(name),
                )
            )
    return tuple(items)


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


class ClaudeUpstream:
    """The only module coupled to Claude's private subscription transport."""

    def __init__(
        self,
        auth: ClaudeAuth,
        timeout: int,
        tokens_path: Path | None = None,
    ):
        self.auth = auth
        self.timeout = timeout
        # Read-only metadata: it costs no tokens and cannot open a window, and
        # it covers the whole subscription, other clients included.
        self.usage = LimitsStore(
            "claude",
            lambda: _limits(self._get(USAGE_URL, "usage", timeout=USAGE_TIMEOUT)),
        )
        self.ledger = TokenLedger(tokens_path)
        self._opener = transport.opener()

    def models(self) -> list[dict[str, Any]]:
        value = self._get(MODELS_URL, "model list")
        items = value.get("data") if isinstance(value, dict) else None
        if not isinstance(items, list):
            raise ClaudeUpstreamError(502, "Claude model list is malformed")
        return [model for model in map(_normalize_model, items) if model]

    def events(
        self, body: dict[str, Any], betas: tuple[str, ...] = ()
    ) -> Iterator[dict[str, Any]]:
        betas_header = ",".join((CLAUDE_CODE_BETA, OAUTH_BETA, *betas))
        prewarm = body.get("max_tokens") == 0
        outgoing = {**body, "stream": False} if prewarm else body
        try:
            response = self._open(outgoing, betas_header, refresh=False)
        except ClaudeUpstreamError as error:
            # Reported only once the failure is final: the budget retry below
            # recovers on its own, and dumping the turn for it would name a
            # fault that never reached the caller.
            if not _thinking_rejected(error, outgoing):
                _report_block_shape(error, outgoing)
                raise
            previous = outgoing["thinking"]
            display = previous.get("display")
            adaptive = {
                "type": "adaptive",
                **({"display": display} if display is not None else {}),
            }
            outgoing = {**outgoing, "thinking": adaptive}
            try:
                response = self._open(outgoing, betas_header, refresh=False)
            except ClaudeUpstreamError as retried:
                _report_block_shape(retried, outgoing)
                raise
        events = (
            _message_events(response)
            if prewarm
            else transport.read_events(response, {"message_stop", "error"})
        )
        yield from self._tracked(events)

    def _tracked(self, events: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        return track_usage(events, self.ledger, ClaudeUsage().read, {"message_stop"})

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

    def _get(
        self,
        url: str,
        what: str,
        refresh: bool = False,
        timeout: float | None = None,
    ) -> Any:
        """GET a JSON document with the subscription's OAuth credentials."""
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Authorization": f"Bearer {self._token(refresh)}",
                "Accept": "application/json",
                "anthropic-version": ANTHROPIC_VERSION,
                "anthropic-beta": OAUTH_BETA,
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with self._opener.open(
                request, timeout=timeout or self.timeout
            ) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            if error.code == 401 and not refresh:
                error.close()
                return self._get(url, what, refresh=True, timeout=timeout)
            raise _upstream_error(error) from error
        except urllib.error.URLError as error:
            raise ClaudeUpstreamError(502, str(error.reason)) from error
        except (OSError, http.client.HTTPException) as error:
            # A timeout or a cut connection while the body is still arriving.
            raise ClaudeUpstreamError(502, f"Claude {what} is unreadable") from error
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            raise ClaudeUpstreamError(
                502, f"Claude {what} is not valid JSON"
            ) from error

    def _token(self, refresh: bool) -> str:
        try:
            return self.auth.access_token(force_refresh=refresh)
        except ClaudeAuthError as error:
            raise ClaudeUpstreamError(
                error.status,
                str(error),
                account_unavailable=error.status in {400, 401, 403},
            ) from error

    def _open(
        self,
        body: dict[str, Any],
        betas: str,
        refresh: bool,
        url: str = MESSAGES_URL,
    ):
        token = self._token(refresh)
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
            return self._opener.open(request, timeout=self.timeout)
        except urllib.error.HTTPError as error:
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
    max_input = item.get("max_input_tokens")
    if isinstance(max_input, int) and not isinstance(max_input, bool) and max_input > 0:
        value["context_length"] = max_input
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        value["modalities"] = [
            "text",
            *(["image"] if _supported(capabilities.get("image_input")) else []),
        ]
        efforts = capabilities.get("effort")
        if isinstance(efforts, dict):
            supported = [
                str(name) for name, support in efforts.items() if _supported(support)
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
    # A 403 naming a scope is the credential, not the request: the same body
    # succeeds on a login that holds inference access.
    scope_denied = error.code == 403 and "scope" in message.casefold()
    return ClaudeUpstreamError(
        error.code,
        message,
        account_unavailable=error.code == 401 or scope_denied,
    )


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
