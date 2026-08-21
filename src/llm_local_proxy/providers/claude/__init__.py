"""The Claude provider: an Anthropic subscription over the Claude Code edge."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from ...errors import RequestError
from ...ir import ChatRequest
from ...status import ProviderStatus
from ..base import Provider, ProviderContext
from ..reasoning import ReasoningCache
from .auth import ClaudeAuth, ClaudeAuthError
from .catalog import CLAUDE_MODELS, claude_model_name, model_info
from .events import ClaudeDecoder
from .request import build
from .upstream import ClaudeUpstream, ClaudeUpstreamError

CATALOG_TTL_SECONDS = 60
#: The request fields /v1/messages/count_tokens accepts, per the pinned spec.
COUNTED_FIELDS = (
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "cache_control",
)


class Claude:
    def __init__(self, context: ProviderContext):
        config = context.config
        self.invalidate = context.invalidate
        self.auth = ClaudeAuth(context.directory / "claude-credentials.json")
        self.upstream = ClaudeUpstream(
            self.auth,
            config.request_timeout,
            usage_path=context.directory / "claude-usage.json",
            tokens_path=context.directory / "claude-tokens.json",
        )
        self.cache = ReasoningCache()
        self._catalog: tuple[float, list[dict[str, Any]]] | None = None
        self._lock = threading.Lock()

    def chat(
        self, canonical: str, request: ChatRequest
    ) -> tuple[Iterator[dict[str, Any]], ClaudeDecoder]:
        if not self.auth.signed_in():
            raise ClaudeAuthError(
                "not signed in to Claude; use the sign in button on the status page"
            )
        body, betas = build(
            request,
            canonical,
            max_output=self._capability(canonical, "max_output_tokens"),
            thinking=self._capability(canonical, "thinking"),
            reasoning_cache=self.cache,
        )
        return self.upstream.events(body, tuple(betas)), ClaudeDecoder(self.cache)

    def count_tokens(self, canonical: str, request: ChatRequest) -> dict[str, Any]:
        if not self.auth.signed_in():
            raise ClaudeAuthError(
                "not signed in to Claude; use the sign in button on the status page"
            )
        body, betas = build(
            request,
            canonical,
            max_output=self._capability(canonical, "max_output_tokens"),
            thinking=self._capability(canonical, "thinking"),
            reasoning_cache=self.cache,
        )
        # Its schema accepts only prompt fields; the rest are rejected.
        counted = {key: body[key] for key in COUNTED_FIELDS if key in body}
        return self.upstream.count_tokens(counted, tuple(betas))

    def models(self) -> list[dict[str, Any]]:
        if self.auth.signed_in():
            live = self._live_catalog()
            if live:
                return [model_info(item) for item in live]
        return [model_info(item) for item in CLAUDE_MODELS]

    def status(self) -> ProviderStatus:
        return replace(
            self.auth.status(),
            limits=self.upstream.usage.limits(),
            tokens=self.upstream.ledger.windows(),
            updated_at=self.upstream.usage.updated_at(),
        )

    def finish_login(self, body: dict[str, Any]) -> dict[str, Any]:
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RequestError("code is required")
        result = self.auth.finish(code)
        self.forget()
        self.invalidate()
        return result

    def usage(self, body: dict[str, Any]) -> dict[str, Any]:
        return {"usage": self.upstream.ping_usage()}

    def forget(self) -> None:
        with self._lock:
            self._catalog = None

    def _live_catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._catalog
        if cached and time.time() - cached[0] < CATALOG_TTL_SECONDS:
            return cached[1]
        try:
            items = self.upstream.models()
        except ClaudeUpstreamError:
            items = []
        with self._lock:
            self._catalog = (time.time(), items)
        return items

    def _capability(self, model: str, key: str) -> Any:
        if not self.auth.signed_in():
            return None
        for item in self._live_catalog():
            if item.get("id") == model:
                return item.get(key)
        return None


def create(context: ProviderContext) -> Provider:
    claude = Claude(context)
    return Provider(
        name="claude",
        auth=claude.auth,
        login_flow="paste_code",
        match=claude_model_name,
        chat=claude.chat,
        models=claude.models,
        status=claude.status,
        routes={"code": claude.finish_login, "usage": claude.usage},
        count_tokens=claude.count_tokens,
    )
