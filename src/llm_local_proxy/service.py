"""Composition root.

Owns the provider registry, the shared caches and the aggregated catalog and
status views. It is deliberately transport-free: nothing here knows about
HTTP, and nothing here knows which dialect a request arrived in.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from .config import Config
from .protocol import ReasoningCache, RequestError, Translator, build_request
from .providers import Provider
from .providers.claude.auth import ClaudeAuth, ClaudeAuthError
from .providers.claude.protocol import (
    CLAUDE_MODELS,
    ClaudeTranslator,
    build_messages_request,
    claude_model_name,
)
from .providers.claude.upstream import ClaudeUpstream, ClaudeUpstreamError
from .providers.codex.app_server import AppServer, RpcError
from .providers.codex.auth import CodexAuth
from .providers.codex.upstream import Upstream
from .status import ProviderStatus


def _model_info(
    item: dict[str, Any], context_windows: dict[str, int] | None = None
) -> dict[str, Any] | None:
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
    context = (context_windows or {}).get(str(model), 0)
    if context > 0:
        value["context_length"] = context
    return value


def _claude_model_info(item: dict[str, Any]) -> dict[str, Any]:
    value = {
        "id": item["id"],
        "canonical_slug": item["id"],
        "object": "model",
        "created": int(item.get("created") or 0),
        "owned_by": "anthropic",
        "name": item["name"],
        "architecture": {
            "modality": "text+image->text",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        "supported_parameters": [
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "reasoning_effort",
            "web_search",
            "temperature",
            "top_p",
        ],
        "default_parameters": (
            {"max_tokens": item["max_output_tokens"]}
            if item.get("max_output_tokens")
            else None
        ),
        "per_request_limits": None,
        "is_default": False,
        "supported_reasoning_efforts": item.get("reasoning_efforts")
        or ["low", "medium", "high"],
    }
    context = item.get("context_length")
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        context = next(
            (
                model.get("context_length")
                for model in CLAUDE_MODELS
                if model["id"] == item["id"]
            ),
            None,
        )
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        value["context_length"] = context
    return value


class Service:
    def __init__(self, config: Config):
        self.config = config
        self.app = AppServer(config.codex_binary, config.codex_home)
        config_dir = config.path.parent
        self.upstream = Upstream(
            self.app,
            config.request_timeout,
            tokens_path=config_dir / "codex-tokens.json",
        )
        self.claude_auth = ClaudeAuth(config_dir / "claude-credentials.json")
        self.claude = ClaudeUpstream(
            self.claude_auth,
            config.request_timeout,
            usage_path=config_dir / "claude-usage.json",
            tokens_path=config_dir / "claude-tokens.json",
        )
        self.codex_auth = CodexAuth(self.app)
        self.cache = ReasoningCache()
        self.claude_reasoning = ReasoningCache()
        self.providers = self._providers()
        self._models: tuple[float, dict[str, Any]] | None = None
        self._claude_catalog: tuple[float, list[dict[str, Any]]] | None = None
        self._models_lock = threading.Lock()

    def _providers(self) -> list[Provider]:
        """The wired provider registry; later entries may fall back on any model."""

        def codex_match(model: str) -> str | None:
            # Codex is the fallback transport: it serves everything the more
            # specific providers did not claim.
            return None if claude_model_name(model) else model

        return [
            Provider(
                name="claude",
                auth=self.claude_auth,
                login_flow="paste_code",
                match=claude_model_name,
                chat=self._claude_chat,
                models=self._claude_items,
                status=self._claude_status,
                routes={
                    "code": self._claude_code,
                    "usage": self._claude_usage,
                },
            ),
            Provider(
                name="codex",
                auth=self.codex_auth,
                login_flow="device_code",
                match=codex_match,
                chat=self._codex_chat,
                models=self._codex_models,
                status=self._codex_status,
                routes={},
            ),
        ]

    def provider(self, name: str) -> Provider | None:
        return next((item for item in self.providers if item.name == name), None)

    def route(self, model: str) -> tuple[Provider, str] | None:
        """First provider whose ``match`` claims the model, or None."""
        for provider in self.providers:
            canonical = provider.match(model)
            if canonical is not None:
                return provider, canonical
        return None

    # -- per-provider handlers -------------------------------------------

    def _codex_chat(
        self, canonical: str, body: dict[str, Any], session: str
    ) -> tuple[Iterator[dict[str, Any]], Translator]:
        request, _ = build_request(body, self.cache, session)
        return self.upstream.events(request), Translator(canonical, self.cache)

    def _codex_models(self) -> list[dict[str, Any]]:
        result = self.app.call("model/list", {"limit": 100, "includeHidden": False})
        context_windows = self.app.model_contexts()
        models = []
        for item in result.get("data", []):
            if not isinstance(item, dict):
                continue
            model = _model_info(item, context_windows)
            if model:
                models.append(model)
        return models

    def _claude_chat(
        self, canonical: str, body: dict[str, Any], session: str
    ) -> tuple[Iterator[dict[str, Any]], ClaudeTranslator]:
        if not self.claude_auth.signed_in():
            raise ClaudeAuthError(
                "not signed in to Claude; use the sign in button on the status page"
            )
        request, betas = build_messages_request(
            body,
            canonical,
            max_output=self._claude_capability(canonical, "max_output_tokens"),
            thinking=self._claude_capability(canonical, "thinking"),
            reasoning_cache=self.claude_reasoning,
        )
        events = self.claude.events(request, tuple(betas))
        return events, ClaudeTranslator(canonical, self.claude_reasoning)

    def _claude_code(self, body: dict[str, Any]) -> dict[str, Any]:
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RequestError("code is required")
        result = self.claude_auth.finish(code)
        self.invalidate_models()
        return result

    def _claude_usage(self, body: dict[str, Any]) -> dict[str, Any]:
        return {"usage": self.claude.ping_usage()}

    def _claude_status(self) -> ProviderStatus:
        return replace(
            self.claude_auth.status(),
            limits=self.claude.usage.limits(),
            tokens=self.claude.ledger.windows(),
            updated_at=self.claude.usage.updated_at(),
        )

    def _codex_status(self) -> ProviderStatus:
        return replace(self.codex_auth.status(), tokens=self.upstream.ledger.windows())

    def models(self) -> dict[str, Any]:
        with self._models_lock:
            if self._models and time.time() - self._models[0] < 60:
                return self._models[1]
        data: list[dict[str, Any]] = []
        for provider in self.providers:
            try:
                data.extend(provider.models())
            except (RpcError, ClaudeUpstreamError, OSError, ValueError):
                # One provider being down must not take the whole catalog
                # down with it; degrade just that provider's slice.
                continue
        value = {"object": "list", "data": data}
        with self._models_lock:
            self._models = (time.time(), value)
        return value

    def invalidate_models(self) -> None:
        with self._models_lock:
            self._models = None
            self._claude_catalog = None

    def _load_claude_catalog(self) -> list[dict[str, Any]]:
        with self._models_lock:
            if self._claude_catalog and time.time() - self._claude_catalog[0] < 60:
                return self._claude_catalog[1]
        try:
            items = self.claude.models()
        except ClaudeUpstreamError:
            items = []
        with self._models_lock:
            self._claude_catalog = (time.time(), items)
        return items

    def _claude_items(self) -> list[dict[str, Any]]:
        if self.claude_auth.signed_in():
            live = self._load_claude_catalog()
            if live:
                return [_claude_model_info(item) for item in live]
        return [_claude_model_info(item) for item in CLAUDE_MODELS]

    def _claude_capability(self, model: str, key: str) -> Any:
        if not self.claude_auth.signed_in():
            return None
        for item in self._load_claude_catalog():
            if item.get("id") == model:
                return item.get(key)
        return None

    def status(self) -> dict[str, Any]:
        cards = []
        for provider in self.providers:
            try:
                value = provider.status()
            except (RpcError, ClaudeUpstreamError, OSError, ValueError) as error:
                # A provider that is unreachable degrades to its own card
                # instead of blanking out the healthy providers.
                value = ProviderStatus(error=str(error) or "unavailable")
            cards.append(
                {
                    "name": provider.name,
                    "login_flow": provider.login_flow,
                    "routes": sorted(provider.routes),
                    **value.payload(),
                }
            )
        return {"base_url": self.config.base_url, "providers": cards}

    def close(self) -> None:
        self.app.close()
