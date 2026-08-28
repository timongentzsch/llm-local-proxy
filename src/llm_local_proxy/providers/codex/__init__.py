"""The Codex provider: a ChatGPT subscription driven through codex app-server."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from ...ir import ChatRequest
from ...status import ProviderStatus
from ..base import Provider, ProviderContext
from ..catalog import match_model
from ..reasoning import ReasoningCache
from .app_server import AppServer
from .auth import CodexAuth
from .catalog import model_info
from .events import CodexDecoder
from .request import build
from .upstream import Upstream

CATALOG_TTL_SECONDS = 60


class Codex:
    def __init__(self, context: ProviderContext):
        config = context.config
        self.app = AppServer(config.codex_binary, config.codex_home)
        self.upstream = Upstream(
            self.app,
            config.request_timeout,
            tokens_path=context.directory / "codex-tokens.json",
        )
        self.auth = CodexAuth(self.app)
        self.cache = ReasoningCache()
        self._catalog: tuple[float, list[dict[str, Any]]] | None = None
        self._lock = threading.Lock()

    def match(self, model: str) -> str | None:
        return match_model(model, self._live_catalog())

    def chat(
        self, canonical: str, request: ChatRequest
    ) -> tuple[Iterator[dict[str, Any]], CodexDecoder]:
        model = next(
            (item for item in self._live_catalog() if item.get("id") == canonical), {}
        )
        efforts = model.get("supported_reasoning_efforts")
        body, _ = build(
            request,
            self.cache,
            reasoning_efforts=efforts if isinstance(efforts, list) else None,
        )
        return self.upstream.events(body), CodexDecoder(self.cache)

    def models(self) -> list[dict[str, Any]]:
        return self._live_catalog()

    def _live_catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._catalog
        if cached and time.time() - cached[0] < CATALOG_TTL_SECONDS:
            return cached[1]
        result = self.app.call("model/list", {"limit": 100, "includeHidden": False})
        contexts = self.app.model_contexts()
        items = [item for item in result.get("data", []) if isinstance(item, dict)]
        first_model = next(
            (
                str(item.get("model") or item.get("id"))
                for item in items
                if item.get("model") or item.get("id")
            ),
            "",
        )
        transport_efforts = (
            self.upstream.reasoning_efforts(first_model) if first_model else None
        )
        models = []
        for item in items:
            model = model_info(item, contexts, transport_efforts)
            if model:
                models.append(model)
        with self._lock:
            self._catalog = (time.time(), models)
        return models

    def status(self) -> ProviderStatus:
        return replace(self.auth.status(), tokens=self.upstream.ledger.windows())


def create(context: ProviderContext) -> Provider:
    codex = Codex(context)
    return Provider(
        name="codex",
        auth=codex.auth,
        login_flow="device_code",
        match=codex.match,
        chat=codex.chat,
        models=codex.models,
        status=codex.status,
        routes={},
        healthy=codex.app.alive,
        close=codex.app.close,
    )
