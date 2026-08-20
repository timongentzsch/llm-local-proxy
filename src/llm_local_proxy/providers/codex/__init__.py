"""The Codex provider: a ChatGPT subscription driven through codex app-server."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from ...ir import ChatRequest
from ...status import ProviderStatus
from ..base import Provider, ProviderContext
from ..reasoning import ReasoningCache
from .app_server import AppServer
from .auth import CodexAuth
from .catalog import model_info
from .events import CodexDecoder
from .request import build
from .upstream import Upstream


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

    def match(self, model: str) -> str | None:
        # Codex is the fallback transport: registered last, it serves
        # everything a more specific provider did not claim.
        return model

    def chat(
        self, canonical: str, request: ChatRequest
    ) -> tuple[Iterator[dict[str, Any]], CodexDecoder]:
        body, _ = build(request, self.cache)
        return self.upstream.events(body), CodexDecoder(self.cache)

    def models(self) -> list[dict[str, Any]]:
        result = self.app.call("model/list", {"limit": 100, "includeHidden": False})
        contexts = self.app.model_contexts()
        models = []
        for item in result.get("data", []):
            if isinstance(item, dict):
                model = model_info(item, contexts)
                if model:
                    models.append(model)
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
