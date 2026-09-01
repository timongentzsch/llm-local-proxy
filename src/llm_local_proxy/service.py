"""Composition root.

Builds the provider registry, then aggregates it: the merged model catalog,
the status cards, health. It is deliberately transport-free — nothing here
knows about HTTP — and provider-agnostic: nothing here names Codex or Claude.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .config import Config
from .dialects import DIALECTS
from .errors import ProviderError
from .providers import REGISTRY, Provider, ProviderContext
from .status import ProviderStatus

CATALOG_TTL_SECONDS = 60

#: An unreachable provider degrades its own slice, not the whole response.
DEGRADES = (ProviderError, OSError, ValueError)


class Service:
    def __init__(self, config: Config):
        self.config = config
        context = ProviderContext(
            config=config,
            directory=config.path.parent,
            invalidate=self.invalidate_models,
        )
        self.providers: list[Provider] = [create(context) for create in REGISTRY]
        self._models: tuple[float, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def provider(self, name: str) -> Provider | None:
        return next((item for item in self.providers if item.name == name), None)

    def route(self, model: str) -> tuple[Provider, str] | None:
        """First provider whose ``match`` claims the model, or None."""
        for provider in self.providers:
            canonical = provider.match(model)
            if canonical is not None:
                return provider, canonical
        return None

    def healthy(self) -> bool:
        return all(provider.healthy() for provider in self.providers)

    def models(self) -> dict[str, Any]:
        with self._lock:
            cached = self._models
        if cached and time.time() - cached[0] < CATALOG_TTL_SECONDS:
            return cached[1]
        data: list[dict[str, Any]] = []
        for provider in self.providers:
            try:
                data.extend(provider.models())
            except DEGRADES:
                continue
        value = {"object": "list", "data": data}
        with self._lock:
            self._models = (time.time(), value)
        return value

    def invalidate_models(self) -> None:
        with self._lock:
            self._models = None

    def status(self) -> dict[str, Any]:
        cards = []
        for provider in self.providers:
            try:
                value = provider.status()
            except DEGRADES as error:
                value = ProviderStatus(error=str(error) or "unavailable")
            cards.append(
                {
                    "name": provider.name,
                    "login_flow": provider.login_flow,
                    "routes": sorted(provider.routes),
                    **value.payload(),
                }
            )
        return {
            "dialects": [
                {
                    "name": dialect.name,
                    "base_url": self.config.origin + dialect.base_path,
                }
                for dialect in DIALECTS
            ],
            "base_url": self.config.base_url,
            "providers": cards,
        }

    def close(self) -> None:
        for provider in self.providers:
            provider.close()
