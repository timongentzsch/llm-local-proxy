"""Composition root.

Builds the provider registry, then aggregates it: the merged model catalog,
the status cards, health. It is deliberately transport-free — nothing here
knows about HTTP — and provider-agnostic: nothing here names Codex or Claude.
"""

from __future__ import annotations

from typing import Any

from .config import Config
from .dialects import DIALECTS
from .errors import ProviderError
from .providers import REGISTRY, Provider, ProviderContext
from .status import ProviderStatus

#: An unreachable provider degrades its own slice, not the whole response.
DEGRADES = (ProviderError, OSError, ValueError)


class Service:
    def __init__(self, config: Config):
        self.config = config
        context = ProviderContext(config=config, directory=config.path.parent)
        self.providers: list[Provider] = [create(context) for create in REGISTRY]

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

    def models(self, refresh: bool = False) -> dict[str, Any]:
        """The merged catalog; each provider caches its own slice."""
        data: list[dict[str, Any]] = []
        for provider in self.providers:
            if refresh:
                provider.forget()
            try:
                data.extend(provider.models())
            except DEGRADES:
                continue
        return {"object": "list", "data": data}

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
            "providers": cards,
        }

    def close(self) -> None:
        for provider in self.providers:
            provider.close()
