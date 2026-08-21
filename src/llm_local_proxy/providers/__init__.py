"""The provider registry.

Order is match priority: each provider is offered a model name in turn and
the first to claim it serves the request. Codex is last because it is the
fallback that claims anything left.

Adding a provider is one package plus one entry here.
"""

from __future__ import annotations

from collections.abc import Callable

from . import claude, codex
from .base import Provider, ProviderContext

REGISTRY: tuple[Callable[[ProviderContext], Provider], ...] = (
    claude.create,
    codex.create,
)

__all__ = ["REGISTRY", "Provider", "ProviderContext"]
