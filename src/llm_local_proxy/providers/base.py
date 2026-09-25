"""Provider registry building blocks.

A :class:`Provider` wires one upstream's model matching, chat handling, model
catalog, status and HTTP routes into a single object the server can iterate.
Subscription providers build theirs with
:meth:`~llm_local_proxy.providers.pool.PooledProvider.provider`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..ir import ChatRequest, Decoder
from ..status import ProviderStatus


@dataclass(frozen=True)
class ProviderContext:
    """What every provider needs from the host to construct itself."""

    config: Config
    #: Where credentials and token ledgers are persisted.
    directory: Path


@dataclass(frozen=True, eq=False)
class Provider:
    #: Route prefix and card name on the status page (e.g. "codex", "claude").
    name: str
    #: Maps a requested model id to a canonical name for this provider, or
    #: None when the model does not belong to it (used to route requests).
    match: Callable[[str], str | None]
    #: (canonical model, parsed request) -> (upstream events, decoder).
    chat: Callable[[str, ChatRequest], tuple[Iterator[dict[str, Any]], Decoder]]
    #: Model catalog entries to merge into the /v1/models listing.
    models: Callable[[], list[dict[str, Any]]]
    #: The provider's card for /api/status, normalised so every provider
    #: renders through the same dashboard component.
    status: Callable[[], ProviderStatus]
    #: POST handlers at /api/<name>/<route>, including login and logout.
    routes: Mapping[str, Callable[[dict[str, Any]], Any]]
    #: None when the upstream cannot count exactly; callers then get a 404
    #: rather than an estimate they would wrongly trust.
    count_tokens: Callable[[str, ChatRequest], dict[str, Any]] | None = None
    #: Drops the cached catalog, e.g. after a login changes what is visible.
    forget: Callable[[], None] = lambda: None
    #: Defaults suit a provider that is just an HTTPS client.
    healthy: Callable[[], bool] = lambda: True
    #: Release anything long-lived. Called once at shutdown.
    close: Callable[[], None] = lambda: None
