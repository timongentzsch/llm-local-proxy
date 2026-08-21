"""Provider registry building blocks.

A :class:`Provider` wires one upstream's auth, model matching, chat handling,
model catalog and any extra HTTP routes into a single object the server can
iterate. Adding a provider then means constructing one ``Provider`` for it
(rather than editing ``server.py``'s dispatch): give it a ``match`` for its
model names, a ``chat`` handler, its catalog and status contribution, and an
:class:`~llm_local_proxy.auth.Auth` implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import Config
from ..ir import ChatRequest
from ..status import ProviderStatus
from .auth import Auth


@dataclass(frozen=True)
class ProviderContext:
    """What every provider needs from the host to construct itself."""

    config: Config
    #: Where credentials and token ledgers are persisted.
    directory: Path
    #: Drops cached catalogs, e.g. after a login changes what is visible.
    invalidate: Callable[[], None]


@dataclass(frozen=True, eq=False)
class Provider:
    #: Route prefix and card name on the status page (e.g. "codex", "claude").
    name: str
    #: OAuth access for the status page and the /api/<name>/login|logout endpoints.
    auth: Auth
    #: "device_code" shows login_start's code; "paste_code" POSTs one back to
    #: the provider's "code" route, which such a provider must register.
    login_flow: str
    #: Maps a requested model id to a canonical name for this provider, or
    #: None when the model does not belong to it (used to route requests).
    match: Callable[[str], str | None]
    #: (canonical model, parsed request) -> (upstream events, decoder).
    chat: Callable[[str, ChatRequest], tuple[Iterator[Any], Any]]
    #: Model catalog entries to merge into the /v1/models listing.
    models: Callable[[], list[dict[str, Any]]]
    #: The provider's card for /api/status, normalised so every provider
    #: renders through the same dashboard component.
    status: Callable[[], ProviderStatus]
    #: POST handlers at /api/<name>/<route>. "login" and "logout" are
    #: reserved and always reach auth directly.
    routes: Mapping[str, Callable[[dict[str, Any]], Any]]
    #: None when the upstream cannot count exactly; callers then get a 404
    #: rather than an estimate they would wrongly trust.
    count_tokens: Callable[[str, ChatRequest], dict[str, Any]] | None = None
    #: Defaults suit a provider that is just an HTTPS client.
    healthy: Callable[[], bool] = lambda: True
    #: Release anything long-lived. Called once at shutdown.
    close: Callable[[], None] = lambda: None
