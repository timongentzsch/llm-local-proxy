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
from typing import Any

from .auth import Auth
from .status import ProviderStatus


@dataclass(frozen=True, eq=False)
class Provider:
    #: Route prefix and card name on the status page (e.g. "codex", "claude").
    name: str
    #: OAuth access for the status page and the /api/<name>/login|logout endpoints.
    auth: Auth
    #: How the dashboard finishes a login: "device_code" shows the code returned
    #: by login_start, "paste_code" prompts for a code and POSTs it to the
    #: provider's "code" route (which such a provider must register).
    login_flow: str
    #: Maps a requested model id to a canonical name for this provider, or
    #: None when the model does not belong to it (used to route requests).
    match: Callable[[str], str | None]
    #: (canonical model, chat body, session id) -> (events iterator, translator).
    chat: Callable[[str, dict[str, Any], str], tuple[Iterator[Any], Any]]
    #: Model catalog entries to merge into the /v1/models listing.
    models: Callable[[], list[dict[str, Any]]]
    #: The provider's card for /api/status, normalised so every provider
    #: renders through the same dashboard component.
    status: Callable[[], ProviderStatus]
    #: Extra provider-specific POST handlers mounted at /api/<name>/<route>.
    #: The names "login" and "logout" are reserved by the server's generic
    #: dispatch and always call auth.login_start()/auth.logout() instead of
    #: any handler registered here.
    routes: Mapping[str, Callable[[dict[str, Any]], Any]]
