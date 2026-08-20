"""Provider OAuth abstraction.

Each provider owns its login differently. For Codex, the ``codex app-server``
binary holds ChatGPT's OAuth state and refreshes the token; the proxy only
drives it over JSON-RPC. For Claude, the proxy itself performs the whole
authorization-code + PKCE exchange and rotates the refresh token. ``Auth`` is
the small shared surface the status page and HTTP handlers need; an
implementation hides where the tokens live and who refreshes them, so a new
provider only has to provide these four operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .status import ProviderStatus


class Auth(ABC):
    """Login lifecycle that any provider can expose to the status page.

    Code-paste flows (Claude) expose an extra ``finish(code)`` on the concrete
    class; device-code flows (Codex) do not. Only the operations every
    provider shares live here, so a device-flow-style provider is not forced
    to stub out a code step.
    """

    @abstractmethod
    def login_start(self) -> dict[str, Any]:
        """Start a login: ``{"url": ...}`` plus ``"code"`` for device flows."""

    @abstractmethod
    def logout(self) -> None:
        """Sign out locally and revoke the stored session if supported."""

    @abstractmethod
    def signed_in(self) -> bool:
        """Whether the provider currently has a usable session."""

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Normalised card for ``/api/status`` (secrets excluded)."""
