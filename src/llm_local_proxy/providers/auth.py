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
from collections.abc import Callable
from typing import Any

from ..errors import RequestError
from ..status import ProviderStatus


class Auth(ABC):
    """Login lifecycle that any provider can expose to the status page.

    Code-paste flows (Claude) expose an extra ``finish(code)`` on the concrete
    class; device-code flows (Codex) do not. Only the operations every
    provider shares live here, so a device-flow-style provider is not forced
    to stub out a code step.
    """

    @abstractmethod
    def login_start(self, account_id: str = "") -> dict[str, Any]:
        """Start a login: ``{"url": ...}`` plus ``"code"`` for device flows."""

    @abstractmethod
    def logout(self, account_id: str = "") -> None:
        """Sign out locally and revoke the stored session if supported."""

    @abstractmethod
    def signed_in(self) -> bool:
        """Whether the provider currently has a usable session."""

    @abstractmethod
    def status(self) -> ProviderStatus:
        """Normalised card for ``/api/status`` (secrets excluded)."""


class MultiAuth(Auth):
    """Expose several independent logins through the provider auth routes."""

    def __init__(self, accounts: Callable[[], tuple[tuple[str, Auth], ...]]):
        self.accounts = accounts

    def _account(self, account_id: str) -> Auth:
        for candidate, auth in self.accounts():
            if candidate == account_id:
                return auth
        raise RequestError(f"unknown account: {account_id}")

    @staticmethod
    def _signed_in(auth: Auth) -> bool:
        try:
            return auth.signed_in()
        except (OSError, RuntimeError, ValueError):
            return False

    def login_start(self, account_id: str = "") -> dict[str, Any]:
        auth = self._account(account_id)
        return {**auth.login_start(), "account": account_id}

    def finish(self, account_id: str, code: str) -> dict[str, Any]:
        auth = self._account(account_id)
        finish = getattr(auth, "finish", None)
        if finish is None:
            raise ValueError("this provider does not accept a pasted code")
        return finish(code)

    def logout(self, account_id: str = "") -> None:
        self._account(account_id).logout()

    def signed_in(self) -> bool:
        return any(self._signed_in(auth) for _, auth in self.accounts())

    def status(self) -> ProviderStatus:
        accounts = self.accounts()
        signed_in = 0
        for _, auth in accounts:
            signed_in += self._signed_in(auth)
        return ProviderStatus(
            signed_in=bool(signed_in),
            account=f"{signed_in} of {len(accounts)} accounts signed in",
        )
