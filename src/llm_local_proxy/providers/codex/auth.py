"""Codex/ChatGPT auth adapter.

Codex owns its own OAuth: ``codex app-server`` holds the token pair and
refreshes it. This adapter only forwards the JSON-RPC calls the status page
and the HTTP handlers need, and reads back whether a session exists. It
implements the :class:`~llm_local_proxy.auth.Auth` interface so the server
treats Codex exactly like any other provider.
"""

from __future__ import annotations

import time
from typing import Any

from ...status import Limit, ProviderStatus, window_name
from ..auth import Auth
from .app_server import AppServer, RpcError


class CodexAuth(Auth):
    def __init__(self, app: AppServer):
        self.app = app

    def login_start(self, account_id: str = "") -> dict[str, Any]:
        value = self.app.call("account/login/start", {"type": "chatgptDeviceCode"})
        return {
            "url": value.get("verificationUrl", ""),
            "code": value.get("userCode", ""),
        }

    def logout(self, account_id: str = "") -> None:
        self.app.call("account/logout")

    def signed_in(self) -> bool:
        return bool(self._account().get("account"))

    def status(self) -> ProviderStatus:
        account = self._account().get("account")
        if not account:
            return ProviderStatus()
        try:
            limits = _limits(self.app.call("account/rateLimits/read"))
        except RpcError:
            return ProviderStatus(signed_in=True, account=_account_line(account))
        return ProviderStatus(
            signed_in=True,
            account=_account_line(account),
            limits=limits,
            updated_at=time.time(),
        )

    def _account(self) -> dict[str, Any]:
        return self.app.call("account/read", {"refreshToken": False})


def _account_line(account: dict[str, Any]) -> str:
    plan = account.get("planType") or account.get("type") or ""
    name = account.get("email") or "ChatGPT"
    return f"{name} · {plan}" if plan else str(name)


def _limits(value: dict[str, Any]) -> tuple[Limit, ...]:
    items: list[tuple[int, str, Limit]] = []
    for entry in (value or {}).get("rateLimitsByLimitId", {}).values():
        if not isinstance(entry, dict):
            continue
        name = entry.get("limitName") or entry.get("limitId") or "limit"
        for window in (entry.get("primary"), entry.get("secondary")):
            if not isinstance(window, dict):
                continue
            minutes = int(window.get("windowDurationMins") or 0)
            items.append(
                (
                    minutes,
                    str(name),
                    Limit(
                        label=f"{name} · {window_name(minutes)}",
                        used_percent=float(window.get("usedPercent") or 0),
                        resets_at=window.get("resetsAt"),
                    ),
                )
            )
    return tuple(limit for _, _, limit in sorted(items, key=lambda i: i[:2]))
