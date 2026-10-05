"""Codex/ChatGPT auth adapter.

Codex owns its own OAuth: ``codex app-server`` holds the token pair and
refreshes it. This adapter only forwards the JSON-RPC calls the status page
and the HTTP handlers need, and reads back whether a session exists. It
implements the :class:`~llm_local_proxy.providers.auth.Auth` interface so the server
treats Codex exactly like any other provider.
"""

from __future__ import annotations

from typing import Any

from ...status import AccountStatus, Limit, window_name
from ..auth import Auth
from ..limits import LimitsStore
from .app_server import AppServer


class CodexAuth(Auth):
    def __init__(self, app: AppServer):
        self.app = app
        self.limits = LimitsStore(
            "codex", lambda: _limits(self.app.call("account/rateLimits/read"))
        )

    def login_start(self) -> dict[str, Any]:
        value = self.app.call("account/login/start", {"type": "chatgptDeviceCode"})
        return {
            "url": value.get("verificationUrl", ""),
            "code": value.get("userCode", ""),
        }

    def logout(self) -> None:
        self.app.call("account/logout")
        self.limits.clear()

    def signed_in(self) -> bool:
        return bool(self._account().get("account"))

    def status(self) -> AccountStatus:
        account = self._account().get("account")
        if not account:
            return AccountStatus()
        limits, updated_at = self.limits.current()
        return AccountStatus(
            signed_in=True,
            account=_account_line(account),
            limits=limits,
            updated_at=updated_at,
        )

    def _account(self) -> dict[str, Any]:
        return self.app.call("account/read", {"refreshToken": False})


def _account_line(account: dict[str, Any]) -> str:
    plan = account.get("planType") or account.get("type") or ""
    name = account.get("email") or "ChatGPT"
    return f"{name} · {plan}" if plan else str(name)


def _limits(value: dict[str, Any]) -> tuple[Limit, ...]:
    items: list[tuple[int, str, Limit]] = []
    # The top-level entry is the account's own limit; any other is one
    # model's (it restricts only that model, not the whole account).
    value = value or {}
    default = value.get("rateLimits")
    default_id = default.get("limitId") if isinstance(default, dict) else None
    # The protocol allows the map to be null; the default limit then stands
    # alone.
    entries = value.get("rateLimitsByLimitId")
    if not isinstance(entries, dict):
        entries = {"": default} if isinstance(default, dict) else {}
    for entry in entries.values():
        if not isinstance(entry, dict):
            continue
        name = entry.get("limitName") or entry.get("limitId") or "limit"
        scoped = default_id is not None and entry.get("limitId") != default_id
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
                        model=str(name) if scoped else "",
                    ),
                )
            )
    return tuple(limit for _, _, limit in sorted(items, key=lambda i: i[:2]))
