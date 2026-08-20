"""Uniform status shape every provider reports to the dashboard.

The page renders one card per provider from these fields alone: sign-in
state, an account line, usage bars, proxy token counts and a freshness
stamp. Providers normalise their own upstream shapes into this, so adding a
provider needs no dashboard change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Rolling windows named the same way across providers and the token ledger.
WINDOW_LABELS = {"5h": "5 hour", "7d": "weekly"}


@dataclass(frozen=True)
class Limit:
    """One subscription usage bar."""

    label: str
    used_percent: float
    #: Epoch seconds or an ISO timestamp; the page formats either.
    resets_at: Any = None


@dataclass(frozen=True)
class ProviderStatus:
    signed_in: bool = False
    #: One-line account description (e.g. "user@example.com · pro").
    account: str = ""
    limits: tuple[Limit, ...] = ()
    #: Token ledger windows: {"5h": {"input": .., "output": .., ...}, ...}.
    tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    #: When the usage numbers were last observed, epoch seconds.
    updated_at: float | None = None
    #: Set when the provider could not be reached; the card degrades to this.
    error: str = ""

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def window_label(key: str) -> str:
    return WINDOW_LABELS.get(key, key)


def window_name(minutes: Any) -> str:
    try:
        value = int(minutes)
    except (TypeError, ValueError):
        return "window"
    if value == 300:
        return "5 hour"
    if value == 10080:
        return "weekly"
    if value % 1440 == 0:
        return f"{value // 1440} day"
    if value % 60 == 0:
        return f"{value // 60} hour"
    return f"{value} minute"
