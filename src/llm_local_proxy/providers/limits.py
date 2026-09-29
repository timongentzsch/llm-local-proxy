"""Subscription utilization bars, read on demand and shared across callers.

Each provider supplies how to read its bars; this spaces the reads so any
number of dashboards and routing decisions cost one upstream call, and keeps
the last bars, with the stamp of when they were observed, through a failure.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable

from ..errors import ProviderError
from ..status import Limit

#: Seconds between reads; after a refusal (credentials or rate limit) the
#: next read waits for the backoff instead.
TTL = 30
BACKOFF = 300

Bars = tuple[Limit, ...]


class LimitsStore:
    """One read at a time, outside the lock: callers meanwhile get the last bars."""

    def __init__(self, name: str, read: Callable[[], Bars]):
        self._name = name
        self._read = read
        self._lock = threading.Lock()
        self._bars: Bars = ()
        self._updated_at: float | None = None
        self._next_read = 0.0
        self._reading = False
        # Bumped by clear(), so a read begun for the previous login is dropped.
        self._generation = 0
        self._last_error = ""

    def current(self, wait: bool = True) -> tuple[Bars, float | None]:
        """The latest bars and when they were observed, refreshed if due.

        Without ``wait`` a due read runs in the background and the call
        returns at once, so request routing never waits on the network.
        """
        with self._lock:
            if self._reading or time.monotonic() < self._next_read:
                return self._bars, self._updated_at
            self._reading = True
            generation = self._generation
            if not wait:
                threading.Thread(
                    target=self._refresh, args=(generation,), daemon=True
                ).start()
                return self._bars, self._updated_at
        return self._refresh(generation)

    def clear(self) -> None:
        """Forget the bars of a login this slot no longer holds."""
        with self._lock:
            self._bars, self._updated_at, self._next_read = (), None, 0.0
            self._generation += 1
            self._last_error = ""

    def _refresh(self, generation: int) -> tuple[Bars, float | None]:
        try:
            bars, error = self._read(), None
        except Exception as caught:  # noqa: BLE001 - never leave the store busy
            bars, error = None, caught
        with self._lock:
            self._reading = False
            if generation != self._generation:
                return self._bars, self._updated_at
            delay = TTL
            if bars is not None:
                self._bars, self._updated_at = bars, time.time()
                self._last_error = ""
            else:
                if isinstance(error, ProviderError) and error.status in {401, 403, 429}:
                    delay = BACKOFF
                # Logged once per distinct failure, not on every retry.
                if str(error) != self._last_error:
                    self._last_error = str(error)
                    sys.stderr.write(f"{self._name}: limits unavailable: {error}\n")
            self._next_read = time.monotonic() + delay
            return self._bars, self._updated_at


def fullest(bars: Bars) -> float | None:
    """The fullest window that limits the whole account, if any is known."""
    values = [bar.used_percent for bar in bars if not bar.model]
    return max(values) if values else None
