"""Rolling per-request token ledger shared by the provider upstreams.

The subscription utilisation bars are weighted and opaque (see the upstream
rate-limit headers); the only hard token numbers come from each request's
usage block in the response body. This ledger sums those per-request numbers
over the same 5h/7d windows the bars use, so the dashboard can show how many
tokens the *proxy* consumed in each window. It reflects only proxy traffic;
other clients of the same subscription are not visible here.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json

#: (label, seconds) windows mirroring the subscription utilisation buckets.
WINDOWS: tuple[tuple[str, int], ...] = (("5h", 5 * 3600), ("7d", 7 * 86400))
MAX_AGE = 7 * 86400


class TokenLedger:
    """Append-only per-request token counts with sliding-window sums.

    Persisted to ``path`` (a JSON list of records) so totals survive restarts;
    records older than ``MAX_AGE`` are pruned on write and on read.
    """

    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.Lock()
        self._records = self._load()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path or not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(value, list):
            return []
        cutoff = time.time() - MAX_AGE
        return [r for r in value if isinstance(r, dict) and r.get("ts", 0) > cutoff]

    def add(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read: int = 0,
        cache_write: int = 0,
    ) -> None:
        record = {
            "ts": int(time.time()),
            "input": int(input_tokens),
            "output": int(output_tokens),
            "cache_read": int(cache_read),
            "cache_write": int(cache_write),
        }
        cutoff = time.time() - MAX_AGE
        with self._lock:
            self._records.append(record)
            self._records = [r for r in self._records if r.get("ts", 0) > cutoff]
            if self.path:
                atomic_write_json(self.path, self._records)

    def windows(self) -> dict[str, dict[str, int]]:
        """Summed tokens per window (``{"5h": {...}, "7d": {...}}``)."""
        now = time.time()
        with self._lock:
            records = list(self._records)
        result: dict[str, dict[str, int]] = {}
        for label, seconds in WINDOWS:
            since = now - seconds
            window = [r for r in records if r.get("ts", 0) > since]
            totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
            for record in window:
                totals["input"] += record.get("input", 0)
                totals["output"] += record.get("output", 0)
                totals["cache_read"] += record.get("cache_read", 0)
                totals["cache_write"] += record.get("cache_write", 0)
            result[label] = totals
        return result
