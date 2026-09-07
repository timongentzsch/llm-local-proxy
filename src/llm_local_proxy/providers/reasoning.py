"""Bounded replay cache for clients that cannot carry signed reasoning.

Responses and Anthropic clients can replay their native opaque blocks directly.
Chat Completions cannot, so providers retain those blocks by tool-call id.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class ReasoningCache:
    def __init__(self, limit: int = 128):
        self._items: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._limit = limit
        self._lock = threading.Lock()

    def get(self, call_ids: list[str]) -> list[dict[str, Any]]:
        with self._lock:
            for call_id in call_ids:
                if call_id in self._items:
                    self._items.move_to_end(call_id)
                    return self._items[call_id]
        return []

    def put(self, call_ids: list[str], items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with self._lock:
            for call_id in call_ids:
                self._items[call_id] = items
                self._items.move_to_end(call_id)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
