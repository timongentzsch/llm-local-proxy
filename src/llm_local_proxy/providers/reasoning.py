"""Reasoning carried between a tool call and its result.

Both upstreams refuse a tool result whose originating reasoning is missing,
and neither downstream format has anywhere to put it: Codex returns an
encrypted blob and Claude a signed thinking block, both opaque and both
required verbatim on the next turn. So the proxy holds them here, keyed by
the tool call ids they belong to, rather than asking clients to round-trip
something they cannot read.
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
