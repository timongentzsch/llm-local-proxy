from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any


class RequestError(ValueError):
    pass


class ReasoningCache:
    """Keeps encrypted reasoning between a tool call and its result."""

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


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("message content must be a string or array")
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "\n".join(parts)
