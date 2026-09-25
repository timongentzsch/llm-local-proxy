"""Server-sent event plumbing.

Anthropic Messages and OpenAI Responses name every frame after its ``type``
and end without a sentinel; Chat Completions sends anonymous frames and ends
with ``data: [DONE]``. The route selects which; the keepalive belongs to the
dialect.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from typing import Any, cast

from ..streaming import closing_iterator

SSE_HEARTBEAT_SECONDS = 15
_DONE = object()


def render(data: dict[str, Any], event: str | None = None) -> bytes:
    """One SSE frame; the event line is written only for named frames."""
    payload = b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"
    return payload if event is None else f"event: {event}\n".encode() + payload


def with_heartbeats(
    events: Iterator[dict[str, Any]], interval: float = SSE_HEARTBEAT_SECONDS
) -> Iterator[dict[str, Any] | None]:
    """Yield upstream events, or None when the upstream has gone quiet.

    The upstream is drained on a worker thread so a slow model cannot stall
    the keepalive; None is the caller's cue to emit one.
    """
    items: queue.Queue[dict[str, Any] | Exception | object] = queue.Queue()
    stopped = threading.Event()

    def read() -> None:
        try:
            with closing_iterator(events):
                for event in events:
                    if stopped.is_set():
                        break
                    items.put(event)
        except Exception as error:  # noqa: BLE001 - cross the thread boundary
            items.put(error)
        finally:
            items.put(_DONE)

    threading.Thread(target=read, daemon=True).start()
    try:
        while True:
            try:
                item = items.get(timeout=interval)
            except queue.Empty:
                yield None
                continue
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield cast(dict[str, Any], item)
    finally:
        stopped.set()


class SseStream:
    """Writes frames for one dialect to one client connection."""

    def __init__(self, wfile: Any, keepalive: bytes, named: bool):
        self._wfile = wfile
        self._keepalive = keepalive
        self._named = named

    def send(self, data: dict[str, Any]) -> None:
        self._write(render(data, data.get("type") if self._named else None))

    def keepalive(self) -> None:
        self._write(self._keepalive)

    def end(self) -> None:
        if not self._named:
            self._write(b"data: [DONE]\n\n")

    def _write(self, payload: bytes) -> None:
        self._wfile.write(payload)
        self._wfile.flush()
