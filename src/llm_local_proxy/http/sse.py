"""Server-sent event plumbing, parameterised by dialect.

Framing is shared: an event name is written only when the dialect's encoder
supplies one, which is what distinguishes named Anthropic frames from
anonymous Chat Completions frames. What differs per dialect — the keepalive
and the terminator — lives on the :class:`~llm_local_proxy.dialects.base.Dialect`.
"""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from typing import Any, cast

from ..dialects import Dialect, Frame
from ..streaming import closing_iterator

SSE_HEARTBEAT_SECONDS = 15
_DONE = object()


def render(frame: Frame) -> bytes:
    data = json.dumps(frame.data, separators=(",", ":")).encode()
    if frame.event is None:
        return b"data: " + data + b"\n\n"
    return f"event: {frame.event}\n".encode() + b"data: " + data + b"\n\n"


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

    def __init__(self, wfile: Any, dialect: Dialect):
        self._wfile = wfile
        self._dialect = dialect

    def send(self, data: dict[str, Any], event: str | None = None) -> None:
        self._write(render(Frame(data, event)))

    def keepalive(self) -> None:
        self._write(self._dialect.keepalive)

    def end(self) -> None:
        if self._dialect.terminator is not None:
            self._write(self._dialect.terminator)

    def _write(self, payload: bytes) -> None:
        self._wfile.write(payload)
        self._wfile.flush()
