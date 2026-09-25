"""Downstream wire formats.

A :class:`Dialect` is one public API the proxy speaks *to clients*, as opposed
to a :class:`~llm_local_proxy.providers.base.Provider`, which is one upstream
subscription the proxy speaks to. The two axes are independent: any dialect can
be served by any provider.

Everything here describes a published specification (see ``docs/specs.md``),
so a claim in this package is checkable. Undocumented, reverse-engineered
behaviour belongs in a provider instead.

Adding a dialect means constructing one ``Dialect`` and registering it, rather
than editing the HTTP handler: give it a mount prefix, its error envelope, its
stream framing and the header its clients authenticate with.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import RequestError
from ..ir import ChatRequest, Decoder, StreamEvent


def block_text(parts: list[Any]) -> str:
    """Flatten text-only content without silently discarding other parts."""
    if any(
        not isinstance(part, dict)
        or part.get("type") != "text"
        or not isinstance(part.get("text"), str)
        for part in parts
    ):
        raise RequestError("content must contain text blocks")
    return "\n".join(part["text"] for part in parts)


class Encoder:
    """Drives one provider's decoder and shapes its events for one dialect.

    Subclasses implement `_one` (one event to frames) and the lifecycle ends:
    `start`, `finish` and `result`.
    """

    def __init__(self, decoder: Decoder):
        self.decoder = decoder
        self._drained = False

    def start(self) -> dict[str, Any]: ...
    def finish(self) -> list[dict[str, Any]]: ...
    def result(self) -> dict[str, Any]: ...
    def _one(self, event: StreamEvent) -> list[dict[str, Any]]: ...

    def error(self, message: str) -> dict[str, Any] | None:
        """A mid-stream failure frame; None sends the dialect's error body."""
        return None

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        return self._encode(self.decoder.decode(event))

    def _drain(self) -> list[dict[str, Any]]:
        """Collect whatever the decoder only knows once the stream ends."""
        if self._drained:
            return []
        self._drained = True
        return self._encode(self.decoder.finish())

    def _encode(self, events: list[StreamEvent]) -> list[dict[str, Any]]:
        return [frame for event in events for frame in self._one(event)]


@dataclass(frozen=True)
class Route:
    #: (body, session) -> the dialect-neutral request every provider reads.
    parse: Callable[[dict[str, Any], str], ChatRequest]
    #: (model, provider decoder, request) -> encoder. Pairing here keeps
    #: neither side naming the other. None: the route counts input tokens.
    encode: Callable[[str, Decoder, ChatRequest], Encoder] | None = None
    #: SSE frames are named after their `type` and need no end sentinel;
    #: otherwise they are anonymous and end with `data: [DONE]`.
    named: bool = False


@dataclass(frozen=True, eq=False)
class Dialect:
    #: Registry key and mount name (e.g. "openai", "anthropic").
    name: str
    #: Mount point; every dialect has one so routes cannot collide.
    prefix: str
    #: What a client is configured with. Not always prefix + "/v1": clients
    #: differ in how much of the path they append themselves.
    base_path: str
    #: Paths below the prefix that accept a request, and how each is served.
    routes: Mapping[str, Route]
    #: Merged provider catalogs -> this dialect's model listing.
    catalog: Callable[[list[dict[str, Any]]], dict[str, Any]]
    #: (status, message) -> the dialect's error body.
    error: Callable[[int, str], dict[str, Any]]
    #: Written while the upstream is silent, so idle connections stay open.
    keepalive: bytes
