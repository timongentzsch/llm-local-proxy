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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..ir import ChatRequest


def block_text(parts: list[Any]) -> str:
    """Join the text blocks of a content list, ignoring every other kind."""
    return "\n".join(
        str(part.get("text", ""))
        for part in parts
        if isinstance(part, dict) and part.get("type") == "text"
    )


@dataclass(frozen=True, eq=False)
class Frame:
    """One server-sent event; ``event`` is unset for anonymous frames."""

    data: dict[str, Any]
    event: str | None = None


@dataclass(frozen=True, eq=False)
class Dialect:
    #: Registry key and mount name (e.g. "openai", "anthropic").
    name: str
    #: Mount point; every dialect has one so routes cannot collide.
    prefix: str
    #: What a client is configured with. Not always prefix + "/v1": clients
    #: differ in how much of the path they append themselves.
    base_path: str
    #: Path, below the prefix, that accepts a chat request.
    chat_route: str
    #: (body, session) -> the dialect-neutral request every provider reads.
    parse: Callable[[dict[str, Any], str], ChatRequest]
    #: (model, provider decoder) -> encoder. Pairing here keeps neither side
    #: naming the other.
    encode: Callable[[str, Any], Any]
    #: Merged provider catalogs -> this dialect's model listing.
    catalog: Callable[[list[dict[str, Any]]], dict[str, Any]]
    #: Payload -> the SSE event name to write, or None for anonymous frames.
    event_name: Callable[[dict[str, Any]], str | None]
    #: (status, message) -> the dialect's error body.
    error: Callable[[int, str], dict[str, Any]]
    keepalive: bytes
    #: Written after the final frame; Anthropic has no such sentinel.
    terminator: bytes | None
    #: Counts input tokens without running the request; None when unsupported.
    count_route: str | None = None
    parse_count: Callable[[dict[str, Any], str], ChatRequest] | None = None
    #: Optional second OpenAI endpoint using Responses item semantics.
    responses_route: str | None = None
    parse_responses: Callable[[dict[str, Any], str], ChatRequest] | None = None
    encode_responses: Callable[[str, Any, ChatRequest], Any] | None = None
