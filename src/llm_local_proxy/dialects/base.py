"""Downstream wire formats.

A :class:`Dialect` is one public API the proxy speaks *to clients*, as opposed
to a :class:`~llm_local_proxy.providers.base.Provider`, which is one upstream
subscription the proxy speaks to. The two axes are independent: any dialect can
be served by any provider.

Everything here describes a published specification (see ``specs/PINNED.md``),
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


@dataclass(frozen=True, eq=False)
class Frame:
    """One server-sent event.

    ``event`` names the SSE event. Chat Completions leaves it unset, since
    that format streams anonymous ``data:`` frames; the Anthropic Messages
    format names every frame.
    """

    data: dict[str, Any]
    event: str | None = None


@dataclass(frozen=True, eq=False)
class Dialect:
    #: Registry key and mount name (e.g. "openai", "anthropic").
    name: str
    #: URL mount point. Every dialect has one, so no two can collide over a
    #: shared route name — notably /v1/models, whose response shape differs
    #: per dialect. The dialect named by DEFAULT additionally answers on the
    #: bare paths, for clients configured before the prefixes existed.
    prefix: str
    #: What a client should be configured with, below the origin. Not always
    #: prefix + "/v1": an OpenAI client wants /v1 included in its base url
    #: and appends "/chat/completions", while an Anthropic client appends
    #: "/v1/messages" itself.
    base_path: str
    #: Path, below the prefix, that accepts a chat request.
    chat_route: str
    #: (body, session) -> the dialect-neutral request every provider reads.
    parse: Callable[[dict[str, Any], str], ChatRequest]
    #: (model, provider decoder) -> the encoder that writes this dialect's
    #: wire output. Pairing happens here so neither side names the other,
    #: which is what keeps dialects and providers at N+M rather than N×M.
    encode: Callable[[str, Any], Any]
    #: Merged provider catalogs -> this dialect's model listing.
    catalog: Callable[[list[dict[str, Any]]], dict[str, Any]]
    #: Payload -> the SSE event name to write, or None for anonymous frames.
    event_name: Callable[[dict[str, Any]], str | None]
    #: (status, message) -> the dialect's error body.
    error: Callable[[int, str], dict[str, Any]]
    #: Sent when a stream has produced nothing for a while, to keep proxies
    #: and load balancers from closing an idle connection.
    keepalive: bytes
    #: Written once after the final frame, when the format has such a
    #: sentinel. Anthropic's does not.
    terminator: bytes | None
    #: Path, below the prefix, that counts a request's input tokens without
    #: running it. None when the format has no such endpoint.
    count_route: str | None = None
    #: Parses a body for count_route, which omits the generation-only fields.
    parse_count: Callable[[dict[str, Any], str], ChatRequest] | None = None
