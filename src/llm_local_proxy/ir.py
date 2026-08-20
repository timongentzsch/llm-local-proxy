"""The intermediate representation shared by every dialect and provider.

A downstream request is parsed once into :class:`ChatRequest`; each provider
renders its own upstream body from that. Without it the proxy would need one
converter per (dialect, provider) pair.

The shape follows Anthropic's message model rather than the intersection of
the formats it serves: typed content blocks, tool results as blocks inside a
user turn, signed thinking. That model is the superset, and an IR built from
the intersection would have to drop exactly those.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Text:
    text: str


@dataclass
class Image:
    #: A data: or http(s) URL as sent; providers encode it their own way.
    url: str


@dataclass
class ToolUse:
    #: May be empty: Chat Completions allows a call without an id, and each
    #: provider decides whether to synthesise one.
    id: str
    name: str
    #: JSON string or already-decoded object, as sent.
    arguments: Any


@dataclass
class ToolResult:
    tool_use_id: str
    text: str
    is_error: bool = False


@dataclass
class Thinking:
    """Signed reasoning replayed to the upstream that produced it.

    Must round-trip byte-exactly or the upstream rejects the turn.
    """

    text: str
    signature: str = ""
    redacted: str = ""


Block = Text | Image | ToolUse | ToolResult | Thinking


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    blocks: list[Block] = field(default_factory=list)


@dataclass
class FunctionTool:
    name: str
    parameters: dict[str, Any]
    description: str = ""


@dataclass
class WebSearchTool:
    context_size: str = ""


Tool = FunctionTool | WebSearchTool


@dataclass
class ToolChoice:
    kind: Literal["auto", "none", "required", "tool"] = "auto"
    name: str = ""


@dataclass
class ChatRequest:
    #: Empty when the client named the model out of band; providers that
    #: require it say so themselves.
    model: str = ""
    system: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    tool_choice: ToolChoice | None = None
    max_tokens: Any = None
    reasoning_effort: Any = None
    parallel_tool_calls: Any = None
    stream: bool = False
    session: str = ""
    #: Sampling knobs exactly as sent, holding only the keys the client sent.
    #: Whether a value is acceptable is the upstream's business, not the wire
    #: format's: Claude honours a temperature that Codex refuses.
    params: dict[str, Any] = field(default_factory=dict)
