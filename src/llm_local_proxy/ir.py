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


# --- response side ---------------------------------------------------------
#
# What a provider decodes an upstream stream into, and what a dialect encodes
# onto the wire. Finish reasons use Anthropic's vocabulary because it is the
# richer of the two; narrowing to Chat Completions' four is the encoder's job.


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ThinkingSignature:
    """Closes a thinking block. Chat Completions has nowhere to put it."""

    signature: str


@dataclass
class RedactedThinkingDelta:
    data: str


@dataclass
class ToolCallStart:
    #: Position the dialect should stream this call under. Providers choose
    #: it differently — Codex by call ordinal, Claude by content block — and
    #: clients only require that it be stable within one response.
    index: Any
    id: str
    name: str
    #: Complete when the upstream sends a call whole, empty when it streams.
    arguments: str = ""


@dataclass
class ToolCallArgs:
    index: Any
    fragment: str


@dataclass
class ToolCallEnd:
    """The assembled call. Carries no new bytes; completes the accumulator."""

    index: Any
    id: str
    name: str
    arguments: str


@dataclass
class Citation:
    url: str
    title: str | None = None
    start_index: int | None = None
    end_index: int | None = None


@dataclass
class Usage:
    #: Total input including cache. Anthropic reports these apart and OpenAI
    #: together, so each decoder normalises to the total here.
    prompt: int = 0
    completion: int = 0
    #: None when the upstream does not report one; encoders then add it up.
    total: int | None = None
    cache_read: int = 0
    cache_write: int = 0
    thinking: int = 0
    web_searches: int = 0


@dataclass
class Finish:
    reason: str = "end_turn"


StreamEvent = (
    TextDelta
    | ThinkingDelta
    | ThinkingSignature
    | RedactedThinkingDelta
    | ToolCallStart
    | ToolCallArgs
    | ToolCallEnd
    | Citation
    | Usage
    | Finish
)


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
    #: An explicit thinking budget, when the client gave one instead of a
    #: coarse effort tier. Preferred over reasoning_effort where supported.
    thinking_budget: int | None = None
    parallel_tool_calls: Any = None
    stream: bool = False
    session: str = ""
    #: Sampling knobs exactly as sent, holding only the keys the client sent.
    #: Whether a value is acceptable is the upstream's business, not the wire
    #: format's: Claude honours a temperature that Codex refuses.
    params: dict[str, Any] = field(default_factory=dict)
