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
    #: Prompt-cache breakpoint; honoured by Claude, ignored by Codex.
    cache: bool = False


@dataclass
class Image:
    url: str


@dataclass
class ToolUse:
    #: May be empty; Chat Completions allows a call without one.
    id: str
    name: str
    arguments: Any


@dataclass
class ToolResult:
    tool_use_id: str
    text: str
    is_error: bool = False


@dataclass
class Thinking:
    """Signed reasoning; must round-trip byte-exactly or upstream rejects it."""

    text: str
    signature: str = ""
    redacted: str = ""


@dataclass
class Reasoning:
    """Opaque Responses reasoning item carried verbatim between turns."""

    item: dict[str, Any]


@dataclass
class NativeResponseItem:
    """A Responses input/output item with no lossless cross-dialect mapping."""

    item: dict[str, Any]


Block = Text | Image | ToolUse | ToolResult | Thinking | Reasoning | NativeResponseItem


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    blocks: list[Block] = field(default_factory=list)


@dataclass
class FunctionTool:
    name: str
    parameters: dict[str, Any]
    description: str = ""
    native: dict[str, Any] | None = None


@dataclass
class WebSearchTool:
    context_size: str = ""
    native: dict[str, Any] | None = None


@dataclass
class NativeTool:
    """A Responses tool definition retained without schema conversion."""

    item: dict[str, Any]


Tool = FunctionTool | WebSearchTool | NativeTool


@dataclass
class ToolChoice:
    kind: Literal["auto", "none", "required", "tool"] = "auto"
    name: str = ""


# --- response side ---------------------------------------------------------
# Finish reasons use Anthropic's seven-value vocabulary; the Chat Completions
# encoder narrows them to four.


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class ThinkingSignature:
    """Closes a thinking block."""

    signature: str


@dataclass
class RedactedThinkingDelta:
    data: str


@dataclass
class ReasoningItem:
    """A complete opaque reasoning item for stateless Responses replay."""

    item: dict[str, Any]


@dataclass
class NativeItem:
    """A complete native Responses output item."""

    item: dict[str, Any]


@dataclass
class ToolCallStart:
    #: Stable within one response; providers number calls differently.
    index: Any
    id: str
    name: str
    arguments: str = ""


@dataclass
class ToolCallArgs:
    index: Any
    fragment: str


@dataclass
class ToolCallEnd:
    """The assembled call; carries no new bytes."""

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
    #: Total input including cache; Anthropic reports these apart.
    prompt: int = 0
    completion: int = 0
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
    | ReasoningItem
    | NativeItem
    | ToolCallStart
    | ToolCallArgs
    | ToolCallEnd
    | Citation
    | Usage
    | Finish
)


@dataclass
class ChatRequest:
    model: str = ""
    #: Blocks rather than one string so cache breakpoints survive.
    system: list[Text] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    tools: list[Tool] = field(default_factory=list)
    tool_choice: ToolChoice | None = None
    max_tokens: Any = None
    reasoning_effort: Any = None
    #: Explicit budget; preferred over reasoning_effort where supported.
    thinking_budget: int | None = None
    #: "adaptive" or "disabled" when named; neither maps to a budget.
    thinking_mode: str = ""
    #: Claude thinking visibility: "summarized" or "omitted".
    thinking_display: str = ""
    parallel_tool_calls: Any = None
    stream: bool = False
    session: str = ""
    #: As sent; each provider validates what it can honour.
    params: dict[str, Any] = field(default_factory=dict)
