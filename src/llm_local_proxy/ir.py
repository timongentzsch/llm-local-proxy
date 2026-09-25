"""The intermediate representation shared by every dialect and provider.

A downstream request is parsed once into :class:`ChatRequest`; each provider
renders its own upstream body from that. Without it the proxy would need one
converter per (dialect, provider) pair.

Common semantics use typed fields. Content without a lossless mapping uses
explicit opaque wire-format records; adapters must preserve or reject them.

Prompt caching never changes output, so every ``cache`` field is a hint: None
places no breakpoint, otherwise the block ends a cacheable prefix kept for that
TTL ("" for the upstream's default). A provider honours the hints its upstream
can express and otherwise relies on the upstream's automatic caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class Text:
    text: str
    cache: str | None = None
    #: Anthropic-format citations on replayed assistant text, for an upstream
    #: that verifies them; others read the text alone.
    citations: list[Any] | None = None


@dataclass
class Image:
    url: str
    cache: str | None = None


@dataclass
class ToolUse:
    #: May be empty; Chat Completions allows a call without one.
    id: str
    name: str
    arguments: Any
    #: The Responses namespace the called tool belongs to, if any.
    namespace: str = ""
    cache: str | None = None


@dataclass
class ToolResult:
    tool_use_id: str
    text: str
    is_error: bool = False
    cache: str | None = None


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


@dataclass
class NativeAnthropicBlock:
    """An Anthropic-only content block retained verbatim for replay."""

    item: dict[str, Any]


@dataclass
class HostedSearch:
    """A finished provider-run search that a client echoes back in history.

    It replays verbatim to an upstream that speaks ``source`` (the pause_turn
    continuation Anthropic requires) and is omitted elsewhere: the search
    already ran, its answer is in the transcript, and the record was written
    for the client by this proxy.
    """

    item: dict[str, Any]
    source: Literal["anthropic", "responses"]


Block = (
    Text
    | Image
    | ToolUse
    | ToolResult
    | Thinking
    | Reasoning
    | NativeResponseItem
    | NativeAnthropicBlock
    | HostedSearch
)


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    blocks: list[Block] = field(default_factory=list)


@dataclass
class FunctionTool:
    name: str
    parameters: dict[str, Any]
    description: str = ""
    strict: bool | None = None
    #: Extra fields belong to a wire format, not to a particular provider.
    source: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    cache: str | None = None


@dataclass
class WebSearchTool:
    """A search the provider runs, as its source format defined it.

    ``tools.responses_web_search`` and ``tools.anthropic_web_search`` translate
    it; options without an equivalent that change what is searched are
    refused, hints the target cannot act on are not.
    """

    native: dict[str, Any]
    source: Literal["responses", "anthropic"]


@dataclass
class NativeTool:
    """A Responses tool definition retained without schema conversion."""

    item: dict[str, Any]


@dataclass
class ToolNamespace:
    """A Responses namespace: tools grouped under one name.

    ``item`` is the definition as sent, for targets that speak Responses;
    others flatten ``tools`` with :func:`llm_local_proxy.tools.flatten`.
    """

    name: str
    tools: list[FunctionTool | NativeTool]
    item: dict[str, Any]


Tool = FunctionTool | WebSearchTool | NativeTool | ToolNamespace


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
    #: The reasoning item this text belongs to, when the upstream names one,
    #: so an item-based client sees one id from `added` through `done`.
    item_id: str = ""


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
    namespace: str = ""


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
    namespace: str = ""


@dataclass
class HostedToolEvent:
    """One lifecycle step of a tool the *provider* runs, not the client.

    Deliberately not :class:`ToolCallStart`/:class:`ToolCallEnd`: those oblige
    the client to execute something and answer with a result, and a hosted
    search has already been executed upstream. It is progress to show, never a
    tool round to take.
    """

    tool: str
    id: str
    phase: str
    #: What the provider searched for, when it said. Carried so an Anthropic
    #: client sees the `server_tool_use` input its upstream actually sent.
    query: str = ""
    #: Provider error code, when a hosted tool returned an error block.
    error_code: str = ""
    #: Native result payload when the provider exposes it for exact replay.
    result: Any = None


#: Ranked so only forward steps are emitted. Providers repeat their terminal
#: event -- a Responses search completes once as `web_search_call.completed`
#: and again as `output_item.done` -- and a replayed phase would duplicate the
#: client's lifecycle and double-count the search.
_PHASE_RANK = {"started": 0, "searching": 1, "completed": 2, "failed": 2}


def hosted_tool_step(seen: dict[str, str], id: str, phase: str) -> bool:
    """Record `phase` for search `id`; True when it advances the lifecycle."""
    rank = _PHASE_RANK.get(phase)
    if rank is None or rank <= _PHASE_RANK.get(seen.get(id, ""), -1):
        return False
    seen[id] = phase
    return True


@dataclass
class Citation:
    url: str
    title: str | None = None
    start_index: int | None = None
    end_index: int | None = None
    #: The Anthropic-format citation as issued. An upstream that verifies it on
    #: replay needs it whole, so an Anthropic client must receive it whole.
    native: dict[str, Any] | None = None


@dataclass
class Usage:
    #: Total input including cache; Anthropic reports these apart.
    prompt: int = 0
    completion: int = 0
    total: int | None = None
    cache_read: int = 0
    cache_write: int = 0
    #: The part of cache_write kept for an hour; None when the upstream does
    #: not report cache TTLs.
    cache_write_1h: int | None = None
    thinking: int = 0
    web_searches: int = 0


@dataclass
class Finish:
    reason: str = "end_turn"
    incomplete_reason: str | None = None
    stop_sequence: str | None = None


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
    | HostedToolEvent
    | Citation
    | Usage
    | Finish
)


class Decoder(Protocol):
    """Translate upstream wire events into the shared response vocabulary."""

    def decode(self, event: dict[str, Any]) -> list[StreamEvent]: ...
    def finish(self) -> list[StreamEvent]: ...


@dataclass
class OutputFormat:
    """A client's request that output be constrained, not merely prompted.

    Dialect-neutral because each dialect names the same capability its own way
    (Responses ``text.format``, Messages ``output_config.format``). ``kind`` is
    ``json_schema`` or ``json_object``; a plain-text format is no constraint at
    all and is dropped at the edge rather than carried as one.
    """

    kind: str
    name: str = ""
    schema: dict[str, Any] | None = None
    strict: bool = False


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
    #: Anthropic thinking visibility: "summarized" or "omitted".
    thinking_display: str = ""
    #: OpenAI reasoning summary mode: "auto", "concise", "detailed" or "none".
    reasoning_summary: str = ""
    #: Responses reasoning context: "auto", "current_turn" or "all_turns".
    reasoning_context: str = ""
    #: OpenAI output verbosity: "low", "medium" or "high".
    verbosity: str = ""
    parallel_tool_calls: bool | None = None
    stream: bool = False
    #: Account affinity: requests of one session start on the same account.
    session: str = ""
    #: The client's own prompt-cache key, for an upstream that takes one.
    cache_key: str = ""
    #: A breakpoint the upstream places automatically at the end of the prompt.
    cache: str | None = None
    #: As sent; each provider validates what it can honour.
    params: dict[str, Any] = field(default_factory=dict)
    #: Schema-constrained output when the client asked for one. A provider that
    #: cannot constrain its upstream must reject this rather than answer with
    #: unconstrained prose the client will fail to parse.
    output_format: OutputFormat | None = None
