"""ChatRequest -> a Claude Messages API request body."""

from __future__ import annotations

import sys
import uuid
from collections.abc import Collection
from typing import Any

from ...errors import RequestError
from ...ir import (
    ChatRequest,
    FunctionTool,
    Image,
    NativeAnthropicBlock,
    NativeResponseItem,
    Reasoning,
    Text,
    Thinking,
    ToolChoice,
    ToolResult,
    ToolUse,
    Turn,
    WebSearchTool,
)
from ...tools import arguments, render_function
from ..reasoning import ReasoningCache
from .subscription import CLAUDE_CODE_SYSTEM_MARKER
from .thinking import Outcome, Unpacked, unpack

WEB_SEARCH_BETA = "web-search-2025-03-05"
WEB_SEARCH_TOOL = "web_search_20250305"
#: Structured outputs remain gated; `output_config.format` needs this header.
STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"

#: Chat Completions knobs the Messages API has no equivalent for.
UNSUPPORTED = (
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "top_logprobs",
    "seed",
    "logit_bias",
)


def _number(value: Any, name: str, low: float, high: float, closed: bool) -> None:
    ok = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (low <= float(value) if closed else low < float(value))
        and float(value) <= high
    )
    if not ok:
        raise RequestError(f"{name} must be a number between {low:g} and {high:g}")


def _check(params: dict[str, Any]) -> None:
    for name in UNSUPPORTED:
        if params.get(name) is not None:
            raise RequestError(f"unsupported parameter: {name}")
    if params.get("temperature") is not None:
        _number(params["temperature"], "temperature", 0, 1, closed=True)
    if params.get("top_p") is not None:
        _number(params["top_p"], "top_p", 0, 1, closed=False)
    top_k = params.get("top_k")
    if top_k is not None and not (
        isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0
    ):
        raise RequestError("top_k must be a positive integer")


def _image(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        if not data:
            raise RequestError("image_url must be a data URL or an http(s) URL")
        media_type = header[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise RequestError("image_url must be a data URL or an http(s) URL")


def _text(block: Text) -> dict[str, Any]:
    return {
        "type": "text",
        "text": block.text,
        **({"cache_control": block.cache} if block.cache is not None else {}),
    }


def _native_thinking(block: Thinking) -> dict[str, Any]:
    if block.redacted:
        return {"type": "redacted_thinking", "data": block.redacted}
    return {"type": "thinking", "thinking": block.text, "signature": block.signature}


def _blocks(
    turn: Turn, cache: ReasoningCache | None, dropped: list[Outcome] | None = None
) -> list[dict[str, Any]]:
    """One turn, in the order its blocks actually occurred.

    Claude interleaves thinking with the tool calls it precedes, and verifies
    what it gets back, so position is part of the payload: grouping blocks by
    kind would rewrite a turn Claude signed.
    """
    blocks: list[dict[str, Any]] = []
    ordinals: list[int] = []
    lost = 0
    native_thinking = False
    for block in turn.blocks:
        if isinstance(block, NativeResponseItem):
            raise RequestError(
                "Claude upstream cannot faithfully represent Responses items: "
                + str(block.item.get("type", "unknown"))
            )
        if turn.role != "assistant" and isinstance(
            block, (Thinking, Reasoning, ToolUse)
        ):
            raise RequestError(
                f"unsupported {turn.role} content: {type(block).__name__}"
            )
        if isinstance(block, Thinking):
            native_thinking = True
            # A client can hand back a block it was given, including one this
            # upstream signed without ever streaming its text.
            native = _native_thinking(block)
            if _replayable(native):
                blocks.append(native)
            else:
                lost += 1
                if dropped is not None:
                    dropped.append(Outcome.WITHHELD)
        elif isinstance(block, Reasoning):
            recovered = unpack(block.item.get("encrypted_content"))
            if recovered.outcome is Outcome.OK and not _replayable(recovered.block):
                recovered = Unpacked(Outcome.WITHHELD)
            if recovered.outcome is Outcome.OK:
                assert recovered.block is not None
                blocks.append(recovered.block)
                ordinals.append(recovered.ordinal)
            else:
                lost += 1
                if dropped is not None:
                    dropped.append(recovered.outcome)
        elif isinstance(block, Text):
            if turn.role == "user" or block.text.strip():
                blocks.append(_text(block))
        elif isinstance(block, Image) and turn.role == "user":
            blocks.append(_image(block.url))
        elif isinstance(block, ToolResult) and turn.role == "user":
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.text,
                    **({"is_error": True} if block.is_error else {}),
                }
            )
        elif isinstance(block, ToolUse):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id or "toolu_" + uuid.uuid4().hex[:24],
                    "name": block.name,
                    "input": arguments(block.arguments, RequestError),
                }
            )
        elif isinstance(block, NativeAnthropicBlock):
            blocks.append(dict(block.item))
        else:
            raise RequestError(
                f"unsupported {turn.role} content: {type(block).__name__}"
            )
    uses = [block.id for block in turn.blocks if isinstance(block, ToolUse)]
    replay = cache.get([use for use in uses if use]) if cache and uses else []
    if native_thinking:
        # Anthropic clients return native thinking blocks themselves. The
        # cache exists for dialects such as Chat Completions that cannot carry
        # those blocks; prepending it here would duplicate a signed block and
        # Claude rejects the modified assistant turn.
        return blocks
    if ordinals:
        # A fraction of a signed turn is altered, where none of it is merely
        # thinner, so a turn that lost any block sends none. The cache gives
        # the count -- never the blocks, which it holds without their
        # positions -- and that is what catches a dropped trailing block,
        # whose ordinals still read 0..n-1.
        if lost or (replay and len(ordinals) < len(replay)):
            return [b for b in blocks if b["type"] not in _SIGNED]
        if ordinals != list(range(len(ordinals))):
            raise RequestError(
                "cannot replay Claude reasoning: the assistant turn's signed "
                "blocks arrived out of order or incomplete"
            )
        return blocks
    # No envelopes: a dialect that cannot carry reasoning, or a history older
    # than the envelope. Claude accepts a turn with no thinking.
    return replay + blocks


#: Block kinds Claude signs, and therefore will not accept rebuilt.
_SIGNED = frozenset({"thinking", "redacted_thinking"})

#: Why a reasoning item could not be replayed, as the operator reads it.
DROP_REASONS = {
    Outcome.FOREIGN: "this proxy did not write and cannot replay",
    Outcome.MALFORMED: "whose envelope arrived damaged",
    Outcome.BAD_VERSION: "whose envelope an unsupported version wrote",
    Outcome.WITHHELD: "whose thinking text the upstream never streamed",
}


def _replayable(block: dict[str, Any] | None) -> bool:
    """False for a signed block Claude will not take back.

    A thinking block whose text never arrived is one. Histories written before
    that was understood hold them by the hundred, so they are refused on the
    way in as well as on the way out.
    """
    if not isinstance(block, dict):
        return False
    if block.get("type") != "thinking":
        return True
    return bool(str(block.get("thinking", "")) and str(block.get("signature", "")))


def _web_tool(tool: WebSearchTool) -> dict[str, Any]:
    if tool.native is None:
        return {"type": WEB_SEARCH_TOOL, "name": "web_search"}
    if tool.source == "anthropic":
        return dict(tool.native)
    if tool.source != "responses" or set(tool.native) - {"type"}:
        raise RequestError(
            "Claude upstream cannot faithfully represent Responses web_search options"
        )
    return {"type": WEB_SEARCH_TOOL, "name": "web_search"}


def _tool_choice(choice: ToolChoice | None) -> dict[str, Any]:
    if choice is None or choice.kind == "auto":
        return {"type": "auto"}
    if choice.kind == "required":
        return {"type": "any"}
    if choice.kind == "tool":
        return {"type": "tool", "name": choice.name}
    raise RequestError("unsupported tool_choice")


def _stop_sequences(stop: Any) -> list[str]:
    if isinstance(stop, str):
        return [stop] if stop else []
    if isinstance(stop, list):
        return [str(item) for item in stop if item]
    return []


def build(
    request: ChatRequest,
    model: str,
    max_output: int | None = None,
    thinking: str | None = None,
    reasoning_efforts: Collection[str] | None = None,
    reasoning_cache: ReasoningCache | None = None,
) -> tuple[dict[str, Any], list[str]]:
    _check(request.params)

    messages = []
    dropped: list[Outcome] = []
    for turn in request.turns:
        blocks = _blocks(turn, reasoning_cache, dropped)
        if blocks:
            messages.append({"role": turn.role, "content": blocks})
    # Visible rather than silent: the turn still runs, but Claude is no longer
    # seeing reasoning it signed, and each reason is a different operator
    # problem -- a foreign history, a damaged blob, a version skew.
    for outcome, reason in DROP_REASONS.items():
        count = dropped.count(outcome)
        if count:
            sys.stderr.write(
                f"claude: dropped {count} reasoning item"
                f"{'' if count == 1 else 's'} {reason}\n"
            )
    if not messages or messages[0]["role"] != "user":
        raise RequestError("first message must be a user message")

    max_tokens = request.max_tokens
    if max_tokens is None:
        if (
            not isinstance(max_output, int)
            or isinstance(max_output, bool)
            or max_output <= 0
        ):
            raise RequestError(
                "model catalog did not report max output tokens; provide max_tokens"
            )
        max_tokens = max_output
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 0
    ):
        raise RequestError("max_tokens must not be negative")

    # Must be first to bill against the subscription pool; real clients
    # already send it.
    blocks = [block for block in request.system if block.text.strip()]
    marker = next(
        (block for block in blocks if block.text.strip() == CLAUDE_CODE_SYSTEM_MARKER),
        Text(CLAUDE_CODE_SYSTEM_MARKER),
    )
    system = [_text(marker)] + [
        _text(block)
        for block in blocks
        if block.text.strip() != CLAUDE_CODE_SYSTEM_MARKER
    ]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "stream": True,
        "cache_control": {"type": "ephemeral"},
        "system": system,
    }
    if request.params.get("temperature") is not None:
        body["temperature"] = float(request.params["temperature"])
    if request.params.get("top_p") is not None:
        body["top_p"] = float(request.params["top_p"])
    if request.params.get("top_k") is not None:
        body["top_k"] = int(request.params["top_k"])
    sequences = _stop_sequences(request.params.get("stop"))
    if sequences:
        body["stop_sequences"] = sequences

    betas: list[str] = []
    tools = []
    for tool in request.tools:
        if isinstance(tool, FunctionTool):
            if tool.options.get("defer_loading"):
                raise RequestError(
                    "Anthropic deferred tools require unsupported tool search"
                )
            tools.append(render_function(tool, "anthropic", "input_schema"))
        elif isinstance(tool, WebSearchTool):
            tools.append(_web_tool(tool))
            if WEB_SEARCH_BETA not in betas:
                betas.append(WEB_SEARCH_BETA)
        else:
            raise RequestError(
                "Claude upstream cannot faithfully represent Responses tools: "
                + str(tool.item.get("type", "unknown"))
            )
    choice = request.tool_choice
    if tools and not (choice and choice.kind == "none"):
        body["tools"] = tools
        body["tool_choice"] = _tool_choice(choice)
        if request.parallel_tool_calls is not None:
            body["tool_choice"][
                "disable_parallel_tool_use"
            ] = not request.parallel_tool_calls

    if request.output_format is not None:
        if request.output_format.kind != "json_schema":
            # Messages constrains output with a schema or not at all; a bare
            # "must be JSON" mode would have to be faked in the prompt.
            raise RequestError(
                "Claude upstream can constrain output only with a JSON schema; "
                "json_object has no Messages equivalent"
            )
        body["output_config"] = {
            "format": {"type": "json_schema", "schema": request.output_format.schema}
        }
        betas.append(STRUCTURED_OUTPUTS_BETA)

    # Anthropic's zero-token prewarm generates nothing. The transport switches
    # just this request to non-streaming and synthesizes the ordinary event
    # lifecycle, so generation-only controls have no work to do here.
    if max_tokens == 0:
        return body, betas

    effort = None
    if request.reasoning_effort:
        effort = str(request.reasoning_effort).casefold()
        supported = {str(item).casefold() for item in reasoning_efforts or ()}
        if supported and effort not in supported:
            raise RequestError(
                f"unsupported reasoning_effort: {request.reasoning_effort}"
            )
        # Claude's native effort control is independent of its thinking mode.
        # Do not approximate named effort tiers with fabricated token budgets.
        body.setdefault("output_config", {})["effort"] = effort

    budget = request.thinking_budget
    display = request.thinking_display or "summarized"
    if request.thinking_mode == "disabled":
        return body, betas
    if request.thinking_mode == "adaptive":
        # The client asked the model to size its own reasoning.
        body["thinking"] = {"type": "adaptive", "display": display}
        return body, betas
    if budget is not None:
        budget = min(budget, max_tokens - 1)
        if budget < 1024:
            raise RequestError(
                "max_tokens is too small for the requested thinking budget"
            )
        # The catalog can report enabled as unsupported yet honour it.
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget,
            "display": display,
        }
    elif thinking == "adaptive" or effort is not None or request.thinking_display:
        # Some live catalog entries advertise effort but omit their thinking
        # capability even though the model accepts adaptive thinking. An
        # explicit OpenAI-shaped reasoning request must therefore activate it
        # without relying solely on catalog metadata.
        body["thinking"] = {"type": "adaptive", "display": display}
    return body, betas
