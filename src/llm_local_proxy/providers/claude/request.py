"""ChatRequest -> a Claude Messages API request body."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ...errors import RequestError
from ...ir import (
    ChatRequest,
    FunctionTool,
    Image,
    NativeResponseItem,
    NativeTool,
    Reasoning,
    Text,
    Thinking,
    ToolChoice,
    ToolResult,
    ToolUse,
    Turn,
    WebSearchTool,
)
from ..reasoning import ReasoningCache
from .subscription import CLAUDE_CODE_SYSTEM_MARKER

WEB_SEARCH_BETA = "web-search-2025-03-05"
WEB_SEARCH_TOOL = "web_search_20250305"
DEFAULT_MAX_OUTPUT_TOKENS = 32768

#: Chat Completions knobs the Messages API has no equivalent for.
UNSUPPORTED = (
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "top_logprobs",
    "seed",
    "response_format",
    "logit_bias",
)

EFFORTS = {"low", "medium", "high", "xhigh", "max"}


def _reject_unsupported(params: dict[str, Any]) -> None:
    for name in UNSUPPORTED:
        if params.get(name) is not None:
            raise RequestError(f"unsupported parameter: {name}")


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
    _reject_unsupported(params)
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


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            raise RequestError("tool call arguments must be a JSON object") from None
    if not isinstance(value, dict):
        raise RequestError("tool call arguments must be an object")
    return value


def _user_blocks(turn: Turn) -> list[dict[str, Any]]:
    blocks = []
    for block in turn.blocks:
        if isinstance(block, Text):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, Image):
            blocks.append(_image(block.url))
        elif isinstance(block, ToolResult):
            item: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.text,
            }
            if block.is_error:
                item["is_error"] = True
            blocks.append(item)
    return blocks


def _assistant_blocks(turn: Turn, cache: ReasoningCache | None) -> list[dict[str, Any]]:
    # An empty text block is rejected upstream.
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": block.text}
        for block in turn.blocks
        if isinstance(block, Text) and block.text.strip()
    ]
    blocks.extend(
        {"type": "thinking", "thinking": block.text, "signature": block.signature}
        if not block.redacted
        else {"type": "redacted_thinking", "data": block.redacted}
        for block in turn.blocks
        if isinstance(block, Thinking)
    )
    blocks.extend(
        _responses_reasoning(block.item)
        for block in turn.blocks
        if isinstance(block, Reasoning)
    )
    uses = [block for block in turn.blocks if isinstance(block, ToolUse)]
    for use in uses:
        blocks.append(
            {
                "type": "tool_use",
                "id": use.id or "toolu_" + uuid.uuid4().hex[:24],
                "name": use.name,
                "input": _arguments(use.arguments),
            }
        )
    has_carried_reasoning = any(
        isinstance(block, (Thinking, Reasoning)) for block in turn.blocks
    )
    # Compatibility cache is only for dialects that cannot carry reasoning.
    if cache is not None and uses and not has_carried_reasoning:
        blocks = cache.get([use.id for use in uses if use.id]) + blocks
    return blocks


def _responses_reasoning(item: dict[str, Any]) -> dict[str, Any]:
    """Recover a Claude thinking block from a stateless Responses item."""
    encrypted = item.get("encrypted_content")
    if not isinstance(encrypted, str) or not encrypted:
        raise RequestError("Claude reasoning replay requires encrypted_content")
    summary = item.get("summary")
    text = (
        "".join(str(part.get("text", "")) for part in summary if isinstance(part, dict))
        if isinstance(summary, list)
        else ""
    )
    if text:
        return {"type": "thinking", "thinking": text, "signature": encrypted}
    return {"type": "redacted_thinking", "data": encrypted}


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
    reasoning_cache: ReasoningCache | None = None,
) -> tuple[dict[str, Any], list[str]]:
    _check(request.params)

    native_items = [
        block.item
        for turn in request.turns
        for block in turn.blocks
        if isinstance(block, NativeResponseItem)
    ]
    if native_items:
        kinds = sorted({str(item.get("type", "unknown")) for item in native_items})
        raise RequestError(
            "Claude upstream cannot faithfully represent Responses items: "
            + ", ".join(kinds)
        )
    messages = []
    for turn in request.turns:
        blocks = (
            _user_blocks(turn)
            if turn.role == "user"
            else _assistant_blocks(turn, reasoning_cache)
        )
        if blocks:
            messages.append({"role": turn.role, "content": blocks})
    if not messages or messages[0]["role"] != "user":
        raise RequestError("first message must be a user message")

    max_tokens = request.max_tokens
    if max_tokens is None:
        max_tokens = max_output if isinstance(max_output, int) and max_output > 0 else 0
        max_tokens = max_tokens or DEFAULT_MAX_OUTPUT_TOKENS
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise RequestError("max_tokens must be a positive integer")

    # Must be first to bill against the subscription pool; real clients
    # already send it.
    blocks = [block for block in request.system if block.text.strip()]
    system: list[dict[str, Any]] = []
    if not any(block.text.strip() == CLAUDE_CODE_SYSTEM_MARKER for block in blocks):
        system.append({"type": "text", "text": CLAUDE_CODE_SYSTEM_MARKER})
    for block in blocks:
        item: dict[str, Any] = {"type": "text", "text": block.text}
        if block.cache:
            item["cache_control"] = {"type": "ephemeral"}
        system.append(item)
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
    native = [tool for tool in request.tools if isinstance(tool, NativeTool)]
    if native:
        kinds = sorted({str(tool.item.get("type", "unknown")) for tool in native})
        raise RequestError(
            "Claude upstream cannot faithfully represent Responses tools: "
            + ", ".join(kinds)
        )
    function_tools = [tool for tool in request.tools if isinstance(tool, FunctionTool)]
    for tool in function_tools:
        if tool.native is not None:
            unsupported = sorted(
                set(tool.native)
                - {"type", "name", "parameters", "description", "strict"}
            )
            if unsupported:
                raise RequestError(
                    "Claude upstream cannot faithfully represent function tool fields: "
                    + ", ".join(unsupported)
                )
    web_tools = [tool for tool in request.tools if isinstance(tool, WebSearchTool)]
    for tool in web_tools:
        if tool.native is not None and set(tool.native) - {"type"}:
            raise RequestError(
                "Claude upstream cannot faithfully represent Responses web_search options"
            )
    tools = [
        {
            "name": tool.name,
            "input_schema": tool.parameters,
            **({"description": tool.description} if tool.description else {}),
            **(
                {"strict": tool.native["strict"]}
                if tool.native is not None and "strict" in tool.native
                else {}
            ),
        }
        for tool in function_tools
    ]
    if web_tools:
        tools.append({"type": WEB_SEARCH_TOOL, "name": "web_search"})
        betas.append(WEB_SEARCH_BETA)
    choice = request.tool_choice
    if tools and not (choice and choice.kind == "none"):
        body["tools"] = tools
        body["tool_choice"] = _tool_choice(choice)

    effort = None
    if request.reasoning_effort:
        effort = str(request.reasoning_effort).casefold()
        if effort not in EFFORTS:
            raise RequestError(
                f"unsupported reasoning_effort: {request.reasoning_effort}"
            )
        # Claude's native effort control is independent of its thinking mode.
        # Do not approximate named effort tiers with fabricated token budgets.
        body["output_config"] = {"effort": effort}

    budget = request.thinking_budget
    if request.thinking_mode == "disabled":
        return body, betas
    if request.thinking_mode == "adaptive":
        # The client asked the model to size its own reasoning.
        body["thinking"] = {"type": "adaptive"}
        return body, betas
    if budget is not None:
        budget = min(budget, max_tokens - 1)
        if budget < 1024:
            raise RequestError(
                "max_tokens is too small for the requested thinking budget"
            )
        # The catalog can report enabled as unsupported yet honour it.
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}
    elif thinking == "adaptive":
        body["thinking"] = {"type": "adaptive"}
    return body, betas
