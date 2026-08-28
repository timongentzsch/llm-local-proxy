"""ChatRequest -> a Codex Responses API request body."""

from __future__ import annotations

import hashlib
from collections.abc import Collection
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
    Tool,
    ToolChoice,
    ToolResult,
    ToolUse,
    WebSearchTool,
)
from ..reasoning import ReasoningCache

#: Knobs Codex does not expose, and the value of each that means "unset".
UNSUPPORTED = (
    "logprobs",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "top_logprobs",
    "top_p",
)
NEUTRAL: dict[str, Any] = {
    "temperature": 1,
    "top_p": 1,
    "logprobs": False,
    "response_format": {"type": "text"},
}
_UNSET = object()


def _reject_unsupported(params: dict[str, Any]) -> None:
    named = sorted(
        name
        for name in UNSUPPORTED
        if name in params
        and params[name] is not None
        and params[name] != NEUTRAL.get(name, _UNSET)
    )
    if named:
        raise RequestError(f"unsupported parameters: {', '.join(named)}")


def _content(turn_blocks: list[Any], role: str) -> list[dict[str, Any]]:
    kind = "output_text" if role == "assistant" else "input_text"
    content = []
    for block in turn_blocks:
        if isinstance(block, Text):
            content.append({"type": kind, "text": block.text})
        elif isinstance(block, Image):
            if role == "assistant":
                raise RequestError("unsupported assistant content type: image_url")
            content.append({"type": "input_image", "image_url": block.url})
    return content


def _tool(tool: Tool) -> dict[str, Any]:
    if isinstance(tool, NativeTool):
        return dict(tool.item)
    if isinstance(tool, (FunctionTool, WebSearchTool)) and tool.native is not None:
        return dict(tool.native)
    if isinstance(tool, FunctionTool):
        item = {"type": "function", "name": tool.name, "parameters": tool.parameters}
        if tool.description:
            item["description"] = tool.description
        return item
    item = {"type": "web_search"}
    if tool.context_size:
        item["search_context_size"] = tool.context_size
    return item


def _tool_choice(choice: ToolChoice | None) -> Any:
    if choice is None:
        return "auto"
    if choice.kind == "tool":
        return {"type": "function", "name": choice.name}
    return choice.kind


def _flush_content(items: list[dict[str, Any]], pending: list[Any], role: str) -> None:
    content = _content(pending, role)
    if content:
        items.append({"role": role, "content": content})
    pending.clear()


def _turn_items(turn: Any, cache: ReasoningCache) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pending: list[Any] = []
    uses = [block for block in turn.blocks if isinstance(block, ToolUse)]
    has_reasoning = any(isinstance(block, Reasoning) for block in turn.blocks)
    cached = [] if has_reasoning else cache.get([use.id for use in uses if use.id])
    cache_inserted = False
    for block in turn.blocks:
        if isinstance(block, (Text, Image)):
            pending.append(block)
            continue
        _flush_content(items, pending, turn.role)
        if isinstance(block, (Reasoning, NativeResponseItem)):
            items.append(dict(block.item))
        elif isinstance(block, ToolUse):
            if not cache_inserted:
                items.extend(cached)
                cache_inserted = True
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": str(block.arguments),
                }
            )
        elif isinstance(block, ToolResult):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": block.text,
                }
            )
    _flush_content(items, pending, turn.role)
    return items


def build(
    request: ChatRequest,
    cache: ReasoningCache,
    reasoning_efforts: Collection[str] | None = None,
) -> tuple[dict[str, Any], str]:
    if not request.model:
        raise RequestError("model is required")
    _reject_unsupported(request.params)

    # Codex caches prefixes implicitly, so breakpoints do not apply.
    instructions = "\n\n".join(block.text for block in request.system)
    items: list[dict[str, Any]] = []
    first_user = ""
    for turn in request.turns:
        if turn.role == "user" and not first_user:
            first_user = "\n".join(
                block.text for block in turn.blocks if isinstance(block, Text)
            )
        items.extend(_turn_items(turn, cache))

    session = request.session
    if not session:
        seed = f"{instructions}\0{first_user}".encode()
        session = "proxy-" + hashlib.sha256(seed).hexdigest()[:24]

    body: dict[str, Any] = {
        "model": request.model,
        "instructions": instructions,
        "input": items,
        "store": False,
        "stream": True,
        "prompt_cache_key": session,
    }
    tools = [_tool(tool) for tool in request.tools]
    if tools:
        body["tools"] = tools
        body["tool_choice"] = _tool_choice(request.tool_choice)
        parallel = request.parallel_tool_calls
        body["parallel_tool_calls"] = bool(True if parallel is None else parallel)
    # Always request the completed encrypted item: models can reason at their
    # catalog default even when the client omits an explicit effort.
    body["include"] = ["reasoning.encrypted_content"]
    if request.reasoning_effort:
        effort = str(request.reasoning_effort).casefold()
        supported = {str(item).casefold() for item in reasoning_efforts or ()}
        if supported and effort not in supported:
            raise RequestError(
                f"unsupported reasoning_effort: {request.reasoning_effort}"
            )
        body["reasoning"] = {"effort": effort, "summary": "auto"}
    return body, session
