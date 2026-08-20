"""ChatRequest -> a Codex Responses API request body."""

from __future__ import annotations

import hashlib
from typing import Any

from ...ir import (
    ChatRequest,
    FunctionTool,
    Image,
    Text,
    Tool,
    ToolChoice,
    ToolResult,
    ToolUse,
)
from ...protocol import ReasoningCache, RequestError

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


def build(request: ChatRequest, cache: ReasoningCache) -> tuple[dict[str, Any], str]:
    if not request.model:
        raise RequestError("model is required")
    _reject_unsupported(request.params)

    instructions = "\n\n".join(request.system)
    items: list[dict[str, Any]] = []
    first_user = ""
    for turn in request.turns:
        if turn.role == "user" and not first_user:
            first_user = "\n".join(
                block.text for block in turn.blocks if isinstance(block, Text)
            )
        content = _content(turn.blocks, turn.role)
        if content:
            items.append({"role": turn.role, "content": content})
        uses = [block for block in turn.blocks if isinstance(block, ToolUse)]
        if uses:
            # Codex will not accept a tool result unless the encrypted
            # reasoning that produced the call is replayed with it.
            items.extend(cache.get([use.id for use in uses if use.id]))
            items.extend(
                {
                    "type": "function_call",
                    "call_id": use.id,
                    "name": use.name,
                    "arguments": str(use.arguments),
                }
                for use in uses
            )
        items.extend(
            {
                "type": "function_call_output",
                "call_id": block.tool_use_id,
                "output": block.text,
            }
            for block in turn.blocks
            if isinstance(block, ToolResult)
        )

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
    if request.reasoning_effort:
        body["reasoning"] = {"effort": request.reasoning_effort, "summary": "auto"}
        body["include"] = ["reasoning.encrypted_content"]
    return body, session
