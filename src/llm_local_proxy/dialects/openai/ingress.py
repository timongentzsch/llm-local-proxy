"""Chat Completions request body -> :class:`~llm_local_proxy.ir.ChatRequest`.

Structural parsing only. Whether a parameter is *supported* depends on the
upstream that will serve it, so providers make that call; this module only
rejects bodies that are not valid Chat Completions at all.
"""

from __future__ import annotations

from typing import Any

from ...errors import RequestError
from ...ir import (
    Block,
    ChatRequest,
    FunctionTool,
    Image,
    Text,
    Tool,
    ToolChoice,
    ToolResult,
    ToolUse,
    Turn,
    WebSearchTool,
)
from ..base import block_text

SYSTEM_ROLES = {"system", "developer"}
PARAMS = (
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "logprobs",
    "top_logprobs",
    "seed",
    "response_format",
    "logit_bias",
    "stop",
)


def _text(content: Any) -> str:
    """Flatten a content field that may be a string or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("message content must be a string or array")
    return block_text(content)


def _content(value: Any, role: str) -> list[Block]:
    if value is None:
        return []
    if isinstance(value, str):
        return [Text(value)] if value else []
    if not isinstance(value, list):
        raise RequestError("message content must be a string or array")
    blocks: list[Block] = []
    for part in value:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            blocks.append(Text(str(part.get("text", ""))))
        elif kind == "image_url":
            image = part.get("image_url", {})
            url = image.get("url") if isinstance(image, dict) else image
            if url:
                blocks.append(Image(str(url)))
        else:
            raise RequestError(f"unsupported {role} content type: {kind}")
    return blocks


def _tool_calls(message: dict[str, Any]) -> list[Block]:
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list):
        raise RequestError("tool_calls must be an array")
    blocks: list[Block] = []
    for call in calls:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict) or not function.get("name"):
            raise RequestError("invalid assistant tool call")
        blocks.append(
            ToolUse(
                id=str(call.get("id") or ""),
                name=str(function["name"]),
                arguments=function.get("arguments", "{}"),
            )
        )
    return blocks


def _add_tool_result(turns: list[Turn], message: dict[str, Any]) -> None:
    """Group consecutive tool results into one user turn.

    Anthropic wants every tool_result of a turn in a single user message;
    Codex reads them back one item at a time, so grouping costs it nothing
    and leaves one representation for both.
    """
    tool_use_id = message.get("tool_call_id") or message.get("tool_use_id")
    if not tool_use_id:
        raise RequestError("tool message is missing tool_call_id")
    block = ToolResult(
        tool_use_id=str(tool_use_id),
        text=_text(message.get("content")),
        is_error=bool(message.get("is_error")),
    )
    last = turns[-1] if turns else None
    if (
        last
        and last.role == "user"
        and all(isinstance(item, ToolResult) for item in last.blocks)
    ):
        last.blocks.append(block)
    else:
        turns.append(Turn("user", [block]))


def _tools(value: Any) -> list[Tool]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RequestError("tools must be an array")
    tools: list[Tool] = []
    for item in value:
        if not isinstance(item, dict):
            raise RequestError("invalid tool")
        if item.get("type") == "openrouter:web_search":
            parameters = item.get("parameters")
            size = (
                parameters.get("search_context_size")
                if isinstance(parameters, dict)
                else None
            )
            tools.append(
                WebSearchTool(size if size in {"low", "medium", "high"} else "")
            )
            continue
        function = item.get("function")
        if item.get("type") != "function" or not isinstance(function, dict):
            raise RequestError(
                "only function and openrouter:web_search tools are supported"
            )
        if not function.get("name"):
            raise RequestError("function tool name is required")
        tools.append(
            FunctionTool(
                name=str(function["name"]),
                parameters=function.get("parameters") or {"type": "object"},
                description=str(function.get("description") or ""),
            )
        )
    return tools


def _tool_choice(value: Any) -> ToolChoice | None:
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return ToolChoice(value)
    if isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function", {})
        if isinstance(function, dict) and function.get("name"):
            return ToolChoice("tool", str(function["name"]))
    raise RequestError("unsupported tool_choice")


def parse(body: dict[str, Any], session: str = "") -> ChatRequest:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    if body.get("n", 1) != 1:
        raise RequestError("n must be 1")

    system: list[str] = []
    turns: list[Turn] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role in SYSTEM_ROLES:
            text = _text(message.get("content"))
            if text:
                system.append(text)
        elif role == "user":
            blocks = _content(message.get("content"), "user")
            if blocks:
                turns.append(Turn("user", blocks))
        elif role == "assistant":
            blocks = _content(message.get("content"), "assistant")
            blocks += _tool_calls(message)
            if blocks:
                turns.append(Turn("assistant", blocks))
        elif role == "tool":
            _add_tool_result(turns, message)
        else:
            raise RequestError(f"unsupported message role: {role}")

    reasoning = body.get("reasoning")
    effort = body.get("reasoning_effort")
    if not effort and isinstance(reasoning, dict):
        effort = reasoning.get("effort")

    return ChatRequest(
        model=body["model"] if isinstance(body.get("model"), str) else "",
        # One block: Chat Completions has no cache breakpoints to preserve, and
        # every system and developer turn is one prompt to the upstream.
        system=[Text("\n\n".join(system))] if system else [],
        turns=turns,
        tools=_tools(body.get("tools")),
        tool_choice=_tool_choice(body.get("tool_choice")),
        max_tokens=body.get("max_tokens", body.get("max_completion_tokens")),
        reasoning_effort=effort,
        parallel_tool_calls=body.get("parallel_tool_calls"),
        stream=bool(body.get("stream", False)),
        session=session or str(body.get("session_id", "")),
        params={name: body[name] for name in PARAMS if name in body},
    )
