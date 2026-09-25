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
    Image,
    OutputFormat,
    Text,
    Tool,
    ToolResult,
    ToolUse,
    Turn,
    WebSearchTool,
)
from ...tools import definitions, optional_bool, parse_choice, parse_function
from ..base import block_text
from .output import VERBOSITY, enum_value, format_of
from .reasoning import options as reasoning_options

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
    "logit_bias",
    "stop",
)


def _output_format(value: Any) -> OutputFormat | None:
    """Read `response_format`, whose schema sits one level deeper than Responses."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RequestError("response_format must be an object")
    kind = value.get("type")
    nested = value.get("json_schema") if kind == "json_schema" else {}
    if kind == "json_schema" and not isinstance(nested, dict):
        raise RequestError("response_format.json_schema must be an object")
    return format_of(kind, nested or {})


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
            raise RequestError("message content parts must be objects")
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


def _web_search_options(value: Any) -> list[Tool]:
    """OpenAI's Chat Completions search parameter, as the Responses search tool.

    Both describe one search: the same context size, and the same approximate
    location, which Chat Completions nests one level deeper.
    """
    if value is None:
        return []
    if not isinstance(value, dict):
        raise RequestError("web_search_options must be an object")
    extra = sorted(set(value) - {"search_context_size", "user_location"})
    if extra:
        raise RequestError(f"unsupported web_search_options: {', '.join(extra)}")
    tool: dict[str, Any] = {"type": "web_search"}
    size = value.get("search_context_size")
    if size is not None:
        if size not in {"low", "medium", "high"}:
            raise RequestError(
                "web_search_options.search_context_size must be low, medium or high"
            )
        tool["search_context_size"] = size
    location = value.get("user_location")
    if location is not None:
        approximate = (
            isinstance(location, dict) and location.get("type") == "approximate"
        )
        detail = location.get("approximate") if approximate else None
        if not isinstance(detail, dict):
            raise RequestError("web_search_options.user_location must be approximate")
        tool["user_location"] = {"type": "approximate", **detail}
    return [WebSearchTool(tool, "responses")]


def _tools(value: Any) -> list[Tool]:
    tools: list[Tool] = []
    for item in definitions(value):
        function = item.get("function")
        if item.get("type") != "function" or not isinstance(function, dict):
            raise RequestError(
                "only function tools are supported; request web search with "
                "web_search_options"
            )
        tools.append(parse_function(function, "chat_completions"))
    return tools


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
        if not isinstance(role, str):
            raise RequestError("message role must be a string")
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

    nested_effort, thinking_display, summary = reasoning_options(body.get("reasoning"))
    effort = body.get("reasoning_effort")
    if not effort:
        effort = nested_effort

    return ChatRequest(
        model=body["model"] if isinstance(body.get("model"), str) else "",
        # One block: Chat Completions has no cache breakpoints to preserve, and
        # every system and developer turn is one prompt to the upstream.
        system=[Text("\n\n".join(system))] if system else [],
        turns=turns,
        tools=[
            *_tools(body.get("tools")),
            *_web_search_options(body.get("web_search_options")),
        ],
        tool_choice=parse_choice(body.get("tool_choice"), nested=True),
        max_tokens=body.get("max_tokens", body.get("max_completion_tokens")),
        reasoning_effort=effort,
        thinking_display=thinking_display,
        reasoning_summary=summary,
        verbosity=enum_value(body.get("verbosity"), VERBOSITY, "verbosity"),
        parallel_tool_calls=optional_bool(
            body.get("parallel_tool_calls"), "parallel_tool_calls"
        ),
        stream=optional_bool(body.get("stream"), "stream") or False,
        session=session or str(body.get("session_id", "")),
        params={name: body[name] for name in PARAMS if name in body},
        output_format=_output_format(body.get("response_format")),
    )
