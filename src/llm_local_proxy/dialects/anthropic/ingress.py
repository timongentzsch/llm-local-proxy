"""Anthropic Messages request body -> :class:`~llm_local_proxy.ir.ChatRequest`.

Verified against specs/anthropic-openapi.json. Structural parsing only;
whether an upstream can honour a parameter is that provider's judgement.
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
    Thinking,
    Tool,
    ToolChoice,
    ToolResult,
    ToolUse,
    Turn,
    WebSearchTool,
)
from ..base import block_text

#: The only server tool the proxy can serve; the rest are refused.
WEB_SEARCH_PREFIX = "web_search_"
#: No proxy implementation; refused rather than silently ignored. Fields the
#: proxy cannot act on but real clients send -- context_management, metadata --
#: are deliberately absent: they are accepted and dropped.
REJECTED = ("container", "mcp_servers", "service_tier")
PARAMS = ("temperature", "top_p", "top_k")
CHOICES = {"auto": "auto", "any": "required", "tool": "tool", "none": "none"}


def _text(value: Any) -> str:
    """Text of a content field that may be a string or a block list."""
    if isinstance(value, str):
        return value
    return block_text(value) if isinstance(value, list) else ""


def _image(source: Any) -> Image:
    if not isinstance(source, dict):
        raise RequestError("image source must be an object")
    kind = source.get("type")
    if kind == "url" and source.get("url"):
        return Image(str(source["url"]))
    if kind == "base64" and source.get("data"):
        media = source.get("media_type") or "image/png"
        return Image(f"data:{media};base64,{source['data']}")
    raise RequestError("image source must be a url or base64 source")


def _block(part: Any) -> Block | None:
    if not isinstance(part, dict):
        raise RequestError("each content block must be an object")
    kind = part.get("type")
    if kind == "text":
        return Text(str(part.get("text", "")))
    if kind == "image":
        return _image(part.get("source"))
    if kind == "tool_use":
        return ToolUse(
            id=str(part.get("id") or ""),
            name=str(part.get("name") or ""),
            arguments=part.get("input", {}),
        )
    if kind == "tool_result":
        tool_use_id = part.get("tool_use_id")
        if not tool_use_id:
            raise RequestError("tool_result is missing tool_use_id")
        return ToolResult(
            tool_use_id=str(tool_use_id),
            text=_text(part.get("content")),
            is_error=bool(part.get("is_error")),
        )
    if kind == "thinking":
        # Must survive verbatim or the upstream refuses the turn.
        return Thinking(
            text=str(part.get("thinking", "")),
            signature=str(part.get("signature", "")),
        )
    if kind == "redacted_thinking":
        return Thinking(text="", redacted=str(part.get("data", "")))
    raise RequestError(f"unsupported content block: {kind}")


def _turn(message: Any) -> Turn:
    if not isinstance(message, dict):
        raise RequestError("each message must be an object")
    role = message.get("role")
    if role not in {"user", "assistant", "system"}:
        raise RequestError(f"unsupported message role: {role}")
    # Spec allows a system role inside messages and Claude Code uses it;
    # neither upstream has a third role, so keep the text in place.
    if role == "system":
        role = "user"
    content = message.get("content")
    if isinstance(content, str):
        blocks: list[Block] = [Text(content)] if content else []
    elif isinstance(content, list):
        blocks = [block for block in map(_block, content) if block is not None]
    else:
        raise RequestError("message content must be a string or array")
    return Turn(role, blocks)


def _tools(value: Any) -> list[Tool]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RequestError("tools must be an array")
    tools: list[Tool] = []
    for item in value:
        if not isinstance(item, dict):
            raise RequestError("invalid tool")
        kind = str(item.get("type") or "")
        if kind.startswith(WEB_SEARCH_PREFIX):
            tools.append(WebSearchTool(native=dict(item)))
            continue
        if kind and not item.get("input_schema"):
            raise RequestError(f"unsupported server tool: {kind}")
        name = item.get("name")
        if not name:
            raise RequestError("tool name is required")
        tools.append(
            FunctionTool(
                name=str(name),
                parameters=item.get("input_schema") or {"type": "object"},
                description=str(item.get("description") or ""),
            )
        )
    return tools


def _tool_choice(value: Any) -> tuple[ToolChoice | None, Any]:
    """The choice, and whether parallel calls were disabled."""
    if value is None:
        return None, None
    if not isinstance(value, dict) or value.get("type") not in CHOICES:
        raise RequestError("unsupported tool_choice")
    parallel = None
    if value.get("disable_parallel_tool_use"):
        parallel = False
    kind = CHOICES[value["type"]]
    return ToolChoice(kind, str(value.get("name") or "")), parallel


def _system(value: Any) -> list[Text]:
    """System blocks in order, preserving cache breakpoints."""
    if isinstance(value, str):
        return [Text(value)] if value else []
    if not isinstance(value, list):
        return []
    blocks = []
    for part in value:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            blocks.append(
                Text(str(part["text"]), cache=bool(part.get("cache_control")))
            )
    return blocks


def parse(body: dict[str, Any], session: str = "") -> ChatRequest:
    return _parse(body, session, generating=True)


def parse_count(body: dict[str, Any], session: str = "") -> ChatRequest:
    """Parse a count_tokens body, which has no max_tokens."""
    return _parse(body, session, generating=False)


def _parse(body: dict[str, Any], session: str, generating: bool) -> ChatRequest:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise RequestError("model is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    max_tokens = body.get("max_tokens")
    if generating:
        # Zero is legal: it pre-warms the prompt cache without generating.
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise RequestError("max_tokens is required and must be an integer")
        if max_tokens < 0:
            raise RequestError("max_tokens must not be negative")
    else:
        max_tokens = None
    for name in REJECTED:
        if body.get(name) is not None:
            raise RequestError(f"unsupported parameter: {name}")

    turns = [_turn(message) for message in messages]
    choice, parallel = _tool_choice(body.get("tool_choice"))
    thinking = body.get("thinking")
    mode = thinking.get("type") if isinstance(thinking, dict) else None
    budget = thinking.get("budget_tokens") if mode == "enabled" else None
    display = thinking.get("display") if isinstance(thinking, dict) else None
    if display is not None and display not in {"summarized", "omitted"}:
        raise RequestError("thinking.display must be summarized or omitted")
    # The provider validates this against the selected model's live catalog.
    output_config = body.get("output_config")
    effort = output_config.get("effort") if isinstance(output_config, dict) else None

    return ChatRequest(
        model=model,
        system=_system(body.get("system")),
        turns=[turn for turn in turns if turn.blocks],
        tools=_tools(body.get("tools")),
        tool_choice=choice,
        max_tokens=max_tokens,
        reasoning_effort=effort,
        thinking_budget=budget if isinstance(budget, int) else None,
        thinking_mode=mode if mode in {"adaptive", "disabled"} else "",
        thinking_display=str(display or ""),
        parallel_tool_calls=parallel,
        stream=bool(body.get("stream", False)),
        session=session,
        params={
            **{name: body[name] for name in PARAMS if name in body},
            **({"stop": body["stop_sequences"]} if "stop_sequences" in body else {}),
        },
    )
