"""Anthropic Messages request body -> :class:`~llm_local_proxy.ir.ChatRequest`.

Verified against specs/anthropic-openapi.json. Structural parsing only;
whether an upstream can honour a parameter is that provider's judgement.
"""

from __future__ import annotations

from typing import Any

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
from ...protocol import RequestError

#: Server tools the proxy can actually serve. Everything else in the tools
#: union runs inside Anthropic's infrastructure and has no Codex equivalent,
#: so it is refused rather than silently dropped.
WEB_SEARCH_PREFIX = "web_search_"
#: Top-level fields with no proxy implementation. Refusing loudly beats
#: pretending they took effect.
REJECTED = ("container", "mcp_servers", "service_tier", "output_config")
PARAMS = ("temperature", "top_p", "top_k")
CHOICES = {"auto": "auto", "any": "required", "tool": "tool", "none": "none"}


def _text(value: Any) -> str:
    """Text of a content field that may be a string or a block list."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(
        str(part.get("text", ""))
        for part in value
        if isinstance(part, dict) and part.get("type") == "text"
    )


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
        # Signed blocks must survive verbatim or the upstream refuses the turn.
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
    if role not in {"user", "assistant"}:
        raise RequestError(f"unsupported message role: {role}")
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
            tools.append(WebSearchTool())
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
    """Returns the choice and whether parallel calls were disabled."""
    if value is None:
        return None, None
    if not isinstance(value, dict) or value.get("type") not in CHOICES:
        raise RequestError("unsupported tool_choice")
    parallel = None
    if value.get("disable_parallel_tool_use"):
        parallel = False
    kind = CHOICES[value["type"]]
    return ToolChoice(kind, str(value.get("name") or "")), parallel


def _system(value: Any) -> list[str]:
    if value is None:
        return []
    text = _text(value)
    return [text] if text else []


def parse(body: dict[str, Any], session: str = "") -> ChatRequest:
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise RequestError("model is required")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    max_tokens = body.get("max_tokens")
    # max_tokens is required, and zero is legal: it pre-warms the prompt
    # cache without generating.
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise RequestError("max_tokens is required and must be an integer")
    if max_tokens < 0:
        raise RequestError("max_tokens must not be negative")
    for name in REJECTED:
        if body.get(name) is not None:
            raise RequestError(f"unsupported parameter: {name}")

    turns = [_turn(message) for message in messages]
    choice, parallel = _tool_choice(body.get("tool_choice"))
    thinking = body.get("thinking")
    budget = None
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens")

    return ChatRequest(
        model=model,
        system=_system(body.get("system")),
        turns=[turn for turn in turns if turn.blocks],
        tools=_tools(body.get("tools")),
        tool_choice=choice,
        max_tokens=max_tokens,
        thinking_budget=budget if isinstance(budget, int) else None,
        parallel_tool_calls=parallel,
        stream=bool(body.get("stream", False)),
        session=session,
        params={
            **{name: body[name] for name in PARAMS if name in body},
            **({"stop": body["stop_sequences"]} if "stop_sequences" in body else {}),
        },
    )
