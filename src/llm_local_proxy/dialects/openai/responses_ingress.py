"""OpenAI Responses request body -> dialect-neutral request."""

from __future__ import annotations

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
    ToolChoice,
    ToolResult,
    ToolUse,
    Turn,
    WebSearchTool,
)
from .reasoning import options as reasoning_options


def _content(value: Any) -> list[Text | Image]:
    if isinstance(value, str):
        return [Text(value)] if value else []
    if not isinstance(value, list):
        raise RequestError("message content must be a string or array")
    blocks: list[Text | Image] = []
    for part in value:
        if not isinstance(part, dict):
            raise RequestError("message content parts must be objects")
        kind = part.get("type")
        if kind in {"input_text", "output_text", "text"}:
            blocks.append(Text(str(part.get("text", ""))))
        elif kind == "input_image" and part.get("image_url"):
            blocks.append(Image(str(part["image_url"])))
        else:
            raise RequestError(f"unsupported Responses content type: {kind}")
    return blocks


def _append(turns: list[Turn], role: str, blocks: list[Any]) -> None:
    if not blocks:
        return
    if turns and turns[-1].role == role:
        turns[-1].blocks.extend(blocks)
    else:
        turns.append(Turn(role, blocks))


def _input(value: Any, system: list[Text], additional_tools: list[Any]) -> list[Turn]:
    if isinstance(value, str):
        return [Turn("user", [Text(value)])] if value else []
    if not isinstance(value, list):
        raise RequestError("input must be a string or array")
    turns: list[Turn] = []
    for item in value:
        if not isinstance(item, dict):
            raise RequestError("input items must be objects")
        kind = item.get("type")
        if kind == "additional_tools":
            additional_tools.extend(_tools(item.get("tools")))
        elif kind == "message" or (kind is None and item.get("role")):
            role = item.get("role")
            blocks = _content(item.get("content", []))
            if role in {"system", "developer"}:
                system.extend(block for block in blocks if isinstance(block, Text))
            elif role in {"user", "assistant"}:
                _append(turns, role, blocks)
            else:
                raise RequestError(f"unsupported Responses role: {role}")
        elif kind == "reasoning":
            # Keep all provider-owned fields opaque. Only type is interpreted.
            _append(turns, "assistant", [Reasoning(dict(item))])
        elif kind == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if not call_id or not item.get("name"):
                raise RequestError("function_call requires call_id and name")
            _append(
                turns,
                "assistant",
                [ToolUse(str(call_id), str(item["name"]), item.get("arguments", "{}"))],
            )
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not call_id:
                raise RequestError("function_call_output requires call_id")
            output = item.get("output", "")
            if isinstance(output, str):
                _append(turns, "user", [ToolResult(str(call_id), output)])
            else:
                # Structured text/image/file outputs have no lossless legacy IR
                # representation; keep the entire Responses item for Codex.
                _append(turns, "user", [NativeResponseItem(dict(item))])
        else:
            # Current Responses adds native item kinds regularly (custom calls,
            # programs, shell/patch calls, tool search, compaction). Preserve
            # unknown typed items for a native upstream instead of dropping them.
            if not isinstance(kind, str) or not kind:
                raise RequestError("Responses input item requires type")
            role = "user" if kind.endswith("_output") else "assistant"
            _append(turns, role, [NativeResponseItem(dict(item))])
    return turns


def _tools(value: Any) -> list[FunctionTool | WebSearchTool | NativeTool]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise RequestError("tools must be an array")
    tools: list[FunctionTool | WebSearchTool | NativeTool] = []
    for item in value:
        if not isinstance(item, dict):
            raise RequestError("invalid tool")
        kind = item.get("type")
        if kind in {"web_search", "web_search_preview"}:
            tools.append(
                WebSearchTool(
                    str(item.get("search_context_size") or ""), native=dict(item)
                )
            )
        elif kind == "function" and item.get("name"):
            tools.append(
                FunctionTool(
                    str(item["name"]),
                    item.get("parameters") or {"type": "object"},
                    str(item.get("description") or ""),
                    native=dict(item),
                )
            )
        elif kind in {"custom", "namespace", "tool_search"}:
            # These newer Responses definitions have no Chat Completions
            # equivalent. Providers that speak Responses can forward them;
            # other providers reject them rather than silently weakening tools.
            tools.append(NativeTool(dict(item)))
        else:
            raise RequestError(f"unsupported Responses tool: {kind}")
    return tools


def _choice(value: Any) -> ToolChoice | None:
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return ToolChoice(value)
    if (
        isinstance(value, dict)
        and value.get("type") == "function"
        and value.get("name")
    ):
        return ToolChoice("tool", str(value["name"]))
    raise RequestError("unsupported tool_choice")


def parse(body: dict[str, Any], session: str = "") -> ChatRequest:
    if body.get("store") is True:
        raise RequestError(
            "only stateless Responses requests are supported; set store to false"
        )
    if body.get("previous_response_id") is not None:
        raise RequestError(
            "previous_response_id is not supported; resend complete input history"
        )
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise RequestError("model is required")
    instructions = body.get("instructions", "")
    system = [Text(str(instructions))] if instructions else []
    additional_tools: list[Any] = []
    turns = _input(body.get("input", ""), system, additional_tools)
    effort, thinking_display = reasoning_options(body.get("reasoning"))
    params = {key: body[key] for key in ("temperature", "top_p", "stop") if key in body}
    return ChatRequest(
        model=model,
        system=system,
        turns=turns,
        tools=[*_tools(body.get("tools")), *additional_tools],
        tool_choice=_choice(body.get("tool_choice")),
        max_tokens=body.get("max_output_tokens"),
        reasoning_effort=effort,
        thinking_display=thinking_display,
        parallel_tool_calls=body.get("parallel_tool_calls"),
        stream=bool(body.get("stream")),
        session=session or str(body.get("prompt_cache_key") or ""),
        params=params,
    )
