"""Shared tool contracts and wire encoders, independent of provider implementations.

Common fields cross formats unchanged. Extra options retain their origin and
are rejected on a different format unless an adapter explicitly maps them.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .errors import RequestError
from .ir import FunctionTool, NativeTool, Tool, ToolChoice, WebSearchTool


def definitions(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RequestError("tools must be an array of objects")
    return value


def optional_bool(value: Any, name: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise RequestError(f"{name} must be a boolean")
    return value


def parse_function(
    value: dict[str, Any], source: str, schema_key: str = "parameters"
) -> FunctionTool:
    name = value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RequestError("function tool name is required")
    schema = value.get(schema_key, {"type": "object"})
    if not isinstance(schema, dict):
        raise RequestError(f"tool {schema_key} must be an object")
    common = {"type", "name", schema_key, "description", "strict"}
    return FunctionTool(
        name,
        copy.deepcopy(schema),
        str(value.get("description") or ""),
        strict=optional_bool(value.get("strict"), "strict"),
        source=source,
        options=copy.deepcopy({k: v for k, v in value.items() if k not in common}),
    )


def render_function(
    tool: FunctionTool, target: str, schema_key: str = "parameters"
) -> dict[str, Any]:
    if tool.options and tool.source != target:
        raise RequestError(
            f"{target} cannot faithfully represent {tool.source} function tool fields: "
            + ", ".join(sorted(tool.options))
        )
    result = {"name": tool.name, schema_key: tool.parameters, **tool.options}
    if tool.description:
        result["description"] = tool.description
    if tool.strict is not None:
        result["strict"] = tool.strict
    return copy.deepcopy(result)


def arguments(value: Any, error: type[ValueError] = ValueError) -> dict[str, Any]:
    """Require a JSON object; never turn a broken call into an empty call.

    Request renderers use RequestError (400); response translators use the
    default ValueError (502). Empty input remains valid for parameterless tools.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else {}
        except json.JSONDecodeError:
            raise error("tool call arguments must be a JSON object") from None
    if not isinstance(value, dict):
        raise error("tool call arguments must be a JSON object")
    return value


def responses_tool(tool: Tool) -> dict[str, Any]:
    if isinstance(tool, NativeTool):
        return copy.deepcopy(tool.item)
    if isinstance(tool, WebSearchTool) and tool.native is not None:
        if tool.source == "responses":
            return copy.deepcopy(tool.native)
        if tool.source != "anthropic":
            raise RequestError(f"unsupported web_search source: {tool.source}")
        unsupported = sorted(set(tool.native) - {"type", "name"})
        if unsupported:
            raise RequestError(
                "Responses cannot faithfully represent Anthropic web_search "
                "options: " + ", ".join(unsupported)
            )
        return {"type": "web_search"}
    if isinstance(tool, FunctionTool):
        return {"type": "function", **render_function(tool, "responses")}
    item = {"type": "web_search"}
    if tool.context_size:
        item["search_context_size"] = tool.context_size
    return item


def parse_choice(value: Any, *, nested: bool = False) -> ToolChoice | None:
    """OpenAI tool choice, with the Chat Completions function wrapper removed."""
    if value is None:
        return None
    if isinstance(value, str) and value in {"auto", "none", "required"}:
        return ToolChoice(value)
    if isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function") if nested else value
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            return ToolChoice("tool", name)
    raise RequestError("unsupported tool_choice")


def responses_choice(choice: ToolChoice | None) -> Any:
    if choice is None:
        return "auto"
    if choice.kind == "tool":
        return {"type": "function", "name": choice.name}
    return choice.kind
