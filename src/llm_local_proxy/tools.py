"""Shared tool contracts and wire encoders, independent of provider implementations.

Common fields cross formats unchanged. Extra options retain their origin and
are rejected on a different format unless an adapter explicitly maps them.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from typing import Any

from .errors import RequestError
from .ir import FunctionTool, NativeTool, Tool, ToolChoice, ToolNamespace, WebSearchTool

#: The Messages web search tool version the proxy requests.
ANTHROPIC_WEB_SEARCH = "web_search_20250305"
#: The longest tool name Anthropic accepts (`^[a-zA-Z0-9_-]{1,64}$`).
MAX_TOOL_NAME = 64


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


def qualified_name(namespace: str, name: str) -> str:
    """One flat, deterministic tool name for a namespaced tool.

    Too-long names keep a readable prefix and end in a hash of the full name,
    so a request and every later replay of it agree without shared state.
    """
    if not namespace:
        return name
    full = f"{namespace}__{name}"
    if len(full) <= MAX_TOOL_NAME:
        return full
    digest = hashlib.sha1(full.encode()).hexdigest()[:8]
    return f"{full[: MAX_TOOL_NAME - 9]}_{digest}"


def flatten(tools: list[Tool]) -> tuple[list[Tool], dict[str, tuple[str, str]]]:
    """Namespaced tools as plain function tools, for targets without namespaces.

    Returns the flat tool list and ``{qualified name: (namespace, name)}`` to
    restore calls. A member that is not a function tool, or two tools that
    would share a name, cannot be represented and is refused.
    """
    flat: list[Tool] = []
    names: dict[str, tuple[str, str]] = {}
    for tool in tools:
        members = tool.tools if isinstance(tool, ToolNamespace) else [tool]
        namespace = tool.name if isinstance(tool, ToolNamespace) else ""
        for member in members:
            if namespace and not isinstance(member, FunctionTool):
                raise RequestError(
                    f"namespace {namespace} contains a non-function tool: "
                    + str(member.item.get("type", "unknown"))
                )
            if isinstance(member, FunctionTool):
                name = qualified_name(namespace, member.name)
                if name in names:
                    raise RequestError(f"duplicate tool name: {name}")
                names[name] = (namespace, member.name)
                member = replace(member, name=name)
            flat.append(member)
    return flat, {key: value for key, value in names.items() if value[0]}


def responses_tool(tool: Tool) -> dict[str, Any]:
    if isinstance(tool, (NativeTool, ToolNamespace)):
        return copy.deepcopy(tool.item)
    if isinstance(tool, WebSearchTool):
        return responses_web_search(tool)
    return {"type": "function", **render_function(tool, "responses")}


def responses_web_search(tool: WebSearchTool) -> dict[str, Any]:
    """A search tool as Responses defines it."""
    native = tool.native
    if tool.source == "responses":
        return copy.deepcopy(native)
    if tool.source != "anthropic":
        raise RequestError(f"unsupported web_search source: {tool.source}")
    # Anthropic `max_uses` caps how often the model searches; Responses has no
    # cap, and exceeding it changes cost, not what is searched. A cache
    # breakpoint is a hint as well.
    unsupported = sorted(
        set(native)
        - {
            "type",
            "name",
            "max_uses",
            "cache_control",
            "allowed_domains",
            "blocked_domains",
            "user_location",
        }
    )
    if native.get("blocked_domains"):
        unsupported.append("blocked_domains")
    if unsupported:
        raise RequestError(
            "Responses cannot faithfully represent Anthropic web_search options: "
            + ", ".join(unsupported)
        )
    item: dict[str, Any] = {"type": "web_search"}
    if native.get("allowed_domains"):
        item["filters"] = {"allowed_domains": list(native["allowed_domains"])}
    if native.get("user_location"):
        item["user_location"] = dict(native["user_location"])
    return item


def anthropic_web_search(tool: WebSearchTool) -> dict[str, Any]:
    """A search tool as the Messages API defines it."""
    native = tool.native
    if tool.source == "anthropic":
        return dict(native)
    if tool.source != "responses":
        raise RequestError(f"unsupported web_search source: {tool.source}")
    # The context size is a hint Messages has no control for, and live access
    # is its only mode; options that change what is searched must map.
    filters = native.get("filters") or {}
    unsupported = sorted(
        set(native)
        - {
            "type",
            "search_context_size",
            "external_web_access",
            "filters",
            "user_location",
        }
    )
    if native.get("external_web_access") is False:
        unsupported.append("external_web_access")
    if not isinstance(filters, dict) or set(filters) - {"allowed_domains"}:
        unsupported.append("filters")
    if unsupported:
        raise RequestError(
            "Messages cannot faithfully represent Responses web_search options: "
            + ", ".join(unsupported)
        )
    item: dict[str, Any] = {"type": ANTHROPIC_WEB_SEARCH, "name": "web_search"}
    if filters.get("allowed_domains"):
        item["allowed_domains"] = list(filters["allowed_domains"])
    if native.get("user_location"):
        item["user_location"] = dict(native["user_location"])
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
