"""Output formats shared by the OpenAI-shaped request dialects."""

from __future__ import annotations

from typing import Any

from ...errors import RequestError
from ...ir import OutputFormat
from ...tools import optional_bool


def enum_value(value: Any, allowed: frozenset[str], name: str) -> str:
    """An optional string option restricted to its specified values."""
    if value is None:
        return ""
    if value not in allowed:
        raise RequestError(f"{name} must be one of: {', '.join(sorted(allowed))}")
    return value


VERBOSITY = frozenset({"low", "medium", "high"})


def format_of(kind: Any, fields: dict[str, Any]) -> OutputFormat | None:
    """One output format, however its dialect wrapped the schema.

    Chat Completions nests the schema under ``response_format.json_schema``
    while Responses spreads the same fields across ``text.format``; only the
    wrapper differs, so both hand the unwrapped fields to this check. A plain
    text format constrains nothing and is reported as no format at all.
    """
    if kind == "text":
        return None
    if kind == "json_object":
        return OutputFormat("json_object")
    if kind != "json_schema":
        raise RequestError(f"unsupported output format type: {kind}")
    name, schema = fields.get("name"), fields.get("schema")
    if not isinstance(name, str) or not name or not isinstance(schema, dict):
        raise RequestError("json_schema output format requires name and schema")
    return OutputFormat(
        "json_schema",
        name,
        schema,
        optional_bool(fields.get("strict"), "strict") or False,
    )
