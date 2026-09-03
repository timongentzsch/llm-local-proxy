"""Output formats shared by the OpenAI-shaped request dialects."""

from __future__ import annotations

from typing import Any

from ...errors import RequestError
from ...ir import OutputFormat


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
    return OutputFormat("json_schema", name, schema, bool(fields.get("strict")))
