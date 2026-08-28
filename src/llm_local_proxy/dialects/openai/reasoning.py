"""Reasoning controls shared by the OpenAI-shaped request dialects."""

from __future__ import annotations

from typing import Any


def options(value: Any) -> tuple[Any, str]:
    """Return the requested effort and its Claude thinking-display mapping."""
    if not isinstance(value, dict):
        return None, ""
    summary = value.get("summary")
    display = ""
    if summary in {"none", "omitted"}:
        display = "omitted"
    elif summary is not None:
        # OpenAI summary modes all ask for readable reasoning. Claude calls
        # that one wire mode "summarized".
        display = "summarized"
    return value.get("effort"), display
