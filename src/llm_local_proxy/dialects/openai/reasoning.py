"""Reasoning controls shared by the OpenAI-shaped request dialects."""

from __future__ import annotations

from typing import Any

#: Summary modes Responses defines; Codex receives the requested one verbatim.
SUMMARY_MODES = frozenset({"auto", "concise", "detailed"})


def options(value: Any) -> tuple[Any, str, str]:
    """The requested effort, its Claude thinking display, and summary mode."""
    if not isinstance(value, dict):
        return None, "", ""
    summary = value.get("summary")
    display = ""
    if summary in {"none", "omitted"}:
        display = "omitted"
    elif summary is not None:
        # OpenAI summary modes all ask for readable reasoning. Claude calls
        # that one wire mode "summarized".
        display = "summarized"
    mode = summary if summary in SUMMARY_MODES else ""
    return value.get("effort"), display, mode
