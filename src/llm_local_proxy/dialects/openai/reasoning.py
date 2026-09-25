"""Reasoning controls shared by the OpenAI-shaped request dialects."""

from __future__ import annotations

from typing import Any

from ...errors import RequestError

#: Responses summary modes, plus "none", which Codex CLI sends to ask for none.
SUMMARY_MODES = frozenset({"auto", "concise", "detailed", "none"})


def options(value: Any) -> tuple[Any, str]:
    """The requested effort and summary mode."""
    if not isinstance(value, dict):
        return None, ""
    summary = value.get("summary")
    if summary is not None and summary not in SUMMARY_MODES:
        raise RequestError("reasoning.summary must be auto, concise, detailed or none")
    return value.get("effort"), summary or ""
