"""Codex terminal usage, shared by translation and accounting."""

from __future__ import annotations

from typing import Any

from ...ir import Usage

TERMINAL_EVENTS = {"response.completed", "response.incomplete", "response.failed"}


def read_usage(event: dict[str, Any], web_searches: int = 0) -> Usage | None:
    if event.get("type") not in TERMINAL_EVENTS:
        return None
    response = event.get("response")
    value = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(value, dict):
        return None
    input_details = value.get("input_tokens_details")
    output_details = value.get("output_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}
    prompt = int(value.get("input_tokens") or 0)
    completion = int(value.get("output_tokens") or 0)
    return Usage(
        prompt=prompt,
        completion=completion,
        total=int(value["total_tokens"])
        if value.get("total_tokens") is not None
        else prompt + completion,
        cache_read=int(input_details.get("cached_tokens") or 0),
        cache_write=int(input_details.get("cache_write_tokens") or 0),
        thinking=int(output_details.get("reasoning_tokens") or 0),
        web_searches=web_searches,
    )
