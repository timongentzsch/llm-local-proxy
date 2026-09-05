"""Anthropic's cumulative usage, shared by translation and accounting."""

from __future__ import annotations

from typing import Any

from ...ir import Usage


class ClaudeUsage:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def read(self, event: dict[str, Any]) -> Usage | None:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message")
            value = message.get("usage") if isinstance(message, dict) else None
        elif kind == "message_delta":
            value = event.get("usage")
        else:
            return None
        if not isinstance(value, dict):
            return None
        # These are snapshots, not increments. An omitted field retains its
        # previous value; an explicit zero replaces it.
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            count = value.get(name)
            if type(count) is int and count >= 0:
                self._counts[name] = count
        details = value.get("output_tokens_details")
        count = details.get("thinking_tokens") if isinstance(details, dict) else None
        if type(count) is int and count >= 0:
            self._counts["thinking_tokens"] = count
        return self.snapshot()

    def snapshot(self, web_searches: int = 0) -> Usage | None:
        if not self._counts:
            return None
        read = self._counts.get("cache_read_input_tokens", 0)
        write = self._counts.get("cache_creation_input_tokens", 0)
        return Usage(
            prompt=self._counts.get("input_tokens", 0) + read + write,
            completion=self._counts.get("output_tokens", 0),
            cache_read=read,
            cache_write=write,
            thinking=self._counts.get("thinking_tokens", 0),
            web_searches=web_searches,
        )
