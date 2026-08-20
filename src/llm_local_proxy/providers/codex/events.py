"""Codex Responses API SSE events -> canonical stream events."""

from __future__ import annotations

from typing import Any

from ...ir import (
    Citation,
    Finish,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)
from ..reasoning import ReasoningCache


class CodexDecoder:
    """Decodes one Codex response stream.

    Codex delivers a tool call whole rather than in pieces, so each one
    produces a start carrying its complete arguments and an immediate end.
    """

    def __init__(self, cache: ReasoningCache):
        self.cache = cache
        self.calls: list[str] = []
        self.reasoning_items: list[dict[str, Any]] = []
        self.web_searches: set[str] = set()
        self._usage: Usage | None = None

    def decode(self, event: dict[str, Any]) -> list[StreamEvent]:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            text = str(event.get("delta", ""))
            return [TextDelta(text)] if text else []
        if kind == "response.reasoning_summary_text.delta":
            text = str(event.get("delta", ""))
            return [ThinkingDelta(text)] if text else []
        if kind == "response.output_text.annotation.added":
            return _citation(event.get("annotation"))
        if kind == "response.output_item.added":
            self._web_search(event.get("item"))
            return []
        if kind == "response.output_item.done":
            return self._item(event.get("item"))
        if kind == "response.completed":
            response = event.get("response", {})
            if not isinstance(response, dict):
                return []
            events: list[StreamEvent] = []
            for item in response.get("output", []):
                events.extend(self._item(item))
            usage = response.get("usage")
            if isinstance(usage, dict):
                self._usage = _usage(usage, len(self.web_searches))
            return events
        if kind in {"response.failed", "response.incomplete", "error"}:
            detail = event.get("error") or event.get("response") or event
            raise RuntimeError(f"Codex response failed: {detail}")
        return []

    def finish(self) -> list[StreamEvent]:
        self.cache.put(self.calls, self.reasoning_items)
        events: list[StreamEvent] = [Finish("tool_use" if self.calls else "end_turn")]
        if self._usage is not None:
            events.append(self._usage)
        return events

    def _item(self, item: Any) -> list[StreamEvent]:
        if not isinstance(item, dict):
            return []
        self._web_search(item)
        events: list[StreamEvent] = []
        content = item.get("content", [])
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict):
                for annotation in part.get("annotations", []):
                    events.extend(_citation(annotation))
        if item.get("type") == "reasoning" and item.get("encrypted_content"):
            # Replayed with the next request; Codex refuses a tool result
            # whose originating reasoning is missing.
            kept = {
                key: item[key]
                for key in ("type", "id", "summary", "encrypted_content")
                if key in item
            }
            if kept not in self.reasoning_items:
                self.reasoning_items.append(kept)
        if item.get("type") != "function_call":
            return events
        call_id = str(item.get("call_id") or item.get("id") or "")
        if not call_id or call_id in self.calls:
            return events
        name = str(item.get("name", ""))
        arguments = str(item.get("arguments", "{}"))
        index = len(self.calls)
        self.calls.append(call_id)
        events.append(ToolCallStart(index, call_id, name, arguments))
        events.append(ToolCallEnd(index, call_id, name, arguments))
        return events

    def _web_search(self, item: Any) -> None:
        if isinstance(item, dict) and item.get("type") == "web_search_call":
            self.web_searches.add(str(item.get("id") or "web_search"))


def _citation(value: Any) -> list[StreamEvent]:
    if not isinstance(value, dict) or value.get("type") != "url_citation":
        return []
    url = value.get("url")
    if not isinstance(url, str) or not url:
        return []
    return [
        Citation(
            url,
            value.get("title"),
            value.get("start_index"),
            value.get("end_index"),
        )
    ]


def _usage(value: dict[str, Any], web_searches: int) -> Usage:
    input_details = value.get("input_tokens_details", {})
    output_details = value.get("output_tokens_details", {})
    prompt = int(value.get("input_tokens", 0) or 0)
    completion = int(value.get("output_tokens", 0) or 0)
    return Usage(
        prompt=prompt,
        completion=completion,
        total=int(value.get("total_tokens", prompt + completion) or 0),
        cache_read=int(
            input_details.get("cached_tokens", 0)
            if isinstance(input_details, dict)
            else 0
        ),
        thinking=int(
            output_details.get("reasoning_tokens", 0)
            if isinstance(output_details, dict)
            else 0
        ),
        web_searches=web_searches,
    )
