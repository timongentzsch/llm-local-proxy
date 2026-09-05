"""Codex Responses API SSE events -> canonical stream events."""

from __future__ import annotations

import json
from typing import Any

from ...ir import (
    Citation,
    Finish,
    HostedToolEvent,
    NativeItem,
    ReasoningItem,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    hosted_tool_step,
)
from ..reasoning import ReasoningCache
from .thinking import pack
from .usage import read_usage


class CodexDecoder:
    """Decodes one Codex response stream.

    Tool calls arrive whole, so each yields a start and an immediate end.
    """

    def __init__(self, cache: ReasoningCache):
        self.cache = cache
        self.calls: list[str] = []
        self.reasoning_items: list[dict[str, Any]] = []
        self.web_searches: set[str] = set()
        self._search_phase: dict[str, str] = {}
        self._native_seen: set[str] = set()
        self._thinking = ""
        self._usage: Usage | None = None
        self._stop: str | None = None
        self._incomplete_reason: str | None = None

    def decode(self, event: dict[str, Any]) -> list[StreamEvent]:
        kind = event.get("type")
        if kind == "response.output_text.delta":
            text = str(event.get("delta", ""))
            return [TextDelta(text)] if text else []
        if kind == "response.reasoning_summary_text.delta":
            text = str(event.get("delta", ""))
            if text:
                self._thinking += text
            return [ThinkingDelta(text)] if text else []
        if kind == "response.output_text.annotation.added":
            return _citation(event.get("annotation"))
        if kind == "response.output_item.added":
            return self._web_search(event.get("item"), "started")
        if kind in _SEARCH_PHASES:
            return self._hosted(str(event.get("item_id") or ""), _SEARCH_PHASES[kind])
        if kind == "response.output_item.done":
            return self._item(event.get("item"))
        if kind in {"response.completed", "response.incomplete"}:
            response = event.get("response", {})
            if not isinstance(response, dict):
                return []
            events: list[StreamEvent] = []
            for item in response.get("output", []):
                events.extend(self._item(item))
            self._usage = read_usage(event, len(self.web_searches))
            if kind == "response.incomplete":
                reason = (response.get("incomplete_details") or {}).get("reason")
                self._incomplete_reason = reason or "max_output_tokens"
                self._stop = "refusal" if reason == "content_filter" else "max_tokens"
            return events
        if kind in {"response.failed", "error"}:
            detail = event.get("error") or event.get("response") or event
            raise RuntimeError(f"Codex response failed: {detail}")
        return []

    def finish(self) -> list[StreamEvent]:
        self.cache.put(self.calls, self.reasoning_items)
        events: list[StreamEvent] = [
            Finish(
                self._stop or ("tool_use" if self.calls else "end_turn"),
                self._incomplete_reason,
            )
        ]
        if self._usage is not None:
            events.append(self._usage)
        return events

    def _item(self, item: Any) -> list[StreamEvent]:
        if not isinstance(item, dict):
            return []
        events: list[StreamEvent] = self._web_search(item, _terminal(item))
        content = item.get("content", [])
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict):
                for annotation in part.get("annotations", []):
                    events.extend(_citation(annotation))
        if item.get("type") == "reasoning":
            return events + self._reasoning(item)
        if item.get("type") == "function_call":
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
        if item.get("type") in {"message", "web_search_call"}:
            return events
        key = str(item.get("id") or item.get("call_id") or "")
        key = key or json.dumps(item, sort_keys=True, separators=(",", ":"))
        if key not in self._native_seen:
            self._native_seen.add(key)
            events.append(NativeItem(dict(item)))
        return events

    def _reasoning(self, item: dict[str, Any]) -> list[StreamEvent]:
        """Keep one opaque item and bridge it through Anthropic's signature."""
        if not item.get("encrypted_content"):
            self._thinking = ""
            return []
        kept = {
            key: item[key]
            for key in ("type", "id", "summary", "encrypted_content")
            if key in item
        }
        if kept in self.reasoning_items:
            self._thinking = ""
            return []
        self.reasoning_items.append(kept)
        events: list[StreamEvent] = []
        summary = _summary_text(kept)
        if summary and not self._thinking:
            self._thinking = summary
            events.append(ThinkingDelta(summary))
        events.extend(
            [
                ReasoningItem(dict(kept)),
                ThinkingSignature(pack(kept, self._thinking)),
            ]
        )
        self._thinking = ""
        return events

    def _web_search(self, item: Any, phase: str) -> list[StreamEvent]:
        if not isinstance(item, dict) or item.get("type") != "web_search_call":
            return []
        search_id = str(item.get("id") or "web_search")
        self.web_searches.add(search_id)
        return self._hosted(search_id, phase)

    def _hosted(self, search_id: str, phase: str) -> list[StreamEvent]:
        """One lifecycle step, dropped unless it advances this search."""
        if not search_id or not hosted_tool_step(self._search_phase, search_id, phase):
            return []
        return [HostedToolEvent("web_search", search_id, phase)]


#: Responses names the middle of a hosted search in its own events; the ends
#: arrive as ordinary output items.
_SEARCH_PHASES = {
    "response.web_search_call.in_progress": "started",
    "response.web_search_call.searching": "searching",
    "response.web_search_call.completed": "completed",
}


def _terminal(item: Any) -> str:
    """How a finished search item ended. Absent status means it simply did."""
    status = item.get("status") if isinstance(item, dict) else None
    return "failed" if status in {"failed", "incomplete"} else "completed"


def _summary_text(item: dict[str, Any]) -> str:
    summary = item.get("summary")
    if not isinstance(summary, list):
        return ""
    return "".join(
        str(part.get("text", ""))
        for part in summary
        if isinstance(part, dict) and part.get("type") == "summary_text"
    )


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
