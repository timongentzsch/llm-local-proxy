from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from typing import Any


class RequestError(ValueError):
    pass


class ReasoningCache:
    """Keeps encrypted reasoning between a tool call and its result."""

    def __init__(self, limit: int = 128):
        self._items: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._limit = limit
        self._lock = threading.Lock()

    def get(self, call_ids: list[str]) -> list[dict[str, Any]]:
        with self._lock:
            for call_id in call_ids:
                if call_id in self._items:
                    self._items.move_to_end(call_id)
                    return self._items[call_id]
        return []

    def put(self, call_ids: list[str], items: list[dict[str, Any]]) -> None:
        if not items:
            return
        with self._lock:
            for call_id in call_ids:
                self._items[call_id] = items
                self._items.move_to_end(call_id)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("message content must be a string or array")
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
    return "\n".join(parts)


def _usage(value: Any, web_searches: int = 0) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    input_details = value.get("input_tokens_details", {})
    output_details = value.get("output_tokens_details", {})
    prompt = int(value.get("input_tokens", 0) or 0)
    completion = int(value.get("output_tokens", 0) or 0)
    result = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(value.get("total_tokens", prompt + completion) or 0),
        "prompt_tokens_details": {
            "cached_tokens": int(
                input_details.get("cached_tokens", 0)
                if isinstance(input_details, dict)
                else 0
            )
        },
        "completion_tokens_details": {
            "reasoning_tokens": int(
                output_details.get("reasoning_tokens", 0)
                if isinstance(output_details, dict)
                else 0
            )
        },
    }
    if web_searches:
        result["server_tool_use"] = {"web_search_requests": web_searches}
    return result


def _annotation(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("type") != "url_citation":
        return None
    url = value.get("url")
    if not isinstance(url, str) or not url:
        return None
    citation = {
        key: value[key]
        for key in ("url", "title", "start_index", "end_index")
        if key in value
    }
    return {"type": "url_citation", "url_citation": citation}


class Translator:
    def __init__(self, model: str, cache: ReasoningCache):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.cache = cache
        self.content = ""
        self.reasoning = ""
        self.calls: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.reasoning_items: list[dict[str, Any]] = []
        self.web_searches: set[str] = set()
        self.usage: dict[str, Any] | None = None

    def chunk(self, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def start(self) -> dict[str, Any]:
        return self.chunk({"role": "assistant", "content": ""})

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            delta = str(event.get("delta", ""))
            self.content += delta
            return [self.chunk({"content": delta})] if delta else []
        if event_type == "response.reasoning_summary_text.delta":
            delta = str(event.get("delta", ""))
            self.reasoning += delta
            return [self.chunk({"reasoning_content": delta})] if delta else []
        if event_type == "response.output_text.annotation.added":
            return self._add_annotation(event.get("annotation"))
        if event_type == "response.output_item.added":
            self._web_search(event.get("item"))
            return []
        if event_type == "response.output_item.done":
            return self._item(event.get("item"))
        if event_type == "response.completed":
            response = event.get("response", {})
            chunks = []
            if isinstance(response, dict):
                for item in response.get("output", []):
                    chunks.extend(self._item(item))
                self.usage = _usage(response.get("usage"), len(self.web_searches))
            return chunks
        if event_type in {"response.failed", "response.incomplete", "error"}:
            detail = event.get("error") or event.get("response") or event
            raise RuntimeError(f"Codex response failed: {detail}")
        return []

    def _item(self, item: Any) -> list[dict[str, Any]]:
        if not isinstance(item, dict):
            return []
        self._web_search(item)
        chunks = []
        content = item.get("content", [])
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict):
                for annotation in part.get("annotations", []):
                    chunks.extend(self._add_annotation(annotation))
        if item.get("type") == "reasoning" and item.get("encrypted_content"):
            kept = {
                key: item[key]
                for key in ("type", "id", "summary", "encrypted_content")
                if key in item
            }
            if kept not in self.reasoning_items:
                self.reasoning_items.append(kept)
        if item.get("type") != "function_call":
            return chunks
        call_id = str(item.get("call_id") or item.get("id") or "")
        if not call_id or any(call["id"] == call_id for call in self.calls):
            return chunks
        call = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": str(item.get("name", "")),
                "arguments": str(item.get("arguments", "{}")),
            },
        }
        self.calls.append(call)
        index = len(self.calls) - 1
        chunks.append(self.chunk({"tool_calls": [{"index": index, **call}]}))
        return chunks

    def _web_search(self, item: Any) -> None:
        if isinstance(item, dict) and item.get("type") == "web_search_call":
            self.web_searches.add(str(item.get("id") or "web_search"))

    def _add_annotation(self, value: Any) -> list[dict[str, Any]]:
        annotation = _annotation(value)
        if not annotation or annotation in self.annotations:
            return []
        self.annotations.append(annotation)
        return [self.chunk({"annotations": [annotation]})]

    def finish(self) -> list[dict[str, Any]]:
        self.cache.put([call["id"] for call in self.calls], self.reasoning_items)
        chunks = [self.chunk({}, "tool_calls" if self.calls else "stop")]
        if self.usage:
            chunks.append(
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [],
                    "usage": self.usage,
                }
            )
        return chunks

    def result(self) -> dict[str, Any]:
        self.cache.put([call["id"] for call in self.calls], self.reasoning_items)
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content or None,
        }
        if self.calls:
            message["tool_calls"] = self.calls
        if self.annotations:
            message["annotations"] = self.annotations
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if self.calls else "stop",
                }
            ],
            "usage": self.usage,
        }
