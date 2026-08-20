"""Canonical stream events -> Chat Completions chunks and completions."""

from __future__ import annotations

import time
import uuid
from typing import Any

from ...ir import (
    Citation,
    Finish,
    RedactedThinkingDelta,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolCallArgs,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)

#: Anthropic's seven stop reasons narrowed onto Chat Completions' four.
FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "refusal": "content_filter",
    "pause_turn": "stop",
}


class ChunkEncoder:
    """Turns one provider's decoded stream into Chat Completions output.

    Holds only wire shaping. Everything upstream-specific — which events a
    given API produces, and what has to be cached to replay a tool call —
    lives in the decoder it wraps.
    """

    def __init__(self, model: str, decoder: Any):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.decoder = decoder
        self.content = ""
        self.reasoning = ""
        self.calls: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.usage: dict[str, Any] | None = None
        self._finish: str | None = None
        self._drained = False

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
        return self._encode(self.decoder.decode(event))

    def finish(self) -> list[dict[str, Any]]:
        chunks = self._drain()
        chunks.append(self.chunk({}, self._finish_reason()))
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
        self._drain()
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
                    "finish_reason": self._finish_reason(),
                }
            ],
            "usage": self.usage,
        }

    def _finish_reason(self) -> str:
        return self._finish or ("tool_calls" if self.calls else "stop")

    def _drain(self) -> list[dict[str, Any]]:
        """Collect whatever the decoder only knows once the stream ends."""
        if self._drained:
            return []
        self._drained = True
        return self._encode(self.decoder.finish())

    def _encode(self, events: list[StreamEvent]) -> list[dict[str, Any]]:
        chunks = []
        for event in events:
            chunk = self._one(event)
            if chunk is not None:
                chunks.append(chunk)
        return chunks

    def _one(self, event: StreamEvent) -> dict[str, Any] | None:
        if isinstance(event, TextDelta):
            self.content += event.text
            return self.chunk({"content": event.text})
        if isinstance(event, ThinkingDelta):
            self.reasoning += event.text
            return self.chunk({"reasoning_content": event.text})
        if isinstance(event, ToolCallStart):
            return self.chunk(
                {
                    "tool_calls": [
                        {
                            "index": event.index,
                            "id": event.id,
                            "type": "function",
                            "function": {
                                "name": event.name,
                                "arguments": event.arguments,
                            },
                        }
                    ]
                }
            )
        if isinstance(event, ToolCallArgs):
            return self.chunk(
                {
                    "tool_calls": [
                        {
                            "index": event.index,
                            "function": {"arguments": event.fragment},
                        }
                    ]
                }
            )
        if isinstance(event, ToolCallEnd):
            self.calls.append(
                {
                    "id": event.id,
                    "type": "function",
                    "function": {"name": event.name, "arguments": event.arguments},
                }
            )
            return None
        if isinstance(event, Citation):
            return self._citation(event)
        if isinstance(event, Usage):
            self.usage = _usage(event)
            return None
        if isinstance(event, Finish):
            self._finish = FINISH_REASONS.get(event.reason, "stop")
            return None
        # Signed and redacted thinking have no Chat Completions representation;
        # the decoder keeps them for replay upstream.
        if isinstance(event, (ThinkingSignature, RedactedThinkingDelta)):
            return None
        return None

    def _citation(self, event: Citation) -> dict[str, Any] | None:
        fields = {
            "url": event.url,
            "title": event.title,
            "start_index": event.start_index,
            "end_index": event.end_index,
        }
        annotation = {
            "type": "url_citation",
            "url_citation": {k: v for k, v in fields.items() if v is not None},
        }
        if annotation in self.annotations:
            return None
        self.annotations.append(annotation)
        return self.chunk({"annotations": [annotation]})


def _usage(event: Usage) -> dict[str, Any]:
    prompt_details: dict[str, Any] = {"cached_tokens": event.cache_read}
    if event.cache_write:
        prompt_details["cache_write_tokens"] = event.cache_write
    result = {
        "prompt_tokens": event.prompt,
        "completion_tokens": event.completion,
        "total_tokens": (
            event.total if event.total is not None else event.prompt + event.completion
        ),
        "prompt_tokens_details": prompt_details,
        "completion_tokens_details": {"reasoning_tokens": event.thinking},
    }
    if event.web_searches:
        result["server_tool_use"] = {"web_search_requests": event.web_searches}
    return result
