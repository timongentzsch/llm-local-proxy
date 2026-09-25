"""Canonical stream events -> Chat Completions chunks and completions."""

from __future__ import annotations

import time
import uuid
from typing import Any

from ...errors import ProviderError
from ...ir import (
    Citation,
    Decoder,
    Finish,
    NativeItem,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallArgs,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)
from ..base import Encoder

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


class ChunkEncoder(Encoder):
    """Turns one provider's decoded stream into Chat Completions output.

    Holds only wire shaping; upstream specifics live in the decoder.
    """

    def __init__(self, model: str, decoder: Decoder):
        super().__init__(decoder)
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.content = ""
        self.reasoning = ""
        self.calls: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.usage: dict[str, Any] | None = None
        self._finish: str | None = None

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

    def _one(self, event: StreamEvent) -> list[dict[str, Any]]:
        if isinstance(event, NativeItem):
            raise ProviderError(
                "native Responses output requires the Responses endpoint"
            )
        if isinstance(event, TextDelta):
            self.content += event.text
            return [self.chunk({"content": event.text})]
        if isinstance(event, ThinkingDelta):
            self.reasoning += event.text
            return [self.chunk({"reasoning_content": event.text})]
        if isinstance(event, ToolCallStart):
            return [
                self.chunk(
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
            ]
        if isinstance(event, ToolCallArgs):
            return [
                self.chunk(
                    {
                        "tool_calls": [
                            {
                                "index": event.index,
                                "function": {"arguments": event.fragment},
                            }
                        ]
                    }
                )
            ]
        if isinstance(event, ToolCallEnd):
            self.calls.append(
                {
                    "id": event.id,
                    "type": "function",
                    "function": {"name": event.name, "arguments": event.arguments},
                }
            )
            return []
        if isinstance(event, Citation):
            return self._citation(event)
        if isinstance(event, Usage):
            self.usage = _usage(event)
            return []
        if isinstance(event, Finish):
            self._finish = FINISH_REASONS.get(event.reason, "stop")
            return []
        # Signed reasoning and hosted tool lifecycles have no Chat representation.
        return []

    def _citation(self, event: Citation) -> list[dict[str, Any]]:
        if not event.url:
            return []
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
            return []
        self.annotations.append(annotation)
        return [self.chunk({"annotations": [annotation]})]


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
