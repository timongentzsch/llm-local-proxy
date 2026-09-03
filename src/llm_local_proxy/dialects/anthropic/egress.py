"""Canonical stream events -> Anthropic Messages frames and messages.

Two constraints from specs/anthropic-openapi.json shape this file. Exactly
one content block may be open at a time, under monotonically increasing
indices, so the encoder is a small state machine that closes the open block
whenever the kind changes. And message_start carries a Message whose
usage.input_tokens is non-nullable, while a Codex stream reports no input
count until it ends: the opening frame therefore claims zero and the
authoritative totals arrive in message_delta.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ...ir import (
    Citation,
    Finish,
    HostedToolEvent,
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


class MessageEncoder:
    """Turns one provider's decoded stream into Anthropic Messages output."""

    def __init__(self, model: str, decoder: Any):
        self.id = "msg_" + uuid.uuid4().hex[:24]
        self.model = model
        self.decoder = decoder
        self.blocks: list[dict[str, Any]] = []
        self.usage: Usage | None = None
        self.stop_reason: str | None = None
        self._index = -1
        self._searches: set[str] = set()
        self._open: dict[str, Any] | None = None
        self._drained = False

    def start(self) -> dict[str, Any]:
        return {"type": "message_start", "message": self._message(streaming=True)}

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        return self._encode(self.decoder.decode(event))

    def finish(self) -> list[dict[str, Any]]:
        frames = self._drain()
        frames.extend(self._close())
        frames.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": self._stop(), "stop_sequence": None},
                "usage": _usage(self.usage),
            }
        )
        frames.append({"type": "message_stop"})
        return frames

    def result(self) -> dict[str, Any]:
        self._drain()
        self._close()
        return self._message(streaming=False)

    # -- block state machine ---------------------------------------------

    def _open_block(self, kind: str, block: dict[str, Any]) -> list[dict[str, Any]]:
        frames = self._close()
        self._index += 1
        self._open = {"kind": kind, "block": block, "json": ""}
        frames.append(
            {
                "type": "content_block_start",
                "index": self._index,
                # A copy: the retained block keeps accumulating, and the
                # opening frame must show the block as it was at the start.
                "content_block": dict(block),
            }
        )
        return frames

    def _close(self) -> list[dict[str, Any]]:
        if self._open is None:
            return []
        block = self._open["block"]
        if self._open["kind"] == "tool_use":
            block["input"] = _arguments(self._open["json"])
        self.blocks.append(block)
        self._open = None
        return [{"type": "content_block_stop", "index": self._index}]

    def _delta(self, delta: dict[str, Any]) -> dict[str, Any]:
        return {"type": "content_block_delta", "index": self._index, "delta": delta}

    def _encode(self, events: list[StreamEvent]) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for event in events:
            frames.extend(self._one(event))
        return frames

    def _one(self, event: StreamEvent) -> list[dict[str, Any]]:
        if isinstance(event, TextDelta):
            frames = self._ensure("text", {"type": "text", "text": ""})
            self._open["block"]["text"] += event.text
            return frames + [self._delta({"type": "text_delta", "text": event.text})]
        if isinstance(event, ThinkingDelta):
            frames = self._ensure(
                "thinking", {"type": "thinking", "thinking": "", "signature": ""}
            )
            self._open["block"]["thinking"] += event.text
            return frames + [
                self._delta({"type": "thinking_delta", "thinking": event.text})
            ]
        if isinstance(event, ThinkingSignature):
            frames = self._ensure(
                "thinking", {"type": "thinking", "thinking": "", "signature": ""}
            )
            assert self._open is not None
            self._open["block"]["signature"] = event.signature
            return frames + [
                self._delta({"type": "signature_delta", "signature": event.signature})
            ]
        if isinstance(event, RedactedThinkingDelta):
            frames = self._open_block(
                "redacted_thinking",
                {"type": "redacted_thinking", "data": event.data},
            )
            return frames + self._close()
        if isinstance(event, ToolCallStart):
            frames = self._open_block(
                "tool_use",
                {"type": "tool_use", "id": event.id, "name": event.name, "input": {}},
            )
            if event.arguments:
                # Codex hands over a complete call; Anthropic clients still
                # expect the arguments to arrive as a delta.
                self._open["json"] += event.arguments
                frames.append(
                    self._delta(
                        {
                            "type": "input_json_delta",
                            "partial_json": event.arguments,
                        }
                    )
                )
            return frames
        if isinstance(event, ToolCallArgs):
            if self._open is None or self._open["kind"] != "tool_use":
                return []
            self._open["json"] += event.fragment
            return [
                self._delta(
                    {"type": "input_json_delta", "partial_json": event.fragment}
                )
            ]
        if isinstance(event, ToolCallEnd):
            return self._close()
        if isinstance(event, HostedToolEvent):
            return self._hosted(event)
        if isinstance(event, Citation):
            if self._open is None or self._open["kind"] != "text":
                return []
            return [
                self._delta({"type": "citations_delta", "citation": _citation(event)})
            ]
        if isinstance(event, Usage):
            self.usage = event
            return []
        if isinstance(event, Finish):
            self.stop_reason = event.reason
            return []
        return []

    def _hosted(self, event: HostedToolEvent) -> list[dict[str, Any]]:
        """A provider-run search as Anthropic's own server-tool blocks.

        Never `tool_use`: that block obliges the client to run the search and
        return a result, and this one has already run upstream. `_stop()`
        matches `tool_use` exactly, so `stop_reason` is unaffected.
        """
        frames: list[dict[str, Any]] = []
        if event.id not in self._searches:
            self._searches.add(event.id)
            block: dict[str, Any] = {
                "type": "server_tool_use",
                "id": event.id,
                "name": "web_search",
                "input": {"query": event.query} if event.query else {},
            }
            # Left open: the span until the result block is the search, and a
            # client showing it needs both ends to arrive when they happened.
            frames = self._open_block("server_tool_use", block)
        if event.phase == "completed":
            # Present and empty rather than invented: no upstream forwards the
            # individual result records through this proxy.
            frames.extend(
                self._open_block(
                    "web_search_tool_result",
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": event.id,
                        "content": [],
                    },
                )
            )
        if event.phase in {"completed", "failed"}:
            # A failure closes the request block and stops there; Anthropic's
            # error record carries a code this proxy would have to make up.
            frames.extend(self._close())
        return frames

    def _ensure(self, kind: str, block: dict[str, Any]) -> list[dict[str, Any]]:
        if self._open is not None and self._open["kind"] == kind:
            return []
        return self._open_block(kind, block)

    # -- message assembly -------------------------------------------------

    def _stop(self) -> str:
        if self.stop_reason:
            return self.stop_reason
        return (
            "tool_use"
            if any(b["type"] == "tool_use" for b in self.blocks)
            else "end_turn"
        )

    def _drain(self) -> list[dict[str, Any]]:
        if self._drained:
            return []
        self._drained = True
        return self._encode(self.decoder.finish())

    def _message(self, streaming: bool) -> dict[str, Any]:
        # Every field below is required by the schema; the nullable ones must
        # still be present, so a client SDK can read them unconditionally.
        return {
            "id": self.id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [] if streaming else self.blocks,
            "stop_reason": None if streaming else self._stop(),
            "stop_sequence": None,
            "stop_details": None,
            "container": None,
            "usage": _usage(None if streaming else self.usage, full=True),
        }


def _arguments(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return {}


def _citation(event: Citation) -> dict[str, Any]:
    citation = {"type": "web_search_result_location", "url": event.url}
    if event.title is not None:
        citation["title"] = event.title
    return citation


def _usage(usage: Usage | None, full: bool = False) -> dict[str, Any]:
    cache_read = usage.cache_read if usage else 0
    cache_write = usage.cache_write if usage else 0
    # The canonical prompt count includes cache; Anthropic reports the three
    # apart, so back the plain input tokens out of the total.
    result: dict[str, Any] = {
        "input_tokens": max(
            (usage.prompt if usage else 0) - cache_read - cache_write, 0
        ),
        "output_tokens": usage.completion if usage else 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_write,
    }
    if usage and usage.thinking:
        result["output_tokens_details"] = {"thinking_tokens": usage.thinking}
    if usage and usage.web_searches:
        result["server_tool_use"] = {"web_search_requests": usage.web_searches}
    if full:
        result.setdefault("cache_creation", None)
        result.setdefault("service_tier", None)
    return result
