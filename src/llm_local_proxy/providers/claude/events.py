"""Claude Messages API SSE events -> canonical stream events."""

from __future__ import annotations

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
from ..reasoning import ReasoningCache
from .request import WEB_SEARCH_TOOL

#: Passed through; anything else means the turn simply ended.
PASSTHROUGH_STOP = {"tool_use", "max_tokens"}


class ClaudeDecoder:
    """Decodes one Claude response stream.

    Tool calls arrive in pieces. Signed thinking is accumulated rather than
    forwarded: it exists to be replayed upstream on the next turn.
    """

    def __init__(self, reasoning_cache: ReasoningCache | None = None):
        self.reasoning_cache = reasoning_cache
        self.reasoning_blocks: list[dict[str, Any]] = []
        self.calls: list[str] = []
        self.web_searches: set[Any] = set()
        self._open_call: dict[str, Any] | None = None
        self._open_thinking: dict[str, Any] | None = None
        self._stop: str | None = None
        self._usage = {
            "input": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "output": 0,
            "thinking": 0,
        }

    def decode(self, event: dict[str, Any]) -> list[StreamEvent]:
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message", {})
            self._merge(message.get("usage") if isinstance(message, dict) else None)
            return []
        if kind == "content_block_start":
            return self._block_start(event)
        if kind == "content_block_delta":
            return self._delta(event.get("delta"))
        if kind == "content_block_stop":
            return self._block_stop()
        if kind == "message_delta":
            delta = event.get("delta")
            reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            self._stop = reason if reason in PASSTHROUGH_STOP else "end_turn"
            self._merge(event.get("usage"))
            return []
        if kind == "error":
            detail = event.get("error") or event
            raise RuntimeError(f"Claude response failed: {detail}")
        return []

    def finish(self) -> list[StreamEvent]:
        self._cache_reasoning()
        events: list[StreamEvent] = [
            Finish(self._stop or ("tool_use" if self.calls else "end_turn"))
        ]
        usage = self._usage_event()
        if usage is not None:
            events.append(usage)
        return events

    def _block_start(self, event: dict[str, Any]) -> list[StreamEvent]:
        block = event.get("content_block")
        kind = block.get("type") if isinstance(block, dict) else None
        index = event.get("index")
        if kind == "tool_use":
            call_id = str(block.get("id") or "toolu_" + uuid.uuid4().hex[:24])
            name = str(block.get("name", ""))
            self._open_call = {
                "index": index,
                "id": call_id,
                "name": name,
                "arguments": "",
            }
            # Announced immediately so time-to-first-token is not stalled.
            return [ToolCallStart(index, call_id, name)]
        if kind == "thinking":
            self._open_thinking = {"type": "thinking", "thinking": "", "signature": ""}
        elif kind == "redacted_thinking":
            # Opaque, safety-flagged portion; replay verbatim on round-trip.
            data = block.get("data") or ""
            self.reasoning_blocks.append({"type": "redacted_thinking", "data": data})
            return [RedactedThinkingDelta(data)] if data else []
        elif kind in {WEB_SEARCH_TOOL, "web_search_tool_result"}:
            self.web_searches.add(index)
        return []

    def _block_stop(self) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if self._open_thinking is not None:
            signature = self._open_thinking.get("signature")
            if signature:
                # Only a signed block can be replayed upstream.
                self.reasoning_blocks.append(self._open_thinking)
                events.append(ThinkingSignature(signature))
            self._open_thinking = None
        return events + self._close_call()

    def _delta(self, delta: Any) -> list[StreamEvent]:
        if not isinstance(delta, dict):
            return []
        kind = delta.get("type")
        if kind == "text_delta":
            text = str(delta.get("text", ""))
            return [TextDelta(text)] if text else []
        if kind == "thinking_delta":
            text = str(delta.get("thinking", ""))
            if self._open_thinking is not None and text:
                self._open_thinking["thinking"] += text
            return [ThinkingDelta(text)] if text else []
        if kind == "signature_delta":
            signature = str(delta.get("signature", ""))
            if self._open_thinking is not None and signature:
                self._open_thinking["signature"] += signature
            return []
        if kind == "redacted_thinking_delta":
            data = str(delta.get("data", ""))
            last = self.reasoning_blocks[-1] if self.reasoning_blocks else None
            if data and last and last.get("type") == "redacted_thinking":
                last["data"] += data
            return [RedactedThinkingDelta(data)] if data else []
        if kind == "input_json_delta":
            piece = str(delta.get("partial_json", ""))
            if not self._open_call or not piece:
                return []
            self._open_call["arguments"] += piece
            return [ToolCallArgs(self._open_call["index"], piece)]
        if kind == "citations_delta":
            return _citation(delta.get("citation"))
        return []

    def _close_call(self) -> list[StreamEvent]:
        call = self._open_call
        self._open_call = None
        if not call or not call["name"] or call["id"] in self.calls:
            return []
        streamed = call["arguments"].strip()
        self.calls.append(call["id"])
        events: list[StreamEvent] = []
        if not streamed:
            # No input_json_delta arrives, so the client would see "".
            events.append(ToolCallArgs(call["index"], "{}"))
        events.append(
            ToolCallEnd(call["index"], call["id"], call["name"], streamed or "{}")
        )
        return events

    def _merge(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key, name in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("cache_read", "cache_read_input_tokens"),
            ("cache_creation", "cache_creation_input_tokens"),
        ):
            if isinstance(value.get(name), int):
                self._usage[key] = value[name]
        details = value.get("output_tokens_details")
        if isinstance(details, dict) and isinstance(
            details.get("thinking_tokens"), int
        ):
            self._usage["thinking"] = details["thinking_tokens"]

    def _usage_event(self) -> Usage | None:
        if not any(self._usage.values()):
            return None
        return Usage(
            # Anthropic reports cache tokens apart from input_tokens.
            prompt=self._usage["input"]
            + self._usage["cache_read"]
            + self._usage["cache_creation"],
            completion=self._usage["output"],
            cache_read=self._usage["cache_read"],
            cache_write=self._usage["cache_creation"],
            thinking=self._usage["thinking"],
            web_searches=len(self.web_searches),
        )

    def _cache_reasoning(self) -> None:
        """Persist signed thinking blocks keyed by their tool call ids.

        Only assistant turns that ended in tool calls need them replayed, so
        this is a no-op unless they exist.
        """
        if self.reasoning_cache is None or not self.reasoning_blocks:
            return
        if self.calls:
            self.reasoning_cache.put(self.calls, self.reasoning_blocks)


def _citation(value: Any) -> list[StreamEvent]:
    # A url is what makes a citation citable, whatever its location kind.
    if not isinstance(value, dict):
        return []
    url = value.get("url")
    if not isinstance(url, str) or not url:
        return []
    return [Citation(url, value.get("title"))]
