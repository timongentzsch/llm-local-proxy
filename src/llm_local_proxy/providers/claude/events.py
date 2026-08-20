"""Claude Messages SSE events -> Chat Completions chunks."""

from __future__ import annotations

import time
import uuid
from typing import Any

from ...protocol import ReasoningCache
from .request import WEB_SEARCH_TOOL


class ClaudeTranslator:
    """Claude Messages SSE events to Chat Completions chunks."""

    def __init__(self, model: str, reasoning_cache: ReasoningCache | None = None):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.content = ""
        self.reasoning = ""
        self.calls: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.web_searches: set[Any] = set()
        self.reasoning_cache = reasoning_cache
        self.reasoning_blocks: list[dict[str, Any]] = []
        self._open_thinking: tuple[str, str] | None = None  # (kind, index)
        self._usage = {
            "input": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "output": 0,
            "thinking": 0,
        }
        self._finish: str | None = None
        self._open_call: dict[str, Any] | None = None
        self._open_thinking: dict[str, Any] | None = None  # pending thinking block

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
        if event_type == "message_start":
            message = event.get("message", {})
            self._usage_merge(
                message.get("usage") if isinstance(message, dict) else None
            )
            return []
        if event_type == "content_block_start":
            block = event.get("content_block")
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "tool_use":
                call_id = str(block.get("id") or "toolu_" + uuid.uuid4().hex[:24])
                name = str(block.get("name", ""))
                self._open_call = {
                    "index": event.get("index"),
                    "id": call_id,
                    "name": name,
                    "arguments": "",
                }
                # Announce the call immediately so the client's time-to-first-
                # token isn't stalled until the full JSON arguments assemble.
                return self._tool_call_chunk(
                    event.get("index"),
                    {"name": name, "arguments": ""},
                    id=call_id,
                )
            elif kind == "thinking":
                self._open_thinking = {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                }
            elif kind == "redacted_thinking":
                # Opaque, safety-flagged portion; replay verbatim on round-trip.
                self.reasoning_blocks.append(
                    {"type": "redacted_thinking", "data": block.get("data") or ""}
                )
            elif kind == WEB_SEARCH_TOOL or kind == "web_search_tool_result":
                self.web_searches.add(event.get("index"))
            return []
        if event_type == "content_block_delta":
            return self._delta(event.get("delta"))
        if event_type == "content_block_stop":
            if self._open_thinking is not None:
                # Only a signed block can be replayed upstream.
                if self._open_thinking.get("signature"):
                    self.reasoning_blocks.append(self._open_thinking)
                self._open_thinking = None
            return self._close_call()
        if event_type == "message_delta":
            delta = event.get("delta")
            stop_reason = delta.get("stop_reason") if isinstance(delta, dict) else None
            if stop_reason == "tool_use":
                self._finish = "tool_calls"
            elif stop_reason == "max_tokens":
                self._finish = "length"
            else:
                self._finish = "stop"
            self._usage_merge(event.get("usage"))
            return []
        if event_type == "error":
            detail = event.get("error") or event
            raise RuntimeError(f"Claude response failed: {detail}")
        return []

    def _delta(self, delta: Any) -> list[dict[str, Any]]:
        if not isinstance(delta, dict):
            return []
        kind = delta.get("type")
        if kind == "text_delta":
            text = str(delta.get("text", ""))
            if not text:
                return []
            self.content += text
            return [self.chunk({"content": text})]
        if kind == "thinking_delta":
            text = str(delta.get("thinking", ""))
            if self._open_thinking is not None and text:
                self._open_thinking["thinking"] += text
            if not text:
                return []
            self.reasoning += text
            return [self.chunk({"reasoning_content": text})]
        if kind == "signature_delta":
            sig = str(delta.get("signature", ""))
            if self._open_thinking is not None and sig:
                self._open_thinking["signature"] += sig
            return []
        if kind == "redacted_thinking_delta":
            data = str(delta.get("data", ""))
            last = self.reasoning_blocks[-1] if self.reasoning_blocks else None
            if data and last and last.get("type") == "redacted_thinking":
                last["data"] += data
            return []
        if kind == "input_json_delta":
            piece = str(delta.get("partial_json", ""))
            if not self._open_call or not piece:
                return []
            self._open_call["arguments"] += piece
            return self._tool_call_chunk(self._open_call["index"], {"arguments": piece})
        if kind == "citations_delta":
            return self._citation(delta.get("citation"))
        return []

    def _close_call(self) -> list[dict[str, Any]]:
        call = self._open_call
        self._open_call = None
        if not call or not call["name"]:
            return []
        if any(existing["id"] == call["id"] for existing in self.calls):
            return []
        streamed = call["arguments"].strip()
        arguments = streamed or "{}"
        item = {
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": arguments},
        }
        self.calls.append(item)
        if streamed:
            # Already sent by content_block_start plus input_json_delta.
            return []
        # A no-argument call streams no input_json_delta at all, so the client
        # would be left with "" instead of parseable JSON.
        return self._tool_call_chunk(call["index"], {"arguments": "{}"})

    def _tool_call_chunk(
        self, index: Any, function: dict[str, Any], **fields: Any
    ) -> list[dict[str, Any]]:
        call: dict[str, Any] = {"index": index, **fields, "function": function}
        if "id" in fields:
            call["type"] = "function"
        return [self.chunk({"tool_calls": [call]})]

    def _citation(self, value: Any) -> list[dict[str, Any]]:
        # Claude names web results web_search_result_location and document
        # citations by their location kind; a url is what makes one citable.
        if not isinstance(value, dict):
            return []
        url = value.get("url")
        if not isinstance(url, str) or not url:
            return []
        citation = {key: value[key] for key in ("url", "title") if key in value}
        annotation = {"type": "url_citation", "url_citation": citation}
        if annotation in self.annotations:
            return []
        self.annotations.append(annotation)
        return [self.chunk({"annotations": [annotation]})]

    def _usage_merge(self, value: Any) -> None:
        if not isinstance(value, dict):
            return
        if isinstance(value.get("input_tokens"), int):
            self._usage["input"] = value["input_tokens"]
        if isinstance(value.get("output_tokens"), int):
            self._usage["output"] = value["output_tokens"]
        if isinstance(value.get("cache_read_input_tokens"), int):
            self._usage["cache_read"] = value["cache_read_input_tokens"]
        if isinstance(value.get("cache_creation_input_tokens"), int):
            self._usage["cache_creation"] = value["cache_creation_input_tokens"]
        details = value.get("output_tokens_details")
        if isinstance(details, dict) and isinstance(
            details.get("thinking_tokens"), int
        ):
            self._usage["thinking"] = details["thinking_tokens"]

    @property
    def usage(self) -> dict[str, Any] | None:
        if not any(self._usage.values()):
            return None
        prompt = (
            self._usage["input"]
            + self._usage["cache_read"]
            + self._usage["cache_creation"]
        )
        result: dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": self._usage["output"],
            "total_tokens": prompt + self._usage["output"],
            "prompt_tokens_details": {"cached_tokens": self._usage["cache_read"]},
            "completion_tokens_details": {"reasoning_tokens": self._usage["thinking"]},
        }
        if self._usage["cache_creation"]:
            result["prompt_tokens_details"]["cache_write_tokens"] = self._usage[
                "cache_creation"
            ]
        if self.web_searches:
            result["server_tool_use"] = {"web_search_requests": len(self.web_searches)}
        return result

    def finish(self) -> list[dict[str, Any]]:
        self._cache_reasoning()
        finish_reason = self._finish or ("tool_calls" if self.calls else "stop")
        chunks = [self.chunk({}, finish_reason)]
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
        self._cache_reasoning()
        message: dict[str, Any] = {"role": "assistant", "content": self.content or None}
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
                    "finish_reason": self._finish
                    or ("tool_calls" if self.calls else "stop"),
                }
            ],
            "usage": self.usage,
        }

    def _cache_reasoning(self) -> None:
        """Persist signed thinking blocks keyed by their tool call ids.

        Only assistant turns that ended in tool calls need them replayed, so
        this is a no-op unless they exist.
        """
        if self.reasoning_cache is None or not self.reasoning_blocks:
            return
        call_ids = [str(call.get("id")) for call in self.calls if call.get("id")]
        if call_ids:
            self.reasoning_cache.put(call_ids, self.reasoning_blocks)
