"""Canonical stream events -> schema-valid OpenAI Responses objects/events."""

from __future__ import annotations

import copy
import time
import uuid
from typing import Any

from ...ir import (
    ChatRequest,
    Citation,
    Finish,
    FunctionTool,
    NativeItem,
    NativeTool,
    ReasoningItem,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallArgs,
    ToolCallEnd,
    ToolCallStart,
    Usage,
    WebSearchTool,
)


def _tool(tool: Any) -> dict[str, Any]:
    if isinstance(tool, NativeTool):
        return copy.deepcopy(tool.item)
    if isinstance(tool, (FunctionTool, WebSearchTool)) and tool.native is not None:
        return copy.deepcopy(tool.native)
    if isinstance(tool, FunctionTool):
        value: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "parameters": tool.parameters,
        }
        if tool.description:
            value["description"] = tool.description
        return value
    value = {"type": "web_search"}
    if tool.context_size:
        value["search_context_size"] = tool.context_size
    return value


def _tool_choice(request: ChatRequest | None) -> Any:
    choice = request.tool_choice if request else None
    if choice is None:
        return "auto"
    if choice.kind == "tool":
        return {"type": "function", "name": choice.name}
    return choice.kind


class ResponseEncoder:
    def __init__(self, model: str, decoder: Any, request: ChatRequest | None = None):
        self.id = "resp_" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.decoder = decoder
        self.request = request
        self.output: list[dict[str, Any]] = []
        self.usage: dict[str, Any] | None = None
        self._sequence = 0
        self._message: dict[str, Any] | None = None
        self._reasoning: dict[str, Any] | None = None
        self._reasoning_part_open = False
        self._calls: dict[Any, dict[str, Any]] = {}
        self._finish_reason = "end_turn"
        self._drained = False
        self._terminal = False

    def _response(self, status: str, *, output: bool) -> dict[str, Any]:
        request = self.request
        tools = [_tool(tool) for tool in request.tools] if request else []
        incomplete = {"reason": "max_output_tokens"} if status == "incomplete" else None
        return {
            "id": self.id,
            "object": "response",
            "created_at": self.created,
            "status": status,
            "model": self.model,
            "output": copy.deepcopy(self.output) if output else [],
            "parallel_tool_calls": bool(
                True
                if request is None or request.parallel_tool_calls is None
                else request.parallel_tool_calls
            ),
            "tool_choice": _tool_choice(request),
            "tools": tools,
            "usage": copy.deepcopy(self.usage) if output else None,
            "error": None,
            "incomplete_details": incomplete,
        }

    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"type": kind, "sequence_number": self._sequence, **fields}
        self._sequence += 1
        return event

    def start(self) -> dict[str, Any]:
        return self._event(
            "response.created", response=self._response("in_progress", output=False)
        )

    def error(self, message: str) -> dict[str, Any]:
        self._terminal = True
        return self._event("error", code="upstream_error", message=message, param=None)

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        return self._encode(self.decoder.decode(event))

    def finish(self) -> list[dict[str, Any]]:
        chunks = self._drain()
        chunks.extend(self._close_open_items())
        if self._terminal:
            return chunks
        self._terminal = True
        incomplete = self._finish_reason in {"length", "max_tokens"}
        status = "incomplete" if incomplete else "completed"
        kind = "response.incomplete" if incomplete else "response.completed"
        chunks.append(self._event(kind, response=self._response(status, output=True)))
        return chunks

    def result(self) -> dict[str, Any]:
        self._drain()
        self._close_open_items()
        self._terminal = True
        status = (
            "incomplete"
            if self._finish_reason in {"length", "max_tokens"}
            else "completed"
        )
        return self._response(status, output=True)

    def _drain(self) -> list[dict[str, Any]]:
        if self._drained:
            return []
        self._drained = True
        return self._encode(self.decoder.finish())

    def _encode(self, events: list[StreamEvent]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for event in events:
            chunks.extend(self._one(event))
        return chunks

    def _one(self, event: StreamEvent) -> list[dict[str, Any]]:
        if isinstance(event, ThinkingDelta):
            chunks = self._complete_message()
            chunks.extend(self._ensure_reasoning(summary=True))
            assert self._reasoning is not None
            summary = self._reasoning["summary"][0]
            summary["text"] += event.text
            chunks.append(
                self._event(
                    "response.reasoning_summary_text.delta",
                    item_id=self._reasoning["id"],
                    output_index=self.output.index(self._reasoning),
                    summary_index=0,
                    delta=event.text,
                )
            )
            return chunks
        if isinstance(event, ReasoningItem):
            return self._complete_reasoning(event.item)
        if isinstance(event, NativeItem):
            chunks = self._close_open_items()
            item = copy.deepcopy(event.item)
            self.output.append(item)
            index = len(self.output) - 1
            added = copy.deepcopy(item)
            if "status" in added:
                added["status"] = "in_progress"
            chunks.extend(
                [
                    self._event(
                        "response.output_item.added",
                        output_index=index,
                        item=added,
                    ),
                    self._event(
                        "response.output_item.done",
                        output_index=index,
                        item=copy.deepcopy(item),
                    ),
                ]
            )
            return chunks
        if isinstance(event, TextDelta):
            chunks = self._complete_reasoning_open()
            chunks.extend(self._ensure_message())
            assert self._message is not None
            part = self._message["content"][0]
            part["text"] += event.text
            chunks.append(
                self._event(
                    "response.output_text.delta",
                    item_id=self._message["id"],
                    output_index=self.output.index(self._message),
                    content_index=0,
                    delta=event.text,
                    logprobs=[],
                )
            )
            return chunks
        if isinstance(event, ToolCallStart):
            chunks = self._close_open_items()
            item = {
                "type": "function_call",
                "id": "fc_" + uuid.uuid4().hex,
                "call_id": event.id,
                "name": event.name,
                "arguments": event.arguments,
                "status": "in_progress",
            }
            self._calls[event.index] = item
            self.output.append(item)
            chunks.append(
                self._event(
                    "response.output_item.added",
                    output_index=len(self.output) - 1,
                    item=copy.deepcopy(item),
                )
            )
            return chunks
        if isinstance(event, ToolCallArgs):
            item = self._calls.get(event.index)
            if item is None:
                return []
            item["arguments"] += event.fragment
            return [
                self._event(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=self.output.index(item),
                    delta=event.fragment,
                )
            ]
        if isinstance(event, ToolCallEnd):
            item = self._calls.get(event.index)
            if item is None:
                return []
            item["arguments"] = event.arguments
            item["status"] = "completed"
            index = self.output.index(item)
            return [
                self._event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=index,
                    name=item["name"],
                    arguments=event.arguments,
                ),
                self._event(
                    "response.output_item.done",
                    output_index=index,
                    item=copy.deepcopy(item),
                ),
            ]
        if isinstance(event, Citation):
            if self._message is None:
                return []
            annotation = {
                "type": "url_citation",
                "url": event.url,
                "title": event.title or "",
                "start_index": event.start_index or 0,
                "end_index": event.end_index or 0,
            }
            part = self._message["content"][0]
            part["annotations"].append(annotation)
            return [
                self._event(
                    "response.output_text.annotation.added",
                    item_id=self._message["id"],
                    output_index=self.output.index(self._message),
                    content_index=0,
                    annotation_index=len(part["annotations"]) - 1,
                    annotation=copy.deepcopy(annotation),
                )
            ]
        if isinstance(event, Usage):
            self.usage = {
                "input_tokens": event.prompt,
                "output_tokens": event.completion,
                "total_tokens": event.total
                if event.total is not None
                else event.prompt + event.completion,
                "input_tokens_details": {
                    "cached_tokens": event.cache_read,
                    "cache_write_tokens": event.cache_write,
                },
                "output_tokens_details": {"reasoning_tokens": event.thinking},
            }
            return []
        if isinstance(event, Finish):
            self._finish_reason = event.reason
        return []

    def _ensure_reasoning(self, *, summary: bool) -> list[dict[str, Any]]:
        if self._reasoning is not None:
            return []
        item = {
            "type": "reasoning",
            "id": "rs_" + uuid.uuid4().hex,
            "summary": [],
        }
        self._reasoning = item
        self.output.append(item)
        index = len(self.output) - 1
        chunks = [
            self._event(
                "response.output_item.added",
                output_index=index,
                item=copy.deepcopy(item),
            )
        ]
        if summary:
            item["summary"].append({"type": "summary_text", "text": ""})
            self._reasoning_part_open = True
            chunks.append(
                self._event(
                    "response.reasoning_summary_part.added",
                    item_id=item["id"],
                    output_index=index,
                    summary_index=0,
                    part={"type": "summary_text", "text": ""},
                )
            )
        return chunks

    def _complete_reasoning(self, opaque: dict[str, Any]) -> list[dict[str, Any]]:
        if self._reasoning is None:
            item = copy.deepcopy(opaque)
            self.output.append(item)
            index = len(self.output) - 1
            return [
                self._event(
                    "response.output_item.added",
                    output_index=index,
                    item={**copy.deepcopy(item), "status": "in_progress"},
                ),
                self._event(
                    "response.output_item.done",
                    output_index=index,
                    item=copy.deepcopy(item),
                ),
            ]
        item = self._reasoning
        item.update(copy.deepcopy(opaque))
        index = self.output.index(item)
        chunks: list[dict[str, Any]] = []
        summary = item.get("summary")
        if self._reasoning_part_open and isinstance(summary, list) and summary:
            text = str(summary[0].get("text", ""))
            chunks.extend(
                [
                    self._event(
                        "response.reasoning_summary_text.done",
                        item_id=item["id"],
                        output_index=index,
                        summary_index=0,
                        text=text,
                    ),
                    self._event(
                        "response.reasoning_summary_part.done",
                        item_id=item["id"],
                        output_index=index,
                        summary_index=0,
                        part=copy.deepcopy(summary[0]),
                    ),
                ]
            )
        chunks.append(
            self._event(
                "response.output_item.done",
                output_index=index,
                item=copy.deepcopy(item),
            )
        )
        self._reasoning = None
        self._reasoning_part_open = False
        return chunks

    def _complete_reasoning_open(self) -> list[dict[str, Any]]:
        if self._reasoning is None:
            return []
        return self._complete_reasoning(copy.deepcopy(self._reasoning))

    def _ensure_message(self) -> list[dict[str, Any]]:
        if self._message is not None:
            return []
        item = {
            "type": "message",
            "id": "msg_" + uuid.uuid4().hex,
            "status": "in_progress",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
            ],
        }
        self._message = item
        self.output.append(item)
        index = len(self.output) - 1
        return [
            self._event(
                "response.output_item.added",
                output_index=index,
                item=copy.deepcopy(item),
            ),
            self._event(
                "response.content_part.added",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                part=copy.deepcopy(item["content"][0]),
            ),
        ]

    def _complete_message(self) -> list[dict[str, Any]]:
        if self._message is None:
            return []
        item = self._message
        item["status"] = "completed"
        index = self.output.index(item)
        part = item["content"][0]
        self._message = None
        return [
            self._event(
                "response.output_text.done",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                text=part["text"],
                logprobs=[],
            ),
            self._event(
                "response.content_part.done",
                item_id=item["id"],
                output_index=index,
                content_index=0,
                part=copy.deepcopy(part),
            ),
            self._event(
                "response.output_item.done",
                output_index=index,
                item=copy.deepcopy(item),
            ),
        ]

    def _close_open_items(self) -> list[dict[str, Any]]:
        chunks = self._complete_reasoning_open()
        chunks.extend(self._complete_message())
        return chunks
