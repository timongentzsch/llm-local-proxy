from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any


class RequestError(ValueError):
    pass


_UNSUPPORTED = {
    "logprobs",
    "response_format",
    "seed",
    "stop",
    "temperature",
    "top_logprobs",
    "top_p",
}


def _unsupported(body: dict[str, Any]) -> list[str]:
    result = []
    for name in _UNSUPPORTED & body.keys():
        value = body[name]
        neutral = value is None
        neutral |= name in {"temperature", "top_p"} and value == 1
        neutral |= name == "logprobs" and value is False
        neutral |= name == "response_format" and value == {"type": "text"}
        if not neutral:
            result.append(name)
    return sorted(result)


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


def _content(content: Any, role: str) -> list[dict[str, Any]]:
    kind = "output_text" if role == "assistant" else "input_text"
    if isinstance(content, str):
        return [{"type": kind, "text": content}] if content else []
    if content is None:
        return []
    if not isinstance(content, list):
        raise RequestError("message content must be a string or array")
    result = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            result.append({"type": kind, "text": str(part.get("text", ""))})
        elif role == "user" and part.get("type") == "image_url":
            image = part.get("image_url", {})
            url = image.get("url") if isinstance(image, dict) else image
            if url:
                result.append({"type": "input_image", "image_url": url})
        else:
            raise RequestError(f"unsupported {role} content type")
    return result


def _tool_choice(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value or "auto"
    if isinstance(value, dict) and value.get("type") == "function":
        function = value.get("function", {})
        if isinstance(function, dict) and function.get("name"):
            return {"type": "function", "name": function["name"]}
    raise RequestError("unsupported tool_choice")


def _tool(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RequestError("invalid tool")
    if value.get("type") == "openrouter:web_search":
        item: dict[str, Any] = {"type": "web_search"}
        parameters = value.get("parameters", {})
        if isinstance(parameters, dict):
            context = parameters.get("search_context_size")
            if context in {"low", "medium", "high"}:
                item["search_context_size"] = context
        return item
    function = value.get("function", {})
    if value.get("type") != "function" or not isinstance(function, dict):
        raise RequestError(
            "only function and openrouter:web_search tools are supported"
        )
    item = {
        "type": "function",
        "name": function.get("name"),
        "parameters": function.get("parameters", {"type": "object"}),
    }
    if function.get("description"):
        item["description"] = function["description"]
    if not item["name"]:
        raise RequestError("function tool name is required")
    return item


def build_request(
    body: dict[str, Any],
    cache: ReasoningCache,
    session_id: str = "",
) -> tuple[dict[str, Any], str]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise RequestError("model is required")
    if body.get("n", 1) != 1:
        raise RequestError("n must be 1")
    unsupported = _unsupported(body)
    if unsupported:
        raise RequestError(f"unsupported parameters: {', '.join(unsupported)}")

    instructions = "\n\n".join(
        _text(message.get("content"))
        for message in messages
        if isinstance(message, dict)
        and message.get("role") in {"system", "developer"}
        and _text(message.get("content"))
    )
    input_items: list[dict[str, Any]] = []
    first_user = ""
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role in {"system", "developer"}:
            continue
        if role in {"user", "assistant"}:
            if role == "user" and not first_user:
                first_user = _text(message.get("content"))
            content = _content(message.get("content"), role)
            if content:
                input_items.append({"role": role, "content": content})
            calls = message.get("tool_calls", [])
            if role == "assistant" and isinstance(calls, list) and calls:
                call_ids = [
                    str(call.get("id", ""))
                    for call in calls
                    if isinstance(call, dict) and call.get("id")
                ]
                input_items.extend(cache.get(call_ids))
                for call in calls:
                    function = (
                        call.get("function", {}) if isinstance(call, dict) else {}
                    )
                    if not isinstance(function, dict) or not function.get("name"):
                        raise RequestError("invalid assistant tool call")
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": str(call.get("id", "")),
                            "name": function["name"],
                            "arguments": str(function.get("arguments", "{}")),
                        }
                    )
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if not call_id:
                raise RequestError("tool message is missing tool_call_id")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(call_id),
                    "output": _text(message.get("content")),
                }
            )
        else:
            raise RequestError(f"unsupported message role: {role}")

    request_tools = body.get("tools") or []
    if not isinstance(request_tools, list):
        raise RequestError("tools must be an array")
    tools = [_tool(tool) for tool in request_tools]

    session = session_id or str(body.get("session_id", ""))
    if not session:
        seed = f"{instructions}\0{first_user}".encode()
        session = "proxy-" + hashlib.sha256(seed).hexdigest()[:24]
    request: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_items,
        "store": False,
        "stream": True,
        "prompt_cache_key": session,
    }
    if tools:
        request["tools"] = tools
        request["tool_choice"] = _tool_choice(body.get("tool_choice"))
        request["parallel_tool_calls"] = bool(body.get("parallel_tool_calls", True))
    effort = body.get("reasoning_effort")
    if not effort and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    if effort:
        request["reasoning"] = {"effort": effort, "summary": "auto"}
        request["include"] = ["reasoning.encrypted_content"]
    return request, session


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
