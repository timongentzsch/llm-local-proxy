"""Translation between Chat Completions and Claude's Messages API."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .protocol import RequestError, _text

WEB_SEARCH_BETA = "web-search-2025-03-05"
WEB_SEARCH_TOOL = "web_search_20250305"
DEFAULT_MAX_OUTPUT_TOKENS = 32768

# Fallback catalog used when the live /v1/models listing is unavailable (not
# signed in, or the transport fails); requests for any `claude-*` name are
# routed, the table only supplies display names and output-token defaults.
CLAUDE_MODELS = [
    {"id": "claude-opus-5", "name": "Claude Opus 5", "max_output_tokens": 128000},
    {"id": "claude-sonnet-5", "name": "Claude Sonnet 5", "max_output_tokens": 128000},
    {"id": "claude-haiku-4-5", "name": "Claude Haiku 4.5", "max_output_tokens": 64000},
]

_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 32768,
    "max": 65536,
}


def claude_model_name(model: Any) -> str | None:
    if not isinstance(model, str) or not model:
        return None
    bare = model.split("/", 1)[1] if "/" in model else model
    return bare if bare.startswith("claude-") else None


def _max_output_tokens(model: str) -> int:
    for item in CLAUDE_MODELS:
        if item["id"] == model:
            return int(item["max_output_tokens"])
    return DEFAULT_MAX_OUTPUT_TOKENS


def build_messages_request(
    body: dict[str, Any],
    model: str,
    max_output: int | None = None,
    thinking: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty array")
    if body.get("n", 1) != 1:
        raise RequestError("n must be 1")
    for name in (
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "seed",
        "response_format",
        "logit_bias",
    ):
        if name in body and body[name] is not None:
            raise RequestError(f"unsupported parameter: {name}")
    temperature = body.get("temperature")
    if temperature is not None and not (
        isinstance(temperature, (int, float))
        and not isinstance(temperature, bool)
        and 0 <= float(temperature) <= 1
    ):
        raise RequestError("temperature must be a number between 0 and 1")
    top_p = body.get("top_p")
    if top_p is not None and not (
        isinstance(top_p, (int, float))
        and not isinstance(top_p, bool)
        and 0 < float(top_p) <= 1
    ):
        raise RequestError("top_p must be a number between 0 and 1")
    top_k = body.get("top_k")
    if top_k is not None and not (
        isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0
    ):
        raise RequestError("top_k must be a positive integer")

    system_parts: list[str] = []
    request_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("each message must be an object")
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _text(message.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            blocks = _user_blocks(message.get("content"))
            if blocks:
                request_messages.append({"role": "user", "content": blocks})
        elif role == "assistant":
            blocks = _assistant_blocks(message)
            if blocks:
                request_messages.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            _append_tool_result(request_messages, message)
        else:
            raise RequestError(f"unsupported message role: {role}")
    if not request_messages or request_messages[0]["role"] != "user":
        raise RequestError("first message must be a user message")

    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    if max_tokens is None:
        if isinstance(max_output, int) and not isinstance(max_output, bool) and max_output > 0:
            max_tokens = max_output
        else:
            max_tokens = _max_output_tokens(model)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise RequestError("max_tokens must be a positive integer")

    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": request_messages,
        "stream": True,
    }
    if system_parts:
        request["system"] = "\n\n".join(system_parts)
    if temperature is not None:
        request["temperature"] = float(temperature)
    if top_p is not None:
        request["top_p"] = float(top_p)
    if top_k is not None:
        request["top_k"] = int(top_k)
    stop = body.get("stop")
    if isinstance(stop, str) and stop:
        request["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        sequences = [str(item) for item in stop if item]
        if sequences:
            request["stop_sequences"] = sequences

    betas: list[str] = []
    tools_in = body.get("tools")
    if tools_in is not None:
        if not isinstance(tools_in, list):
            raise RequestError("tools must be an array")
        tools: list[dict[str, Any]] = []
        web_searches = 0
        for tool in tools_in:
            if isinstance(tool, dict) and tool.get("type") == "openrouter:web_search":
                web_searches += 1
                continue
            item = _claude_tool(tool)
            if item:
                tools.append(item)
        if web_searches:
            tools.append({"type": WEB_SEARCH_TOOL, "name": "web_search"})
            betas.append(WEB_SEARCH_BETA)
        tool_choice = body.get("tool_choice")
        if tool_choice == "none":
            tools = []
        if tools:
            request["tools"] = tools
            request["tool_choice"] = _claude_tool_choice(tool_choice)

    effort = body.get("reasoning_effort")
    if not effort and isinstance(body.get("reasoning"), dict):
        effort = body["reasoning"].get("effort")
    if effort:
        if thinking == "adaptive":
            request["thinking"] = {"type": "adaptive"}
        else:
            budget = _THINKING_BUDGETS.get(str(effort).casefold())
            if not budget:
                raise RequestError(f"unsupported reasoning_effort: {effort}")
            budget = min(budget, max_tokens - 1)
            if budget < 1024:
                raise RequestError(
                    "max_tokens is too small for the requested thinking budget"
                )
            request["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return request, betas


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        raise RequestError("message content must be a string or array")
    blocks = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": str(part.get("text", ""))})
        elif kind == "image_url":
            image = part.get("image_url", {})
            url = image.get("url") if isinstance(image, dict) else image
            block = _image_block(url)
            if block:
                blocks.append(block)
        else:
            raise RequestError(f"unsupported user content type: {kind}")
    return blocks


def _image_block(url: Any) -> dict[str, Any] | None:
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        if not data:
            return None
        media_type = header[5:].split(";")[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    if url.startswith(("http://", "https://")):
        return {"type": "image", "source": {"type": "url", "url": url}}
    raise RequestError("image_url must be a data URL or an http(s) URL")


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = [
        {"type": "text", "text": str(part.get("text", ""))}
        for part in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else [{"type": "text", "text": message.get("content")}]
            if message.get("content") is not None
            else []
        )
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    calls = message.get("tool_calls", [])
    if not isinstance(calls, list):
        raise RequestError("tool_calls must be an array")
    for call in calls:
        if not isinstance(call, dict):
            raise RequestError("invalid assistant tool call")
        function = call.get("function", {})
        if not isinstance(function, dict) or not function.get("name"):
            raise RequestError("invalid assistant tool call")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                raise RequestError("tool call arguments must be a JSON object") from None
        if not isinstance(arguments, dict):
            raise RequestError("tool call arguments must be an object")
        call_id = str(call.get("id") or "")
        if not call_id:
            call_id = "toolu_" + uuid.uuid4().hex[:24]
        blocks.append({"type": "tool_use", "id": call_id, "name": str(function["name"]), "input": arguments})
    return blocks


def _append_tool_result(messages: list[dict[str, Any]], message: dict[str, Any]) -> None:
    tool_use_id = message.get("tool_call_id") or message.get("tool_use_id")
    if not tool_use_id:
        raise RequestError("tool message is missing tool_call_id")
    content = message.get("content")
    if isinstance(content, list):
        text = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = _text(content)
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": str(tool_use_id),
        "content": text,
    }
    if message.get("is_error"):
        block["is_error"] = True
    last = messages[-1] if messages else None
    if (
        last
        and last["role"] == "user"
        and isinstance(last.get("content"), list)
        and all(isinstance(item, dict) and item.get("type") == "tool_result" for item in last["content"])
    ):
        last["content"].append(block)
    else:
        messages.append({"role": "user", "content": [block]})


def _claude_tool(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        raise RequestError("invalid tool")
    function = value.get("function")
    if value.get("type") != "function" or not isinstance(function, dict):
        raise RequestError("only function and openrouter:web_search tools are supported")
    name = function.get("name")
    if not name:
        raise RequestError("function tool name is required")
    item: dict[str, Any] = {
        "name": str(name),
        "input_schema": function.get("parameters") or {"type": "object"},
    }
    if function.get("description"):
        item["description"] = str(function["description"])
    return item


def _claude_tool_choice(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "auto"}
    if isinstance(value, str):
        if value == "auto":
            return {"type": "auto"}
        if value == "required":
            return {"type": "any"}
    if isinstance(value, dict) and value.get("type") == "function":
        name = value.get("function", {}).get("name")
        if name:
            return {"type": "tool", "name": str(name)}
    raise RequestError("unsupported tool_choice")


class ClaudeTranslator:
    """Claude Messages SSE events to Chat Completions chunks."""

    def __init__(self, model: str):
        self.id = "chatcmpl-" + uuid.uuid4().hex
        self.created = int(time.time())
        self.model = model
        self.content = ""
        self.reasoning = ""
        self.calls: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []
        self.web_searches: set[Any] = set()
        self._usage = {
            "input": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "output": 0,
        }
        self._finish: str | None = None
        self._open_call: dict[str, Any] | None = None

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
            self._usage_merge(message.get("usage") if isinstance(message, dict) else None)
            return []
        if event_type == "content_block_start":
            block = event.get("content_block")
            kind = block.get("type") if isinstance(block, dict) else None
            if kind == "tool_use":
                self._open_call = {
                    "id": str(block.get("id") or "toolu_" + uuid.uuid4().hex[:24]),
                    "name": str(block.get("name", "")),
                    "arguments": "",
                }
            elif kind == WEB_SEARCH_TOOL or kind == "web_search_tool_result":
                self.web_searches.add(event.get("index"))
            return []
        if event_type == "content_block_delta":
            return self._delta(event.get("delta"))
        if event_type == "content_block_stop":
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
            if not text:
                return []
            self.reasoning += text
            return [self.chunk({"reasoning_content": text})]
        if kind == "input_json_delta":
            piece = str(delta.get("partial_json", ""))
            if self._open_call and piece:
                self._open_call["arguments"] += piece
            return []
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
        arguments = call["arguments"].strip() or "{}"
        item = {
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": arguments},
        }
        self.calls.append(item)
        index = len(self.calls) - 1
        return [self.chunk({"tool_calls": [{"index": index, **item}]})]

    def _citation(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, dict) or value.get("type") != "url_citation":
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

    @property
    def usage(self) -> dict[str, Any] | None:
        if not any(self._usage.values()):
            return None
        prompt = self._usage["input"] + self._usage["cache_read"] + self._usage["cache_creation"]
        result: dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": self._usage["output"],
            "total_tokens": prompt + self._usage["output"],
            "prompt_tokens_details": {"cached_tokens": self._usage["cache_read"]},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
        if self._usage["cache_creation"]:
            result["prompt_tokens_details"]["cache_creation_tokens"] = self._usage["cache_creation"]
        if self.web_searches:
            result["server_tool_use"] = {"web_search_requests": len(self.web_searches)}
        return result

    def finish(self) -> list[dict[str, Any]]:
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
                    "finish_reason": self._finish or ("tool_calls" if self.calls else "stop"),
                }
            ],
            "usage": self.usage,
        }
