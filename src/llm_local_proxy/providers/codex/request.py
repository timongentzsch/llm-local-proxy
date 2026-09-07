"""ChatRequest -> a Codex Responses API request body."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from typing import Any

from ...errors import RequestError
from ...ir import (
    ChatRequest,
    Image,
    NativeAnthropicBlock,
    NativeResponseItem,
    OutputFormat,
    Reasoning,
    Text,
    Thinking,
    ToolResult,
    ToolUse,
    Turn,
)
from ...tools import arguments, responses_choice, responses_tool
from ..reasoning import ReasoningCache
from .thinking import unpack as unpack_thinking

#: Knobs Codex does not expose, and the value of each that means "unset".
UNSUPPORTED = (
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "top_k",
    "logprobs",
    "seed",
    "stop",
    "temperature",
    "top_logprobs",
    "top_p",
)
NEUTRAL: dict[str, Any] = {
    "temperature": 1,
    "top_p": 1,
    "logprobs": False,
}
_UNSET = object()


def _reject_unsupported(params: dict[str, Any]) -> None:
    named = sorted(
        name
        for name in UNSUPPORTED
        if name in params
        and params[name] is not None
        and params[name] != NEUTRAL.get(name, _UNSET)
    )
    if named:
        raise RequestError(f"unsupported parameters: {', '.join(named)}")


def _output_format(fmt: OutputFormat) -> dict[str, Any]:
    """The Responses `text.format` item for a neutral output format."""
    if fmt.kind == "json_object":
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        # Responses requires a label Messages never sends; the schema is what
        # constrains the model, so a placeholder costs the client nothing.
        "name": fmt.name or "response",
        "schema": fmt.schema,
        "strict": fmt.strict,
    }


def _flush_content(items: list[dict[str, Any]], pending: list[Any], role: str) -> None:
    if pending:
        items.append({"role": role, "content": list(pending)})
    pending.clear()


def _turn_items(turn: Turn, cache: ReasoningCache) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    uses = [block for block in turn.blocks if isinstance(block, ToolUse)]
    has_reasoning = any(
        isinstance(block, (Reasoning, Thinking)) for block in turn.blocks
    )
    cached = [] if has_reasoning else cache.get([use.id for use in uses if use.id])
    cache_inserted = False
    for block in turn.blocks:
        if isinstance(block, Text):
            kind = "output_text" if turn.role == "assistant" else "input_text"
            pending.append({"type": kind, "text": block.text})
            continue
        if isinstance(block, Image):
            if turn.role == "assistant":
                raise RequestError("unsupported assistant content type: image_url")
            pending.append({"type": "input_image", "image_url": block.url})
            continue
        _flush_content(items, pending, turn.role)
        if isinstance(block, (Reasoning, NativeResponseItem)):
            items.append(dict(block.item))
        elif isinstance(block, NativeAnthropicBlock):
            raise RequestError(
                "Codex upstream cannot faithfully represent Anthropic content block: "
                + str(block.item.get("type", "unknown"))
            )
        elif isinstance(block, Thinking):
            try:
                bridged = unpack_thinking(block.signature)
            except ValueError as exc:
                raise RequestError(str(exc)) from None
            if bridged is None:
                raise RequestError(
                    "Codex upstream cannot faithfully represent Anthropic signed thinking"
                )
            if block.text != bridged.thinking:
                raise RequestError("Codex reasoning thinking text was modified")
            items.append(dict(bridged.item))
        elif isinstance(block, ToolUse):
            value = arguments(block.arguments, RequestError)
            # Keep existing wire bytes stable for replay and prompt caching.
            encoded = (
                block.arguments
                if isinstance(block.arguments, str) and block.arguments.strip()
                else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            )
            if not cache_inserted:
                items.extend(cached)
                cache_inserted = True
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.id,
                    "name": block.name,
                    "arguments": encoded,
                }
            )
        elif isinstance(block, ToolResult):
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.tool_use_id,
                    "output": block.text,
                }
            )
    _flush_content(items, pending, turn.role)
    return items


def build(
    request: ChatRequest,
    cache: ReasoningCache,
    reasoning_efforts: Collection[str] | None = None,
) -> tuple[dict[str, Any], str]:
    if not request.model:
        raise RequestError("model is required")
    _reject_unsupported(request.params)

    # Codex caches prefixes implicitly, so breakpoints do not apply.
    instructions = "\n\n".join(block.text for block in request.system)
    items: list[dict[str, Any]] = []
    # The opening user turn seeds the fallback cache key, and empty text is a
    # legitimate value for an image-only turn: a sentinel that cannot tell
    # "empty" from "not seen yet" would re-seed the key from a later turn and
    # move the whole conversation to a different upstream cache mid-flight.
    first_user: str | None = None
    for turn in request.turns:
        if turn.role == "user" and first_user is None:
            first_user = "\n".join(
                block.text if isinstance(block, Text) else block.url
                for block in turn.blocks
                if isinstance(block, (Text, Image))
            )
        items.extend(_turn_items(turn, cache))

    session = request.session
    if not session:
        seed = f"{instructions}\0{first_user or ''}".encode()
        session = "proxy-" + hashlib.sha256(seed).hexdigest()[:24]

    body: dict[str, Any] = {
        "model": request.model,
        "instructions": instructions,
        "input": items,
        "store": False,
        "stream": True,
        "prompt_cache_key": session,
    }
    tools = [responses_tool(tool) for tool in request.tools]
    if tools:
        body["tools"] = tools
        body["tool_choice"] = responses_choice(request.tool_choice)
        body["parallel_tool_calls"] = request.parallel_tool_calls is not False
    if request.output_format is not None:
        body["text"] = {"format": _output_format(request.output_format)}
    if request.thinking_budget is not None:
        raise RequestError(
            "Codex upstream cannot faithfully represent an Anthropic thinking budget; "
            "use output_config.effort"
        )
    if request.thinking_mode == "disabled":
        raise RequestError("Codex upstream cannot guarantee that reasoning is disabled")
    if request.reasoning_effort:
        effort = str(request.reasoning_effort).casefold()
        supported = {str(item).casefold() for item in reasoning_efforts or ()}
        if supported and effort not in supported:
            raise RequestError(
                f"unsupported reasoning_effort: {request.reasoning_effort}"
            )
        body["reasoning"] = {"effort": effort}
        if request.thinking_display != "omitted":
            body["reasoning"]["summary"] = "auto"
    # Models can reason at their catalog default even when the client omits an
    # explicit effort, so always request the completed encrypted item.
    body["include"] = ["reasoning.encrypted_content"]
    return body, session
