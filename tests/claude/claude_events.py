"""Claude Messages SSE events for one content block or the closing delta."""

from __future__ import annotations


def _delta(index: int, delta: dict) -> dict:
    return {"type": "content_block_delta", "index": index, "delta": delta}


def _block(index: int, start: dict, *deltas: dict) -> list[dict]:
    return [
        {"type": "content_block_start", "index": index, "content_block": start},
        *(_delta(index, delta) for delta in deltas),
        {"type": "content_block_stop", "index": index},
    ]


def server_block(index: int, block: dict) -> list[dict]:
    """A server tool block (search request or result), complete at start."""
    return _block(index, block)


def text(index: int, text: str) -> list[dict]:
    return _block(
        index,
        {"type": "text", "text": ""},
        {"type": "text_delta", "text": text},
    )


def thinking(index: int, text: str, signature: str) -> list[dict]:
    """A signed thinking block; empty text is streamed as no thinking delta."""
    deltas = [{"type": "thinking_delta", "thinking": text}] if text else []
    return _block(
        index,
        {"type": "thinking", "thinking": ""},
        *deltas,
        {"type": "signature_delta", "signature": signature},
    )


def tool_use(index: int, id: str, name: str, args_json: str) -> list[dict]:
    return _block(
        index,
        {"type": "tool_use", "id": id, "name": name},
        {"type": "input_json_delta", "partial_json": args_json},
    )


def stop(reason: str, usage: dict | None = None) -> dict:
    event = {"type": "message_delta", "delta": {"stop_reason": reason}}
    if usage is not None:
        event["usage"] = usage
    return event
