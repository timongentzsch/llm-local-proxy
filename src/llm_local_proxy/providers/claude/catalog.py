"""The Claude model catalog."""

from __future__ import annotations

from typing import Any

CLAUDE_MODELS = [
    {
        "id": "claude-opus-5",
        "name": "Claude Opus 5",
        "context_length": 1_000_000,
        "max_output_tokens": 128_000,
    },
    {
        "id": "claude-sonnet-5",
        "name": "Claude Sonnet 5",
        "context_length": 1_000_000,
        "max_output_tokens": 128_000,
    },
    {
        "id": "claude-haiku-4-5",
        "name": "Claude Haiku 4.5",
        "context_length": 200_000,
        "max_output_tokens": 64_000,
    },
]


def claude_model_name(model: Any) -> str | None:
    if not isinstance(model, str) or not model:
        return None
    bare = model.split("/", 1)[1] if "/" in model else model
    return bare if bare.startswith("claude-") else None
