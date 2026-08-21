"""The Claude model catalog."""

from __future__ import annotations

from typing import Any

from ..catalog import model_info as shared

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


def model_info(item: dict[str, Any]) -> dict[str, Any]:
    context = item.get("context_length")
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        context = next(
            (m.get("context_length") for m in CLAUDE_MODELS if m["id"] == item["id"]),
            0,
        )
    return shared(
        item["id"],
        item["name"],
        "anthropic",
        extra_parameters=("temperature", "top_p"),
        default_parameters=(
            {"max_tokens": item["max_output_tokens"]}
            if item.get("max_output_tokens")
            else None
        ),
        reasoning_efforts=item.get("reasoning_efforts") or [],
        context_length=context if isinstance(context, int) else 0,
        created=int(item.get("created") or 0),
    )


def claude_model_name(model: Any) -> str | None:
    if not isinstance(model, str) or not model:
        return None
    bare = model.split("/", 1)[1] if "/" in model else model
    return bare if bare.startswith("claude-") else None
