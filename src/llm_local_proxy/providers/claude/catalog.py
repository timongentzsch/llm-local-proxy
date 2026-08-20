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


def model_info(item: dict[str, Any]) -> dict[str, Any]:
    value = {
        "id": item["id"],
        "canonical_slug": item["id"],
        "object": "model",
        "created": int(item.get("created") or 0),
        "owned_by": "anthropic",
        "name": item["name"],
        "architecture": {
            "modality": "text+image->text",
            "input_modalities": ["text", "image"],
            "output_modalities": ["text"],
        },
        "supported_parameters": [
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "reasoning_effort",
            "web_search",
            "temperature",
            "top_p",
        ],
        "default_parameters": (
            {"max_tokens": item["max_output_tokens"]}
            if item.get("max_output_tokens")
            else None
        ),
        "per_request_limits": None,
        "is_default": False,
        "supported_reasoning_efforts": item.get("reasoning_efforts")
        or ["low", "medium", "high"],
    }
    context = item.get("context_length")
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        context = next(
            (
                model.get("context_length")
                for model in CLAUDE_MODELS
                if model["id"] == item["id"]
            ),
            None,
        )
    if isinstance(context, int) and not isinstance(context, bool) and context > 0:
        value["context_length"] = context
    return value


def claude_model_name(model: Any) -> str | None:
    if not isinstance(model, str) or not model:
        return None
    bare = model.split("/", 1)[1] if "/" in model else model
    return bare if bare.startswith("claude-") else None
