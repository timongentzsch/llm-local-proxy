"""The Claude model catalog."""

from __future__ import annotations

from typing import Any

from ..catalog import model_info as shared


def model_info(item: dict[str, Any]) -> dict[str, Any]:
    context = item.get("context_length")
    if not isinstance(context, int) or isinstance(context, bool) or context <= 0:
        context = 0
    return shared(
        item["id"],
        item["name"],
        "anthropic",
        modalities=(
            [str(value) for value in item["modalities"]]
            if isinstance(item.get("modalities"), list)
            else None
        ),
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
