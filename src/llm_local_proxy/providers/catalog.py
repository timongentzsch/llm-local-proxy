"""The OpenRouter-compatible model shape both providers report."""

from __future__ import annotations

from typing import Any

#: Every model the proxy serves accepts these; providers add their own.
PARAMETERS = (
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "reasoning",
    "reasoning_effort",
    "web_search",
)
EFFORTS = ("low", "medium", "high")


def model_info(
    model: str,
    name: str,
    owned_by: str,
    *,
    modalities: list[str] | None = None,
    extra_parameters: tuple[str, ...] = (),
    default_parameters: dict[str, Any] | None = None,
    reasoning_efforts: list[str] | None = None,
    context_length: int = 0,
    created: int = 0,
    is_default: bool = False,
) -> dict[str, Any]:
    modalities = modalities or ["text", "image"]
    value = {
        "id": model,
        "canonical_slug": model,
        "object": "model",
        "created": created,
        "owned_by": owned_by,
        "name": name,
        "architecture": {
            "modality": f"{'+'.join(modalities)}->text",
            "input_modalities": modalities,
            "output_modalities": ["text"],
        },
        "supported_parameters": [*PARAMETERS, *extra_parameters],
        "default_parameters": default_parameters,
        "per_request_limits": None,
        "is_default": is_default,
        "supported_reasoning_efforts": reasoning_efforts or list(EFFORTS),
    }
    if context_length > 0:
        value["context_length"] = context_length
    return value
