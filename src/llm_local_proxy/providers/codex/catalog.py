"""The Codex model catalog, shaped for the OpenRouter-compatible listing."""

from __future__ import annotations

from typing import Any


def model_info(
    item: dict[str, Any], context_windows: dict[str, int] | None = None
) -> dict[str, Any] | None:
    model = item.get("model") or item.get("id")
    if not model:
        return None
    raw_modalities = item.get("inputModalities")
    modalities = (
        [str(modality) for modality in raw_modalities]
        if isinstance(raw_modalities, list) and raw_modalities
        else ["text", "image"]
    )
    efforts = [
        effort.get("reasoningEffort")
        for effort in item.get("supportedReasoningEfforts", [])
        if isinstance(effort, dict) and effort.get("reasoningEffort")
    ]
    value = {
        "id": model,
        "canonical_slug": model,
        "object": "model",
        "created": 0,
        "owned_by": "openai",
        "name": item.get("displayName") or model,
        "architecture": {
            "modality": f"{'+'.join(modalities)}->text",
            "input_modalities": modalities,
            "output_modalities": ["text"],
        },
        "supported_parameters": [
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning",
            "reasoning_effort",
            "web_search",
        ],
        "default_parameters": (
            {"reasoning_effort": item["defaultReasoningEffort"]}
            if item.get("defaultReasoningEffort")
            else None
        ),
        "per_request_limits": None,
        "is_default": bool(item.get("isDefault")),
        "supported_reasoning_efforts": efforts,
    }
    context = (context_windows or {}).get(str(model), 0)
    if context > 0:
        value["context_length"] = context
    return value
