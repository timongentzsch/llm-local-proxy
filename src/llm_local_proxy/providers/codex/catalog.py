"""The Codex model catalog."""

from __future__ import annotations

from typing import Any

from ..catalog import model_info as shared


def model_info(
    item: dict[str, Any], context_windows: dict[str, int] | None = None
) -> dict[str, Any] | None:
    model = item.get("model") or item.get("id")
    if not model:
        return None
    modalities = item.get("inputModalities")
    efforts = [
        effort.get("reasoningEffort")
        for effort in item.get("supportedReasoningEfforts", [])
        if isinstance(effort, dict) and effort.get("reasoningEffort")
    ]
    return shared(
        str(model),
        item.get("displayName") or str(model),
        "openai",
        modalities=(
            [str(m) for m in modalities]
            if isinstance(modalities, list) and modalities
            else None
        ),
        default_parameters=(
            {"reasoning_effort": item["defaultReasoningEffort"]}
            if item.get("defaultReasoningEffort")
            else None
        ),
        reasoning_efforts=efforts or [],
        context_length=(context_windows or {}).get(str(model), 0),
        is_default=bool(item.get("isDefault")),
    )
