"""The Codex model catalog."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from ..catalog import model_info as shared


def model_info(
    item: dict[str, Any],
    context_windows: dict[str, int] | None = None,
    transport_efforts: Collection[str] | None = None,
) -> dict[str, Any] | None:
    model = item.get("model") or item.get("id")
    if not model:
        return None
    modalities = item.get("inputModalities")
    advertised = [
        effort.get("reasoningEffort")
        for effort in item.get("supportedReasoningEfforts", [])
        if isinstance(effort, dict) and effort.get("reasoningEffort")
    ]
    accepted = set(transport_efforts) if transport_efforts is not None else None
    efforts = [value for value in advertised if accepted is None or value in accepted]
    default_effort = item.get("defaultReasoningEffort")
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
            {"reasoning_effort": default_effort}
            if default_effort and (accepted is None or default_effort in accepted)
            else None
        ),
        reasoning_efforts=efforts or [],
        context_length=(context_windows or {}).get(str(model), 0),
        is_default=bool(item.get("isDefault")),
    )
