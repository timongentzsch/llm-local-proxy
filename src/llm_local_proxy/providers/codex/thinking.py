"""Codex reasoning items carried through Anthropic thinking blocks.

Anthropic requires every thinking block to have an opaque signature and to be
returned byte-for-byte on the next turn. Codex instead returns one opaque
Responses reasoning item. This envelope uses Anthropic's signature slot to
carry that item without inventing an Anthropic signature or dropping Codex's
encrypted content.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

ENVELOPE_PREFIX = "llpv1-codex-reasoning:"


@dataclass(frozen=True)
class Unpacked:
    item: dict[str, Any]
    thinking: str


def pack(item: dict[str, Any], thinking: str) -> str:
    payload = json.dumps(
        {"item": item, "thinking": thinking}, separators=(",", ":"), sort_keys=True
    )
    return ENVELOPE_PREFIX + base64.urlsafe_b64encode(payload.encode()).decode()


def unpack(signature: Any) -> Unpacked | None:
    """Return a bridge item, None for a real Anthropic signature.

    A signature claiming our prefix but carrying a damaged payload is an
    error, not a foreign signature: silently accepting it would lose required
    Codex reasoning on the following tool-result turn.
    """
    if not isinstance(signature, str) or not signature.startswith(ENVELOPE_PREFIX):
        return None
    try:
        payload = base64.urlsafe_b64decode(signature[len(ENVELOPE_PREFIX) :].encode())
        envelope = json.loads(payload.decode())
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("malformed Codex reasoning signature") from exc
    item = envelope.get("item") if isinstance(envelope, dict) else None
    thinking = envelope.get("thinking") if isinstance(envelope, dict) else None
    if (
        not isinstance(item, dict)
        or item.get("type") != "reasoning"
        or not isinstance(item.get("encrypted_content"), str)
        or not item["encrypted_content"]
        or not isinstance(thinking, str)
    ):
        raise ValueError("malformed Codex reasoning signature")
    return Unpacked(item, thinking)
