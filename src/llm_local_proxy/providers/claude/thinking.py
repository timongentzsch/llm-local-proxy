"""Claude thinking blocks carried verbatim through a Responses reasoning item.

Claude signs the thinking blocks of an assistant turn and requires them back
exactly as it produced them: the block kind, its text and its signature are all
part of what it verifies. A Responses item has one opaque slot for that, so the
whole block is packed into `encrypted_content` and unpacked on replay.

Nothing is rebuilt from the readable `summary`, which a client may shorten or
drop: reconstructing a block from its signature is what upstream rejects as
"blocks must remain as they were in the original response".

Each envelope records where its block sat in the turn, so a client that
reorders them or drops one from the middle is caught here rather than
upstream. A dropped trailing block still leaves ordinals reading 0..n-1 --
the total is not known until the turn ends -- so the reasoning cache's count
catches that one instead.

Only a replayable block is packed [empirical]: the subscription edge signs
reasoning whose text it never streams, and a signature covering text that
never arrived cannot be sent back.

The version tag is part of the payload contract. Any change to the shape below
must bump it, so an older build's blobs are reported as an unreadable version
rather than misread. Clients keep append-only histories, so a build that
rejects what an earlier one wrote strands every later turn of those sessions.
"""

from __future__ import annotations

import base64
import enum
import json
from dataclasses import dataclass
from typing import Any

#: Version tag, so a later shape change is detectable instead of misread.
#: v1 carried the bare block; v2 wraps it with its ordinal.
ENVELOPE_PREFIX = "llpv2-claude-thinking:"

#: The two block kinds Claude signs, and the fields each one must carry.
BLOCK_FIELDS = {"thinking": ("thinking", "signature"), "redacted_thinking": ("data",)}


class Outcome(enum.Enum):
    OK = "ok"
    #: No envelope of ours: another upstream's blob, or a pre-envelope session.
    FOREIGN = "foreign"
    #: Ours by prefix, but a version this build cannot read.
    BAD_VERSION = "bad_version"
    #: Ours by prefix and version, and damaged.
    MALFORMED = "malformed"
    #: Ours and intact, carrying a block Claude will not accept back: it
    #: signed reasoning whose text it never streamed, and the signature
    #: covers that text rather than the empty string stored here.
    WITHHELD = "withheld"


@dataclass(frozen=True)
class Unpacked:
    outcome: Outcome
    block: dict[str, Any] | None = None
    #: Position of this block among the signed blocks of its assistant turn.
    ordinal: int = 0


def pack(block: dict[str, Any], ordinal: int) -> str:
    """The opaque `encrypted_content` carrying one Claude thinking block."""
    payload = json.dumps(
        {"n": ordinal, "block": block}, separators=(",", ":"), sort_keys=True
    )
    return ENVELOPE_PREFIX + base64.urlsafe_b64encode(payload.encode()).decode()


def _valid(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    required = BLOCK_FIELDS.get(block.get("type"))
    return required is not None and all(
        isinstance(block.get(field), str) for field in required
    )


def unpack(encrypted: Any) -> Unpacked:
    """Classify one `encrypted_content` blob and recover its block."""
    if not isinstance(encrypted, str) or not encrypted:
        return Unpacked(Outcome.FOREIGN)
    if not encrypted.startswith(ENVELOPE_PREFIX):
        # A bare `llp` prefix we do not know is a version we cannot honour; any
        # other blob was simply written by something else.
        head = encrypted.split(":", 1)[0]
        if head.startswith("llp") and head.endswith("-claude-thinking"):
            return Unpacked(Outcome.BAD_VERSION)
        return Unpacked(Outcome.FOREIGN)
    try:
        payload = base64.urlsafe_b64decode(encrypted[len(ENVELOPE_PREFIX) :].encode())
        envelope = json.loads(payload.decode())
    except ValueError:
        return Unpacked(Outcome.MALFORMED)
    if not isinstance(envelope, dict) or not _valid(envelope.get("block")):
        return Unpacked(Outcome.MALFORMED)
    ordinal = envelope.get("n")
    if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
        return Unpacked(Outcome.MALFORMED)
    return Unpacked(Outcome.OK, envelope["block"], ordinal)
