"""Claude's decoded stream against the pinned Anthropic spec (docs/specs.md)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.reasoning import ReasoningCache

SPEC = Path(__file__).resolve().parents[2] / "specs" / "anthropic-openapi.json"


def _schemas():
    return json.loads(SPEC.read_text())["components"]["schemas"]


def _members(schema, schemas):
    """Const `type` value of each member of a oneOf/anyOf union."""
    names = []
    for member in schema.get("oneOf") or schema.get("anyOf") or []:
        if "$ref" in member:
            member = schemas[member["$ref"].split("/")[-1]]
        kind = (member.get("properties") or {}).get("type") or {}
        if kind.get("const"):
            names.append(kind["const"])
    return names


class SpecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SPEC.exists():  # pragma: no cover - only when specs are absent
            raise unittest.SkipTest("run scripts/refresh-specs.sh")
        cls.schemas = _schemas()

    def test_stream_event_union_is_covered(self):
        documented = set(_members(self.schemas["MessageStreamEvent"], self.schemas))
        encoder = MessageEncoder("m", ClaudeDecoder(ReasoningCache()))
        emitted = {encoder.start()["type"]}
        emitted.update(frame["type"] for frame in encoder.finish())
        emitted.update(
            {"content_block_start", "content_block_delta", "content_block_stop"}
        )
        self.assertEqual(documented, emitted)

    def test_message_carries_every_required_field(self):
        required = set(self.schemas["Message"]["required"])
        encoder = MessageEncoder("m", ClaudeDecoder(ReasoningCache()))
        self.assertEqual(required - set(encoder.result()), set())

    def test_usage_fields_we_emit_exist_in_the_schema(self):
        documented = set(self.schemas["Usage"]["properties"])
        encoder = MessageEncoder("m", ClaudeDecoder(ReasoningCache()))
        self.assertEqual(set(encoder.result()["usage"]) - documented, set())


if __name__ == "__main__":
    unittest.main()
