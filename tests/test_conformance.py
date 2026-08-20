"""The pinned specifications as an executable oracle.

These tests read specs/ directly, so a spec refresh that changes the wire
contract fails here rather than in a user's client. See specs/PINNED.md for
what each file is and what it does not cover — notably that Anthropic's SSE
framing is documented in prose only and is pinned in tests/test_anthropic.py
instead.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from llm_local_proxy.dialects.anthropic import ERROR_TYPES
from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.anthropic.ingress import CHOICES, parse
from llm_local_proxy.dialects.openai.egress import FINISH_REASONS
from llm_local_proxy.protocol import ReasoningCache, RequestError
from llm_local_proxy.providers.claude.events import ClaudeDecoder

SPEC = Path(__file__).resolve().parents[1] / "specs" / "anthropic-openapi.json"


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

    def test_every_documented_stop_reason_is_handled(self):
        documented = set(self.schemas["StopReason"]["enum"])
        self.assertEqual(documented - set(FINISH_REASONS), set())

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

    def test_request_required_fields_are_enforced(self):
        body = json.loads(SPEC.read_text())["paths"]["/v1/messages"]["post"]
        schema = body["requestBody"]["content"]["application/json"]["schema"]
        if "$ref" in schema:
            schema = self.schemas[schema["$ref"].split("/")[-1]]
        self.assertEqual(set(schema["required"]), {"model", "messages", "max_tokens"})
        complete = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
        }
        for name in schema["required"]:
            partial = {k: v for k, v in complete.items() if k != name}
            with self.assertRaises(RequestError, msg=name):
                parse(partial)

    def test_tool_choice_variants_are_all_accepted(self):
        documented = set(_members(self.schemas["ToolChoice"], self.schemas))
        self.assertEqual(documented, set(CHOICES))

    def test_error_types_are_documented(self):
        documented = set(self.schemas["ErrorType"]["enum"])
        self.assertEqual(set(ERROR_TYPES.values()) - documented, set())

    def test_usage_fields_we_emit_exist_in_the_schema(self):
        documented = set(self.schemas["Usage"]["properties"])
        encoder = MessageEncoder("m", ClaudeDecoder(ReasoningCache()))
        self.assertEqual(set(encoder.result()["usage"]) - documented, set())


if __name__ == "__main__":
    unittest.main()
