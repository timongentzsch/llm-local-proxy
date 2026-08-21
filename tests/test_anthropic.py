"""The Anthropic Messages dialect.

Wire claims here are checked against specs/anthropic-openapi.json and the
streaming prose; see specs/PINNED.md for what each source does and does not
cover.
"""

from __future__ import annotations

import unittest

from llm_local_proxy.dialects import ANTHROPIC, resolve
from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.anthropic.ingress import parse
from llm_local_proxy.errors import RequestError
from llm_local_proxy.http import security
from llm_local_proxy.ir import Image, Text, Thinking, ToolResult, ToolUse
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as build_claude
from llm_local_proxy.providers.reasoning import ReasoningCache

BASE = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}],
}


class IngressTest(unittest.TestCase):
    def test_minimal_body(self):
        request = parse(BASE)
        self.assertEqual(request.model, "claude-sonnet-5")
        self.assertEqual(request.max_tokens, 1024)
        self.assertEqual(request.turns[0].blocks, [Text("Hello")])

    def test_model_and_messages_are_required(self):
        with self.assertRaises(RequestError):
            parse({"max_tokens": 1, "messages": []})
        with self.assertRaises(RequestError):
            parse({"model": "m", "max_tokens": 1, "messages": []})

    def test_max_tokens_is_required(self):
        body = {k: v for k, v in BASE.items() if k != "max_tokens"}
        with self.assertRaises(RequestError):
            parse(body)

    def test_max_tokens_zero_is_legal(self):
        # Documented: zero pre-warms the prompt cache without generating.
        self.assertEqual(parse({**BASE, "max_tokens": 0}).max_tokens, 0)

    def test_negative_max_tokens_rejected(self):
        with self.assertRaises(RequestError):
            parse({**BASE, "max_tokens": -1})

    def test_system_accepts_string_or_blocks(self):
        self.assertEqual(parse({**BASE, "system": "Be terse."}).system, ["Be terse."])
        blocks = [{"type": "text", "text": "Be terse."}]
        self.assertEqual(parse({**BASE, "system": blocks}).system, ["Be terse."])

    def test_assistant_prefill_is_preserved(self):
        # A trailing assistant turn continues the response; it must survive.
        body = {
            **BASE,
            "messages": [
                {"role": "user", "content": "The Greek sun god is"},
                {"role": "assistant", "content": "The best answer is ("},
            ],
        }
        request = parse(body)
        self.assertEqual(request.turns[-1].role, "assistant")
        self.assertEqual(request.turns[-1].blocks, [Text("The best answer is (")])

    def test_tool_use_and_result_blocks(self):
        body = {
            **BASE,
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Berlin"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "17C",
                        }
                    ],
                },
            ],
        }
        request = parse(body)
        self.assertEqual(
            request.turns[1].blocks,
            [ToolUse("toolu_1", "get_weather", {"city": "Berlin"})],
        )
        self.assertEqual(request.turns[2].blocks, [ToolResult("toolu_1", "17C")])

    def test_signed_thinking_survives(self):
        body = {
            **BASE,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm", "signature": "SIG"}
                    ],
                },
            ],
        }
        self.assertEqual(parse(body).turns[1].blocks, [Thinking("hmm", "SIG")])

    def test_images(self):
        def block(source):
            return (
                parse(
                    {
                        **BASE,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "image", "source": source}],
                            }
                        ],
                    }
                )
                .turns[0]
                .blocks[0]
            )

        self.assertEqual(
            block({"type": "url", "url": "https://example.com/a.png"}),
            Image("https://example.com/a.png"),
        )
        self.assertEqual(
            block({"type": "base64", "media_type": "image/png", "data": "AAAA"}),
            Image("data:image/png;base64,AAAA"),
        )

    def test_tool_choice_maps_and_disables_parallel(self):
        request = parse(
            {**BASE, "tool_choice": {"type": "any", "disable_parallel_tool_use": True}}
        )
        self.assertEqual(request.tool_choice.kind, "required")
        self.assertIs(request.parallel_tool_calls, False)
        self.assertEqual(
            parse({**BASE, "tool_choice": {"type": "auto"}}).tool_choice.kind, "auto"
        )
        named = parse({**BASE, "tool_choice": {"type": "tool", "name": "f"}})
        self.assertEqual(
            (named.tool_choice.kind, named.tool_choice.name), ("tool", "f")
        )

    def test_web_search_accepted_other_server_tools_refused(self):
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        self.assertEqual(len(parse({**BASE, "tools": tools}).tools), 1)
        with self.assertRaises(RequestError):
            parse({**BASE, "tools": [{"type": "bash_20250124", "name": "bash"}]})

    def test_thinking_budget_is_carried(self):
        body = {**BASE, "thinking": {"type": "enabled", "budget_tokens": 4096}}
        self.assertEqual(parse(body).thinking_budget, 4096)

    def test_unsupported_top_level_parameters_are_refused(self):
        for name in ("container", "mcp_servers", "service_tier", "output_config"):
            with self.assertRaises(RequestError, msg=name):
                parse({**BASE, name: {"any": "value"}})

    def test_stop_sequences_and_sampling(self):
        request = parse({**BASE, "stop_sequences": ["END"], "top_k": 5})
        self.assertEqual(request.params["stop"], ["END"])
        self.assertEqual(request.params["top_k"], 5)


class RoundTripTest(unittest.TestCase):
    def test_signed_thinking_reaches_claude_verbatim(self):
        """The whole point of the Anthropic lane: signatures must survive."""
        body = {
            **BASE,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm", "signature": "SIG"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "f",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "x",
                        }
                    ],
                },
            ],
        }
        upstream, _ = build_claude(parse(body), "claude-sonnet-5")
        blocks = upstream["messages"][1]["content"]
        self.assertIn(
            {"type": "thinking", "thinking": "hmm", "signature": "SIG"}, blocks
        )

    def test_explicit_budget_beats_effort_tiers(self):
        body = {
            **BASE,
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
        upstream, _ = build_claude(parse(body), "claude-sonnet-5")
        self.assertEqual(
            upstream["thinking"], {"type": "enabled", "budget_tokens": 2000}
        )

    def test_budget_must_leave_room_to_answer(self):
        body = {**BASE, "thinking": {"type": "enabled", "budget_tokens": 2000}}
        with self.assertRaises(RequestError):
            build_claude(parse(body), "claude-sonnet-5")


class EgressTest(unittest.TestCase):
    def _stream(self, events):
        encoder = MessageEncoder("claude-sonnet-5", ClaudeDecoder(ReasoningCache()))
        frames = [encoder.start()]
        for event in events:
            frames.extend(encoder.feed(event))
        frames.extend(encoder.finish())
        return frames

    def test_stream_opens_and_closes_correctly(self):
        frames = self._stream(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
                {"type": "message_stop"},
            ]
        )
        self.assertEqual(frames[0]["type"], "message_start")
        self.assertEqual(frames[-1]["type"], "message_stop")
        self.assertEqual(frames[-2]["type"], "message_delta")
        self.assertEqual(frames[-2]["delta"]["stop_reason"], "end_turn")

    def test_message_start_carries_required_usage(self):
        # input_tokens is non-nullable even before anything is known.
        start = self._stream([])[0]
        self.assertEqual(start["message"]["usage"]["input_tokens"], 0)
        self.assertIsNone(start["message"]["stop_reason"])

    def test_blocks_stay_singly_open_with_rising_indices(self):
        frames = self._stream(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "t"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "x"},
                },
                {"type": "content_block_stop", "index": 1},
            ]
        )
        opens = [f["index"] for f in frames if f["type"] == "content_block_start"]
        stops = [f["index"] for f in frames if f["type"] == "content_block_stop"]
        self.assertEqual(opens, [0, 1])
        self.assertEqual(stops, [0, 1])

    def test_non_stream_message_has_every_required_field(self):
        encoder = MessageEncoder("claude-sonnet-5", ClaudeDecoder(ReasoningCache()))
        message = encoder.result()
        for field in (
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "stop_sequence",
            "stop_details",
            "usage",
            "container",
        ):
            self.assertIn(field, message)
        self.assertEqual(message["type"], "message")
        self.assertEqual(message["role"], "assistant")


class DialectTest(unittest.TestCase):
    def test_mounted_under_its_own_prefix(self):
        dialect, path = resolve("/anthropic/v1/messages")
        self.assertEqual(dialect.name, "anthropic")
        self.assertEqual(path, dialect.chat_route)

    def test_bare_paths_still_belong_to_chat_completions(self):
        self.assertEqual(resolve("/v1/models")[0].name, "openai")

    def test_authenticates_with_either_credential_header(self):
        self.assertTrue(security.authorized({"x-api-key": "k"}, "k"))
        self.assertTrue(security.authorized({"Authorization": "Bearer k"}, "k"))
        self.assertFalse(security.authorized({"x-api-key": "no"}, "k"))

    def test_error_envelope(self):
        error = ANTHROPIC.error(400, "bad")
        self.assertEqual(error["type"], "error")
        self.assertIn("request_id", error)
        self.assertEqual(error["error"]["type"], "invalid_request_error")
        self.assertEqual(
            ANTHROPIC.error(429, "slow")["error"]["type"], "rate_limit_error"
        )

    def test_stream_has_named_frames_and_no_done_sentinel(self):
        self.assertIsNone(ANTHROPIC.terminator)
        self.assertEqual(ANTHROPIC.event_name({"type": "message_stop"}), "message_stop")
        self.assertIn(b"event: ping", ANTHROPIC.keepalive)

    def test_catalog_shape(self):
        catalog = ANTHROPIC.catalog([{"id": "claude-sonnet-5", "name": "Sonnet"}])
        self.assertEqual(catalog["data"][0]["type"], "model")
        self.assertEqual(catalog["data"][0]["display_name"], "Sonnet")
        self.assertFalse(catalog["has_more"])
        self.assertEqual(catalog["first_id"], "claude-sonnet-5")


if __name__ == "__main__":
    unittest.main()
