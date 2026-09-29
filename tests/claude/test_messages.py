"""Anthropic Messages clients served by Claude: requests reach it as sent."""

from __future__ import annotations

import contextlib
import io
import unittest

from claude import claude_events
from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.anthropic.ingress import parse
from llm_local_proxy.errors import RequestError
from llm_local_proxy.ir import (
    HostedSearch,
)
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as build_claude
from llm_local_proxy.providers.reasoning import ReasoningCache

BASE = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}],
}


class IngressTest(unittest.TestCase):
    def test_a_client_cannot_hand_back_thinking_the_upstream_withheld(self):
        # An Anthropic client resends the blocks it was given, and the
        # subscription edge signs some whose text it never streamed. Those
        # arrive here as native Thinking rather than as an envelope, and
        # forwarding one is the modification upstream refuses.
        def turn(block):
            return parse(
                {
                    **BASE,
                    "messages": [
                        {"role": "user", "content": "go"},
                        {
                            "role": "assistant",
                            "content": [
                                block,
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
                                    "content": "ok",
                                }
                            ],
                        },
                    ],
                }
            )

        def kinds(request):
            with contextlib.redirect_stderr(io.StringIO()):
                body, _ = build_claude(request, "claude-test")
            return [block["type"] for block in body["messages"][1]["content"]]

        signed = {"type": "thinking", "thinking": "reasoned", "signature": "S"}
        withheld = {"type": "thinking", "thinking": "", "signature": "S"}
        unsigned = {"type": "thinking", "thinking": "reasoned", "signature": ""}
        redacted = {"type": "redacted_thinking", "data": "OPAQUE"}
        self.assertEqual(kinds(turn(signed)), ["thinking", "tool_use"])
        self.assertEqual(kinds(turn(withheld)), ["tool_use"])
        self.assertEqual(kinds(turn(unsigned)), ["tool_use"])
        self.assertEqual(kinds(turn(redacted)), ["redacted_thinking", "tool_use"])

    def test_effort_and_schema_share_one_output_config(self):
        request = parse(
            {
                **BASE,
                "output_config": {
                    "effort": "high",
                    "format": {"type": "json_schema", "schema": {"type": "object"}},
                },
            }
        )
        upstream, _ = build_claude(
            request,
            "claude-sonnet-5",
            max_output=32768,
            reasoning_efforts=["high"],
            reasoning_cache=ReasoningCache(),
        )
        self.assertEqual(upstream["output_config"]["effort"], "high")
        self.assertEqual(
            upstream["output_config"]["format"],
            {"type": "json_schema", "schema": {"type": "object"}},
        )


class RoundTripTest(unittest.TestCase):
    def test_hosted_search_pause_turn_reaches_claude_verbatim(self):
        search = {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "current model"},
        }
        result = {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com/result",
                    "title": "Result",
                    "encrypted_content": "opaque-index",
                }
            ],
        }
        body = {
            **BASE,
            "messages": [
                {"role": "user", "content": "find it"},
                {"role": "assistant", "content": [search, result]},
                {"role": "user", "content": "continue"},
            ],
        }

        request = parse(body)
        self.assertEqual(
            request.turns[1].blocks,
            [HostedSearch(search, "anthropic"), HostedSearch(result, "anthropic")],
        )
        upstream, _ = build_claude(request, "claude-sonnet-5")
        self.assertEqual(upstream["messages"][1]["content"], [search, result])

    def test_adaptive_thinking_is_forwarded_not_converted(self):
        body = {
            **BASE,
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {"effort": "high"},
        }
        upstream, _ = build_claude(parse(body), "claude-sonnet-5")
        self.assertEqual(
            upstream["thinking"], {"type": "adaptive", "display": "omitted"}
        )

    def test_explicit_budget_beats_effort_tiers(self):
        body = {
            **BASE,
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
        upstream, _ = build_claude(parse(body), "claude-sonnet-5")
        self.assertEqual(
            upstream["thinking"],
            {
                "type": "enabled",
                "budget_tokens": 2000,
                "display": "summarized",
            },
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

    def test_pause_turn_survives_an_incomplete_hosted_search(self):
        frames = self._stream(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "current model"},
                    },
                },
                claude_events.stop("pause_turn", {"output_tokens": 1}),
            ]
        )
        blocks = [
            frame["content_block"]
            for frame in frames
            if frame["type"] == "content_block_start"
        ]
        self.assertEqual(blocks, [])
        self.assertEqual(frames[-2]["delta"]["stop_reason"], "pause_turn")

    def test_hosted_search_stays_a_server_tool_and_never_a_tool_use(self):
        """A search the provider ran must not read as a call the client owes.

        A `tool_use` block plus a `tool_use` stop reason would send an
        Anthropic client into a tool round for work already done upstream.
        """
        frames = self._stream(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {"query": "python 3.14"},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com/result",
                                "encrypted_content": "opaque-index",
                            }
                        ],
                    },
                },
            ]
        )
        blocks = [
            f["content_block"] for f in frames if f["type"] == "content_block_start"
        ]
        self.assertEqual(
            [block["type"] for block in blocks],
            ["server_tool_use", "web_search_tool_result"],
        )
        self.assertEqual(blocks[0]["input"], {"query": "python 3.14"})
        self.assertEqual(blocks[1]["tool_use_id"], "srvtoolu_1")
        self.assertEqual(
            blocks[1]["content"],
            [
                {
                    "type": "web_search_result",
                    "url": "https://example.com/result",
                    "encrypted_content": "opaque-index",
                }
            ],
        )
        self.assertEqual(frames[-2]["delta"]["stop_reason"], "end_turn")
        # One block open at a time, under monotonic indices.
        opened = [f["index"] for f in frames if f["type"] == "content_block_start"]
        closed = [f["index"] for f in frames if f["type"] == "content_block_stop"]
        self.assertEqual(opened, [0, 1])
        self.assertEqual(closed, [0, 1])

    def test_a_failed_search_emits_the_required_matching_result(self):
        frames = self._stream(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "unavailable",
                        },
                    },
                },
            ]
        )
        blocks = [
            f["content_block"] for f in frames if f["type"] == "content_block_start"
        ]
        self.assertEqual(
            [block["type"] for block in blocks],
            ["server_tool_use", "web_search_tool_result"],
        )
        self.assertEqual(blocks[1]["tool_use_id"], "srvtoolu_1")
        self.assertEqual(
            blocks[1]["content"],
            {
                "type": "web_search_tool_result_error",
                "error_code": "unavailable",
            },
        )
        self.assertEqual(frames[-2]["delta"]["stop_reason"], "end_turn")


if __name__ == "__main__":
    unittest.main()
