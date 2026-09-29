"""The Anthropic Messages dialect.

Wire claims here are checked against specs/anthropic-openapi.json and the
streaming prose; see docs/specs.md for what each source does and does not
cover.
"""

from __future__ import annotations

import unittest

from llm_local_proxy.dialects import ANTHROPIC
from llm_local_proxy.dialects.anthropic.ingress import parse
from llm_local_proxy.errors import RequestError
from llm_local_proxy.ir import (
    Image,
    Text,
)

BASE = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}],
}


class IngressTest(unittest.TestCase):
    def test_model_and_messages_are_required(self):
        with self.assertRaises(RequestError):
            parse({"max_tokens": 1, "messages": []})
        with self.assertRaises(RequestError):
            parse({"model": "m", "max_tokens": 1, "messages": []})

    def test_max_tokens_zero_is_legal(self):
        # Documented: zero pre-warms the prompt cache without generating.
        self.assertEqual(parse({**BASE, "max_tokens": 0}).max_tokens, 0)

    def test_negative_max_tokens_rejected(self):
        with self.assertRaises(RequestError):
            parse({**BASE, "max_tokens": -1})

    def test_system_accepts_string_or_blocks(self):
        self.assertEqual(
            parse({**BASE, "system": "Be terse."}).system, [Text("Be terse.")]
        )
        blocks = [{"type": "text", "text": "Be terse."}]
        self.assertEqual(parse({**BASE, "system": blocks}).system, [Text("Be terse.")])

    def test_system_cache_breakpoints_survive(self):
        # Losing these would make every Claude Code turn re-pay full input.
        blocks = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        ]
        self.assertEqual(
            parse({**BASE, "system": blocks}).system,
            [Text("a"), Text("b", cache="")],
        )
        # A null TTL still places a breakpoint, at the default TTL.
        cached = {
            "type": "text",
            "text": "c",
            "cache_control": {"type": "ephemeral", "ttl": None},
        }
        self.assertEqual(
            parse({**BASE, "system": [cached]}).system, [Text("c", cache="")]
        )

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

    def test_system_role_messages_are_kept_in_place(self):
        # The Messages API allows a system role inside messages and real
        # Claude Code uses it; neither upstream has a third role.
        body = {
            **BASE,
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "system", "content": "b"},
            ],
        }
        request = parse(body)
        self.assertEqual([turn.role for turn in request.turns], ["user", "user"])
        self.assertEqual(request.turns[1].blocks, [Text("b")])

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
        body = {
            **BASE,
            "thinking": {
                "type": "enabled",
                "budget_tokens": 4096,
                "display": "omitted",
            },
        }
        request = parse(body)
        self.assertEqual(request.thinking_budget, 4096)
        self.assertEqual(request.thinking_display, "omitted")
        with self.assertRaisesRegex(RequestError, "thinking.display"):
            parse({**BASE, "thinking": {"type": "adaptive", "display": "raw"}})

    def test_unsupported_top_level_parameters_are_refused(self):
        for name in ("container", "mcp_servers", "service_tier"):
            with self.assertRaises(RequestError, msg=name):
                parse({**BASE, name: {"any": "value"}})

    def test_fields_the_proxy_cannot_act_on_are_accepted(self):
        # Rejecting these would break a real client for no benefit.
        body = {
            **BASE,
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "metadata": {"user_id": "someone"},
        }
        self.assertEqual(parse(body).model, "claude-sonnet-5")

    def test_stop_sequences_and_sampling(self):
        request = parse({**BASE, "stop_sequences": ["END"], "top_k": 5})
        self.assertEqual(request.params["stop"], ["END"])
        self.assertEqual(request.params["top_k"], 5)

    def test_output_config_options_we_cannot_honour_are_refused(self):
        with self.assertRaisesRegex(RequestError, "task_budget"):
            parse({**BASE, "output_config": {"task_budget": {"tokens": 10}}})
        with self.assertRaisesRegex(RequestError, "unsupported output format"):
            parse({**BASE, "output_config": {"format": {"type": "grammar"}}})
        with self.assertRaisesRegex(RequestError, "requires schema"):
            parse({**BASE, "output_config": {"format": {"type": "json_schema"}}})


class DialectTest(unittest.TestCase):
    def test_error_envelope(self):
        error = ANTHROPIC.error(400, "bad")
        self.assertEqual(error["type"], "error")
        self.assertIn("request_id", error)
        self.assertEqual(error["error"]["type"], "invalid_request_error")
        self.assertEqual(
            ANTHROPIC.error(429, "slow")["error"]["type"], "rate_limit_error"
        )

    def test_stream_has_named_frames_and_no_done_sentinel(self):
        self.assertTrue(ANTHROPIC.routes["/v1/messages"].named)
        self.assertIn(b"event: ping", ANTHROPIC.keepalive)

    def test_catalog_shape(self):
        catalog = ANTHROPIC.catalog([{"id": "claude-sonnet-5", "name": "Sonnet"}])
        self.assertEqual(catalog["data"][0]["type"], "model")
        self.assertEqual(catalog["data"][0]["display_name"], "Sonnet")
        self.assertFalse(catalog["has_more"])
        self.assertEqual(catalog["first_id"], "claude-sonnet-5")
        # A model with no known window omits the field rather than claiming 0.
        self.assertNotIn("max_input_tokens", catalog["data"][0])

    def test_catalog_carries_the_window(self):
        catalog = ANTHROPIC.catalog(
            [{"id": "claude-sonnet-5", "name": "Sonnet", "context_length": 1000000}]
        )
        self.assertEqual(catalog["data"][0]["max_input_tokens"], 1000000)


if __name__ == "__main__":
    unittest.main()
