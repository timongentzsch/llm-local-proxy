"""Stateless OpenAI Responses downstream protocol contracts."""

from __future__ import annotations

import unittest

from llm_local_proxy.dialects.openai.responses_egress import ResponseEncoder
from llm_local_proxy.dialects.openai.responses_ingress import parse
from llm_local_proxy.errors import RequestError
from llm_local_proxy.ir import NativeItem
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as build_claude
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build as build_codex
from llm_local_proxy.providers.reasoning import ReasoningCache

REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Checked."}],
    "encrypted_content": "opaque-secret",
}


class ResponsesIngressTest(unittest.TestCase):
    def test_rejects_server_side_state(self):
        base = {"model": "gpt-test", "input": "hi"}
        with self.assertRaisesRegex(RequestError, "store"):
            parse({**base, "store": True})
        with self.assertRaisesRegex(RequestError, "previous_response_id"):
            parse({**base, "previous_response_id": "resp_old"})

    def test_codex_reasoning_and_tool_history_round_trip_verbatim(self):
        body = {
            "model": "gpt-test",
            "instructions": "Be concise.",
            "input": [
                {"type": "message", "role": "user", "content": "read it"},
                REASONING,
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "hello",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "strict": True,
                    "parameters": {"type": "object"},
                }
            ],
            "reasoning": {"effort": "high", "summary": "auto"},
            "store": False,
        }
        upstream, _ = build_codex(parse(body), ReasoningCache())
        self.assertEqual(upstream["input"][1], REASONING)
        self.assertEqual(upstream["input"][2]["call_id"], "call_1")
        self.assertEqual(upstream["input"][3]["type"], "function_call_output")
        self.assertFalse(upstream["store"])
        self.assertEqual(upstream["include"], ["reasoning.encrypted_content"])
        self.assertTrue(upstream["tools"][0]["strict"])

    def test_native_items_and_structured_outputs_pass_through_to_codex(self):
        custom_call = {
            "type": "custom_tool_call",
            "id": "ct_1",
            "call_id": "call_1",
            "name": "shell",
            "input": "pwd",
            "status": "completed",
        }
        custom_output = {
            "type": "custom_tool_call_output",
            "call_id": "call_1",
            "output": "project",
        }
        structured_output = {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": [{"type": "input_text", "text": "hello"}],
        }
        request = parse(
            {
                "model": "gpt-test",
                "input": [
                    {"type": "message", "role": "user", "content": "run it"},
                    custom_call,
                    custom_output,
                    structured_output,
                ],
                "tools": [
                    {
                        "type": "custom",
                        "name": "shell",
                        "format": {"type": "text"},
                    }
                ],
            }
        )
        upstream, _ = build_codex(request, ReasoningCache())
        self.assertEqual(
            upstream["input"][1:], [custom_call, custom_output, structured_output]
        )
        self.assertEqual(upstream["tools"][0]["format"], {"type": "text"})

    def test_claude_rejects_unrepresentable_native_items(self):
        request = parse(
            {
                "model": "claude-test",
                "input": [
                    {"type": "message", "role": "user", "content": "run it"},
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_1",
                        "name": "shell",
                        "input": "pwd",
                    },
                ],
            }
        )
        with self.assertRaisesRegex(RequestError, "custom_tool_call"):
            build_claude(request, "claude-test")

    def test_claude_recovers_signed_thinking_from_responses_history(self):
        request = parse(
            {
                "model": "claude-test",
                "input": [
                    {"type": "message", "role": "user", "content": "read it"},
                    REASONING,
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "hello",
                    },
                ],
                "reasoning": {"effort": "low"},
            }
        )
        cache = ReasoningCache()
        cache.put(
            ["call_1"],
            [
                {
                    "type": "thinking",
                    "thinking": "Checked.",
                    "signature": "opaque-secret",
                }
            ],
        )
        upstream, _ = build_claude(request, "claude-test", reasoning_cache=cache)
        self.assertEqual(upstream["output_config"], {"effort": "low"})
        assistant = upstream["messages"][1]["content"]
        self.assertEqual(
            assistant[0],
            {
                "type": "thinking",
                "thinking": "Checked.",
                "signature": "opaque-secret",
            },
        )
        self.assertEqual(assistant[1]["type"], "tool_use")
        self.assertEqual(
            len([block for block in assistant if block["type"] == "thinking"]), 1
        )


class ResponsesEgressTest(unittest.TestCase):
    def test_codex_stream_has_native_order_and_opaque_reasoning(self):
        encoder = ResponseEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        events = [encoder.start()]
        for upstream in (
            {"type": "response.reasoning_summary_text.delta", "delta": "Checked."},
            {"type": "response.output_item.done", "item": REASONING},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{}",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "output": [],
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            },
        ):
            events.extend(encoder.feed(upstream))
        events.extend(encoder.finish())
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[0], "response.created")
        self.assertLess(
            kinds.index("response.output_item.added"),
            kinds.index("response.reasoning_summary_text.delta"),
        )
        self.assertIn("response.function_call_arguments.done", kinds)
        self.assertEqual(kinds[-1], "response.completed")
        completed = events[-1]["response"]
        self.assertEqual(completed["output"][0]["encrypted_content"], "opaque-secret")
        self.assertEqual(completed["output"][1]["call_id"], "call_1")
        self.assertEqual(completed["parallel_tool_calls"], True)
        self.assertEqual(completed["tool_choice"], "auto")
        self.assertEqual(completed["tools"], [])
        self.assertEqual(
            completed["usage"]["input_tokens_details"]["cache_write_tokens"], 0
        )
        done = next(
            event
            for event in events
            if event["type"] == "response.function_call_arguments.done"
        )
        self.assertEqual(done["name"], "read_file")

    def test_empty_reasoning_summary_remains_exactly_empty(self):
        encoder = ResponseEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        encoder.feed(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_empty",
                    "summary": [],
                    "encrypted_content": "opaque",
                },
            }
        )
        self.assertEqual(encoder.result()["output"][0]["summary"], [])

    def test_native_output_item_and_stream_error_are_preserved(self):
        decoder = CodexDecoder(ReasoningCache())
        encoder = ResponseEncoder("gpt-test", decoder)
        native = {
            "type": "custom_tool_call",
            "id": "ct_1",
            "call_id": "call_1",
            "name": "shell",
            "input": "pwd",
            "status": "completed",
        }
        events = encoder.feed({"type": "response.output_item.done", "item": native})
        self.assertIsInstance(
            decoder.decode(
                {"type": "response.output_item.done", "item": {**native, "id": "ct_2"}}
            )[0],
            NativeItem,
        )
        self.assertEqual(events[-1]["item"], native)
        failure = encoder.error("boom")
        self.assertEqual(failure["type"], "error")
        self.assertEqual(failure["message"], "boom")
        self.assertIn("sequence_number", failure)

    def test_claude_signed_thinking_is_exposed_as_replayable_item(self):
        encoder = ResponseEncoder("claude-test", ClaudeDecoder())
        for event in (
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Checked."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "signature"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ):
            encoder.feed(event)
        result = encoder.result()
        self.assertEqual(result["output"][0]["summary"][0]["text"], "Checked.")
        self.assertEqual(result["output"][0]["encrypted_content"], "signature")


if __name__ == "__main__":
    unittest.main()
