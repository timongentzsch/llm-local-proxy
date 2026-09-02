"""One executable contract for the four downstream/upstream pairings.

Each lane covers reasoning, a tool call followed by its result, and web-search
configuration. Streaming/output-shape and usage details remain pinned by the
dialect/provider tests and byte-level goldens.
"""

from __future__ import annotations

import unittest

from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.anthropic.ingress import parse as anthropic
from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.ingress import parse as chat
from llm_local_proxy.dialects.openai.responses_ingress import parse as responses
from llm_local_proxy.errors import RequestError
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as to_claude
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build as to_codex
from llm_local_proxy.providers.codex.thinking import ENVELOPE_PREFIX
from llm_local_proxy.providers.reasoning import ReasoningCache

CLAUDE_THINKING = {
    "type": "thinking",
    "thinking": "Checked.",
    "signature": "claude-signature",
}
CODEX_REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Checked."}],
    "encrypted_content": "codex-encrypted",
}
CALL = {"id": "call_1", "name": "read", "arguments": "{}"}


def _cache_claude_turn(cache: ReasoningCache) -> None:
    encoder = ChunkEncoder("claude-test", ClaudeDecoder(cache))
    events = [
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
            "delta": {
                "type": "signature_delta",
                "signature": "claude-signature",
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": CALL["id"],
                "name": CALL["name"],
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    ]
    for event in events:
        encoder.feed(event)
    encoder.result()


def _cache_codex_turn(cache: ReasoningCache) -> None:
    encoder = ChunkEncoder("gpt-test", CodexDecoder(cache))
    encoder.feed({"type": "response.output_item.done", "item": CODEX_REASONING})
    encoder.feed(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": CALL["id"], **CALL},
        }
    )
    encoder.result()


def _anthropic_history(thinking: dict) -> list[dict]:
    return [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "content": [
                thinking,
                {
                    "type": "tool_use",
                    "id": CALL["id"],
                    "name": CALL["name"],
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": CALL["id"],
                    "content": "ok",
                }
            ],
        },
    ]


def _chat_history() -> list[dict]:
    return [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": CALL["id"],
                    "type": "function",
                    "function": {"name": CALL["name"], "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": CALL["id"], "content": "ok"},
    ]


class ProtocolMatrixTest(unittest.TestCase):
    def test_openai_to_claude(self):
        cache = ReasoningCache()
        _cache_claude_turn(cache)
        request = chat(
            {
                "model": "claude-test",
                "messages": _chat_history(),
                "tools": [{"type": "openrouter:web_search"}],
                "reasoning": {"effort": "high", "summary": "auto"},
            }
        )
        upstream, betas = to_claude(
            request, "claude-test", max_output=32768, reasoning_cache=cache
        )
        self.assertEqual(upstream["messages"][1]["content"][0], CLAUDE_THINKING)
        self.assertEqual(upstream["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(upstream["output_config"], {"effort": "high"})
        self.assertEqual(upstream["thinking"]["display"], "summarized")
        self.assertEqual(upstream["tools"][0]["type"], "web_search_20250305")
        self.assertIn("web-search-2025-03-05", betas)

    def test_openai_to_codex(self):
        cache = ReasoningCache()
        _cache_codex_turn(cache)
        request = responses(
            {
                "model": "gpt-test",
                "input": [
                    {"type": "message", "role": "user", "content": "read"},
                    {
                        "type": "function_call",
                        "call_id": CALL["id"],
                        "name": CALL["name"],
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": CALL["id"],
                        "output": "ok",
                    },
                ],
                "tools": [{"type": "web_search", "search_context_size": "low"}],
                "reasoning": {"effort": "high", "summary": "auto"},
            }
        )
        upstream, _ = to_codex(request, cache)
        self.assertEqual(upstream["input"][1], CODEX_REASONING)
        self.assertEqual(upstream["input"][3]["type"], "function_call_output")
        self.assertEqual(upstream["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertEqual(
            upstream["tools"], [{"type": "web_search", "search_context_size": "low"}]
        )
        self.assertEqual(upstream["include"], ["reasoning.encrypted_content"])

    def test_anthropic_to_claude(self):
        request = anthropic(
            {
                "model": "claude-test",
                "max_tokens": 4096,
                "messages": _anthropic_history(CLAUDE_THINKING),
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "high"},
            }
        )
        upstream, _ = to_claude(request, "claude-test", max_output=32768)
        self.assertEqual(upstream["messages"][1]["content"][0], CLAUDE_THINKING)
        self.assertEqual(upstream["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(upstream["tools"][0]["max_uses"], 3)
        self.assertEqual(upstream["thinking"]["type"], "adaptive")
        self.assertEqual(upstream["output_config"], {"effort": "high"})

    def test_anthropic_to_codex(self):
        encoder = MessageEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        encoder.feed(
            {"type": "response.reasoning_summary_text.delta", "delta": "Checked."}
        )
        encoder.feed({"type": "response.output_item.done", "item": CODEX_REASONING})
        encoder.feed(
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": CALL["id"], **CALL},
            }
        )
        assistant = encoder.result()["content"]
        signature = assistant[0]["signature"]
        self.assertTrue(signature.startswith(ENVELOPE_PREFIX))

        request = anthropic(
            {
                "model": "gpt-test",
                "max_tokens": 4096,
                "messages": _anthropic_history(assistant[0]),
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "high"},
            }
        )
        upstream, _ = to_codex(request, ReasoningCache())
        self.assertEqual(upstream["input"][1], CODEX_REASONING)
        self.assertEqual(upstream["input"][3]["type"], "function_call_output")
        self.assertEqual(upstream["tools"], [{"type": "web_search"}])
        self.assertEqual(upstream["reasoning"], {"effort": "high", "summary": "auto"})

        lossy = (
            (
                {"thinking": {"type": "enabled", "budget_tokens": 2048}},
                "thinking budget",
            ),
            ({"thinking": {"type": "disabled"}}, "reasoning is disabled"),
            (
                {
                    "tools": [
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": 3,
                        }
                    ]
                },
                "web_search options",
            ),
        )
        base = {
            "model": "gpt-test",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
        }
        for fields, message in lossy:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(RequestError, message),
            ):
                to_codex(anthropic({**base, **fields}), ReasoningCache())

        for signature, message in (
            ("foreign-signature", "signed thinking"),
            (ENVELOPE_PREFIX + "damaged", "malformed"),
        ):
            with (
                self.subTest(signature=signature),
                self.assertRaisesRegex(RequestError, message),
            ):
                foreign = dict(CLAUDE_THINKING, signature=signature)
                to_codex(
                    anthropic({**base, "messages": _anthropic_history(foreign)}),
                    ReasoningCache(),
                )


if __name__ == "__main__":
    unittest.main()
