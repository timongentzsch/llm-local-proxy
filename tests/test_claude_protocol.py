import unittest

from llm_local_proxy.claude_protocol import (
    CLAUDE_CODE_SYSTEM_MARKER,
    DEFAULT_MAX_OUTPUT_TOKENS,
    ClaudeTranslator,
    build_messages_request,
    claude_model_name,
)
from llm_local_proxy.protocol import ReasoningCache, RequestError

BASE = {"model": "claude-fake-1", "messages": [{"role": "user", "content": "hi"}]}


class ClaudeRoutingTest(unittest.TestCase):
    def test_routes_only_claude_names(self):
        self.assertEqual(claude_model_name("claude-fake-1"), "claude-fake-1")
        self.assertEqual(claude_model_name("openrouter/claude-fake-2"), "claude-fake-2")
        self.assertIsNone(claude_model_name("acme-gpt-1"))
        self.assertIsNone(claude_model_name(None))


class BuildMessagesRequestTest(unittest.TestCase):
    def test_system_split_and_defaults(self):
        request, betas = build_messages_request(
            {
                "model": "claude-fake-1",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hi"},
                ],
            },
            "claude-fake-1",
        )
        self.assertEqual(betas, [])
        self.assertEqual(
            request["system"],
            [
                {"type": "text", "text": CLAUDE_CODE_SYSTEM_MARKER},
                {"type": "text", "text": "Be brief."},
            ],
        )
        self.assertEqual(
            request["messages"],
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        )
        self.assertEqual(request["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS)
        self.assertTrue(request["stream"])
        self.assertEqual(request["model"], "claude-fake-1")
        self.assertEqual(request["cache_control"], {"type": "ephemeral"})

    def test_marker_system_always_present(self):
        # Without the Claude Code marker in `system` the subscription edge
        # 429s the call even with valid CLI headers and a fresh token.
        request, _ = build_messages_request(BASE, "claude-fake-1")
        self.assertEqual(
            request["system"], [{"type": "text", "text": CLAUDE_CODE_SYSTEM_MARKER}]
        )

    def test_explicit_max_tokens_wins_over_live_default(self):
        request, _ = build_messages_request(BASE, "claude-fake-1", max_output=128000)
        self.assertEqual(request["max_tokens"], 128000)
        request, _ = build_messages_request(
            {**BASE, "max_tokens": 100}, "claude-fake-1", max_output=128000
        )
        self.assertEqual(request["max_tokens"], 100)

    def test_falls_back_to_default_for_unknown_model(self):
        # A model absent from the static catalog uses the protocol default.
        request, _ = build_messages_request(BASE, "claude-fake-1")
        self.assertEqual(request["max_tokens"], DEFAULT_MAX_OUTPUT_TOKENS)

    def test_tools_and_web_search_beta(self):
        request, betas = build_messages_request(
            {
                "model": "claude-fake-2",
                "messages": [{"role": "user", "content": "search it"}],
                "tools": [
                    {"type": "openrouter:web_search"},
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                            },
                        },
                    },
                ],
                "tool_choice": "required",
            },
            "claude-fake-2",
        )
        self.assertIn("web-search-2025-03-05", betas)
        tools = request["tools"]
        self.assertEqual(tools[0]["name"], "get_weather")
        self.assertEqual(tools[0]["input_schema"]["type"], "object")
        self.assertEqual(
            tools[-1], {"type": "web_search_20250305", "name": "web_search"}
        )
        self.assertEqual(request["tool_choice"], {"type": "any"})

    def test_tool_choice_none_drops_tools(self):
        request, _ = build_messages_request(
            {
                **BASE,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "tool_choice": "none",
            },
            "claude-fake-1",
        )
        self.assertNotIn("tools", request)

    def test_reasoning_effort_maps_to_thinking(self):
        request, _ = build_messages_request(
            {**BASE, "reasoning_effort": "high", "max_tokens": 32768}, "claude-fake-1"
        )
        self.assertEqual(
            request["thinking"], {"type": "enabled", "budget_tokens": 16384}
        )
        request, _ = build_messages_request(
            {**BASE, "reasoning_effort": "high", "max_tokens": 4096}, "claude-fake-1"
        )
        self.assertEqual(
            request["thinking"], {"type": "enabled", "budget_tokens": 4095}
        )
        # An explicit effort outranks an adaptive-capable model, because
        # adaptive silently discards the requested tier.
        request, _ = build_messages_request(
            {**BASE, "reasoning_effort": "xhigh", "max_tokens": 65537},
            "claude-fake-1",
            thinking="adaptive",
        )
        self.assertEqual(
            request["thinking"], {"type": "enabled", "budget_tokens": 32768}
        )
        # Without an effort, an adaptive model still gets adaptive thinking.
        request, _ = build_messages_request(
            {**BASE, "max_tokens": 65537}, "claude-fake-1", thinking="adaptive"
        )
        self.assertEqual(request["thinking"], {"type": "adaptive"})
        # No effort and no adaptive capability means no thinking at all.
        request, _ = build_messages_request(
            {**BASE, "max_tokens": 4096}, "claude-fake-1"
        )
        self.assertNotIn("thinking", request)

    def test_rejects_unsupported_parameters(self):
        with self.assertRaises(RequestError):
            build_messages_request({**BASE, "frequency_penalty": 0.2}, "claude-fake-1")
        with self.assertRaises(RequestError):
            build_messages_request(
                {"model": "claude-fake-1", "messages": []}, "claude-fake-1"
            )
        with self.assertRaises(RequestError):
            build_messages_request({**BASE, "temperature": 1.5}, "claude-fake-1")
        with self.assertRaises(RequestError):
            build_messages_request(
                {**BASE, "reasoning_effort": "ultra"}, "claude-fake-1"
            )

    def test_tool_roundtrip_blocks(self):
        request, _ = build_messages_request(
            {
                "model": "claude-fake-2",
                "messages": [
                    {"role": "user", "content": "weather in berlin?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "toolu_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Berlin"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
                ],
            },
            "claude-fake-2",
        )
        self.assertEqual(
            request["messages"][1],
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
        )
        self.assertEqual(
            request["messages"][2],
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "sunny",
                    }
                ],
            },
        )


class ClaudeTranslatorTest(unittest.TestCase):
    def test_text_stream_and_result(self):
        translator = ClaudeTranslator("claude-fake-1")
        chunks = []
        for event in [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 7, "cache_read_input_tokens": 3}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello "},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "world"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 12},
            },
        ]:
            chunks.extend(translator.feed(event))
        chunks.extend(translator.finish())
        text = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks
            if chunk["choices"]
        )
        self.assertEqual(text, "hello world")
        result = translator.result()
        self.assertEqual(result["choices"][0]["message"]["content"], "hello world")
        self.assertEqual(result["choices"][0]["finish_reason"], "stop")
        usage = result["usage"]
        self.assertEqual(usage["prompt_tokens"], 10)
        self.assertEqual(usage["completion_tokens"], 12)
        self.assertEqual(usage["total_tokens"], 22)
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 3)

    def test_web_search_citation_becomes_url_citation_annotation(self):
        translator = ClaudeTranslator("claude-fake-1")
        chunks = []
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "citations_delta",
                    # Claude's own name for a web result, not OpenAI's.
                    "citation": {
                        "type": "web_search_result_location",
                        "url": "https://python.org/downloads/",
                        "title": "Downloads",
                        "cited_text": "Python 3.14",
                        "encrypted_index": "abc",
                    },
                },
            },
        ]:
            chunks.extend(translator.feed(event))
        annotations = [
            annotation
            for chunk in chunks
            for annotation in chunk["choices"][0]["delta"].get("annotations", [])
        ]
        self.assertEqual(
            annotations,
            [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://python.org/downloads/",
                        "title": "Downloads",
                    },
                }
            ],
        )
        message = translator.result()["choices"][0]["message"]
        self.assertEqual(message["annotations"], annotations)

    def test_citation_without_url_is_ignored(self):
        translator = ClaudeTranslator("claude-fake-1")
        chunks = translator.feed(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "citations_delta",
                    "citation": {"type": "char_location", "document_index": 0},
                },
            }
        )
        self.assertEqual(chunks, [])

    def test_tool_call_without_arguments_streams_empty_object(self):
        translator = ClaudeTranslator("claude-fake-1")
        chunks = []
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "list_files",
                },
            },
            {"type": "content_block_stop", "index": 0},
        ]:
            chunks.extend(translator.feed(event))
        arguments = "".join(
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get(
                "arguments", ""
            )
            for chunk in chunks
            if chunk["choices"] and chunk["choices"][0]["delta"].get("tool_calls")
        )
        self.assertEqual(arguments, "{}")
        call = translator.result()["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], "{}")

    def test_cache_creation_uses_openai_compatible_usage_field(self):
        translator = ClaudeTranslator("claude-fake-1")
        translator.feed(
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 7,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 5,
                    }
                },
            }
        )
        self.assertEqual(
            translator.usage["prompt_tokens_details"],
            {"cached_tokens": 3, "cache_write_tokens": 5},
        )

    def test_tool_call_stream(self):
        translator = ClaudeTranslator("claude-fake-1")
        chunks = []
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_9",
                    "name": "get_weather",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"Berlin"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 5},
            },
        ]:
            chunks.extend(translator.feed(event))
        chunks.extend(translator.finish())
        tool_chunks = [
            chunk
            for chunk in chunks
            if chunk["choices"] and chunk["choices"][0]["delta"].get("tool_calls")
        ]
        # Streams incrementally: an announce chunk (id/name, empty arguments),
        # then one argument-only chunk per input_json_delta.
        self.assertEqual(len(tool_chunks), 3)
        announce = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(announce["id"], "toolu_9")
        self.assertEqual(announce["function"]["name"], "get_weather")
        self.assertEqual(announce["function"]["arguments"], "")
        arguments = "".join(
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"].get(
                "arguments", ""
            )
            for chunk in tool_chunks
        )
        self.assertEqual(arguments, '{"city":"Berlin"}')
        self.assertEqual(
            translator.result()["choices"][0]["finish_reason"], "tool_calls"
        )


class ClaudeReasoningCaptureTest(unittest.TestCase):
    def test_captures_thinking_signature_and_replays_across_tool_use(self):
        cache = ReasoningCache()
        translator = ClaudeTranslator("claude-fake-1", reasoning_cache=cache)
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": "I should check the weather ",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "abc123"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Berlin"},
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 9},
            },
        ]:
            translator.feed(event)
        result = translator.result()
        self.assertEqual(
            result["choices"][0]["message"]["tool_calls"][0]["id"], "toolu_1"
        )
        # The signed thinking block is cached under the tool call id.
        self.assertEqual(
            cache.get(["toolu_1"]),
            [
                {
                    "type": "thinking",
                    "thinking": "I should check the weather ",
                    "signature": "abc123",
                }
            ],
        )

    def test_streaming_finish_caches_signed_blocks(self):
        cache = ReasoningCache()
        translator = ClaudeTranslator("claude-fake-1", reasoning_cache=cache)
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "ponder"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "SIG"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_9", "name": "f"},
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ]:
            translator.feed(event)
        translator.finish()
        self.assertEqual(
            cache.get(["toolu_9"]),
            [{"type": "thinking", "thinking": "ponder", "signature": "SIG"}],
        )

    def test_unsigned_thinking_block_is_not_cached_and_resets(self):
        cache = ReasoningCache()
        translator = ClaudeTranslator("claude-fake-1", reasoning_cache=cache)
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "truncated"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_2", "name": "f"},
            },
            {"type": "content_block_stop", "index": 1},
        ]:
            translator.feed(event)
        translator.result()
        self.assertEqual(translator.reasoning_blocks, [])
        self.assertIsNone(translator._open_thinking)
        self.assertEqual(cache.get(["toolu_2"]), [])

    def test_thinking_without_tool_calls_is_not_cached(self):
        cache = ReasoningCache()
        translator = ClaudeTranslator("claude-fake-1", reasoning_cache=cache)
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "SIG"},
            },
            {"type": "content_block_stop", "index": 0},
        ]:
            translator.feed(event)
        translator.result()
        self.assertEqual(translator.reasoning_blocks[0]["signature"], "SIG")
        self.assertEqual(cache.get(["toolu_absent"]), [])

    def test_replay_without_cached_blocks_sends_plain_tool_use(self):
        request, _ = build_messages_request(
            {
                "model": "claude-fake-2",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "toolu_missing",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "toolu_missing",
                        "content": "sunny",
                    },
                ],
            },
            "claude-fake-2",
            reasoning_cache=ReasoningCache(),
        )
        self.assertEqual(
            [block["type"] for block in request["messages"][1]["content"]], ["tool_use"]
        )

    def test_replay_prepends_thinking_in_followup_request(self):
        cache = ReasoningCache()
        cache.put(
            ["toolu_1"],
            [{"type": "thinking", "thinking": "think", "signature": "SIG"}],
        )
        request, _ = build_messages_request(
            {
                "model": "claude-fake-2",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "toolu_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Berlin"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
                ],
            },
            "claude-fake-2",
            reasoning_cache=cache,
        )
        # The signed thinking block must come before the tool_use block.
        self.assertEqual(
            request["messages"][1]["content"][0],
            {"type": "thinking", "thinking": "think", "signature": "SIG"},
        )
        self.assertEqual(request["messages"][1]["content"][1]["type"], "tool_use")


if __name__ == "__main__":
    unittest.main()
