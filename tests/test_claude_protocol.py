import unittest

from codex_local_proxy.claude_protocol import (
    ClaudeTranslator,
    build_messages_request,
    claude_model_name,
)
from codex_local_proxy.protocol import RequestError

BASE = {"model": "claude-opus-5", "messages": [{"role": "user", "content": "hi"}]}


class ClaudeRoutingTest(unittest.TestCase):
    def test_routes_only_claude_names(self):
        self.assertEqual(claude_model_name("claude-opus-5"), "claude-opus-5")
        self.assertEqual(
            claude_model_name("openrouter/claude-sonnet-5"), "claude-sonnet-5"
        )
        self.assertIsNone(claude_model_name("gpt-5.4"))
        self.assertIsNone(claude_model_name(None))


class BuildMessagesRequestTest(unittest.TestCase):
    def test_system_split_and_defaults(self):
        request, betas = build_messages_request(
            {
                "model": "claude-opus-5",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "hi"},
                ],
            },
            "claude-opus-5",
        )
        self.assertEqual(betas, [])
        self.assertEqual(request["system"], "Be brief.")
        self.assertEqual(request["messages"], [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(request["max_tokens"], 128000)
        self.assertTrue(request["stream"])
        self.assertEqual(request["model"], "claude-opus-5")

    def test_explicit_max_tokens_wins_over_live_default(self):
        request, _ = build_messages_request(BASE, "claude-opus-5", max_output=128000)
        self.assertEqual(request["max_tokens"], 128000)
        request, _ = build_messages_request(
            {**BASE, "max_tokens": 100}, "claude-opus-5", max_output=128000
        )
        self.assertEqual(request["max_tokens"], 100)

    def test_falls_back_to_static_table_without_live_default(self):
        request, _ = build_messages_request(BASE, "claude-opus-5")
        self.assertEqual(request["max_tokens"], 128000)

    def test_tools_and_web_search_beta(self):
        request, betas = build_messages_request(
            {
                "model": "claude-sonnet-5",
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
            "claude-sonnet-5",
        )
        self.assertIn("web-search-2025-03-05", betas)
        tools = request["tools"]
        self.assertEqual(tools[0]["name"], "get_weather")
        self.assertEqual(tools[0]["input_schema"]["type"], "object")
        self.assertEqual(tools[-1], {"type": "web_search_20250305", "name": "web_search"})
        self.assertEqual(request["tool_choice"], {"type": "any"})

    def test_tool_choice_none_drops_tools(self):
        request, _ = build_messages_request(
            {
                **BASE,
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
                "tool_choice": "none",
            },
            "claude-opus-5",
        )
        self.assertNotIn("tools", request)

    def test_reasoning_effort_maps_to_thinking(self):
        request, _ = build_messages_request(
            {**BASE, "reasoning_effort": "high", "max_tokens": 32768}, "claude-opus-5"
        )
        self.assertEqual(
            request["thinking"], {"type": "enabled", "budget_tokens": 16384}
        )
        request, _ = build_messages_request(
            {**BASE, "reasoning_effort": "high", "max_tokens": 4096}, "claude-opus-5"
        )
        self.assertEqual(
            request["thinking"], {"type": "enabled", "budget_tokens": 4095}
        )
        request, _ = build_messages_request(
            {**BASE, "reasoning_effort": "xhigh", "max_tokens": 65537},
            "claude-opus-5",
            thinking="adaptive",
        )
        self.assertEqual(request["thinking"], {"type": "adaptive"})

    def test_rejects_unsupported_parameters(self):
        with self.assertRaises(RequestError):
            build_messages_request({**BASE, "frequency_penalty": 0.2}, "claude-opus-5")
        with self.assertRaises(RequestError):
            build_messages_request(
                {"model": "claude-opus-5", "messages": []}, "claude-opus-5"
            )
        with self.assertRaises(RequestError):
            build_messages_request(
                {**BASE, "temperature": 1.5}, "claude-opus-5"
            )
        with self.assertRaises(RequestError):
            build_messages_request(
                {**BASE, "reasoning_effort": "ultra"}, "claude-opus-5"
            )

    def test_tool_roundtrip_blocks(self):
        request, _ = build_messages_request(
            {
                "model": "claude-sonnet-5",
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
            "claude-sonnet-5",
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
        translator = ClaudeTranslator("claude-opus-5")
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

    def test_tool_call_stream(self):
        translator = ClaudeTranslator("claude-opus-5")
        chunks = []
        for event in [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_9", "name": "get_weather"},
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
        call = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(call["id"], "toolu_9")
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(call["function"]["arguments"], '{"city":"Berlin"}')
        self.assertEqual(
            translator.result()["choices"][0]["finish_reason"], "tool_calls"
        )


if __name__ == "__main__":
    unittest.main()
