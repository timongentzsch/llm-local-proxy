"""The Claude provider: Messages requests out, Messages events back."""

import unittest

from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.ingress import parse
from llm_local_proxy.dialects.openai.responses_ingress import parse as parse_responses
from llm_local_proxy.errors import RequestError
from llm_local_proxy.ir import ToolCallArgs, ToolCallEnd
from llm_local_proxy.providers.catalog import match_model
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build
from llm_local_proxy.providers.reasoning import ReasoningCache


def claude_request(body, model, **kwargs):
    """The whole path a Chat Completions request takes to Claude."""
    kwargs.setdefault("max_output", 32768)
    return build(parse(body), model, **kwargs)


BASE = {"model": "claude-fake-1", "messages": [{"role": "user", "content": "hi"}]}


class ClaudeRoutingTest(unittest.TestCase):
    def test_routes_only_names_in_the_live_catalog(self):
        models = [{"id": "any-runtime-name"}]
        self.assertEqual(match_model("any-runtime-name", models), "any-runtime-name")
        self.assertEqual(
            match_model("provider/any-runtime-name", models), "any-runtime-name"
        )
        self.assertIsNone(match_model("claude-but-not-listed", models))
        self.assertIsNone(match_model(None, models))


class BuildMessagesRequestTest(unittest.TestCase):
    def test_explicit_max_tokens_wins_over_live_default(self):
        request, _ = claude_request(BASE, "claude-fake-1", max_output=128000)
        self.assertEqual(request["max_tokens"], 128000)
        request, _ = claude_request(
            {**BASE, "max_tokens": 100}, "claude-fake-1", max_output=128000
        )
        self.assertEqual(request["max_tokens"], 100)

    def test_zero_tokens_builds_a_prewarm_request(self):
        request, _ = claude_request(
            {**BASE, "max_tokens": 0, "reasoning_effort": "high"},
            "claude-fake-1",
            thinking="adaptive",
        )
        self.assertEqual(request["max_tokens"], 0)
        self.assertNotIn("thinking", request)
        self.assertNotIn("output_config", request)

    def test_requires_a_runtime_limit_when_the_client_omits_one(self):
        with self.assertRaisesRegex(RequestError, "catalog"):
            build(parse(BASE), "claude-fake-1")

    def test_tools_and_web_search_beta(self):
        request, betas = claude_request(
            {
                "model": "claude-fake-2",
                "messages": [{"role": "user", "content": "search it"}],
                "web_search_options": {},
                "tools": [
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
            tools[1], {"type": "web_search_20250305", "name": "web_search"}
        )
        self.assertEqual(request["tool_choice"], {"type": "any"})

    def test_tool_choice_none_drops_tools(self):
        request, _ = claude_request(
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

    def test_reasoning_effort_uses_claudes_native_output_config(self):
        request, _ = claude_request(
            {**BASE, "reasoning_effort": "high", "max_tokens": 32768},
            "claude-fake-1",
        )
        self.assertEqual(request["output_config"], {"effort": "high"})
        self.assertEqual(
            request["thinking"], {"type": "adaptive", "display": "summarized"}
        )
        # Effort and thinking remain independent native controls: effort is
        # forwarded as its tier, while thinking remains adaptively sized.
        request, _ = claude_request(
            {**BASE, "reasoning_effort": "xhigh", "max_tokens": 65537},
            "claude-fake-1",
            thinking="adaptive",
        )
        self.assertEqual(request["output_config"], {"effort": "xhigh"})
        self.assertEqual(
            request["thinking"], {"type": "adaptive", "display": "summarized"}
        )
        # Without an effort, an adaptive model still gets adaptive thinking.
        request, _ = claude_request(
            {**BASE, "max_tokens": 65537}, "claude-fake-1", thinking="adaptive"
        )
        self.assertEqual(
            request["thinking"], {"type": "adaptive", "display": "summarized"}
        )
        # A Responses request to hide the summary maps to Claude's native
        # omitted display mode.
        request, _ = build(
            parse_responses(
                {
                    "model": "claude-fake-1",
                    "input": "hi",
                    "max_output_tokens": 4096,
                    "reasoning": {"effort": "high", "summary": "none"},
                }
            ),
            "claude-fake-1",
            max_output=32768,
            thinking="adaptive",
        )
        self.assertEqual(
            request["thinking"], {"type": "adaptive", "display": "omitted"}
        )
        # Asking for a summary alone asks for reasoning to summarize.
        request, _ = build(
            parse_responses(
                {
                    "model": "claude-fake-1",
                    "input": "hi",
                    "max_output_tokens": 4096,
                    "reasoning": {"summary": "detailed"},
                }
            ),
            "claude-fake-1",
            max_output=32768,
        )
        self.assertEqual(
            request["thinking"], {"type": "adaptive", "display": "summarized"}
        )
        # No effort and no adaptive capability means no thinking at all.
        request, _ = claude_request({**BASE, "max_tokens": 4096}, "claude-fake-1")
        self.assertNotIn("thinking", request)

    def test_rejects_unsupported_parameters(self):
        with self.assertRaises(RequestError):
            claude_request({**BASE, "frequency_penalty": 0.2}, "claude-fake-1")
        with self.assertRaises(RequestError):
            claude_request({"model": "claude-fake-1", "messages": []}, "claude-fake-1")
        with self.assertRaises(RequestError):
            claude_request({**BASE, "temperature": 1.5}, "claude-fake-1")
        with self.assertRaises(RequestError):
            claude_request(
                {**BASE, "reasoning_effort": "ultra"},
                "claude-fake-1",
                reasoning_efforts=["low", "medium", "high"],
            )

    def test_unknown_catalog_efforts_are_forwarded_dynamically(self):
        request, _ = claude_request(
            {**BASE, "reasoning_effort": "future-tier"}, "claude-fake-1"
        )
        self.assertEqual(request["output_config"], {"effort": "future-tier"})


class ClaudeTranslatorTest(unittest.TestCase):
    def test_a_call_without_arguments_still_carries_an_empty_object(self):
        # Claude streams no input_json_delta for a parameterless tool; clients
        # would otherwise see an empty string where a JSON object belongs.
        decoder = ClaudeDecoder()
        decoder.decode(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "now"},
            }
        )
        closed = decoder.decode({"type": "content_block_stop", "index": 0})
        self.assertEqual(
            closed,
            [ToolCallArgs(0, "{}"), ToolCallEnd(0, "toolu_1", "now", "{}")],
        )

    def test_server_tool_use_and_result_bracket_the_search(self):
        """Claude's own server-tool blocks, which the decoder used to drop.

        The stream says `server_tool_use`; `web_search_20250305` is the tool
        *definition*'s name, so a decoder keyed only on the versioned spelling
        saw neither end of the search.
        """
        decoder = ClaudeDecoder()
        started = decoder.decode(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "python 3.14"},
                },
            }
        )
        self.assertEqual(
            [(e.tool, e.id, e.phase, e.query) for e in started],
            [("web_search", "srvtoolu_1", "searching", "python 3.14")],
        )
        done = decoder.decode(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [{"type": "web_search_result", "url": "https://x"}],
                },
            }
        )
        self.assertEqual([(e.id, e.phase) for e in done], [("srvtoolu_1", "completed")])
        # Counted from its request and again from its result: still one search.
        self.assertEqual(len(decoder.web_searches), 1)
        self.assertNotIn(
            "tool_use", [e.reason for e in decoder.finish() if hasattr(e, "reason")]
        )

    def test_a_search_error_result_is_a_failed_search(self):
        decoder = ClaudeDecoder()
        decoder.decode(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {},
                },
            }
        )
        failed = decoder.decode(
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
            }
        )
        self.assertEqual([(e.id, e.phase) for e in failed], [("srvtoolu_1", "failed")])

    def test_citation_without_url_is_ignored(self):
        translator = ChunkEncoder("claude-fake-1", ClaudeDecoder())
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


class ClaudeReasoningCaptureTest(unittest.TestCase):
    def test_streaming_finish_caches_signed_blocks(self):
        cache = ReasoningCache()
        translator = ChunkEncoder("claude-fake-1", ClaudeDecoder(reasoning_cache=cache))
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
        decoder = ClaudeDecoder(reasoning_cache=cache)
        translator = ChunkEncoder("claude-fake-1", decoder)
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
        self.assertEqual(decoder.reasoning_blocks, [])
        self.assertIsNone(decoder._open_thinking)
        self.assertEqual(cache.get(["toolu_2"]), [])


if __name__ == "__main__":
    unittest.main()
