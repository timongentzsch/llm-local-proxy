"""The Codex provider: Responses requests out, Responses events back."""

import unittest

from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.ingress import parse
from llm_local_proxy.errors import RequestError
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build
from llm_local_proxy.providers.reasoning import ReasoningCache


def codex_request(body, cache, session=""):
    """The whole path a Chat Completions request takes to Codex.

    Which layer rejects a bad body — the dialect's ingress or the provider's
    renderer — is an implementation detail these tests should not pin.
    """
    return build(parse(body, session), cache)


class ProtocolTest(unittest.TestCase):
    def test_rejects_invalid_tools_and_multiple_choices(self):
        base = {"model": "acme-gpt-1", "messages": [{"role": "user", "content": "hi"}]}
        with self.assertRaises(RequestError):
            codex_request({**base, "tools": [None]}, ReasoningCache())
        with self.assertRaises(RequestError):
            codex_request({**base, "n": 2}, ReasoningCache())
        with self.assertRaises(RequestError):
            codex_request({**base, "temperature": 0}, ReasoningCache())
        codex_request(
            {
                **base,
                "temperature": 1,
                "top_p": 1,
                "stop": None,
                "logprobs": False,
                "response_format": {"type": "text"},
            },
            ReasoningCache(),
        )

    def test_does_not_inject_default_instructions(self):
        request, _ = codex_request(
            {"model": "acme-gpt-1", "messages": [{"role": "user", "content": "hi"}]},
            ReasoningCache(),
        )
        self.assertEqual(request["instructions"], "")

    def test_accepts_max_tokens_as_compatibility_hint(self):
        request, _ = codex_request(
            {
                "model": "acme-gpt-1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 200,
            },
            ReasoningCache(),
        )
        self.assertNotIn("max_output_tokens", request)

    def test_chat_tools_become_responses_items(self):
        cache = ReasoningCache()
        cache.put(["call_1"], [{"type": "reasoning", "encrypted_content": "secret"}])
        request, session = codex_request(
            {
                "model": "acme-gpt-1",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Read a file."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"a"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "hello",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "reasoning_effort": "medium",
            },
            cache,
        )
        self.assertTrue(session.startswith("proxy-"))
        self.assertEqual(request["instructions"], "Be concise.")
        self.assertEqual(request["input"][1]["type"], "reasoning")
        self.assertEqual(request["input"][2]["type"], "function_call")
        self.assertEqual(request["input"][3]["type"], "function_call_output")
        self.assertEqual(request["tools"][0]["name"], "read_file")

    def test_openrouter_web_search_becomes_responses_tool(self):
        request, _ = codex_request(
            {
                "model": "acme-gpt-1",
                "messages": [{"role": "user", "content": "search"}],
                "tools": [
                    {
                        "type": "openrouter:web_search",
                        "parameters": {
                            "engine": "auto",
                            "search_context_size": "low",
                        },
                    }
                ],
            },
            ReasoningCache(),
        )
        self.assertEqual(
            request["tools"], [{"type": "web_search", "search_context_size": "low"}]
        )

    def test_response_tool_call_becomes_chat_chunk(self):
        cache = ReasoningCache()
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(cache))
        translator.feed(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "encrypted",
                    "summary": [],
                },
            }
        )
        chunks = translator.feed(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                },
            }
        )
        call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(
            translator.finish()[0]["choices"][0]["finish_reason"], "tool_calls"
        )
        self.assertEqual(cache.get(["call_1"])[0]["encrypted_content"], "encrypted")

    def test_usage_is_mapped(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 6,
                        "total_tokens": 16,
                        "input_tokens_details": {"cached_tokens": 4},
                        "output_tokens_details": {"reasoning_tokens": 2},
                    }
                },
            }
        )
        self.assertEqual(translator.result()["usage"]["prompt_tokens"], 10)
        self.assertEqual(
            translator.result()["usage"]["completion_tokens_details"][
                "reasoning_tokens"
            ],
            2,
        )

    def test_non_stream_result_keeps_reasoning(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "Checked.",
            }
        )
        self.assertEqual(
            translator.result()["choices"][0]["message"]["reasoning_content"],
            "Checked.",
        )

    def test_web_search_citation_and_usage_are_mapped(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.output_item.added",
                "item": {"type": "web_search_call", "id": "ws_1"},
            }
        )
        chunks = translator.feed(
            {
                "type": "response.output_text.annotation.added",
                "annotation": {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Example",
                    "start_index": 0,
                    "end_index": 7,
                },
            }
        )
        citation = chunks[0]["choices"][0]["delta"]["annotations"][0]
        self.assertEqual(citation["url_citation"]["url"], "https://example.com")
        translator.feed(
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 2, "output_tokens": 1}},
            }
        )
        result = translator.result()
        self.assertEqual(result["usage"]["server_tool_use"]["web_search_requests"], 1)
        self.assertEqual(result["choices"][0]["message"]["annotations"], [citation])

    def test_completed_output_can_supply_web_search_count(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.completed",
                "response": {
                    "output": [{"type": "web_search_call", "id": "ws_1"}],
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            }
        )
        self.assertEqual(
            translator.result()["usage"]["server_tool_use"]["web_search_requests"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
