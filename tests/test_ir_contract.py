"""Regressions for information lost while crossing the shared IR."""

import unittest

from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.anthropic.ingress import parse as messages
from llm_local_proxy.dialects.openai.egress import FINISH_REASONS, ChunkEncoder
from llm_local_proxy.dialects.openai.ingress import parse as chat
from llm_local_proxy.dialects.openai.responses_egress import ResponseEncoder
from llm_local_proxy.errors import ProviderError, RequestError
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as claude
from llm_local_proxy.providers.claude.subscription import CLAUDE_CODE_SYSTEM_MARKER
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build as codex
from llm_local_proxy.providers.reasoning import ReasoningCache

BASE = {
    "model": "test",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "hi"}],
}
CACHE = {"type": "ephemeral", "ttl": "1h"}


class IRContractTest(unittest.TestCase):
    def test_stop_reasons_survive_streaming_and_buffering(self):
        for reason, narrowed in FINISH_REASONS.items():
            for encoder_type in (MessageEncoder, ChunkEncoder, ResponseEncoder):
                for streaming in (False, True):
                    with self.subTest(
                        reason=reason, encoder=encoder_type, stream=streaming
                    ):
                        encoder = encoder_type("test", ClaudeDecoder())
                        sequence = "END" if reason == "stop_sequence" else None
                        encoder.feed(
                            {
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": reason,
                                    "stop_sequence": sequence,
                                },
                            }
                        )
                        if streaming:
                            frames = encoder.finish()
                            result = frames[-1].get(
                                "response",
                                frames[-2]
                                if encoder_type is MessageEncoder
                                else frames[-1],
                            )
                        else:
                            result = encoder.result()
                        if encoder_type is MessageEncoder:
                            fields = result["delta"] if streaming else result
                            self.assertEqual(fields["stop_reason"], reason)
                            self.assertEqual(fields["stop_sequence"], sequence)
                        elif encoder_type is ChunkEncoder:
                            self.assertEqual(
                                result["choices"][0]["finish_reason"], narrowed
                            )
                        else:
                            expected = (
                                "incomplete"
                                if reason
                                in {"max_tokens", "model_context_window_exceeded"}
                                else "completed"
                            )
                            self.assertEqual(result["status"], expected)

    def test_citation_is_present_once_in_stream_and_buffer(self):
        encoder = MessageEncoder("test", ClaudeDecoder())
        encoder.feed(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "source"},
            }
        )
        citation = {
            "type": "web_search_result_location",
            "url": "https://example.com",
            "title": "Source",
        }
        event = {
            "type": "content_block_delta",
            "delta": {"type": "citations_delta", "citation": citation},
        }
        self.assertEqual(len(encoder.feed(event)), 1)
        self.assertEqual(encoder.feed(event), [])
        self.assertEqual(encoder.result()["content"][0]["citations"], [citation])

    def test_cache_ttl_survives_system_and_message_text(self):
        text = {"type": "text", "text": "cached", "cache_control": CACHE}
        request = messages(
            {
                **BASE,
                "system": [text],
                "messages": [{"role": "user", "content": [text]}],
            }
        )
        body, _ = claude(request, "test")
        self.assertEqual(body["system"][1], text)
        self.assertEqual(body["messages"][0]["content"], [text])

    def test_native_content_options_survive_or_are_rejected(self):
        for block in (
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image.png"},
                "cache_control": CACHE,
            },
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "done",
                "cache_control": CACHE,
            },
            {
                "type": "text",
                "text": "source",
                "citations": [{"url": "https://example.com"}],
            },
        ):
            with self.subTest(block=block):
                request = messages(
                    {**BASE, "messages": [{"role": "user", "content": [block]}]}
                )
                body, _ = claude(request, "test")
                self.assertEqual(body["messages"][0]["content"], [block])
                with self.assertRaises(RequestError):
                    codex(request, ReasoningCache())

    def test_subscription_marker_is_first_and_unique(self):
        marker = {
            "type": "text",
            "text": CLAUDE_CODE_SYSTEM_MARKER,
            "cache_control": CACHE,
        }
        body, _ = claude(
            messages(
                {**BASE, "system": [{"type": "text", "text": "header"}, marker, marker]}
            ),
            "test",
        )
        self.assertEqual(body["system"], [marker, {"type": "text", "text": "header"}])

    def test_codex_rejects_unsupported_sampling_controls(self):
        for name in ("frequency_penalty", "presence_penalty", "logit_bias", "top_k"):
            with self.subTest(name=name), self.assertRaisesRegex(RequestError, name):
                codex(chat({**BASE, name: 1}), ReasoningCache())

    def test_invalid_chat_content_is_not_silently_dropped(self):
        for role in ("system", "developer", "tool", "user", "assistant"):
            for content in ([42], [{"type": "unknown", "text": "lost"}]):
                with (
                    self.subTest(role=role, content=content),
                    self.assertRaises(RequestError),
                ):
                    chat(
                        {
                            **BASE,
                            "messages": [
                                {
                                    "role": role,
                                    "content": content,
                                    "tool_call_id": "call_1",
                                }
                            ],
                        }
                    )

    def test_output_strictness_requires_a_boolean(self):
        with self.assertRaises(RequestError):
            chat(
                {
                    **BASE,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "result",
                            "schema": {},
                            "strict": "false",
                        },
                    },
                }
            )

    def test_native_response_output_requires_a_lossless_endpoint(self):
        event = {
            "type": "response.output_item.done",
            "item": {
                "type": "custom_tool_call",
                "id": "ct_1",
                "call_id": "call_1",
                "name": "read",
                "input": "x",
            },
        }
        for encoder_type in (MessageEncoder, ChunkEncoder):
            encoder = encoder_type("test", CodexDecoder(ReasoningCache()))
            with self.subTest(encoder=encoder_type), self.assertRaises(ProviderError):
                encoder.feed(event)
        encoder = ResponseEncoder("test", CodexDecoder(ReasoningCache()))
        encoder.feed(event)
        self.assertEqual(encoder.result()["output"], [event["item"]])

    def test_malformed_anthropic_thinking_is_rejected(self):
        for value in (
            True,
            "enabled",
            {"type": "unknown"},
            {"type": "enabled"},
            {"type": "enabled", "budget_tokens": True},
            {"type": "enabled", "budget_tokens": "1024"},
        ):
            with self.subTest(value=value), self.assertRaises(RequestError):
                messages({**BASE, "thinking": value})
