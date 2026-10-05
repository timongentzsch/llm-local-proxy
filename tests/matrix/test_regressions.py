"""Defects found by audit, each pinned by the smallest stream that showed it."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.responses_egress import ResponseEncoder
from llm_local_proxy.errors import ProviderError
from llm_local_proxy.ir import (
    TextDelta,
    ThinkingDelta,
    ThinkingSignature,
    ToolCallArgs,
    ToolCallEnd,
    ToolCallStart,
)
from llm_local_proxy.providers.base import ProviderContext
from llm_local_proxy.providers.codex.auth import _limits
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.pool import Account, PooledProvider
from llm_local_proxy.providers.reasoning import ReasoningCache


class _Events:
    """A decoder that hands the encoder canonical events as they are."""

    def decode(self, event):
        return [event]

    def finish(self):
        return []


CITATION = {
    "type": "url_citation",
    "url": "https://example.com/a",
    "title": "A",
    "start_index": 0,
    "end_index": 3,
}


WINDOW = {"usedPercent": 12, "windowDurationMins": 300}


def _message(item_id, text, annotations):
    return {
        "type": "message",
        "id": item_id,
        "content": [{"type": "output_text", "text": text, "annotations": annotations}],
    }


class ChatToolIndexTest(unittest.TestCase):
    def test_tool_calls_are_numbered_from_zero_whatever_the_upstream_index(self):
        # Claude numbers a call among its content blocks, so text and thinking
        # push the first call past zero; a client accumulates by position.
        encoder = ChunkEncoder("m", _Events())
        chunks = []
        for event in (
            TextDelta("hi"),
            ToolCallStart(1, "a", "alpha"),
            ToolCallArgs(1, "{}"),
            ToolCallEnd(1, "a", "alpha", "{}"),
            ToolCallStart(2, "b", "beta"),
            ToolCallArgs(2, "{}"),
            ToolCallEnd(2, "b", "beta", "{}"),
        ):
            chunks.extend(encoder.feed(event))
        indices = [
            call["index"]
            for chunk in chunks
            for call in chunk["choices"][0]["delta"].get("tool_calls", [])
        ]
        self.assertEqual(indices, [0, 0, 1, 1])


class CodexCitationTest(unittest.TestCase):
    def _stream(self):
        first = _message("msg_1", "one", [CITATION])
        return [
            {"type": "response.output_text.delta", "delta": "one"},
            {"type": "response.output_text.annotation.added", "annotation": CITATION},
            {"type": "response.output_item.done", "item": first},
            {"type": "response.completed", "response": {"output": [first]}},
        ]

    def test_a_repeated_annotation_reaches_a_responses_client_once(self):
        encoder = ResponseEncoder("m", CodexDecoder(ReasoningCache()))
        for event in self._stream():
            encoder.feed(event)
        message = next(i for i in encoder.result()["output"] if i["type"] == "message")
        self.assertEqual(len(message["content"][0]["annotations"]), 1)

    def test_a_finished_message_does_not_cite_into_the_next_one(self):
        first = _message("msg_1", "one", [CITATION])
        second = _message("msg_2", "two", [])
        encoder = MessageEncoder("m", CodexDecoder(ReasoningCache()))
        for event in (
            {"type": "response.output_text.delta", "delta": "one"},
            {"type": "response.output_item.done", "item": first},
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": "c", "name": "f"},
            },
            {"type": "response.output_text.delta", "delta": "two"},
            {"type": "response.completed", "response": {"output": [first, second]}},
        ):
            encoder.feed(event)
        texts = [b for b in encoder.result()["content"] if b["type"] == "text"]
        self.assertEqual(len(texts[0]["citations"]), 1)
        self.assertNotIn("citations", texts[1])


class ThinkingBlockTest(unittest.TestCase):
    def test_a_signature_closes_its_block(self):
        encoder = MessageEncoder("m", _Events())
        for event in (
            ThinkingDelta("T0"),
            ThinkingSignature("SIG0"),
            ThinkingDelta("T1"),
            ThinkingSignature("SIG1"),
        ):
            encoder.feed(event)
        self.assertEqual(
            [(b["thinking"], b["signature"]) for b in encoder.result()["content"]],
            [("T0", "SIG0"), ("T1", "SIG1")],
        )


class CodexLimitsTest(unittest.TestCase):
    def test_a_null_limit_map_falls_back_to_the_default_limit(self):
        limits = _limits(
            {
                "rateLimits": {"limitId": "codex", "primary": WINDOW},
                "rateLimitsByLimitId": None,
            }
        )
        self.assertEqual([limit.used_percent for limit in limits], [12.0])
        self.assertEqual(limits[0].model, "")

    def test_no_limits_at_all_is_no_bars(self):
        self.assertEqual(_limits({"rateLimitsByLimitId": None}), ())


class _Auth:
    def signed_in(self):
        return True


class CatalogCacheTest(unittest.TestCase):
    def _provider(self, directory, fetch):
        class Fake(PooledProvider[str]):
            name = "fake"

            def new_account(self, slot):
                return Account(slot, _Auth(), slot)

            def fetch_catalog(self, account):
                return fetch()

            def no_account(self):
                return ProviderError("no account", 503)

        provider = Fake(ProviderContext(config=None, directory=Path(directory)))
        provider.store.add()
        provider.pool.add(provider.new_account("1"))
        return provider

    def test_a_failed_discovery_keeps_the_last_catalog(self):
        answers = [[{"id": "m"}], ProviderError("upstream down", 503)]

        def fetch():
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory, fetch)
            self.assertEqual(provider._live_catalog(), [{"id": "m"}])
            provider._catalog = (time.time() - 3600, provider._catalog[1])
            self.assertEqual(provider._live_catalog(), [{"id": "m"}])
            self.assertEqual(answers, [], "the expired catalog was refreshed")

    def test_a_failed_first_discovery_is_retried_soon_not_cached_as_empty(self):
        answers = [ProviderError("upstream down", 503), [{"id": "m"}]]

        def fetch():
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory, fetch)
            self.assertEqual(provider._live_catalog(), [])
            self.assertIsNone(provider._catalog)
            provider._catalog_retry_at = 0.0
            self.assertEqual(provider._live_catalog(), [{"id": "m"}])

    def test_concurrent_callers_share_one_discovery(self):
        calls = []

        def fetch():
            calls.append(1)
            time.sleep(0.05)
            return [{"id": "m"}]

        with tempfile.TemporaryDirectory() as directory:
            provider = self._provider(directory, fetch)
            threads = [
                threading.Thread(target=provider._live_catalog) for _ in range(8)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
