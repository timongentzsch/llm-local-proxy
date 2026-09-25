"""Prompt-cache locality on the Codex path: the derived cache key and its account."""

from __future__ import annotations

import threading
import time
import unittest

from llm_local_proxy.dialects.openai.ingress import parse
from llm_local_proxy.dialects.openai.responses_ingress import parse as parse_responses
from llm_local_proxy.providers.codex import Codex
from llm_local_proxy.providers.codex.request import build
from llm_local_proxy.providers.reasoning import ReasoningCache

MODEL = "acme-gpt-1"


def body_for(turns, system="You are helpful.", tools=None):
    value = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, *turns],
    }
    if tools:
        value["tools"] = tools
    return value


def built(turns, session="", system="You are helpful."):
    return build(parse(body_for(turns, system), session), ReasoningCache())


class CacheKeyTest(unittest.TestCase):
    """``prompt_cache_key`` decides which upstream cache a request may reuse."""

    def test_a_growing_conversation_keeps_one_derived_key(self):
        first = [{"role": "user", "content": "port the parser"}]
        later = [
            *first,
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "now the tests"},
        ]
        _, key = built(first)
        _, grown = built(later)
        self.assertTrue(key.startswith("proxy-"))
        self.assertEqual(key, grown)

    def test_an_image_only_opening_turn_still_seeds_the_key(self):
        """Empty text is a real value there; re-seeding from a later turn would
        move a live conversation to another upstream cache."""

        opening = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
            ],
        }
        _, first = built([opening])
        _, second = built([opening, {"role": "user", "content": "what is this"}])
        self.assertEqual(first, second)

    def test_two_image_only_conversations_do_not_share_a_key(self):
        def opening(url):
            return {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": url}}],
            }

        _, first = built([opening("data:image/png;base64,AA")])
        _, second = built([opening("data:image/png;base64,BB")])
        self.assertNotEqual(first, second)

    def test_a_client_session_wins_over_the_derived_key(self):
        turns = [{"role": "user", "content": "hi"}]
        _, key = built(turns, session="session-42")
        self.assertEqual(key, "session-42")

    def test_a_client_cache_key_reaches_codex_verbatim(self):
        """An explicit key names the cache; a session header only picks the
        account, so it must not replace the key."""

        chat = {
            **body_for([{"role": "user", "content": "hi"}]),
            "prompt_cache_key": "k1",
        }
        self.assertEqual(build(parse(chat, "header-1"), ReasoningCache())[1], "k1")
        responses = {"model": MODEL, "input": "hi", "prompt_cache_key": "k2"}
        sent, _ = build(parse_responses(responses, "header-2"), ReasoningCache())
        self.assertEqual(sent["prompt_cache_key"], "k2")

    def test_a_rewritten_first_turn_moves_the_key(self):
        """The fallback hashes the first user text, so a refreshed timestamp
        there costs the conversation both its key and its prefix."""

        _, first = built([{"role": "user", "content": "[10:00] hi"}])
        _, second = built([{"role": "user", "content": "[10:01] hi"}])
        self.assertNotEqual(first, second)

    def test_a_changed_system_prompt_moves_the_key(self):
        turns = [{"role": "user", "content": "hi"}]
        _, first = built(turns, system="You are helpful.")
        _, second = built(turns, system="You are helpful. It is 10:01.")
        self.assertNotEqual(first, second)


class _Pool:
    def __init__(self):
        self.sessions = []

    def stream(self, session, create, no_account):
        self.sessions.append(session)
        return iter(())


class PoolLocalityTest(unittest.TestCase):
    @staticmethod
    def provider(pool):
        codex = object.__new__(Codex)
        codex._lock = threading.Lock()
        codex._catalog = (time.time(), [{"id": MODEL}])
        codex.cache = ReasoningCache()
        codex.pool = pool
        return codex

    def test_a_sessionless_request_still_pins_one_account(self):
        pool = _Pool()
        codex = self.provider(pool)
        turns = [{"role": "user", "content": "port the parser"}]
        request = parse(body_for(turns), "")
        codex.chat(MODEL, request)
        codex.chat(MODEL, parse(body_for(turns), ""))
        _, key = built(turns)
        self.assertEqual(pool.sessions, [key, key])

    def test_a_client_session_reaches_the_pool_unchanged(self):
        pool = _Pool()
        codex = self.provider(pool)
        request = parse(body_for([{"role": "user", "content": "hi"}]), "session-42")
        codex.chat(MODEL, request)
        self.assertEqual(pool.sessions, ["session-42"])


if __name__ == "__main__":
    unittest.main()
