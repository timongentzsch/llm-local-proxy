"""Prompt-cache locality on the Codex path: the key, and where a prefix broke."""

from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from llm_local_proxy.dialects.openai.ingress import parse
from llm_local_proxy.providers.codex import Codex
from llm_local_proxy.providers.codex.prefix import PrefixProbe
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
        codex.probe = PrefixProbe()
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


class PrefixProbeTest(unittest.TestCase):
    @staticmethod
    def probe(bodies=False):
        lines = []
        return PrefixProbe(lines.append, bodies), lines

    def test_it_is_off_unless_the_environment_names_a_sink(self):
        probe = PrefixProbe.from_env({})
        body, key = built([{"role": "user", "content": "hi"}])
        self.assertFalse(probe.enabled)
        self.assertIsNone(probe.record(key, body))

    def test_an_appended_turn_keeps_the_whole_previous_prefix(self):
        probe, lines = self.probe()
        first = [{"role": "user", "content": "port the parser"}]
        body, key = built(first)
        probe.record(key, body)
        grown, _ = built(
            [
                *first,
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "now the tests"},
            ]
        )
        report = probe.record(key, grown)
        self.assertEqual(report.kind, "appended")
        # instructions plus the first user turn survive; two items are new.
        self.assertEqual((report.stable, report.items), (2, 4))
        self.assertIn("kind=appended", lines[1])

    def test_a_rewritten_early_item_is_named_with_its_position(self):
        probe, _ = self.probe()
        stable = [
            {"role": "user", "content": "[10:00] hi"},
            {"role": "assistant", "content": "hello"},
        ]
        body, _ = built(stable)
        probe.record("session-42", body)
        edited, _ = built(
            [{"role": "user", "content": "[10:01] hi"}, stable[1]],
        )
        report = probe.record("session-42", edited)
        self.assertEqual(report.kind, "changed")
        self.assertEqual(report.stable, 1)
        self.assertEqual(report.label, "1:user")
        self.assertLess(report.stable_chars, report.total_chars)

    def test_a_shortened_history_is_not_reported_as_a_rewrite(self):
        probe, _ = self.probe()
        turns = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        long_body, _ = built(turns)
        probe.record("session-42", long_body)
        short_body, _ = built(turns[:1])
        self.assertEqual(probe.record("session-42", short_body).kind, "shortened")

    def test_one_conversation_under_two_keys_is_reported_as_churn(self):
        probe, _ = self.probe()
        body, _ = built([{"role": "user", "content": "hi"}])
        probe.record("session-a", body)
        self.assertEqual(probe.record("session-b", body).churn, "session-a")

    def test_a_first_sighting_claims_no_stable_prefix(self):
        probe, _ = self.probe()
        body, key = built([{"role": "user", "content": "hi"}])
        report = probe.record(key, body)
        self.assertEqual((report.kind, report.stable), ("first", 0))

    def test_prompt_text_is_withheld_unless_bodies_are_requested(self):
        quiet, quiet_lines = self.probe()
        loud, loud_lines = self.probe(bodies=True)
        first, _ = built([{"role": "user", "content": "SECRET-ONE"}])
        second, _ = built([{"role": "user", "content": "SECRET-TWO"}])
        for probe in (quiet, loud):
            probe.record("session-42", first)
            probe.record("session-42", second)
        self.assertNotIn("SECRET", "".join(quiet_lines))
        self.assertIn("SECRET-ONE", loud_lines[1])
        self.assertIn("SECRET-TWO", loud_lines[1])

    def test_a_reordered_tool_list_breaks_the_prefix_at_the_tools(self):
        probe, _ = self.probe()
        tools = [
            {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }
            for name in ("read", "write")
        ]
        turns = [{"role": "user", "content": "hi"}]
        first, _ = build(parse(body_for(turns, tools=tools), "x"), ReasoningCache())
        flipped, _ = build(
            parse(body_for(turns, tools=list(reversed(tools))), "x"), ReasoningCache()
        )
        probe.record("x", first)
        report = probe.record("x", flipped)
        self.assertEqual((report.kind, report.label), ("changed", "1:tools"))

    def test_shared_tools_do_not_make_two_conversations_one(self):
        probe, _ = self.probe()
        tools = [
            {
                "type": "function",
                "function": {"name": "read", "parameters": {"type": "object"}},
            }
        ]
        first, first_key = build(
            parse(body_for([{"role": "user", "content": "one"}], tools=tools), ""),
            ReasoningCache(),
        )
        second, second_key = build(
            parse(body_for([{"role": "user", "content": "two"}], tools=tools), ""),
            ReasoningCache(),
        )
        probe.record(first_key, first)
        self.assertEqual(probe.record(second_key, second).churn, "")

    def test_a_file_sink_appends_one_line_per_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.log"
            probe = PrefixProbe.from_env({"LLM_PROXY_PREFIX_DEBUG": str(path)})
            body, key = built([{"role": "user", "content": "hi"}])
            probe.record(key, body)
            probe.record(key, body)
            lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("kind=identical", lines[1])

    def test_a_dash_writes_to_stderr(self):
        probe = PrefixProbe.from_env({"LLM_PROXY_PREFIX_DEBUG": "-"})
        body, key = built([{"role": "user", "content": "hi"}])
        stream = io.StringIO()
        with redirect_stderr(stream):
            probe.record(key, body)
        self.assertIn("codex-prefix", stream.getvalue())

    def test_an_unwritable_sink_disables_the_probe_instead_of_failing(self):
        def refuse(line):
            raise OSError("read-only file system")

        probe = PrefixProbe(refuse)
        body, key = built([{"role": "user", "content": "hi"}])
        self.assertIsNotNone(probe.record(key, body))
        self.assertFalse(probe.enabled)
        self.assertIsNone(probe.record(key, body))

    def test_remembered_conversations_are_bounded(self):
        probe = PrefixProbe(lambda line: None, limit=2)
        body, _ = built([{"role": "user", "content": "hi"}])
        for name in ("a", "b", "c"):
            probe.record(name, body)
        # "a" was evicted, so its next request looks like a first sighting.
        self.assertEqual(probe.record("a", body).kind, "first")


if __name__ == "__main__":
    unittest.main()
