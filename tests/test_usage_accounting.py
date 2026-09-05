"""Client usage and persisted accounting must agree across every dialect."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event

from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.responses_egress import ResponseEncoder
from llm_local_proxy.http.sse import with_heartbeats
from llm_local_proxy.ir import Usage
from llm_local_proxy.ledger import TokenLedger
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.upstream import ClaudeUpstream
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.upstream import Upstream
from llm_local_proxy.providers.pool import Account, AccountPool
from llm_local_proxy.providers.reasoning import ReasoningCache


def upstream(provider):
    cls = ClaudeUpstream if provider == "claude" else Upstream
    result = cls.__new__(cls)
    result.ledger = TokenLedger(input_includes_cache=provider == "codex")
    return result


def codex_terminal(kind="response.completed", reason="max_output_tokens"):
    return {
        "type": kind,
        "response": {
            "incomplete_details": {"reason": reason},
            "usage": {
                "input_tokens": 170,
                "output_tokens": 15,
                "input_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        },
    }


CLAUDE_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "usage": {
                "input_tokens": 25,
                "output_tokens": 1,
                "cache_read_input_tokens": 60,
                "cache_creation_input_tokens": 10,
            }
        },
    },
    {"type": "message_delta", "usage": {"input_tokens": 100, "output_tokens": 8}},
    {
        "type": "message_delta",
        "usage": {
            "output_tokens": 15,
            "output_tokens_details": {"thinking_tokens": 5},
        },
    },
    {"type": "message_stop"},
]


class UsageAccountingTest(unittest.TestCase):
    def test_all_client_formats_agree_with_the_ledger(self):
        for provider in ("claude", "codex"):
            for encoder in (MessageEncoder, ChunkEncoder, ResponseEncoder):
                for streaming in (False, True):
                    with self.subTest(
                        provider=provider, encoder=encoder, streaming=streaming
                    ):
                        client = upstream(provider)
                        decoder = (
                            ClaudeDecoder()
                            if provider == "claude"
                            else CodexDecoder(ReasoningCache())
                        )
                        output = encoder("test", decoder)
                        events = (
                            CLAUDE_EVENTS
                            if provider == "claude"
                            else [codex_terminal()]
                        )
                        for event in client._tracked(iter(events)):
                            output.feed(event)
                        if streaming:
                            frames = output.finish()
                            if encoder is ResponseEncoder:
                                result = frames[-1]["response"]
                            else:
                                result = next(
                                    frame for frame in frames if frame.get("usage")
                                )
                        else:
                            result = output.result()
                        usage = result["usage"]
                        if encoder is MessageEncoder:
                            self.assertEqual(usage["input_tokens"], 100)
                            self.assertEqual(usage["cache_read_input_tokens"], 60)
                            self.assertEqual(usage["cache_creation_input_tokens"], 10)
                            self.assertEqual(usage["output_tokens"], 15)
                        else:
                            name = "prompt" if encoder is ChunkEncoder else "input"
                            out = "completion" if encoder is ChunkEncoder else "output"
                            self.assertEqual(usage[f"{name}_tokens"], 170)
                            self.assertEqual(
                                usage[f"{name}_tokens_details"],
                                {"cached_tokens": 60, "cache_write_tokens": 10},
                            )
                            self.assertEqual(usage[f"{out}_tokens"], 15)
                            self.assertEqual(usage["total_tokens"], 185)
                            self.assertEqual(
                                usage[f"{out}_tokens_details"]["reasoning_tokens"], 5
                            )
                        self.assertEqual(
                            client.ledger.windows()["5h"],
                            {
                                "input": 100,
                                "output": 15,
                                "cache_read": 60,
                                "cache_write": 10,
                            },
                        )

    def test_claude_explicit_zero_replaces_previous_snapshot(self):
        client = upstream("claude")
        decoder = ClaudeDecoder()
        events = CLAUDE_EVENTS[:-1] + [
            {"type": "message_delta", "usage": {"cache_creation_input_tokens": 0}},
            {"type": "message_stop"},
        ]
        for event in client._tracked(iter(events)):
            decoder.decode(event)
        usage = next(event for event in decoder.finish() if isinstance(event, Usage))
        self.assertEqual(usage.prompt, 160)
        self.assertEqual(usage.cache_write, 0)
        self.assertEqual(client.ledger.windows()["5h"]["cache_write"], 0)

    def test_terminal_usage_is_committed_before_yield_and_only_once(self):
        for provider in ("claude", "codex"):
            kinds = (
                ["message_stop"]
                if provider == "claude"
                else ["response.completed", "response.incomplete", "response.failed"]
            )
            for kind in kinds:
                with self.subTest(provider=provider, kind=kind):
                    client = upstream(provider)
                    events = (
                        CLAUDE_EVENTS
                        if provider == "claude"
                        else [codex_terminal(kind)]
                    )
                    tracked = client._tracked(iter(events + [events[-1]]))
                    for _ in events:
                        next(tracked)
                    self.assertEqual(client.ledger.windows()["5h"]["output"], 15)
                    next(tracked)  # duplicate terminal must not double the counts
                    tracked.close()
                    self.assertEqual(len(client.ledger._records), 1)
                    self.assertNotIn("partial_requests", client.ledger.windows()["5h"])

    def test_incomplete_codex_keeps_text_usage_and_stop_reason(self):
        for reason in ("max_output_tokens", "content_filter"):
            for encoder in (MessageEncoder, ChunkEncoder, ResponseEncoder):
                for streaming in (False, True):
                    with self.subTest(
                        reason=reason, encoder=encoder, streaming=streaming
                    ):
                        output = encoder("test", CodexDecoder(ReasoningCache()))
                        output.feed(
                            {
                                "type": "response.output_text.delta",
                                "delta": "partial answer",
                            }
                        )
                        output.feed(codex_terminal("response.incomplete", reason))
                        frames = output.finish() if streaming else []
                        result = output.result()
                        if encoder is ResponseEncoder:
                            self.assertEqual(result["status"], "incomplete")
                            self.assertEqual(
                                result["incomplete_details"], {"reason": reason}
                            )
                            self.assertEqual(
                                result["output"][0]["content"][0]["text"],
                                "partial answer",
                            )
                            if streaming:
                                self.assertEqual(
                                    frames[-1]["type"], "response.incomplete"
                                )
                                self.assertEqual(
                                    frames[-1]["response"]["usage"], result["usage"]
                                )
                        elif encoder is ChunkEncoder:
                            self.assertEqual(
                                result["choices"][0]["finish_reason"],
                                "length"
                                if reason == "max_output_tokens"
                                else "content_filter",
                            )
                            self.assertEqual(
                                result["choices"][0]["message"]["content"],
                                "partial answer",
                            )
                        else:
                            self.assertEqual(
                                result["stop_reason"],
                                "max_tokens"
                                if reason == "max_output_tokens"
                                else "refusal",
                            )
                            self.assertEqual(
                                result["content"][0]["text"], "partial answer"
                            )
                        self.assertTrue(result["usage"])

    def test_failed_codex_records_usage_without_hiding_error(self):
        client = upstream("codex")
        decoder = CodexDecoder(ReasoningCache())
        tracked = client._tracked(iter([codex_terminal("response.failed")]))
        try:
            with self.assertRaisesRegex(RuntimeError, "Codex response failed"):
                decoder.decode(next(tracked))
        finally:
            tracked.close()
        self.assertEqual(client.ledger.windows()["5h"]["output"], 15)

    def test_error_after_claude_start_keeps_reported_output_and_closes_source(self):
        client = upstream("claude")
        closed = Event()

        def broken():
            try:
                yield CLAUDE_EVENTS[0]
                raise OSError("connection lost")
            finally:
                closed.set()

        with self.assertRaisesRegex(OSError, "connection lost"):
            list(client._tracked(broken()))
        self.assertTrue(closed.is_set())
        self.assertEqual(client.ledger.windows()["5h"]["output"], 1)
        self.assertEqual(client.ledger.windows()["5h"]["partial_requests"], 1)

    def test_cancellation_closes_pool_and_records_partial_on_worker(self):
        client = upstream("claude")
        release, closed = Event(), Event()

        def delayed():
            try:
                yield CLAUDE_EVENTS[0]
                release.wait(2)
                yield {"type": "ping"}
            finally:
                closed.set()

        class Auth:
            def signed_in(self):
                return True

        source = delayed()  # retained references must not defer finalization
        tracked = client._tracked(source)
        pool = AccountPool([Account("1", Auth(), client)])
        pooled = pool.stream("audit", lambda _: tracked, RuntimeError)
        heartbeat = with_heartbeats(pooled)
        try:
            self.assertEqual(next(heartbeat), CLAUDE_EVENTS[0])
            heartbeat.close()
        finally:
            release.set()
        self.assertTrue(closed.wait(2))
        self.assertEqual(client.ledger.windows()["5h"]["partial_requests"], 1)
        self.assertEqual(client.ledger.windows()["5h"]["output"], 1)

    def test_unknown_usage_creates_no_phantom_record(self):
        for provider in ("claude", "codex"):
            client = upstream(provider)
            list(client._tracked(iter([{"type": "error"}])))
            self.assertEqual(client.ledger._records, [])

    def test_old_history_and_partial_records_survive_reload(self):
        for includes_cache in (False, True):
            with (
                self.subTest(includes_cache=includes_cache),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / "tokens.json"
                old = {
                    "ts": int(time.time()),
                    "input": 170 if includes_cache else 100,
                    "output": 15,
                    "cache_read": 60,
                    "cache_write": 10,
                }
                path.write_text(json.dumps([old]))
                ledger = TokenLedger(path, input_includes_cache=includes_cache)
                before = ledger.windows()["5h"]
                ledger.record(
                    Usage(prompt=170, completion=15, cache_read=60, cache_write=10),
                    partial=True,
                )
                reloaded = TokenLedger(path, input_includes_cache=includes_cache)
                totals = reloaded.windows()["5h"]
                for key, count in before.items():
                    self.assertEqual(totals[key], count * 2)
                self.assertEqual(totals["partial_requests"], 1)
                self.assertEqual(json.loads(path.read_text())[0], old)
