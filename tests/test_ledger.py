import pathlib
import tempfile
import time
import unittest

from llm_local_proxy.ledger import WINDOWS, TokenLedger
from llm_local_proxy.providers.claude.upstream import ClaudeUpstream
from llm_local_proxy.providers.codex.upstream import Upstream


class TokenLedgerTest(unittest.TestCase):
    def test_windows_sum_only_recent_requests(self):
        ledger = TokenLedger()
        ledger.add(input_tokens=100, output_tokens=50, cache_read=20, cache_write=5)
        now = time.time()
        with ledger._lock:
            ledger._records.insert(
                0,
                {
                    "ts": int(now) - 6 * 3600,  # inside 7d, outside 5h
                    "input": 400,
                    "output": 200,
                    "cache_read": 0,
                    "cache_write": 0,
                },
            )
        windows = ledger.windows()
        self.assertEqual(windows["5h"]["input"], 100)
        self.assertEqual(windows["5h"]["output"], 50)
        self.assertEqual(windows["5h"]["cache_read"], 20)
        # The 6h-old record is outside the 5h window, so cache_write stays 5.
        self.assertEqual(windows["5h"]["cache_write"], 5)
        self.assertEqual(windows["7d"]["input"], 500)
        self.assertEqual(windows["7d"]["output"], 250)

    def test_windows_are_ordered_and_complete(self):
        ledger = TokenLedger()
        ledger.add(input_tokens=1, output_tokens=2, cache_read=3, cache_write=4)
        windows = ledger.windows()
        # Keys mirror the WINDOWS definition order (5h then 7d).
        self.assertEqual(list(windows), [label for label, _ in WINDOWS])
        for label, _ in WINDOWS:
            self.assertEqual(
                list(windows[label]),
                ["input", "output", "cache_read", "cache_write"],
            )
            self.assertEqual(windows[label]["input"], 1)
            self.assertEqual(windows[label]["cache_write"], 4)

    def test_persists_across_reload(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "tokens.json"
        ledger = TokenLedger(path)
        ledger.add(input_tokens=7, output_tokens=3)
        reloaded = TokenLedger(path)
        self.assertEqual(reloaded.windows()["7d"]["input"], 7)
        self.assertEqual(reloaded.windows()["7d"]["output"], 3)

    def test_prunes_expired_records(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "tokens.json"
        ledger = TokenLedger(path)
        with ledger._lock:
            ledger._records = [
                {
                    "ts": time.time() - 8 * 86400,
                    "input": 999,
                    "output": 0,
                    "cache_read": 0,
                    "cache_write": 0,
                }
            ]
        ledger.add(input_tokens=1, output_tokens=1)
        self.assertEqual(ledger.windows()["7d"]["input"], 1)


class TokenCaptureTest(unittest.TestCase):
    def test_claude_extracts_message_usage(self):
        claude = ClaudeUpstream.__new__(ClaudeUpstream)
        claude.ledger = TokenLedger()
        events = [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 42,
                        "cache_read_input_tokens": 7,
                        "cache_creation_input_tokens": 3,
                    }
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
            {"type": "message_delta", "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ]
        # Events pass through unchanged and the ledger accumulates per request,
        # flushed only at message_stop.
        seen = list(claude._tracked(iter(events)))
        self.assertEqual(len(seen), 4)
        window = claude.ledger.windows()["7d"]
        self.assertEqual(window["input"], 42)
        self.assertEqual(window["output"], 9)
        self.assertEqual(window["cache_read"], 7)
        self.assertEqual(window["cache_write"], 3)

    def test_claude_ignores_non_usage_events(self):
        claude = ClaudeUpstream.__new__(ClaudeUpstream)
        claude.ledger = TokenLedger()
        seen = list(
            claude._tracked(
                iter([{"type": "ping"}, {"type": "content_block_delta", "delta": {}}])
            )
        )
        self.assertEqual(len(seen), 2)
        self.assertEqual(claude.ledger.windows()["7d"]["input"], 0)

    def test_claude_multiple_deltas_accumulate_once(self):
        claude = ClaudeUpstream.__new__(ClaudeUpstream)
        claude.ledger = TokenLedger()
        events = [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 2,
                        "cache_creation_input_tokens": 3,
                    }
                },
            },
            {
                "type": "message_delta",
                "usage": {"output_tokens": 4, "cache_read_input_tokens": 5},
            },
            {
                "type": "message_delta",
                "usage": {"output_tokens": 9, "cache_read_input_tokens": 6},
            },
            {"type": "message_stop"},
        ]
        seen = list(claude._tracked(iter(events)))
        self.assertEqual(len(seen), 4)
        window = claude.ledger.windows()["7d"]
        # Cumulative fields take their latest (max) value, and the request is
        # counted exactly once (only flushed at message_stop).
        self.assertEqual(window["input"], 10)
        self.assertEqual(window["output"], 9)
        self.assertEqual(window["cache_read"], 6)
        self.assertEqual(window["cache_write"], 3)

    def test_claude_interrupted_stream_not_recorded(self):
        claude = ClaudeUpstream.__new__(ClaudeUpstream)
        claude.ledger = TokenLedger()
        events = [
            {
                "type": "message_start",
                "message": {"usage": {"input_tokens": 7}},
            },
            {"type": "message_delta", "usage": {"output_tokens": 3}},
            # no message_stop: client disconnected / error
        ]
        list(claude._tracked(iter(events)))
        self.assertEqual(claude.ledger.windows()["7d"]["input"], 0)

    def test_codex_extracts_completed_usage(self):
        upstream = _upstream_with_ledger()
        events = [
            {"type": "response.output_item.done", "item": {}},
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 5,
                        "input_tokens_details": {
                            "cached_tokens": 4,
                            "cache_write_tokens": 2,
                        },
                    }
                },
            },
        ]
        out = list(upstream._tracked(iter(events)))
        self.assertEqual(len(out), 2)
        window = upstream.ledger.windows()["7d"]
        self.assertEqual(window["input"], 11)
        self.assertEqual(window["output"], 5)
        self.assertEqual(window["cache_read"], 4)
        self.assertEqual(window["cache_write"], 2)


def _upstream_with_ledger():
    upstream = Upstream.__new__(Upstream)
    upstream.ledger = TokenLedger()
    return upstream


if __name__ == "__main__":
    unittest.main()
