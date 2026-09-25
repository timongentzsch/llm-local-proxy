import pathlib
import tempfile
import time
import unittest

from llm_local_proxy.ledger import WINDOWS, TokenLedger


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


if __name__ == "__main__":
    unittest.main()
