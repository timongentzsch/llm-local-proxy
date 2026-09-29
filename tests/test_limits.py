"""The shared usage-bar store: spacing, backoff and non-blocking reads."""

import contextlib
import io
import threading
import time
import unittest

from llm_local_proxy.errors import UpstreamError
from llm_local_proxy.providers.limits import TTL, LimitsStore, fullest
from llm_local_proxy.status import Limit

BARS = (
    Limit("5 hour", 40.0),
    Limit("weekly", 75.0),
    Limit("Fable weekly", 99.0, model="Fable"),
)


class LimitsStoreTest(unittest.TestCase):
    def test_reads_are_spaced_and_a_refusal_backs_off_keeping_the_last_bars(self):
        answers = [BARS, UpstreamError(429, "rate limited")]
        reads = []

        def read():
            reads.append(1)
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        store = LimitsStore("test", read)
        first = store.current()
        self.assertEqual(store.current(), first)
        self.assertEqual(len(reads), 1)

        store._next_read = 0
        with contextlib.redirect_stderr(io.StringIO()) as log:
            self.assertEqual(store.current(), first)
        self.assertEqual(len(reads), 2)
        self.assertIn("test: limits unavailable: rate limited", log.getvalue())
        self.assertGreater(store._next_read - time.monotonic(), TTL)

    def test_callers_do_not_wait_for_a_read_in_flight(self):
        started, release = threading.Event(), threading.Event()

        def read():
            started.set()
            release.wait(5)
            return BARS

        store = LimitsStore("test", read)
        reader = threading.Thread(target=store.current)
        reader.start()
        started.wait(5)
        self.assertEqual(store.current(), ((), None))
        release.set()
        reader.join(5)
        self.assertEqual(store.current()[0], BARS)

    def test_a_non_waiting_read_runs_in_the_background(self):
        release, done = threading.Event(), threading.Event()

        def read():
            release.wait(5)
            done.set()
            return BARS

        store = LimitsStore("test", read)
        self.assertEqual(store.current(wait=False), ((), None))
        release.set()
        done.wait(5)
        while store._reading:
            time.sleep(0.001)
        self.assertEqual(store.current(wait=False)[0], BARS)

    def test_an_unexpected_failure_does_not_freeze_the_store(self):
        answers = [OSError("pipe closed"), BARS]

        def read():
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer

        store = LimitsStore("test", read)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(store.current(), ((), None))
        store._next_read = 0
        self.assertEqual(store.current()[0], BARS)

    def test_clear_forgets_the_bars_and_drops_a_read_in_flight(self):
        reads = []
        store = LimitsStore("test", lambda: reads.append(1) or BARS)
        store.current()
        store.clear()
        self.assertEqual(store.current()[0], BARS)
        self.assertEqual(len(reads), 2)

        def read_then_clear():
            store.clear()
            return BARS

        store = LimitsStore("test", read_then_clear)
        self.assertEqual(store.current(), ((), None))

    def test_fullest_is_the_fullest_whole_account_window(self):
        self.assertEqual(fullest(BARS), 75.0)
        self.assertIsNone(fullest(BARS[2:]))
        self.assertIsNone(fullest(()))


if __name__ == "__main__":
    unittest.main()
