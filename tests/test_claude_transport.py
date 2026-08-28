"""The Claude subscription transport: usage capture and error mapping."""

import contextlib
import io
import pathlib
import tempfile
import unittest
import urllib.error
from email.message import Message

from llm_local_proxy.providers.claude.upstream import (
    ClaudeUpstreamError,
    UsageStore,
    _report_block_shape,
    _thinking_rejected,
    _upstream_error,
)


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.anthropic.com/v1/messages",
        code,
        "error",
        None,
        io.BytesIO(body.encode()),
    )


class UpstreamErrorTest(unittest.TestCase):
    def test_rate_limit_becomes_meaningful_message(self):
        error = _upstream_error(
            _http_error(
                429,
                '{"type":"error","error":{"type":"rate_limit_error","message":"Error"}}',
            )
        )
        self.assertEqual(error.status, 429)
        self.assertIn("usage limit", str(error))

    def test_keeps_informative_messages(self):
        error = _upstream_error(
            _http_error(
                400,
                '{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens too large"}}',
            )
        )
        self.assertEqual(str(error), "max_tokens too large")


class BlockShapeReportTest(unittest.TestCase):
    """The diagnostic runs inside an error path and must not raise there."""

    def _report(self, error, body):
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            _report_block_shape(error, body)
        return stream.getvalue()

    def test_names_each_turn_without_quoting_the_conversation(self):
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "secret"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "abcd", "signature": "xy"},
                        {"type": "redacted_thinking", "data": "zzz"},
                        {"type": "tool_use"},
                    ],
                },
            ]
        }
        report = self._report(ClaudeUpstreamError(400, "thinking blocks"), body)
        self.assertIn("[0] user: text", report)
        self.assertIn("thinking(text=4,sig=2)", report)
        self.assertIn("redacted(data=3)", report)
        self.assertNotIn("secret", report)
        self.assertNotIn("abcd", report)

    def test_stays_silent_unless_upstream_faulted_the_thinking(self):
        body = {"messages": [{"role": "user", "content": [{"type": "text"}]}]}
        self.assertEqual(self._report(ClaudeUpstreamError(429, "slow down"), body), "")
        self.assertEqual(
            self._report(ClaudeUpstreamError(400, "max_tokens too large"), body), ""
        )

    def test_survives_a_body_it_did_not_expect(self):
        # It reports on the way out of a failure; raising here would replace
        # the upstream error with its own.
        error = ClaudeUpstreamError(400, "thinking blocks")
        for body in (
            {},
            {"messages": "not a list"},
            {"messages": [None, 7, {"role": "user", "content": "flat"}]},
            {"messages": [{"role": "assistant", "content": [None, {"type": None}]}]},
            {"messages": [{"content": [{"type": "thinking", "thinking": 5}]}]},
        ):
            self._report(error, body)


def _headers(**values):
    message = Message()
    for key, value in values.items():
        message[key] = value
    return message


ENABLED = {"thinking": {"type": "enabled", "budget_tokens": 4096}}


class ThinkingFallbackTest(unittest.TestCase):
    def test_rejected_budget_falls_back(self):
        error = ClaudeUpstreamError(400, "thinking.enabled: not permitted")
        self.assertTrue(_thinking_rejected(error, ENABLED))

    def test_unrelated_400_is_not_retried(self):
        error = ClaudeUpstreamError(400, "messages: must not be empty")
        self.assertFalse(_thinking_rejected(error, ENABLED))

    def test_other_statuses_are_not_retried(self):
        error = ClaudeUpstreamError(429, "thinking")
        self.assertFalse(_thinking_rejected(error, ENABLED))

    def test_adaptive_request_is_never_retried(self):
        error = ClaudeUpstreamError(400, "thinking")
        self.assertFalse(_thinking_rejected(error, {"thinking": {"type": "adaptive"}}))


class UsageStoreTest(unittest.TestCase):
    def test_captures_only_unified_ratelimit_headers(self):
        value = UsageStore.capture(
            _headers(
                **{
                    "anthropic-ratelimit-unified-5h-utilization": "0.07",
                    "anthropic-ratelimit-unified-5h-reset": "1784900000",
                    "anthropic-ratelimit-unified-overage-status": "rejected",
                    "x-request-id": "ignored",
                }
            )
        )
        self.assertIn("updated_at", value)
        self.assertEqual(value["anthropic-ratelimit-unified-5h-utilization"], "0.07")
        self.assertNotIn("x-request-id", value)

    def test_ignores_responses_without_usage(self):
        self.assertIsNone(UsageStore.capture(_headers(**{"x-request-id": "abc"})))
        self.assertIsNone(UsageStore.capture(None))

    def test_update_persists_and_survives_reload(self):
        path = pathlib.Path(tempfile.mkdtemp()) / "usage.json"
        store = UsageStore(path)
        self.assertIsNone(store.get())
        store.update(_headers(**{"anthropic-ratelimit-unified-5h-utilization": "0.1"}))
        reloaded = UsageStore(path)
        value = reloaded.get()
        self.assertEqual(value["anthropic-ratelimit-unified-5h-utilization"], "0.1")

    def test_limits_normalize_headers_into_dashboard_bars(self):
        store = UsageStore()
        store.update(
            _headers(
                **{
                    "anthropic-ratelimit-unified-5h-utilization": "0.57",
                    "anthropic-ratelimit-unified-5h-reset": "1787234400",
                    "anthropic-ratelimit-unified-7d-utilization": "0.28",
                    "anthropic-ratelimit-unified-fallback-percentage": "0.5",
                    "anthropic-ratelimit-unified-overage-status": "rejected",
                }
            )
        )
        limits = {limit.label: limit for limit in store.limits()}
        self.assertEqual(sorted(limits), ["5 hour", "weekly"])
        self.assertAlmostEqual(limits["5 hour"].used_percent, 57.0)
        self.assertEqual(limits["5 hour"].resets_at, "1787234400")
        self.assertIsNone(limits["weekly"].resets_at)
        self.assertIsNotNone(store.updated_at())

    def test_limits_are_empty_without_usage(self):
        self.assertEqual(UsageStore().limits(), ())
        self.assertIsNone(UsageStore().updated_at())


if __name__ == "__main__":
    unittest.main()
