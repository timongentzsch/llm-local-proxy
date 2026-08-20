import io
import pathlib
import tempfile
import unittest
import urllib.error
from email.message import Message

from llm_local_proxy.claude_upstream import UsageStore, _upstream_error


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


def _headers(**values):
    message = Message()
    for key, value in values.items():
        message[key] = value
    return message


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
