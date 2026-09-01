"""The Claude subscription transport: usage capture and error mapping."""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
import urllib.error
from email.message import Message

from llm_local_proxy.providers.claude.auth import ClaudeAuthError
from llm_local_proxy.providers.claude.upstream import (
    ClaudeUpstream,
    ClaudeUpstreamError,
    UsageStore,
    _normalize_model,
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

    def test_only_unauthorized_responses_mark_the_account_unavailable(self):
        unauthorized = _upstream_error(
            _http_error(401, '{"error":{"message":"expired"}}')
        )
        forbidden = _upstream_error(_http_error(403, '{"error":{"message":"revoked"}}'))
        malformed = _upstream_error(
            _http_error(400, '{"error":{"message":"bad request"}}')
        )
        self.assertTrue(unauthorized.account_unavailable)
        self.assertFalse(forbidden.account_unavailable)
        self.assertFalse(malformed.account_unavailable)


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


class _FakeAuth:
    def __init__(self, tokens=("tok",)):
        self.tokens = list(tokens)
        self.forced = []

    def access_token(self, force_refresh: bool = False) -> str:
        self.forced.append(force_refresh)
        return self.tokens[min(len(self.forced) - 1, len(self.tokens) - 1)]


class _StaleAuth:
    def access_token(self, force_refresh: bool = False) -> str:
        raise ClaudeAuthError("refresh token invalid", 400)


class _FakeOpener:
    """Answers each open() with the next queued response or error."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.requests = []

    def open(self, request, timeout=None):
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class _SseResponse(io.BytesIO):
    """An SSE body that also carries headers, as a real response does."""

    def __init__(self, payload: bytes, headers=None):
        super().__init__(payload)
        self.headers = headers if headers is not None else Message()


def _sse(*events: str, headers=None) -> _SseResponse:
    body = "".join(f"data: {event}\n\n" for event in events)
    return _SseResponse(body.encode(), headers)


def _upstream(*answers, tokens=("tok",), tmp: pathlib.Path | None = None):
    upstream = ClaudeUpstream(
        _FakeAuth(tokens),
        timeout=5,
        tokens_path=(tmp / "tokens.json") if tmp else None,
    )
    upstream._opener = _FakeOpener(*answers)
    return upstream


class UpstreamRequestTest(unittest.TestCase):
    """The paths a live subscription takes: refresh, retry, usage, interruption."""

    def test_expired_token_is_refreshed_once_and_the_call_repeats(self):
        upstream = _upstream(_http_error(401, "{}"), _sse('{"type":"message_stop"}'))
        events = list(upstream.events({"model": "m", "messages": []}))
        self.assertEqual([event["type"] for event in events], ["message_stop"])
        self.assertEqual(upstream.auth.forced, [False, True])

    def test_terminal_refresh_failure_marks_the_account_unavailable(self):
        upstream = ClaudeUpstream(_StaleAuth(), timeout=5)
        with self.assertRaises(ClaudeUpstreamError) as caught:
            upstream.models()
        self.assertTrue(caught.exception.account_unavailable)

    def test_a_second_401_is_reported_rather_than_retried_forever(self):
        upstream = _upstream(_http_error(401, "{}"), _http_error(401, "{}"))
        with self.assertRaises(ClaudeUpstreamError) as caught:
            list(upstream.events({"model": "m", "messages": []}))
        self.assertEqual(caught.exception.status, 401)
        self.assertEqual(upstream.auth.forced, [False, True])

    def test_a_refused_thinking_budget_is_retried_as_adaptive(self):
        # The model accepts thinking but not the explicit budget: rather than
        # failing the turn, the request repeats with the native adaptive mode.
        refusal = _http_error(
            400,
            '{"error":{"message":"thinking.budget_tokens is too large"}}',
        )
        upstream = _upstream(refusal, _sse('{"type":"message_stop"}'))
        body = {
            "model": "m",
            "messages": [],
            "thinking": {
                "type": "enabled",
                "budget_tokens": 99,
                "display": "summarized",
            },
        }
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream):
            list(upstream.events(body))
        retried = json.loads(upstream._opener.requests[1].data)
        self.assertEqual(
            retried["thinking"], {"type": "adaptive", "display": "summarized"}
        )
        self.assertEqual(body["thinking"]["type"], "enabled", "caller's body intact")
        # The retry succeeded, so nothing was worth reporting to the operator.
        self.assertEqual(stream.getvalue(), "")

    def test_zero_token_prewarm_uses_non_streaming_upstream(self):
        message = {
            "type": "message",
            "content": [],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 8, "output_tokens": 0},
        }
        response = _SseResponse(json.dumps(message).encode())
        upstream = _upstream(response)
        body = {"model": "m", "messages": [], "max_tokens": 0, "stream": True}
        events = list(upstream.events(body))
        sent = json.loads(upstream._opener.requests[0].data)
        self.assertFalse(sent["stream"])
        self.assertTrue(body["stream"], "caller's body remains unchanged")
        self.assertEqual(
            [event["type"] for event in events],
            ["message_start", "message_delta", "message_stop"],
        )

    def test_an_unrelated_400_is_not_retried(self):
        upstream = _upstream(
            _http_error(400, '{"error":{"message":"max_tokens too large"}}')
        )
        with self.assertRaises(ClaudeUpstreamError):
            list(upstream.events({"model": "m", "messages": []}))
        self.assertEqual(len(upstream._opener.requests), 1)

    def test_a_final_thinking_rejection_reports_the_turn(self):
        refusal = _http_error(
            400, '{"error":{"message":"thinking blocks cannot be modified"}}'
        )
        upstream = _upstream(refusal)
        body = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "", "signature": "S"},
                        {"type": "tool_use"},
                    ],
                }
            ],
        }
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), self.assertRaises(ClaudeUpstreamError):
            list(upstream.events(body))
        self.assertIn("thinking(text=0,sig=1)", stream.getvalue())

    def test_usage_is_recorded_once_the_message_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = pathlib.Path(directory)
            upstream = _upstream(
                _sse(
                    '{"type":"message_start","message":{"usage":'
                    '{"input_tokens":10,"cache_read_input_tokens":4,'
                    '"cache_creation_input_tokens":2}}}',
                    '{"type":"message_delta","usage":{"output_tokens":3}}',
                    '{"type":"message_delta","usage":{"output_tokens":7}}',
                    '{"type":"message_stop"}',
                ),
                tmp=tmp,
            )
            list(upstream.events({"model": "m", "messages": []}))
            totals = upstream.ledger.windows()["5h"]
            # Output is cumulative per delta, so the last value is the total.
            self.assertEqual(totals["output"], 7)
            self.assertEqual(totals["input"], 10)
            self.assertEqual(totals["cache_read"], 4)
            self.assertEqual(totals["cache_write"], 2)

    def test_an_interrupted_stream_records_no_usage(self):
        # Without message_stop the numbers are partial; a half-recorded
        # request would understate every window it lands in.
        with tempfile.TemporaryDirectory() as directory:
            tmp = pathlib.Path(directory)
            upstream = _upstream(
                _sse(
                    '{"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                    '{"type":"message_delta","usage":{"output_tokens":3}}',
                ),
                tmp=tmp,
            )
            list(upstream.events({"model": "m", "messages": []}))
            self.assertEqual(upstream.ledger.windows()["5h"]["input"], 0)

    def test_rate_limit_headers_are_captured_from_a_failure_too(self):
        error = _http_error(429, "{}")
        error.headers = _headers(
            **{"anthropic-ratelimit-unified-5h-utilization": "0.9"}
        )
        upstream = _upstream(error)
        with self.assertRaises(ClaudeUpstreamError):
            list(upstream.events({"model": "m", "messages": []}))
        limits = {limit.label: limit for limit in upstream.usage.limits()}
        self.assertAlmostEqual(limits["5 hour"].used_percent, 90.0)


class ModelNormalizationTest(unittest.TestCase):
    def test_effort_levels_come_from_the_live_catalog(self):
        model = _normalize_model(
            {
                "id": "claude-future",
                "max_input_tokens": 345678,
                "max_tokens": 12345,
                "capabilities": {
                    "image_input": {"supported": True},
                    "effort": {
                        "high": {"supported": True},
                        "future-tier": {"supported": True},
                        "retired": {"supported": False},
                    },
                },
            }
        )
        self.assertEqual(model["reasoning_efforts"], ["high", "future-tier"])
        self.assertEqual(model["context_length"], 345678)
        self.assertEqual(model["max_output_tokens"], 12345)
        self.assertEqual(model["modalities"], ["text", "image"])


if __name__ == "__main__":
    unittest.main()
