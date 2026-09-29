"""The Claude subscription transport: usage reads and error mapping."""

import contextlib
import io
import json
import pathlib
import tempfile
import unittest
import urllib.error

from llm_local_proxy.providers.claude.auth import OAUTH_BETA, ClaudeAuthError
from llm_local_proxy.providers.claude.upstream import (
    USAGE_URL,
    ClaudeUpstream,
    ClaudeUpstreamError,
    _limits,
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

    def test_missing_scope_marks_the_account_unavailable(self):
        error = _upstream_error(
            _http_error(
                403,
                '{"type":"error","error":{"type":"permission_error","message":'
                '"OAuth token does not meet scope requirement any_of(user:inference)"}}',
            )
        )
        self.assertTrue(error.account_unavailable)


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


ENABLED = {"thinking": {"type": "enabled", "budget_tokens": 4096}}


class ThinkingFallbackTest(unittest.TestCase):
    def test_only_a_rejected_explicit_budget_falls_back(self):
        adaptive = {"thinking": {"type": "adaptive"}}
        cases = (
            (400, "thinking.enabled: not permitted", ENABLED, True),
            (400, "messages: must not be empty", ENABLED, False),
            (429, "thinking", ENABLED, False),
            (400, "thinking", adaptive, False),
        )
        for status, message, body, expected in cases:
            with self.subTest(status=status, message=message):
                error = ClaudeUpstreamError(status, message)
                self.assertIs(_thinking_rejected(error, body), expected)


#: The fields of a live /api/oauth/usage response the dashboard reads, among
#: the placeholders and nulls it carries alongside them.
USAGE = {
    "five_hour": {"utilization": 8.0, "resets_at": "2026-09-29T22:19:59+00:00"},
    "seven_day": {"utilization": 75.0, "resets_at": "2026-09-29T21:59:59+00:00"},
    "seven_day_opus": None,
    "nimbus_quill": {"utilization": 0.0, "resets_at": None},
    "limits": [
        {"kind": "session", "percent": 8, "scope": None},
        {"kind": "weekly_all", "percent": 75, "scope": None},
        {
            "kind": "weekly_scoped",
            "percent": 3,
            "resets_at": "2026-09-29T22:00:00+00:00",
            "scope": {"model": {"id": None, "display_name": "Fable"}},
        },
    ],
}


class UsageLimitsTest(unittest.TestCase):
    def test_bars_come_from_the_session_weekly_and_model_scoped_windows(self):
        self.assertEqual(
            [
                (limit.label, limit.used_percent, limit.resets_at, limit.model)
                for limit in _limits(USAGE)
            ],
            [
                ("5 hour", 8.0, "2026-09-29T22:19:59+00:00", ""),
                ("weekly", 75.0, "2026-09-29T21:59:59+00:00", ""),
                ("Fable weekly", 3.0, "2026-09-29T22:00:00+00:00", "Fable"),
            ],
        )

    def test_odd_payloads_leave_no_bars_instead_of_failing(self):
        for payload in (
            {"limits": 3},
            {"five_hour": {"utilization": float("nan")}},
            {"seven_day": {"utilization": True}},
            {"limits": [None, {"kind": "weekly_scoped", "scope": None}]},
        ):
            with self.subTest(payload=payload):
                self.assertEqual(_limits(payload), ())
        with self.assertRaises(ClaudeUpstreamError):
            _limits([])


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
    """A response body, readable and closable as a real one is."""


def _sse(*events: str) -> _SseResponse:
    body = "".join(f"data: {event}\n\n" for event in events)
    return _SseResponse(body.encode())


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

    def test_an_interrupted_stream_records_flagged_partial_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp = pathlib.Path(directory)
            upstream = _upstream(
                _sse(
                    '{"type":"message_start","message":{"usage":{"input_tokens":10}}}',
                    '{"type":"message_delta","usage":{"output_tokens":3}}',
                ),
                tmp=tmp,
            )
            with self.assertRaisesRegex(RuntimeError, "terminal event"):
                list(upstream.events({"model": "m", "messages": []}))
            self.assertEqual(upstream.ledger.windows()["5h"]["input"], 10)
            self.assertEqual(upstream.ledger.windows()["5h"]["output"], 3)
            self.assertEqual(upstream.ledger.windows()["5h"]["partial_requests"], 1)

    def test_usage_is_read_from_the_oauth_usage_endpoint(self):
        upstream = _upstream(
            _http_error(401, "{}"), _SseResponse(json.dumps(USAGE).encode())
        )
        limits, _ = upstream.limits.current()
        self.assertEqual(len(limits), 3)
        request = upstream._opener.requests[-1]
        self.assertEqual(request.full_url, USAGE_URL)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.get_header("Anthropic-beta"), OAUTH_BETA)
        # An expired token is refreshed once and the read repeats.
        self.assertEqual(upstream.auth.forced, [False, True])

    def test_a_body_that_times_out_is_an_upstream_error(self):
        class Stalled(_SseResponse):
            def read(self, *args):
                raise TimeoutError("timed out")

        upstream = _upstream(Stalled())
        with self.assertRaisesRegex(ClaudeUpstreamError, "usage is unreadable"):
            upstream._get(USAGE_URL, "usage")


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
