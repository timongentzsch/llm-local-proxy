"""HTTP-level contracts: SSE framing, dialect resolution, auth headers.

The golden files pin what the translators emit; these pin the bytes that
wrap it. Together they cover the whole downstream surface.
"""

from __future__ import annotations

import io
import unittest
from threading import Event

from llm_local_proxy.dialects import DEFAULT, DIALECTS, OPENAI, Frame, resolve
from llm_local_proxy.http import security
from llm_local_proxy.http.handler import api_path
from llm_local_proxy.http.sse import SseStream, render, with_heartbeats
from llm_local_proxy.providers.transport import read_events


class FramingTest(unittest.TestCase):
    def test_anonymous_frame_matches_chat_completions(self):
        self.assertEqual(render(Frame({"a": 1})), b'data: {"a":1}\n\n')

    def test_named_frame_prefixes_the_event(self):
        self.assertEqual(
            render(Frame({"a": 1}, "message_start")),
            b'event: message_start\ndata: {"a":1}\n\n',
        )

    def test_openai_stream_bytes_are_unchanged(self):
        buffer = io.BytesIO()
        buffer.flush = lambda: None  # type: ignore[method-assign]
        stream = SseStream(buffer, OPENAI)
        stream.send({"id": "chatcmpl-1"})
        stream.keepalive()
        stream.end()
        self.assertEqual(
            buffer.getvalue(),
            b'data: {"id":"chatcmpl-1"}\n\n: keepalive\n\ndata: [DONE]\n\n',
        )

    def test_error_envelope_is_unchanged(self):
        self.assertEqual(
            OPENAI.error(400, "nope"),
            {"error": {"message": "nope", "type": "proxy_error"}},
        )


class UpstreamFramingTest(unittest.TestCase):
    def test_multiline_frame_and_terminal_event(self):
        response = io.BytesIO(
            b': keepalive\r\nevent: done\r\ndata: {"type":\r\n'
            b'data: "done", "text": "hi"}\r\n\r\ndata: [DONE]\n\n'
        )
        self.assertEqual(
            list(read_events(response, {"done"})), [{"type": "done", "text": "hi"}]
        )
        self.assertTrue(response.closed)

    def test_premature_eof_is_not_a_successful_response(self):
        for body in (
            b"",
            b'data: {"type":"delta"}\n\n',
            b'data: {"type":"done"}\n',
            b"data: [DONE]\n\n",
        ):
            response = io.BytesIO(body)
            with (
                self.subTest(body=body),
                self.assertRaisesRegex(RuntimeError, "terminal event"),
            ):
                list(read_events(response, {"done"}))
            self.assertTrue(response.closed)

    def test_malformed_json_closes_the_response(self):
        response = io.BytesIO(b"data: broken\n\n")
        with self.assertRaises(ValueError):
            list(read_events(response))
        self.assertTrue(response.closed)


class ResolveTest(unittest.TestCase):
    def test_each_dialect_answers_under_its_own_prefix(self):
        for dialect in DIALECTS:
            with self.subTest(dialect=dialect.name):
                found, path = resolve(f"{dialect.prefix}/v1/models")
                self.assertEqual(found.name, dialect.name)
                self.assertEqual(path, "/v1/models")

    def test_bare_paths_still_reach_the_default_dialect(self):
        # Configured before the prefixes existed; must keep working.
        dialect, path = resolve("/v1/chat/completions")
        self.assertIs(dialect, DEFAULT)
        self.assertEqual(path, "/v1/chat/completions")

    def test_prefixed_and_bare_default_paths_agree(self):
        self.assertEqual(resolve(f"{DEFAULT.prefix}/v1/models"), resolve("/v1/models"))

    def test_a_bare_prefix_serves_the_dialect_root(self):
        self.assertEqual(resolve("/anthropic")[1], "/")

    def test_every_dialect_has_a_distinct_nonempty_mount(self):
        prefixes = [dialect.prefix for dialect in DIALECTS]
        self.assertEqual(len(prefixes), len(set(prefixes)))
        self.assertTrue(all(prefixes))

    def test_dashboard_alias_maps_onto_the_api_path(self):
        self.assertEqual(api_path("/api/v1/models"), "/v1/models")
        self.assertEqual(api_path("/api/v1/chat/completions"), "/v1/chat/completions")
        # Already-canonical paths and non-/v1 routes pass through untouched.
        self.assertEqual(api_path("/v1/models"), "/v1/models")
        self.assertEqual(api_path("/api/status"), "/api/status")


class HeartbeatTest(unittest.TestCase):
    def test_heartbeat_while_upstream_is_silent(self):
        release = Event()

        def delayed():
            release.wait()
            yield {"type": "response.completed"}

        stream = with_heartbeats(delayed(), interval=0.01)
        self.assertIsNone(next(stream))
        release.set()
        self.assertEqual(next(stream), {"type": "response.completed"})
        with self.assertRaises(StopIteration):
            next(stream)

    def test_upstream_exception_is_propagated(self):
        def broken():
            yield from ()
            raise RuntimeError("upstream failed")

        stream = with_heartbeats(broken(), interval=1)
        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            next(stream)


class OriginValidationTest(unittest.TestCase):
    def test_malformed_origins_are_rejected_without_crashing(self):
        for origin in (
            "http://[",
            "http://localhost:invalid",
            "file://localhost:8787",
            "https://localhost",
        ):
            with self.subTest(origin=origin):
                self.assertFalse(
                    security.same_origin({"Host": "localhost:8787", "Origin": origin})
                )

    def test_malformed_hosts_cannot_bypass_the_host_check(self):
        for host in (
            "localhost/path",
            "attacker@localhost",
            "[::1]junk",
            "localhost:99999",
        ):
            with self.subTest(host=host):
                self.assertFalse(security.valid_host({"Host": host}, "127.0.0.1"))


class AuthTest(unittest.TestCase):
    def test_every_credential_header_is_accepted(self):
        # The key is the proxy's own, not a vendor's, so a mount must not
        # refuse it merely for arriving in the other vendor's header.
        for headers in (
            {"Authorization": "Bearer secret"},
            {"x-api-key": "secret"},
        ):
            with self.subTest(header=next(iter(headers))):
                self.assertTrue(security.authorized(headers, "secret"))

    def test_wrong_token_rejected(self):
        self.assertFalse(
            security.authorized({"Authorization": "Bearer nope"}, "secret")
        )
        self.assertFalse(security.authorized({"x-api-key": "nope"}, "secret"))

    def test_missing_scheme_rejected(self):
        self.assertFalse(security.authorized({"Authorization": "secret"}, "s"))

    def test_a_valid_header_wins_over_a_stale_one(self):
        headers = {"Authorization": "Bearer stale", "x-api-key": "secret"}
        self.assertTrue(security.authorized(headers, "secret"))

    def test_no_configured_key_allows_everything(self):
        self.assertTrue(security.authorized({}, ""))

    def test_loopback_hosts_accepted(self):
        self.assertTrue(security.valid_host({"Host": "127.0.0.1:8787"}, "127.0.0.1"))
        self.assertFalse(security.valid_host({"Host": "evil.test"}, "127.0.0.1"))

    def test_cross_origin_rejected(self):
        headers = {"Host": "127.0.0.1:8787", "Origin": "https://evil.test"}
        self.assertFalse(security.same_origin(headers))

    def test_same_origin_accepted(self):
        headers = {"Host": "127.0.0.1:8787", "Origin": "http://127.0.0.1:8787"}
        self.assertTrue(security.same_origin(headers))


if __name__ == "__main__":
    unittest.main()
