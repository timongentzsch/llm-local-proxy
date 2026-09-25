"""HTTP-level contracts: SSE framing, dialect resolution, auth headers.

The golden files pin what the translators emit; these pin the bytes that
wrap it. Together they cover the whole downstream surface.
"""

from __future__ import annotations

import io
import unittest
from threading import Event

from llm_local_proxy.dialects import DEFAULT, DIALECTS, OPENAI, resolve
from llm_local_proxy.http import security
from llm_local_proxy.http.sse import SseStream, render, with_heartbeats
from llm_local_proxy.providers.transport import read_events


class FramingTest(unittest.TestCase):
    def test_named_frame_prefixes_the_event(self):
        self.assertEqual(
            render({"a": 1}, "message_start"),
            b'event: message_start\ndata: {"a":1}\n\n',
        )

    def test_openai_stream_bytes_are_unchanged(self):
        buffer = io.BytesIO()
        buffer.flush = lambda: None  # type: ignore[method-assign]
        stream = SseStream(buffer, OPENAI.keepalive, named=False)
        stream.send({"id": "chatcmpl-1"})
        stream.keepalive()
        stream.end()
        self.assertEqual(
            buffer.getvalue(),
            b'data: {"id":"chatcmpl-1"}\n\n: keepalive\n\ndata: [DONE]\n\n',
        )

    def test_named_streams_name_each_frame_and_end_without_a_sentinel(self):
        buffer = io.BytesIO()
        buffer.flush = lambda: None  # type: ignore[method-assign]
        stream = SseStream(buffer, OPENAI.keepalive, named=True)
        stream.send({"type": "response.completed"})
        stream.end()
        self.assertEqual(
            buffer.getvalue(),
            b'event: response.completed\ndata: {"type":"response.completed"}\n\n',
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
    def test_local_key_is_checked_in_every_accepted_form(self):
        # The key is the proxy's own, not a vendor's, so a mount must not
        # refuse it merely for arriving in the other vendor's header.
        cases = (
            ({"Authorization": "Bearer secret"}, "secret", True),
            ({"x-api-key": "secret"}, "secret", True),
            ({"Authorization": "Bearer nope"}, "secret", False),
            ({"x-api-key": "nope"}, "secret", False),
            ({"Authorization": "secret"}, "secret", False),  # missing scheme
            ({"Authorization": "Bearer stale", "x-api-key": "secret"}, "secret", True),
            ({}, "", True),  # no configured key
        )
        for headers, key, expected in cases:
            with self.subTest(headers=headers, key=key):
                self.assertIs(security.authorized(headers, key), expected)

    def test_host_and_origin_must_be_local(self):
        self.assertTrue(security.valid_host({"Host": "127.0.0.1:8787"}, "127.0.0.1"))
        self.assertFalse(security.valid_host({"Host": "evil.test"}, "127.0.0.1"))
        host = {"Host": "127.0.0.1:8787"}
        self.assertTrue(
            security.same_origin({**host, "Origin": "http://127.0.0.1:8787"})
        )
        self.assertFalse(security.same_origin({**host, "Origin": "https://evil.test"}))


if __name__ == "__main__":
    unittest.main()
