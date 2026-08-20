"""HTTP-level contracts: SSE framing, dialect resolution, auth headers.

The golden files pin what the translators emit; these pin the bytes that
wrap it. Together they cover the whole downstream surface.
"""

from __future__ import annotations

import io
import unittest

from llm_local_proxy.dialects import DIALECTS, OPENAI, Frame, resolve
from llm_local_proxy.http import security
from llm_local_proxy.http.handler import api_path
from llm_local_proxy.http.sse import SseStream, render


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


class ResolveTest(unittest.TestCase):
    def test_bare_paths_belong_to_the_default_dialect(self):
        dialect, path = resolve("/v1/chat/completions")
        self.assertEqual(dialect.name, "openai")
        self.assertEqual(path, "/v1/chat/completions")

    def test_every_dialect_has_a_distinct_mount(self):
        prefixes = [dialect.prefix for dialect in DIALECTS]
        self.assertEqual(len(prefixes), len(set(prefixes)))

    def test_dashboard_alias_maps_onto_the_api_path(self):
        self.assertEqual(api_path("/api/v1/models"), "/v1/models")
        self.assertEqual(api_path("/v1/models"), "/v1/models")


class AuthTest(unittest.TestCase):
    def test_bearer_token_accepted(self):
        headers = {"Authorization": "Bearer secret"}
        self.assertTrue(security.authorized(headers, OPENAI, "secret"))

    def test_wrong_token_rejected(self):
        headers = {"Authorization": "Bearer nope"}
        self.assertFalse(security.authorized(headers, OPENAI, "secret"))

    def test_missing_scheme_rejected(self):
        self.assertFalse(security.authorized({"Authorization": "secret"}, OPENAI, "s"))

    def test_no_configured_key_allows_everything(self):
        self.assertTrue(security.authorized({}, OPENAI, ""))

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
