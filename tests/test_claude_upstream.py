import io
import unittest
import urllib.error

from codex_local_proxy.claude_upstream import _upstream_error


def _http_error(code: int, body: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.anthropic.com/v1/messages", code, "error", None, io.BytesIO(body.encode())
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


if __name__ == "__main__":
    unittest.main()
