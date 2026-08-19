import unittest

from codex_local_proxy.claude_auth import ClaudeAuth, ClaudeAuthError, _normalize


class ClaudeAuthTest(unittest.TestCase):
    def test_extracts_code_from_callback_url_or_raw_code(self):
        self.assertEqual(ClaudeAuth._extract_code("abc123"), "abc123")
        self.assertEqual(
            ClaudeAuth._extract_code(
                "https://platform.claude.com/oauth/code/callback?state=1&code=abc&scope=y"
            ),
            "abc",
        )
        with self.assertRaises(ClaudeAuthError):
            ClaudeAuth._extract_code("  ")

    def test_normalizes_token_response(self):
        token = _normalize(
            '{"access_token":"sk-ant-oat01-x","refresh_token":"rt","expires_in":3600,'
            '"scope":"user:inference user:profile","subscriptionType":"max"}'
        )
        self.assertEqual(token["access_token"], "sk-ant-oat01-x")
        self.assertEqual(token["refresh_token"], "rt")
        self.assertGreater(token["expires_at"], 0)
        self.assertEqual(token["scopes"], ["user:inference", "user:profile"])
        self.assertEqual(token["subscription_type"], "max")

    def test_normalizes_malformed_response(self):
        self.assertEqual(_normalize("not json"), {})
        self.assertEqual(_normalize('{"refresh_token":"rt"}'), {"refresh_token": "rt"})


if __name__ == "__main__":
    unittest.main()
