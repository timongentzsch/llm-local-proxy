import pathlib
import tempfile
import unittest
from urllib.parse import parse_qs, urlparse

from llm_local_proxy.claude_auth import ClaudeAuth, ClaudeAuthError, _normalize


class ClaudeAuthTest(unittest.TestCase):
    def test_login_url_contains_state_and_finish_validates_it(self):
        auth = ClaudeAuth(pathlib.Path(tempfile.mkdtemp()) / "credentials.json")
        url = auth.login_start()["url"]
        query = parse_qs(urlparse(url).query)
        state = query["state"][0]
        self.assertTrue(state)
        self.assertEqual(query["code"], ["true"])
        auth._token_request = lambda fields: {
            "access_token": "at",
            "refresh_token": "rt",
        }
        callback = "https://platform.claude.com/oauth/code/callback"
        with self.assertRaises(ClaudeAuthError):
            auth.finish(f"{callback}?code=abc&state=stale")
        status = auth.finish(f"{callback}?code=abc&state={state}")
        self.assertTrue(status["signed_in"])

    def test_exchange_body_matches_cli_json_shape(self):
        # The token endpoint 400s form bodies and JSON bodies without `state`
        # ("Invalid request format"); the CLI sends exactly these fields.
        auth = ClaudeAuth(pathlib.Path(tempfile.mkdtemp()) / "credentials.json")
        url = auth.login_start()["url"]
        state = parse_qs(urlparse(url).query)["state"][0]
        seen: dict = {}
        auth._token_request = lambda fields: (
            seen.update(fields)
            or {
                "access_token": "at",
                "refresh_token": "rt",
            }
        )
        callback = "https://platform.claude.com/oauth/code/callback"
        auth.finish(f"{callback}?code=abc&state={state}")
        self.assertEqual(
            set(seen),
            {
                "grant_type",
                "code",
                "redirect_uri",
                "client_id",
                "code_verifier",
                "state",
            },
        )
        self.assertEqual(seen["grant_type"], "authorization_code")
        self.assertEqual(seen["code"], "abc")
        self.assertEqual(seen["state"], state)
        self.assertNotIn("scope", seen)

    def test_refresh_uses_granted_scope(self):
        root = pathlib.Path(tempfile.mkdtemp())
        path = root / "credentials.json"
        auth = ClaudeAuth(path)
        path.write_text(
            '{"access_token":"at","refresh_token":"rt","expires_at":1,'
            '"scopes":["user:profile","user:inference"]}'
        )
        seen: dict = {}
        auth._token_request = lambda fields: (
            seen.update(fields)
            or {
                "access_token": "at2",
                "expires_in": 3600,
            }
        )
        auth.access_token(force_refresh=True)
        self.assertEqual(seen["grant_type"], "refresh_token")
        self.assertEqual(seen["refresh_token"], "rt")
        self.assertEqual(seen["scope"], "user:profile user:inference")

    def test_login_survives_restart(self):
        root = pathlib.Path(tempfile.mkdtemp())
        started = ClaudeAuth(root / "credentials.json").login_start()["url"]
        state = parse_qs(urlparse(started).query)["state"][0]
        restarted = ClaudeAuth(root / "credentials.json")
        restarted._token_request = lambda fields: {
            "access_token": "at",
            "refresh_token": "rt",
        }
        status = restarted.finish(
            f"https://platform.claude.com/oauth/code/callback?code=abc&state={state}"
        )
        self.assertTrue(status["signed_in"])

    def test_extracts_code_from_callback_url_or_raw_code(self):
        self.assertEqual(ClaudeAuth._extract_code("abc123"), "abc123")
        self.assertEqual(
            ClaudeAuth._extract_code("9XBbSsSVxPuq#zaSL6zlRlxRu25_BaPaQddXxqt6Fg2Ge"),
            "9XBbSsSVxPuq",
        )
        self.assertEqual(ClaudeAuth._extract_state("9XBbSsSVxPuq#st-ate"), "st-ate")
        self.assertEqual(
            ClaudeAuth._extract_state("https://x/callback?code=abc&state=st-ate"),
            "st-ate",
        )
        self.assertEqual(ClaudeAuth._extract_state("abc123"), "")
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

    def test_status_never_exposes_secrets(self):
        # The dashboard and /api/status must not leak tokens or expiries;
        # status() only reports sign-in state and the subscription tier.
        auth = ClaudeAuth(pathlib.Path(tempfile.mkdtemp()) / "credentials.json")
        status = auth.status().payload()
        self.assertFalse(status["signed_in"])
        # Seed a full credential set and re-check.
        path = auth.path
        path.write_text(
            '{"access_token":"at","refresh_token":"rt","expires_at":4102444800,'
            '"refresh_expires_at":4102444800,"scopes":["user:inference"],'
            '"subscription_type":"pro"}'
        )
        path.chmod(0o600)
        status = auth.status().payload()
        self.assertTrue(status["signed_in"])
        self.assertEqual(status["account"], "pro")
        for secret in (
            "access_token",
            "refresh_token",
            "refresh_token_masked",
            "expires_at",
            "refresh_expires_at",
            "scopes",
        ):
            self.assertNotIn(secret, status)


if __name__ == "__main__":
    unittest.main()
