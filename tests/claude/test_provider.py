"""The Claude provider: catalog shape, account failover and usage bars."""

import io
import unittest
import urllib.error
from threading import Lock

from mock_provider import FakeClient, two_accounts

from llm_local_proxy.providers.claude import Claude
from llm_local_proxy.providers.claude.catalog import model_info as _claude_model_info
from llm_local_proxy.providers.claude.upstream import _upstream_error


class ClaudeProviderTest(unittest.TestCase):
    def test_claude_model_info_has_the_listing_shape(self):
        model = _claude_model_info(
            {
                "id": "claude-fake-1",
                "name": "Claude Fake 1",
                "created": 1784908800,
                "context_length": 1_000_000,
                "max_output_tokens": 128000,
                "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
            }
        )
        self.assertEqual(model["id"], "claude-fake-1")
        self.assertEqual(model["owned_by"], "anthropic")
        self.assertEqual(model["created"], 1784908800)
        self.assertEqual(model["context_length"], 1_000_000)
        self.assertEqual(model["default_parameters"]["max_tokens"], 128000)
        self.assertEqual(
            model["supported_reasoning_efforts"],
            ["low", "medium", "high", "xhigh", "max"],
        )
        model = _claude_model_info({"id": "claude-fake-1", "name": "Claude Fake 1"})
        self.assertEqual(model["supported_reasoning_efforts"], [])

    def test_claude_login_without_inference_scope_does_not_hide_the_catalog(self):
        # Live failure: a grant without user:inference signs in but every
        # Models call answers 403. It must fail over, not empty the catalog.
        body = (
            '{"type":"error","error":{"type":"permission_error","message":'
            '"OAuth token does not meet scope requirement any_of(user:inference)"}}'
        )
        narrow = FakeClient(
            _upstream_error(
                urllib.error.HTTPError("u", 403, "", None, io.BytesIO(body.encode()))
            )
        )
        live = FakeClient([{"id": "claude-live", "name": "Claude Live"}])
        claude = Claude.__new__(Claude)
        claude.pool = two_accounts(narrow, live)
        claude._lock = Lock()
        claude._catalog = None

        self.assertEqual(claude._live_catalog()[0]["id"], "claude-live")
        self.assertIn("scope requirement", claude.pool.account_error("1"))
        self.assertFalse(claude.status().accounts[0].signed_in)

    def test_claude_usage_is_not_read_for_a_login_awaiting_reauthentication(self):
        claude = Claude.__new__(Claude)
        claude.pool = two_accounts(FakeClient([]), FakeClient([]))
        claude.pool._mark_account_error("1", RuntimeError("expired"))

        claude.status()
        self.assertEqual(claude.pool.get("1").client.usage_reads, 0)
        self.assertEqual(claude.pool.get("2").client.usage_reads, 1)

    def test_a_new_claude_login_starts_without_the_previous_logins_bars(self):
        cleared = []
        claude = Claude.__new__(Claude)
        claude.pool = two_accounts(FakeClient([]), FakeClient([]))
        claude.pool.get("2").client.limits.clear = lambda: cleared.append("2")
        claude.pool.get("2").auth.finish = lambda code: {"ok": True}
        claude._lock = Lock()
        claude._catalog = None

        claude.finish_login({"account": "2", "code": "abc"})
        self.assertEqual(cleared, ["2"])


if __name__ == "__main__":
    unittest.main()
