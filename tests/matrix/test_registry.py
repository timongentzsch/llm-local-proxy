"""The real provider registry: wiring, slots and catalog failover for each."""

import tempfile
import unittest
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from unittest.mock import patch

from mock_provider import FakeClient, two_accounts

from llm_local_proxy.config import load
from llm_local_proxy.errors import RequestError
from llm_local_proxy.providers.claude import Claude
from llm_local_proxy.providers.claude.upstream import ClaudeUpstreamError
from llm_local_proxy.providers.codex import Codex
from llm_local_proxy.providers.codex.upstream import UpstreamError
from llm_local_proxy.providers.pool import AccountStore
from llm_local_proxy.service import Service


class RegistryTest(unittest.TestCase):
    def test_account_slots_change_live_without_a_configured_count(self):
        class FakeApp:
            def __init__(self, *_):
                pass

            def call(self, method, params=None):
                return {}

            def alive(self):
                return True

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                'host="127.0.0.1"\nport=8799\napi_key=""\n'
                f'codex_home="{directory}/codex"\n'
            )
            config_path.chmod(0o600)
            with patch(
                "llm_local_proxy.providers.codex.AppServer", side_effect=FakeApp
            ):
                service = Service(load(config_path))
                for provider in service.providers:
                    self.assertEqual(provider.status().accounts, ())
                    added = provider.routes["accounts"]({"action": "add"})
                    self.assertEqual(len(provider.status().accounts), 1)
                    with self.assertRaisesRegex(
                        RequestError, "existing unsigned account"
                    ):
                        provider.routes["accounts"]({"action": "add"})
                    provider.routes["accounts"](
                        {"action": "remove", "account": added["account"]}
                    )
                    self.assertEqual(provider.status().accounts, ())
                service.close()

    def test_upstreams_get_tokens_paths(self):
        # Regression guard: the token ledgers must be persisted to disk next
        # to the config, otherwise totals reset on every restart.
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                'host="127.0.0.1"\nport=8799\napi_key="123456789012345678901234"\n'
            )
            config_path.chmod(0o600)
            for provider in ("codex", "claude"):
                store = AccountStore(Path(directory), provider)
                store.add()
                store.add()
            config = load(config_path)
            seen = {"codex_tokens": [], "claude_tokens": []}

            def fake_upstream(app, timeout, tokens_path=None):
                seen["codex_tokens"].append(tokens_path)
                return SimpleNamespace(ledger=SimpleNamespace(windows=dict))

            def fake_claude_upstream(auth, timeout, tokens_path=None):
                seen["claude_tokens"].append(tokens_path)
                return SimpleNamespace(
                    ledger=SimpleNamespace(windows=dict),
                    limits=SimpleNamespace(current=lambda: ((), None)),
                )

            with (
                patch("llm_local_proxy.providers.codex.AppServer"),
                patch(
                    "llm_local_proxy.providers.codex.Upstream",
                    side_effect=fake_upstream,
                ),
                patch(
                    "llm_local_proxy.providers.claude.ClaudeUpstream",
                    side_effect=fake_claude_upstream,
                ),
                patch("llm_local_proxy.providers.claude.ClaudeAuth"),
            ):
                Service(config)
            self.assertEqual(
                seen["codex_tokens"],
                [
                    Path(directory) / "accounts/codex/1/tokens.json",
                    Path(directory) / "accounts/codex/2/tokens.json",
                ],
            )
            self.assertEqual(
                seen["claude_tokens"],
                [
                    Path(directory) / "accounts/claude/1/tokens.json",
                    Path(directory) / "accounts/claude/2/tokens.json",
                ],
            )

    def test_catalog_rotates_past_stale_accounts(self):
        cases = [
            (
                Claude,
                ClaudeUpstreamError(
                    400, "refresh token invalid", account_unavailable=True
                ),
                [{"id": "claude-live", "name": "Claude Live"}],
            ),
            (
                Codex,
                UpstreamError(401, "refresh failed", account_unavailable=True),
                [{"id": "gpt-live"}],
            ),
        ]
        for service, error, models in cases:
            with self.subTest(service=service.__name__):
                stale, live = FakeClient(error), FakeClient(models)
                provider = service.__new__(service)
                provider.pool = two_accounts(stale, live)
                provider._lock = Lock()
                provider._catalog = None
                if service is Codex:
                    provider.fetch_catalog = lambda account: account.client.models()

                self.assertEqual(provider._live_catalog(), models)
                self.assertIn(str(error), provider.pool.account_error("1"))
                provider._catalog = None
                self.assertEqual(provider._live_catalog(), models)
                self.assertEqual((stale.calls, live.calls), (1, 2))

                accounts = provider.status().accounts
                self.assertFalse(accounts[0].signed_in)
                self.assertIn("reauthentication required", accounts[0].error)
                self.assertTrue(accounts[1].signed_in)


if __name__ == "__main__":
    unittest.main()
