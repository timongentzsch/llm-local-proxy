import io
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from threading import Lock
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from llm_local_proxy.config import load
from llm_local_proxy.errors import RequestError
from llm_local_proxy.providers import Provider
from llm_local_proxy.providers.catalog import match_model
from llm_local_proxy.providers.claude import Claude
from llm_local_proxy.providers.claude.catalog import model_info as _claude_model_info
from llm_local_proxy.providers.claude.upstream import (
    ClaudeUpstreamError,
    _upstream_error,
)
from llm_local_proxy.providers.codex import Codex
from llm_local_proxy.providers.codex.catalog import model_info as _model_info
from llm_local_proxy.providers.codex.upstream import UpstreamError
from llm_local_proxy.providers.pool import Account, AccountPool, AccountStore
from llm_local_proxy.service import Service
from llm_local_proxy.status import AccountStatus, ProviderStatus


class ServiceWiringTest(unittest.TestCase):
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

            def fake_claude_upstream(auth, timeout, usage_path=None, tokens_path=None):
                seen["claude_tokens"].append(tokens_path)
                return SimpleNamespace(
                    ledger=SimpleNamespace(windows=dict),
                    usage=SimpleNamespace(get=lambda: None),
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


class ServerTest(unittest.TestCase):
    def test_model_info_matches_openrouter_shape(self):
        model = _model_info(
            {
                "model": "acme-gpt-1",
                "displayName": "Acme GPT 1",
                "inputModalities": ["text", "image"],
                "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [{"reasoningEffort": "low"}],
                "isDefault": True,
            }
        )
        self.assertEqual(model["name"], "Acme GPT 1")
        self.assertEqual(model["architecture"]["input_modalities"], ["text", "image"])
        self.assertEqual(model["default_parameters"]["reasoning_effort"], "medium")
        self.assertEqual(model["supported_reasoning_efforts"], ["low"])
        self.assertNotIn("context_length", model)
        model = _model_info({"model": "acme-gpt-1"}, {"acme-gpt-1": 272000})
        self.assertEqual(model["context_length"], 272000)
        self.assertEqual(model["supported_reasoning_efforts"], [])

    def test_codex_catalog_omits_efforts_the_transport_rejects(self):
        model = _model_info(
            {
                "model": "gpt-test",
                "defaultReasoningEffort": "ultra",
                "supportedReasoningEfforts": [
                    {"reasoningEffort": "high"},
                    {"reasoningEffort": "max"},
                    {"reasoningEffort": "ultra"},
                    {"reasoningEffort": "future-tier"},
                ],
            },
            transport_efforts={"high", "max", "future-tier"},
        )
        self.assertEqual(
            model["supported_reasoning_efforts"], ["high", "max", "future-tier"]
        )
        self.assertIsNone(model["default_parameters"])

    def test_claude_model_info_matches_openrouter_shape(self):
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

    @staticmethod
    def _service():
        """A Service-shaped object wired with the real claude/codex match funcs."""
        service = SimpleNamespace(
            _lock=Lock(),
            config=SimpleNamespace(origin="http://127.0.0.1:8787"),
        )
        seen_claude: list[str] = []
        seen_codex: list[str] = []

        def claude_chat(canonical, body, session):
            seen_claude.append(canonical)
            return iter(()), {"provider": "claude"}

        def codex_chat(canonical, body, session):
            seen_codex.append(canonical)
            return iter(()), {"provider": "codex"}

        service._seen = (seen_claude, seen_codex)
        claude_models = [
            _claude_model_info({"id": "claude-fake-1", "name": "Claude Fake 1"})
        ]
        codex_models = [_model_info({"model": "acme-gpt-1"})]
        service.providers = [
            Provider(
                name="claude",
                match=lambda model: match_model(model, claude_models),
                chat=claude_chat,
                models=lambda: claude_models,
                status=lambda: ProviderStatus(signed_in=True),
                routes={"code": lambda body: {}, "usage": lambda body: {}},
            ),
            Provider(
                name="codex",
                match=lambda model: match_model(model, codex_models),
                chat=codex_chat,
                models=lambda: codex_models,
                status=ProviderStatus,
                routes={},
            ),
        ]
        service.route = MethodType(Service.route, service)
        service.provider = MethodType(Service.provider, service)
        return service

    def test_route_picks_the_provider_whose_catalog_claims_the_model(self):
        service = self._service()
        for model, provider, canonical in (
            ("claude-fake-1", "claude", "claude-fake-1"),
            ("anthropic/claude-fake-1", "claude", "claude-fake-1"),
            ("acme-gpt-1", "codex", "acme-gpt-1"),
        ):
            with self.subTest(model=model):
                routed, name = service.route(model)
                self.assertEqual((routed.name, name), (provider, canonical))

    def test_route_no_match_returns_none(self):
        service = self._service()
        # A model no live provider catalog claims makes route() return None.
        service.providers[1] = replace(service.providers[1], match=lambda model: None)
        self.assertIsNone(service.route("acme-gpt-1"))

    def test_models_merges_both_upstreams_without_deadlock(self):
        service = self._service()
        value = Service.models(service)
        ids = [model["id"] for model in value["data"]]
        self.assertIn("claude-fake-1", ids)
        self.assertIn("acme-gpt-1", ids)

    def test_refresh_drops_every_provider_catalog_cache(self):
        service = self._service()
        forgotten = []
        service.providers = [
            replace(provider, forget=lambda name=provider.name: forgotten.append(name))
            for provider in service.providers
        ]
        Service.models(service)
        self.assertEqual(forgotten, [])
        Service.models(service, refresh=True)
        self.assertEqual(forgotten, ["claude", "codex"])

    def test_status_reports_one_uniform_card_per_provider(self):
        service = self._service()
        service.status = MethodType(Service.status, service)
        value = service.status()
        # One client base url per registered dialect, derived from the registry.
        self.assertEqual(
            {dialect["name"]: dialect["base_url"] for dialect in value["dialects"]},
            {
                "openai": "http://127.0.0.1:8787/openai/v1",
                "anthropic": "http://127.0.0.1:8787/anthropic",
            },
        )
        cards = value["providers"]
        self.assertEqual([card["name"] for card in cards], ["claude", "codex"])
        # Every card carries the same keys, whatever the upstream shape is.
        fields = {"name", "routes", "signed_in", "error", "accounts"}
        for card in cards:
            self.assertEqual(set(card), fields)
        self.assertTrue(cards[0]["signed_in"])
        self.assertEqual(cards[0]["routes"], ["code", "usage"])
        self.assertFalse(cards[1]["signed_in"])

    def test_status_isolates_a_failing_provider(self):
        service = self._service()
        service.status = MethodType(Service.status, service)
        # Claude status raises; codex still contributes its card, and claude
        # degrades to an error card rather than disappearing.
        service.providers[0] = replace(
            service.providers[0],
            status=lambda: (_ for _ in ()).throw(
                ClaudeUpstreamError(502, "claude down")
            ),
        )
        cards = {card["name"]: card for card in service.status()["providers"]}
        self.assertEqual(cards["claude"]["error"], "claude down")
        self.assertFalse(cards["claude"]["signed_in"])
        self.assertEqual(cards["codex"]["error"], "")

    def test_models_isolates_a_failing_provider(self):
        service = self._service()
        service.providers[0] = replace(
            service.providers[0],
            models=lambda: (_ for _ in ()).throw(
                ClaudeUpstreamError(502, "catalog down")
            ),
        )
        value = Service.models(service)
        ids = [model["id"] for model in value["data"]]
        self.assertEqual(ids, ["acme-gpt-1"])


class MultiAccountCatalogTest(unittest.TestCase):
    class Auth:
        def __init__(self, account):
            self.account = account

        def signed_in(self):
            return True

        def hydrate_profile(self):
            pass

        def status(self):
            return AccountStatus(signed_in=True, account=self.account)

    class Client:
        def __init__(self, result):
            self.result = result
            self.calls = 0
            self.ledger = SimpleNamespace(windows=dict)
            self.usage = SimpleNamespace(limits=tuple, updated_at=lambda: None)

        def models(self):
            self.calls += 1
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    @classmethod
    def accounts(cls, first, second):
        return AccountPool(
            [
                Account("1", cls.Auth("one"), first),
                Account("2", cls.Auth("two"), second),
            ]
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
                stale, live = self.Client(error), self.Client(models)
                provider = service.__new__(service)
                provider.pool = self.accounts(stale, live)
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

    def test_claude_login_without_inference_scope_does_not_hide_the_catalog(self):
        # Live failure: a grant without user:inference signs in but every
        # Models call answers 403. It must fail over, not empty the catalog.
        body = (
            '{"type":"error","error":{"type":"permission_error","message":'
            '"OAuth token does not meet scope requirement any_of(user:inference)"}}'
        )
        narrow = self.Client(
            _upstream_error(
                urllib.error.HTTPError("u", 403, "", None, io.BytesIO(body.encode()))
            )
        )
        live = self.Client([{"id": "claude-live", "name": "Claude Live"}])
        claude = Claude.__new__(Claude)
        claude.pool = self.accounts(narrow, live)
        claude._lock = Lock()
        claude._catalog = None

        self.assertEqual(claude._live_catalog()[0]["id"], "claude-live")
        self.assertIn("scope requirement", claude.pool.account_error("1"))
        self.assertFalse(claude.status().accounts[0].signed_in)

    def test_claude_usage_refresh_survives_one_failing_account(self):
        class Pinged(self.Client):
            def ping_usage(self, model):
                if isinstance(self.result, Exception):
                    raise self.result
                return self.result

        claude = Claude.__new__(Claude)
        claude.pool = self.accounts(
            Pinged(ClaudeUpstreamError(429, "rate limited")), Pinged({"ok": 1})
        )
        claude._lock = Lock()
        claude._catalog = (float("inf"), [{"id": "claude-live", "name": "x"}])

        usage = claude.usage({})["usage"]
        self.assertEqual(usage["1"], {"error": "rate limited"})
        self.assertEqual(usage["2"], {"ok": 1})

        # A login already known to need reauthentication is not pinged again.
        claude.pool._mark_account_error("1", RuntimeError("expired"))
        self.assertEqual(set(claude.usage({})["usage"]), {"2"})

    def test_a_new_claude_login_starts_without_the_previous_logins_bars(self):
        cleared = []
        claude = Claude.__new__(Claude)
        claude.pool = self.accounts(self.Client([]), self.Client([]))
        claude.pool.get("2").client.usage.clear = lambda: cleared.append("2")
        claude.pool.get("2").auth.finish = lambda code: {"ok": True}
        claude._lock = Lock()
        claude._catalog = None

        claude.finish_login({"account": "2", "code": "abc"})
        self.assertEqual(cleared, ["2"])


if __name__ == "__main__":
    unittest.main()
