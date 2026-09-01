import tempfile
import unittest
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
from llm_local_proxy.providers.claude.upstream import ClaudeUpstreamError
from llm_local_proxy.providers.codex import Codex
from llm_local_proxy.providers.codex.catalog import model_info as _model_info
from llm_local_proxy.providers.codex.upstream import UpstreamError
from llm_local_proxy.providers.pool import Account, AccountPool, AccountStore
from llm_local_proxy.service import Service
from llm_local_proxy.status import ProviderStatus


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
            config=SimpleNamespace(
                base_url="http://127.0.0.1:8787/v1",
                origin="http://127.0.0.1:8787",
            ),
        )
        seen_claude: list[str] = []
        seen_codex: list[str] = []

        def claude_chat(canonical, body, session):
            seen_claude.append(canonical)
            return iter(()), {"provider": "claude"}

        def codex_chat(canonical, body, session):
            seen_codex.append(canonical)
            return iter(()), {"provider": "codex"}

        codex_auth = SimpleNamespace(status=ProviderStatus)
        claude_auth = SimpleNamespace(
            signed_in=lambda: True,
            status=lambda: ProviderStatus(signed_in=True, account="pro"),
        )
        service._seen = (seen_claude, seen_codex)
        claude_models = [
            _claude_model_info({"id": "claude-fake-1", "name": "Claude Fake 1"})
        ]
        codex_models = [_model_info({"model": "acme-gpt-1"})]
        service.providers = [
            Provider(
                name="claude",
                auth=claude_auth,
                login_flow="paste_code",
                match=lambda model: match_model(model, claude_models),
                chat=claude_chat,
                models=lambda: claude_models,
                status=claude_auth.status,
                routes={"code": lambda body: {}, "usage": lambda body: {}},
            ),
            Provider(
                name="codex",
                auth=codex_auth,
                login_flow="device_code",
                match=lambda model: match_model(model, codex_models),
                chat=codex_chat,
                models=lambda: codex_models,
                status=codex_auth.status,
                routes={},
            ),
        ]
        service.route = MethodType(Service.route, service)
        service.provider = MethodType(Service.provider, service)
        return service

    def test_route_sends_claude_models_to_claude(self):
        service = self._service()
        provider, canonical = service.route("claude-fake-1")
        self.assertEqual(provider.name, "claude")
        self.assertEqual(canonical, "claude-fake-1")

    def test_route_strips_provider_prefix_for_claude(self):
        service = self._service()
        provider, canonical = service.route("anthropic/claude-fake-1")
        self.assertEqual(provider.name, "claude")
        self.assertEqual(canonical, "claude-fake-1")

    def test_route_sends_catalogued_models_to_codex(self):
        service = self._service()
        provider, canonical = service.route("acme-gpt-1")
        self.assertEqual(provider.name, "codex")
        self.assertEqual(canonical, "acme-gpt-1")

    def test_route_no_match_returns_none(self):
        service = self._service()
        # A model no live provider catalog claims makes route() return None.
        service.providers[1] = Provider(
            name="codex",
            auth=service.providers[1].auth,
            login_flow="device_code",
            match=lambda model: None,
            chat=service.providers[1].chat,
            models=list,
            status=ProviderStatus,
            routes={},
        )
        self.assertIsNone(service.route("acme-gpt-1"))

    def test_provider_lookup_by_name(self):
        service = self._service()
        self.assertEqual(service.provider("claude").name, "claude")
        self.assertEqual(service.provider("codex").name, "codex")
        self.assertIsNone(service.provider("gemini"))

    def test_models_merges_both_upstreams_without_deadlock(self):
        service = self._service()
        service._models = None
        value = Service.models(service)
        ids = [model["id"] for model in value["data"]]
        self.assertIn("claude-fake-1", ids)
        self.assertIn("acme-gpt-1", ids)

    def test_status_reports_one_uniform_card_per_provider(self):
        service = self._service()
        service.status = MethodType(Service.status, service)
        value = service.status()
        self.assertEqual(value["base_url"], "http://127.0.0.1:8787/v1")
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
        fields = {
            "name",
            "login_flow",
            "routes",
            "signed_in",
            "account",
            "limits",
            "tokens",
            "updated_at",
            "error",
            "accounts",
        }
        for card in cards:
            self.assertEqual(set(card), fields)
        self.assertEqual(cards[0]["account"], "pro")
        self.assertEqual(cards[0]["routes"], ["code", "usage"])
        self.assertFalse(cards[1]["signed_in"])

    def test_status_isolates_a_failing_provider(self):
        service = self._service()
        service.status = MethodType(Service.status, service)
        # Claude status raises; codex still contributes its card, and claude
        # degrades to an error card rather than disappearing.
        service.providers[0] = Provider(
            name="claude",
            auth=service.providers[0].auth,
            login_flow="paste_code",
            match=service.providers[0].match,
            chat=service.providers[0].chat,
            models=service.providers[0].models,
            status=lambda: (_ for _ in ()).throw(
                ClaudeUpstreamError(502, "claude down")
            ),
            routes={},
        )
        cards = {card["name"]: card for card in service.status()["providers"]}
        self.assertEqual(cards["claude"]["error"], "claude down")
        self.assertFalse(cards["claude"]["signed_in"])
        self.assertEqual(cards["codex"]["error"], "")

    def test_models_isolates_a_failing_provider(self):
        service = self._service()
        service._models = None
        service.providers[0] = Provider(
            name="claude",
            auth=service.providers[0].auth,
            login_flow="paste_code",
            match=service.providers[0].match,
            chat=service.providers[0].chat,
            models=lambda: (_ for _ in ()).throw(
                ClaudeUpstreamError(502, "catalog down")
            ),
            status=service.providers[0].status,
            routes={},
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
            return ProviderStatus(signed_in=True, account=self.account)

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

    def test_claude_catalog_rotates_past_stale_accounts(self):
        stale = self.Client(
            ClaudeUpstreamError(400, "refresh token invalid", account_unavailable=True)
        )
        live = self.Client([{"id": "claude-live", "name": "Claude Live"}])
        claude = Claude.__new__(Claude)
        claude.pool = self.accounts(stale, live)
        claude._lock = Lock()
        claude._catalog = None

        self.assertEqual(claude._live_catalog()[0]["id"], "claude-live")
        self.assertIn("refresh token invalid", claude.pool.account_error("1"))
        claude._catalog = None
        self.assertEqual(claude._live_catalog()[0]["id"], "claude-live")
        self.assertEqual((stale.calls, live.calls), (1, 2))

        accounts = claude.status().accounts
        self.assertFalse(accounts[0].signed_in)
        self.assertIn("reauthentication required", accounts[0].error)
        self.assertTrue(accounts[1].signed_in)

    def test_codex_catalog_uses_the_same_rotating_discovery(self):
        stale = self.Client(
            UpstreamError(401, "refresh failed", account_unavailable=True)
        )
        live = self.Client([{"id": "gpt-live"}])
        codex = Codex.__new__(Codex)
        codex.pool = self.accounts(stale, live)
        codex._lock = Lock()
        codex._catalog = None
        codex._catalog_from = lambda client: client.models()

        self.assertEqual(codex._live_catalog(), [{"id": "gpt-live"}])
        self.assertIn("refresh failed", codex.pool.account_error("1"))
        codex._catalog = None
        self.assertEqual(codex._live_catalog(), [{"id": "gpt-live"}])
        self.assertEqual((stale.calls, live.calls), (1, 2))

        accounts = codex.status().accounts
        self.assertFalse(accounts[0].signed_in)
        self.assertIn("reauthentication required", accounts[0].error)
        self.assertTrue(accounts[1].signed_in)


if __name__ == "__main__":
    unittest.main()
