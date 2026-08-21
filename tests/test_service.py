import tempfile
import unittest
from pathlib import Path
from threading import Lock
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from llm_local_proxy.config import load
from llm_local_proxy.providers import Provider
from llm_local_proxy.providers.claude.catalog import claude_model_name
from llm_local_proxy.providers.claude.catalog import model_info as _claude_model_info
from llm_local_proxy.providers.claude.upstream import ClaudeUpstreamError
from llm_local_proxy.providers.codex.catalog import model_info as _model_info
from llm_local_proxy.service import Service
from llm_local_proxy.status import ProviderStatus


class ServiceWiringTest(unittest.TestCase):
    def test_upstreams_get_tokens_paths(self):
        # Regression guard: the token ledgers must be persisted to disk next
        # to the config, otherwise totals reset on every restart.
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                'host="127.0.0.1"\nport=8799\napi_key="123456789012345678901234"\n'
            )
            config_path.chmod(0o600)
            config = load(config_path)
            seen = {}

            def fake_upstream(app, timeout, tokens_path=None):
                seen["codex_tokens"] = tokens_path
                return SimpleNamespace(ledger=SimpleNamespace(windows=dict))

            def fake_claude_upstream(auth, timeout, usage_path=None, tokens_path=None):
                seen["claude_tokens"] = tokens_path
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
                seen["codex_tokens"], Path(directory) / "codex-tokens.json"
            )
            self.assertEqual(
                seen["claude_tokens"], Path(directory) / "claude-tokens.json"
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
        service.providers = [
            Provider(
                name="claude",
                auth=claude_auth,
                login_flow="paste_code",
                match=claude_model_name,
                chat=claude_chat,
                models=lambda: [
                    _claude_model_info({"id": "claude-fake-1", "name": "Claude Fake 1"})
                ],
                status=claude_auth.status,
                routes={"code": lambda body: {}, "usage": lambda body: {}},
            ),
            Provider(
                name="codex",
                auth=codex_auth,
                login_flow="device_code",
                match=lambda model: None if claude_model_name(model) else model,
                chat=codex_chat,
                models=lambda: [_model_info({"model": "acme-gpt-1"})],
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

    def test_route_sends_other_models_to_codex_fallback(self):
        service = self._service()
        provider, canonical = service.route("acme-gpt-1")
        self.assertEqual(provider.name, "codex")
        self.assertEqual(canonical, "acme-gpt-1")

    def test_route_no_match_returns_none(self):
        service = self._service()
        # claude only claims claude-*; codex claims everything else, so a
        # model neither claims makes route() return None.
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


if __name__ == "__main__":
    unittest.main()
