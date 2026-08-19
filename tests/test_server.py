import unittest
from threading import Event, Lock
from types import MethodType, SimpleNamespace

from codex_local_proxy.claude_upstream import ClaudeUpstreamError
from codex_local_proxy.server import (
    Service,
    _api_path,
    _claude_model_info,
    _model_info,
    _with_heartbeats,
)


class ServerTest(unittest.TestCase):
    def test_openrouter_api_prefix_is_an_alias(self):
        self.assertEqual(_api_path("/api/v1/chat/completions"), "/v1/chat/completions")
        self.assertEqual(_api_path("/api/status"), "/api/status")

    def test_model_info_matches_openrouter_shape(self):
        model = _model_info(
            {
                "model": "gpt-5.6-sol",
                "displayName": "GPT-5.6 Sol",
                "inputModalities": ["text", "image"],
                "defaultReasoningEffort": "medium",
                "supportedReasoningEfforts": [{"reasoningEffort": "low"}],
                "isDefault": True,
            }
        )
        self.assertEqual(model["name"], "GPT-5.6 Sol")
        self.assertEqual(model["architecture"]["input_modalities"], ["text", "image"])
        self.assertEqual(model["default_parameters"]["reasoning_effort"], "medium")
        self.assertEqual(model["supported_reasoning_efforts"], ["low"])

    def test_claude_model_info_matches_openrouter_shape(self):
        model = _claude_model_info(
            {
                "id": "claude-opus-5",
                "name": "Claude Opus 5",
                "created": 1784908800,
                "max_output_tokens": 128000,
                "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
            }
        )
        self.assertEqual(model["id"], "claude-opus-5")
        self.assertEqual(model["owned_by"], "anthropic")
        self.assertEqual(model["created"], 1784908800)
        self.assertEqual(model["default_parameters"]["max_tokens"], 128000)
        self.assertEqual(
            model["supported_reasoning_efforts"], ["low", "medium", "high", "xhigh", "max"]
        )

    @staticmethod
    def _service(signed_in, models=None):
        upstream = SimpleNamespace(
            models=lambda: models if models is not None else (_ for _ in ()).throw(
                ClaudeUpstreamError(502, "down")
            )
        )
        service = SimpleNamespace(
            claude_auth=SimpleNamespace(signed_in=lambda: signed_in),
            claude=upstream,
            _models_lock=Lock(),
            _claude_catalog=None,
        )
        service._load_claude_catalog = MethodType(Service._load_claude_catalog, service)
        service._claude_items = MethodType(Service._claude_items, service)
        service._claude_capability = MethodType(Service._claude_capability, service)
        return service

    def test_models_merges_both_upstreams_without_deadlock(self):
        service = self._service(
            True,
            [
                {
                    "id": "claude-opus-5",
                    "name": "Claude Opus 5",
                    "created": 1,
                    "max_output_tokens": 128000,
                }
            ],
        )
        service._models = None
        service.app = SimpleNamespace(
            call=lambda method, params: {"data": [{"model": "gpt-5.4"}]}
        )
        value = Service.models(service)
        ids = [model["id"] for model in value["data"]]
        self.assertIn("gpt-5.4", ids)
        self.assertIn("claude-opus-5", ids)

    def test_claude_items_prefers_live_catalog(self):
        service = self._service(
            True,
            [
                {
                    "id": "claude-sonnet-5",
                    "name": "Claude Sonnet 5",
                    "created": 1782000000,
                    "max_output_tokens": 128000,
                    "reasoning_efforts": ["low", "medium", "high"],
                }
            ],
        )
        items = Service._claude_items(service)
        self.assertEqual(items[0]["id"], "claude-sonnet-5")
        self.assertEqual(items[0]["default_parameters"]["max_tokens"], 128000)

    def test_claude_items_falls_back_when_not_signed_in(self):
        items = Service._claude_items(self._service(False))
        self.assertTrue(all(item["id"].startswith("claude-") for item in items))
        self.assertEqual(items[0]["id"], "claude-opus-5")

    def test_claude_items_falls_back_when_catalog_fails(self):
        items = Service._claude_items(self._service(True))
        self.assertEqual(items[0]["id"], "claude-opus-5")

    def test_heartbeat_while_upstream_is_silent(self):
        release = Event()

        def delayed():
            release.wait()
            yield {"type": "response.completed"}

        stream = _with_heartbeats(delayed(), interval=0.01)
        self.assertIsNone(next(stream))
        release.set()
        self.assertEqual(next(stream), {"type": "response.completed"})
        with self.assertRaises(StopIteration):
            next(stream)

    def test_upstream_exception_is_propagated(self):
        def broken():
            raise RuntimeError("upstream failed")
            yield

        stream = _with_heartbeats(broken(), interval=1)
        with self.assertRaisesRegex(RuntimeError, "upstream failed"):
            next(stream)


if __name__ == "__main__":
    unittest.main()
