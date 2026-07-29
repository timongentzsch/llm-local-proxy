import unittest

from codex_local_proxy.server import _api_path, _model_info


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


if __name__ == "__main__":
    unittest.main()
