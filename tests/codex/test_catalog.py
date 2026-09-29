"""Codex catalog entries in the shared model-listing shape."""

import unittest

from llm_local_proxy.providers.codex.catalog import model_info as _model_info


class CodexCatalogTest(unittest.TestCase):
    def test_codex_model_info_has_the_listing_shape(self):
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


if __name__ == "__main__":
    unittest.main()
