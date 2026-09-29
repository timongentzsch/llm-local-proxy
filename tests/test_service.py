"""The service aggregates providers: routing, the merged catalog, status."""

import unittest
from dataclasses import replace
from types import MethodType, SimpleNamespace

import mock_provider

from llm_local_proxy.errors import UpstreamError
from llm_local_proxy.service import Service
from llm_local_proxy.status import ProviderStatus


class ServiceTest(unittest.TestCase):
    @staticmethod
    def _service():
        """A Service-shaped object over two mock providers."""
        service = SimpleNamespace(
            config=SimpleNamespace(origin="http://127.0.0.1:8787")
        )
        service.providers = [
            mock_provider.provider(
                name="alpha", models=("alpha-1",), routes={"code": lambda body: {}}
            ),
            mock_provider.provider(
                name="beta", models=("beta-1",), status=ProviderStatus
            ),
        ]
        service.route = MethodType(Service.route, service)
        service.provider = MethodType(Service.provider, service)
        return service

    def test_route_picks_the_provider_whose_catalog_claims_the_model(self):
        service = self._service()
        for model, provider, canonical in (
            ("alpha-1", "alpha", "alpha-1"),
            ("vendor/alpha-1", "alpha", "alpha-1"),
            ("beta-1", "beta", "beta-1"),
        ):
            with self.subTest(model=model):
                routed, name = service.route(model)
                self.assertEqual((routed.name, name), (provider, canonical))

    def test_route_no_match_returns_none(self):
        service = self._service()
        # A model no live provider catalog claims makes route() return None.
        service.providers[1] = replace(service.providers[1], match=lambda model: None)
        self.assertIsNone(service.route("beta-1"))

    def test_models_merges_both_upstreams_without_deadlock(self):
        service = self._service()
        value = Service.models(service)
        ids = [model["id"] for model in value["data"]]
        self.assertIn("alpha-1", ids)
        self.assertIn("beta-1", ids)

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
        self.assertEqual(forgotten, ["alpha", "beta"])

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
        self.assertEqual([card["name"] for card in cards], ["alpha", "beta"])
        # Every card carries the same keys, whatever the upstream shape is.
        fields = {"name", "routes", "signed_in", "error", "accounts"}
        for card in cards:
            self.assertEqual(set(card), fields)
        self.assertTrue(cards[0]["signed_in"])
        self.assertEqual(cards[0]["routes"], ["code"])
        self.assertFalse(cards[1]["signed_in"])

    def test_status_isolates_a_failing_provider(self):
        service = self._service()
        service.status = MethodType(Service.status, service)
        # One provider's status raises; the other still contributes its card, and it
        # degrades to an error card rather than disappearing.
        service.providers[0] = replace(
            service.providers[0],
            status=lambda: (_ for _ in ()).throw(UpstreamError(502, "alpha down")),
        )
        cards = {card["name"]: card for card in service.status()["providers"]}
        self.assertEqual(cards["alpha"]["error"], "alpha down")
        self.assertFalse(cards["alpha"]["signed_in"])
        self.assertEqual(cards["beta"]["error"], "")

    def test_models_isolates_a_failing_provider(self):
        service = self._service()
        service.providers[0] = replace(
            service.providers[0],
            models=lambda: (_ for _ in ()).throw(UpstreamError(502, "catalog down")),
        )
        value = Service.models(service)
        ids = [model["id"] for model in value["data"]]
        self.assertEqual(ids, ["beta-1"])


if __name__ == "__main__":
    unittest.main()
