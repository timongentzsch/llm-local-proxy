"""End-to-end HTTP: both dialects served over one socket by one provider.

This is the 2x2 matrix at the level a client actually sees it. The provider
below replays a canned Claude stream, so any difference between the two
responses comes from the dialect and nothing else.
"""

from __future__ import annotations

import http.client
import json
import threading
import unittest
from types import SimpleNamespace

from llm_local_proxy.http.handler import make_handler
from llm_local_proxy.http.server import Server
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.upstream import ClaudeUpstreamError
from llm_local_proxy.providers.reasoning import ReasoningCache

STREAM = [
    {
        "type": "message_start",
        "message": {"usage": {"input_tokens": 11, "output_tokens": 0}},
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 3},
    },
    {"type": "message_stop"},
]

MODELS = {
    "object": "list",
    "data": [{"id": "claude-sonnet-5", "name": "Claude Sonnet 5"}],
}


def _chat(canonical, request):
    """The canned stream, or the failure the model name asks for."""
    if canonical.startswith("claude-fail-"):
        raise ClaudeUpstreamError(int(canonical.rsplit("-", 1)[1]), "upstream said no")
    if canonical == "claude-burst":
        raise OSError("disk fell over")
    return iter(STREAM), ClaudeDecoder(ReasoningCache())


def _service():
    provider = SimpleNamespace(
        name="claude",
        routes={"accounts": lambda body: body},
        auth=SimpleNamespace(),
        chat=_chat,
        count_tokens=lambda canonical, request: {"input_tokens": 42},
    )
    # A provider whose upstream has no way to count, like Codex.
    uncounted = SimpleNamespace(
        name="codex", routes={}, auth=SimpleNamespace(), count_tokens=None
    )
    return SimpleNamespace(
        config=SimpleNamespace(api_key="", host="127.0.0.1"),
        app=SimpleNamespace(alive=lambda: True),
        healthy=lambda: True,
        route=lambda model: (
            (uncounted, model) if model.startswith("gpt") else (provider, model)
        ),
        provider=lambda name: provider if name == "claude" else None,
        models=lambda: MODELS,
        status=lambda: {"providers": []},
        invalidate_models=lambda: None,
    )


class EndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = Server(("127.0.0.1", 0), make_handler(_service()))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = json.dumps(body) if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        text = response.read().decode()
        connection.close()
        return response.status, text

    # -- streaming --------------------------------------------------------

    def test_chat_completions_stream(self):
        status, text = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "claude-sonnet-5",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(text.endswith("data: [DONE]\n\n"))
        self.assertNotIn("event:", text)
        first = json.loads(text.split("\n")[0][len("data: ") :])
        self.assertEqual(first["object"], "chat.completion.chunk")

    def test_responses_stream(self):
        status, text = self.request(
            "POST",
            "/v1/responses",
            {
                "model": "claude-sonnet-5",
                "stream": True,
                "store": False,
                "input": "hi",
            },
        )
        self.assertEqual(status, 200)
        self.assertTrue(text.endswith("data: [DONE]\n\n"))
        payloads = [
            json.loads(line[len("data: ") :])
            for line in text.splitlines()
            if line.startswith("data: {")
        ]
        self.assertEqual(payloads[0]["type"], "response.created")
        self.assertEqual(payloads[-1]["type"], "response.completed")
        self.assertIn("response.output_text.delta", [p["type"] for p in payloads])

    def test_messages_stream(self):
        status, text = self.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": "claude-sonnet-5",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(status, 200)
        # Named frames, and no Chat Completions sentinel.
        self.assertNotIn("[DONE]", text)
        names = [
            line[len("event: ") :]
            for line in text.splitlines()
            if line.startswith("event: ")
        ]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-1], "message_stop")
        self.assertIn("content_block_delta", names)
        # Every frame is named after the type in its own payload.
        payloads = [
            json.loads(line[len("data: ") :])
            for line in text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual([p["type"] for p in payloads], names)

    # -- non streaming ----------------------------------------------------

    def test_chat_completions_body(self):
        status, text = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "claude-sonnet-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(status, 200)
        body = json.loads(text)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Hello")

    def test_responses_body(self):
        status, text = self.request(
            "POST",
            "/v1/responses",
            {"model": "claude-sonnet-5", "store": False, "input": "hi"},
        )
        self.assertEqual(status, 200)
        body = json.loads(text)
        self.assertEqual(body["object"], "response")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["output"][0]["content"][0]["text"], "Hello")

    def test_messages_body(self):
        status, text = self.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": "claude-sonnet-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(status, 200)
        body = json.loads(text)
        self.assertEqual(body["type"], "message")
        self.assertEqual(body["content"], [{"type": "text", "text": "Hello"}])
        self.assertEqual(body["stop_reason"], "end_turn")
        self.assertEqual(body["usage"]["input_tokens"], 11)

    # -- dashboard --------------------------------------------------------

    def test_dashboard_is_served_with_the_auth_flag_substituted(self):
        status, text = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertNotIn("__AUTH_REQUIRED__", text)
        # This service fixture has no api key configured.
        self.assertIn("authRequired=false", text)

    def test_dashboard_persists_the_key_and_keeps_it_out_of_the_url(self):
        _, text = self.request("GET", "/")
        # Read once from the fragment, stored, then stripped from the address
        # bar so a refresh works without re-pasting it.
        self.assertIn("location.hash", text)
        self.assertIn("localStorage.setItem(STORE", text)
        self.assertIn("history.replaceState", text)
        # And a way back out again, deliberately and on rejection.
        self.assertIn("localStorage.removeItem(STORE", text)
        self.assertIn("status===401", text)
        # A fragment-only change is a same-document navigation; without the
        # listener the key is never re-read. Verified in a browser.
        self.assertIn('addEventListener("hashchange"', text)

    def test_dashboard_shows_copyable_login_url_without_opening_a_tab(self):
        _, text = self.request("GET", "/")
        self.assertIn('class="login-url"', text)
        self.assertIn('class="copy-login"', text)
        self.assertIn("navigator.clipboard.writeText(loginUrl.textContent)", text)
        self.assertNotIn("window.open(", text)

    def test_dashboard_builds_commands_for_all_supported_clients(self):
        _, text = self.request("GET", "/")
        self.assertIn("codex --model", text)
        self.assertIn("claude --model", text)
        self.assertIn("opencode --model", text)
        self.assertIn("OPENCODE_CONFIG_CONTENT", text)
        self.assertIn("model_providers.local_proxy", text)
        self.assertIn("CLAUDE_CODE_MAX_CONTEXT_TOKENS", text)
        self.assertIn('class="account-label"', text)
        self.assertIn('box.className="command-code"', text)
        self.assertIn('copy.className="command-copy"', text)
        self.assertIn('reset=setTimeout(()=>copy.textContent="copy",1500)', text)
        self.assertIn('class="add-account"', text)
        self.assertIn('class="remove" hidden', text)
        self.assertIn("/accounts`", text)

    def test_dashboard_has_structured_accessible_loading_skeletons(self):
        _, text = self.request("GET", "/")
        self.assertIn('class="provider skeleton-provider" aria-hidden="true"', text)
        self.assertIn('class="row command skeleton-command" aria-hidden="true"', text)
        self.assertIn('class="model skeleton-model" aria-hidden="true"', text)
        self.assertIn("prefers-reduced-motion:reduce", text)
        self.assertNotIn('class="remote"', text)

    def test_account_auth_routes_require_an_explicit_slot(self):
        for route in ("login", "logout"):
            with self.subTest(route=route):
                status, text = self.request("POST", f"/api/claude/{route}", {})
                self.assertEqual(status, 400)
                self.assertIn("account is required", text)

    def test_account_slots_can_be_managed_over_http(self):
        status, text = self.request("POST", "/api/claude/accounts", {"action": "add"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text), {"action": "add"})

    # -- mounts -----------------------------------------------------------

    def test_openai_prefix_and_bare_path_are_byte_identical(self):
        body = {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
        }

        def normalised(path):
            status, text = self.request("POST", path, body)
            value = json.loads(text)
            # Each response carries a fresh id and timestamp by design.
            value.pop("id"), value.pop("created")
            return status, value

        prefixed = normalised("/openai/v1/chat/completions")
        self.assertEqual(prefixed[0], 200)
        self.assertEqual(prefixed, normalised("/v1/chat/completions"))

    def test_legacy_paths_mirror_the_openai_mount(self):
        """Every route reachable at /openai/... is reachable bare, identically.

        Configs written before the prefixes existed point at /v1, so the two
        must not drift apart.
        """
        chat = {
            "model": "claude-sonnet-5",
            "messages": [{"role": "user", "content": "hi"}],
        }
        for method, path, body in (
            ("GET", "/v1/models", None),
            ("GET", "/v1/models/count", None),
            ("GET", "/api/status", None),
            ("GET", "/healthz", None),
            ("POST", "/v1/chat/completions", chat),
            (
                "POST",
                "/v1/responses",
                {"model": "claude-sonnet-5", "store": False, "input": "hi"},
            ),
        ):
            with self.subTest(route=f"{method} {path}"):
                legacy = self.request(method, path, body)
                prefixed = self.request(method, "/openai" + path, body)
                self.assertEqual(legacy[0], 200)
                self.assertEqual(legacy[0], prefixed[0])
                self.assertEqual(len(legacy[1]), len(prefixed[1]))

    def test_prefixes_do_not_cross_dialects(self):
        # The Anthropic mount has no Chat Completions route, and vice versa.
        status, _ = self.request(
            "POST", "/anthropic/v1/chat/completions", {"model": "m", "messages": []}
        )
        self.assertEqual(status, 404)
        status, _ = self.request(
            "POST", "/openai/v1/messages", {"model": "m", "messages": []}
        )
        self.assertEqual(status, 404)

    # -- token counting ---------------------------------------------------

    def test_count_tokens(self):
        status, text = self.request(
            "POST",
            "/anthropic/v1/messages/count_tokens",
            # No max_tokens: nothing is generated, so its schema omits it.
            {
                "model": "claude-sonnet-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text), {"input_tokens": 42})

    def test_count_tokens_is_404_when_the_upstream_cannot_count(self):
        # Better an honest 404 than a guess the client would trust.
        status, text = self.request(
            "POST",
            "/anthropic/v1/messages/count_tokens",
            {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(text)["error"]["type"], "not_found_error")

    def test_chat_completions_has_no_count_route(self):
        status, _ = self.request(
            "POST",
            "/v1/messages/count_tokens",
            {"model": "claude-sonnet-5", "messages": []},
        )
        self.assertEqual(status, 404)

    # -- auth -------------------------------------------------------------

    def test_every_mount_accepts_every_credential_header(self):
        keyed = _service()
        keyed.config = SimpleNamespace(api_key="secret", host="127.0.0.1")
        server = Server(("127.0.0.1", 0), make_handler(keyed))
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            for path in ("/v1/models", "/openai/v1/models", "/anthropic/v1/models"):
                for header in (
                    {"Authorization": "Bearer secret"},
                    {"x-api-key": "secret"},
                ):
                    with self.subTest(path=path, header=next(iter(header))):
                        connection = http.client.HTTPConnection(
                            "127.0.0.1", port, timeout=10
                        )
                        connection.request("GET", path, headers=header)
                        response = connection.getresponse()
                        status = response.status
                        response.read()
                        connection.close()
                        self.assertEqual(status, 200)
        finally:
            server.shutdown()
            server.server_close()

    # -- catalog and errors -----------------------------------------------

    def test_model_catalogs_differ_per_dialect(self):
        _, openai = self.request("GET", "/v1/models")
        _, anthropic = self.request("GET", "/anthropic/v1/models")
        self.assertEqual(json.loads(openai)["object"], "list")
        listing = json.loads(anthropic)
        self.assertEqual(listing["data"][0]["type"], "model")
        self.assertFalse(listing["has_more"])

    def test_errors_use_each_dialect_envelope(self):
        status, text = self.request(
            "POST", "/v1/chat/completions", {"model": "m", "messages": []}
        )
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(text)["error"]["type"], "proxy_error")

        status, text = self.request(
            "POST", "/anthropic/v1/messages", {"model": "m", "messages": []}
        )
        self.assertEqual(status, 400)
        body = json.loads(text)
        self.assertEqual(body["type"], "error")
        self.assertEqual(body["error"]["type"], "invalid_request_error")

    def test_an_upstream_failure_keeps_its_status(self):
        # What the upstream said is what the client is told: a rate limit must
        # not reach a client as a generic gateway error, or it will retry into
        # the same wall instead of backing off.
        for status in (429, 401, 529):
            with self.subTest(status=status):
                code, text = self.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": f"claude-fail-{status}",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(code, status)
                self.assertIn("upstream said", json.loads(text)["error"]["message"])

    def test_an_unexpected_failure_becomes_a_gateway_error(self):
        # An exception the proxy does not model is still not a 500 with a
        # stack trace: the client gets a bad gateway and the reason.
        code, text = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "claude-burst",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(code, 502)
        self.assertIn("disk fell over", json.loads(text)["error"]["message"])

    def test_unknown_routes_and_methods_are_refused(self):
        self.assertEqual(self.request("GET", "/v1/nope")[0], 404)
        self.assertEqual(self.request("POST", "/v1/nope", {})[0], 404)
        self.assertEqual(self.request("GET", "/api/claude/nope")[0], 404)
        self.assertEqual(self.request("POST", "/api/claude/nope", {})[0], 404)

    def test_malformed_json_is_a_client_error(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request(
            "POST",
            "/v1/chat/completions",
            "{not json",
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        response.read()
        connection.close()
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
