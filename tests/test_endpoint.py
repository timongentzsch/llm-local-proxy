"""End-to-end HTTP: every dialect served over one socket by one provider.

The provider is a mock that replays scripted response events, so any
difference between the responses comes from the dialect and nothing else.
"""

from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import mock_provider

from llm_local_proxy.errors import UpstreamError
from llm_local_proxy.http.handler import make_handler
from llm_local_proxy.http.server import Server
from llm_local_proxy.ir import Finish, ToolCallArgs, ToolCallEnd, ToolCallStart, Usage
from llm_local_proxy.keys import KeyStore
from llm_local_proxy.providers.pool import account_id
from llm_local_proxy.service import Service

SESSIONS = []
CALLERS = []


def _reply(canonical, request):
    """The scripted answer, or the failure the model name asks for."""
    SESSIONS.append(request.session)
    CALLERS.append(request.caller)
    if canonical.startswith("mock-fail-"):
        raise UpstreamError(int(canonical.rsplit("-", 1)[1]), "upstream said no")
    if canonical == "mock-burst":
        raise OSError("disk fell over")
    if canonical == "mock-broken-tool":
        broken = '{"path":'
        return [
            ToolCallStart(0, "call_1", "read"),
            ToolCallArgs(0, broken),
            ToolCallEnd(0, "call_1", "read", broken),
            Usage(prompt=11, completion=3),
            Finish("tool_use"),
        ]
    return mock_provider.REPLY


def _service(api_key="", keys=None):
    provider = mock_provider.provider(
        models=("mock-1", "mock-fail-429", "mock-broken-tool", "mock-burst"),
        reply=_reply,
        routes={
            "accounts": lambda body: body,
            "login": lambda body: {"account": account_id(body)},
            "logout": lambda body: {"account": account_id(body)},
        },
        count_tokens=lambda canonical, request: {"input_tokens": 42},
    )
    # A provider whose upstream has no way to count.
    uncounted = mock_provider.provider(name="uncounted", models=("gpt-1",))
    catalog = {"object": "list", "data": provider.models()}
    service = SimpleNamespace(
        config=SimpleNamespace(
            api_key=api_key,
            host="127.0.0.1",
            origin="http://127.0.0.1:8787",
            public_url="https://proxy.example.ts.net",
        ),
        keys=keys or KeyStore(Path(tempfile.mkdtemp()) / "keys.json"),
        usage=lambda: {"alice": {"mock": {"5h": {"input": 7}}}, "master": {}},
        healthy=lambda: True,
        route=lambda model: (
            (uncounted, model) if model.startswith("gpt") else (provider, model)
        ),
        provider=lambda name: provider if name == "mock" else None,
        models=lambda refresh=False: catalog,
        status=lambda: {"providers": []},
    )
    service.base_urls = MethodType(Service.base_urls, service)
    return service


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

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        payload = json.dumps(body) if body is not None else None
        sent_headers = dict(headers or {})
        if payload:
            sent_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, payload, sent_headers)
        response = connection.getresponse()
        text = response.read().decode()
        connection.close()
        return response.status, text

    def test_broken_tool_arguments_fail_without_successful_completion(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                status, text = self.request(
                    "POST",
                    "/anthropic/v1/messages",
                    {
                        "model": "mock-broken-tool",
                        "max_tokens": 128,
                        "messages": [{"role": "user", "content": "read"}],
                        "stream": streaming,
                    },
                )
                self.assertEqual(status, 200 if streaming else 502)
                self.assertIn("tool call arguments must be a JSON object", text)
                self.assertNotIn('"type": "message_stop"', text)
                self.assertNotIn('"type": "content_block_stop"', text)

    # -- streaming --------------------------------------------------------

    def test_chat_completions_stream(self):
        status, text = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": "mock-1",
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
                "model": "mock-1",
                "stream": True,
                "store": False,
                "input": "hi",
            },
        )
        self.assertEqual(status, 200)
        # As the Responses API streams: every frame named after its type, and
        # no Chat Completions sentinel.
        self.assertNotIn("[DONE]", text)
        lines = text.splitlines()
        payloads = [
            json.loads(line[len("data: ") :])
            for line in lines
            if line.startswith("data: {")
        ]
        names = [line[len("event: ") :] for line in lines if line.startswith("event: ")]
        self.assertEqual(names, [payload["type"] for payload in payloads])
        self.assertEqual(payloads[0]["type"], "response.created")
        self.assertEqual(payloads[-1]["type"], "response.completed")
        self.assertIn("response.output_text.delta", [p["type"] for p in payloads])

    def test_messages_stream(self):
        status, text = self.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": "mock-1",
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
                "model": "mock-1",
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
            {"model": "mock-1", "store": False, "input": "hi"},
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
                "model": "mock-1",
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

    def test_claude_code_session_header_preserves_affinity(self):
        SESSIONS.clear()
        status, _ = self.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": "mock-1",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
            {"X-Claude-Code-Session-Id": "claude-session-42"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(SESSIONS, ["claude-session-42"])

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

    def test_account_auth_routes_require_an_explicit_slot(self):
        for route in ("login", "logout"):
            with self.subTest(route=route):
                status, text = self.request("POST", f"/api/mock/{route}", {})
                self.assertEqual(status, 400)
                self.assertIn("account is required", text)

    def test_account_slots_can_be_managed_over_http(self):
        status, text = self.request("POST", "/api/mock/accounts", {"action": "add"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text), {"action": "add"})

    # -- mounts -----------------------------------------------------------

    def test_legacy_paths_mirror_the_openai_mount(self):
        """Every route reachable at /openai/... is reachable bare, identically.

        Configs written before the prefixes existed point at /v1, so the two
        must not drift apart.
        """
        chat = {
            "model": "mock-1",
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
                {"model": "mock-1", "store": False, "input": "hi"},
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
                "model": "mock-1",
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
            {"model": "mock-1", "messages": []},
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
            # Browsers request these without credentials; neither may 401.
            for path, expected in (("/healthz", 200), ("/favicon.ico", 204)):
                with self.subTest(path=path):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", port, timeout=10
                    )
                    connection.request("GET", path)
                    response = connection.getresponse()
                    response.read()
                    connection.close()
                    self.assertEqual(response.status, expected)
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
                        "model": f"mock-fail-{status}",
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
                "model": "mock-burst",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        self.assertEqual(code, 502)
        self.assertIn("disk fell over", json.loads(text)["error"]["message"])

    def test_unknown_routes_and_methods_are_refused(self):
        self.assertEqual(self.request("GET", "/v1/nope")[0], 404)
        self.assertEqual(self.request("POST", "/v1/nope", {})[0], 404)
        self.assertEqual(self.request("GET", "/api/mock/nope")[0], 404)
        self.assertEqual(self.request("POST", "/api/mock/nope", {})[0], 404)

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


class KeyAccessTest(unittest.TestCase):
    """Named keys on the admin listener and on the public one."""

    MASTER = "m" * 32

    @classmethod
    def setUpClass(cls):
        keys = KeyStore(Path(tempfile.mkdtemp()) / "keys.json")
        cls.alice = keys.add("alice")
        service = _service(api_key=cls.MASTER, keys=keys)
        cls.servers = {}
        for public in (False, True):
            server = Server(("127.0.0.1", 0), make_handler(service, public=public))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            cls.servers[public] = server

    @classmethod
    def tearDownClass(cls):
        for server in cls.servers.values():
            server.shutdown()
            server.server_close()

    def request(self, public, method, path, key, body=None):
        port = self.servers[public].server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        payload = json.dumps(body) if body is not None else None
        if payload:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, payload, headers)
        response = connection.getresponse()
        text = response.read().decode()
        connection.close()
        return response.status, text

    def test_a_named_key_uses_the_model_api_on_both_listeners(self):
        for public in (False, True):
            with self.subTest(public=public):
                CALLERS.clear()
                status, _ = self.request(
                    public,
                    "POST",
                    "/v1/chat/completions",
                    self.alice,
                    {
                        "model": "mock-1",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(CALLERS, ["alice"])
                self.assertEqual(
                    self.request(public, "GET", "/v1/models", self.alice)[0], 200
                )

    def test_a_named_key_sees_only_its_own_dashboard(self):
        for public, origin in (
            (False, "http://127.0.0.1:8787"),
            (True, "https://proxy.example.ts.net"),
        ):
            with self.subTest(public=public):
                status, text = self.request(public, "GET", "/api/me", self.alice)
                self.assertEqual(status, 200)
                me = json.loads(text)
                self.assertEqual((me["name"], me["role"]), ("alice", "user"))
                self.assertEqual(me["usage"], {"mock": {"5h": {"input": 7}}})
                self.assertTrue(me["dialects"][0]["base_url"].startswith(origin))
                for path in ("/api/status", "/api/keys"):
                    code = self.request(public, "GET", path, self.alice)[0]
                    self.assertEqual(code, 404 if public else 403)
                code = self.request(
                    public,
                    "POST",
                    "/api/keys",
                    self.alice,
                    {"action": "add", "name": "x"},
                )[0]
                self.assertEqual(code, 404 if public else 403)

    def test_the_public_listener_refuses_the_master_key_and_admin_routes(self):
        self.assertEqual(self.request(True, "GET", "/v1/models", self.MASTER)[0], 401)
        self.assertEqual(self.request(True, "GET", "/v1/models", "")[0], 401)
        self.assertEqual(self.request(True, "GET", "/healthz", self.alice)[0], 404)
        status, page = self.request(True, "GET", "/", "")
        self.assertEqual(status, 200)
        self.assertIn("authRequired=true", page)

    def test_the_master_key_manages_keys_on_the_admin_listener(self):
        status, text = self.request(
            False, "POST", "/api/keys", self.MASTER, {"action": "add", "name": "bob"}
        )
        self.assertEqual(status, 200)
        bob = json.loads(text)["key"]
        listed = json.loads(self.request(False, "GET", "/api/keys", self.MASTER)[1])
        self.assertIn({"name": "bob", "key": bob}, listed["keys"])
        self.assertEqual(self.request(True, "GET", "/v1/models", bob)[0], 200)
        self.request(
            False, "POST", "/api/keys", self.MASTER, {"action": "remove", "name": "bob"}
        )
        # Revocation takes effect on the next request.
        self.assertEqual(self.request(True, "GET", "/v1/models", bob)[0], 401)


if __name__ == "__main__":
    unittest.main()
