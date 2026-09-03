"""The Codex provider: Responses requests out, Responses events back."""

import base64
import io
import json
import pathlib
import tempfile
import threading
import time
import unittest

from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.ingress import parse
from llm_local_proxy.errors import RequestError
from llm_local_proxy.ir import Finish, HostedToolEvent, ToolCallStart
from llm_local_proxy.providers.codex.app_server import (
    AppServer,
    RpcError,
    _jwt_payload,
)
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build
from llm_local_proxy.providers.codex.upstream import (
    Upstream,
    UpstreamError,
    _effort_values,
)
from llm_local_proxy.providers.reasoning import ReasoningCache


def codex_request(body, cache, session="", reasoning_efforts=None):
    """The whole path a Chat Completions request takes to Codex.

    Which layer rejects a bad body — the dialect's ingress or the provider's
    renderer — is an implementation detail these tests should not pin.
    """
    return build(parse(body, session), cache, reasoning_efforts)


class ProtocolTest(unittest.TestCase):
    def test_response_format_becomes_a_responses_text_format(self):
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "additionalProperties": False,
        }
        body, _ = codex_request(
            {
                "model": "acme-gpt-1",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "city",
                        "schema": schema,
                        "strict": True,
                    },
                },
            },
            ReasoningCache(),
        )
        self.assertEqual(
            body["text"],
            {
                "format": {
                    "type": "json_schema",
                    "name": "city",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

    def test_transport_effort_probe_discovers_the_runtime_enum(self):
        values = _effort_values(
            "Invalid value: 'probe'. Supported values are: 'low', 'max', "
            "and 'future-tier'."
        )
        self.assertEqual(values, {"low", "max", "future-tier"})
        self.assertIsNone(_effort_values("some unrelated request error"))

    def test_transport_effort_probe_propagates_an_unusable_account(self):
        class StaleApp:
            def token(self, force_refresh=False):
                raise RpcError("refresh failed")

        upstream = Upstream(StaleApp(), timeout=5)
        with self.assertRaises(UpstreamError) as caught:
            upstream.reasoning_efforts("gpt-test")
        self.assertTrue(caught.exception.account_unavailable)

    def test_rejects_invalid_tools_and_multiple_choices(self):
        base = {"model": "acme-gpt-1", "messages": [{"role": "user", "content": "hi"}]}
        with self.assertRaises(RequestError):
            codex_request({**base, "tools": [None]}, ReasoningCache())
        with self.assertRaises(RequestError):
            codex_request({**base, "n": 2}, ReasoningCache())
        with self.assertRaises(RequestError):
            codex_request({**base, "temperature": 0}, ReasoningCache())
        codex_request(
            {
                **base,
                "temperature": 1,
                "top_p": 1,
                "stop": None,
                "logprobs": False,
                "response_format": {"type": "text"},
            },
            ReasoningCache(),
        )
        with self.assertRaisesRegex(RequestError, "reasoning_effort"):
            codex_request(
                {**base, "reasoning_effort": "ultra"},
                ReasoningCache(),
                reasoning_efforts=["high", "max"],
            )

    def test_does_not_inject_default_instructions(self):
        request, _ = codex_request(
            {"model": "acme-gpt-1", "messages": [{"role": "user", "content": "hi"}]},
            ReasoningCache(),
        )
        self.assertEqual(request["instructions"], "")

    def test_accepts_max_tokens_as_compatibility_hint(self):
        request, _ = codex_request(
            {
                "model": "acme-gpt-1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 200,
            },
            ReasoningCache(),
        )
        self.assertNotIn("max_output_tokens", request)

    def test_chat_tools_become_responses_items(self):
        cache = ReasoningCache()
        cache.put(["call_1"], [{"type": "reasoning", "encrypted_content": "secret"}])
        request, session = codex_request(
            {
                "model": "acme-gpt-1",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Read a file."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"a"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "hello",
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "reasoning_effort": "medium",
            },
            cache,
        )
        self.assertTrue(session.startswith("proxy-"))
        self.assertEqual(request["instructions"], "Be concise.")
        self.assertEqual(request["input"][1]["type"], "reasoning")
        self.assertEqual(request["input"][2]["type"], "function_call")
        self.assertEqual(request["input"][3]["type"], "function_call_output")
        self.assertEqual(request["tools"][0]["name"], "read_file")

    def test_openrouter_web_search_becomes_responses_tool(self):
        request, _ = codex_request(
            {
                "model": "acme-gpt-1",
                "messages": [{"role": "user", "content": "search"}],
                "tools": [
                    {
                        "type": "openrouter:web_search",
                        "parameters": {
                            "engine": "auto",
                            "search_context_size": "low",
                        },
                    }
                ],
            },
            ReasoningCache(),
        )
        self.assertEqual(
            request["tools"], [{"type": "web_search", "search_context_size": "low"}]
        )

    def test_response_tool_call_becomes_chat_chunk(self):
        cache = ReasoningCache()
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(cache))
        translator.feed(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "encrypted",
                    "summary": [],
                },
            }
        )
        chunks = translator.feed(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                },
            }
        )
        call = chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(
            translator.finish()[0]["choices"][0]["finish_reason"], "tool_calls"
        )
        self.assertEqual(cache.get(["call_1"])[0]["encrypted_content"], "encrypted")

    def test_usage_is_mapped(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 6,
                        "total_tokens": 16,
                        "input_tokens_details": {"cached_tokens": 4},
                        "output_tokens_details": {"reasoning_tokens": 2},
                    }
                },
            }
        )
        self.assertEqual(translator.result()["usage"]["prompt_tokens"], 10)
        self.assertEqual(
            translator.result()["usage"]["completion_tokens_details"][
                "reasoning_tokens"
            ],
            2,
        )

    def test_non_stream_result_keeps_reasoning(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": "Checked.",
            }
        )
        self.assertEqual(
            translator.result()["choices"][0]["message"]["reasoning_content"],
            "Checked.",
        )

    def test_web_search_citation_and_usage_are_mapped(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.output_item.added",
                "item": {"type": "web_search_call", "id": "ws_1"},
            }
        )
        chunks = translator.feed(
            {
                "type": "response.output_text.annotation.added",
                "annotation": {
                    "type": "url_citation",
                    "url": "https://example.com",
                    "title": "Example",
                    "start_index": 0,
                    "end_index": 7,
                },
            }
        )
        citation = chunks[0]["choices"][0]["delta"]["annotations"][0]
        self.assertEqual(citation["url_citation"]["url"], "https://example.com")
        translator.feed(
            {
                "type": "response.completed",
                "response": {"usage": {"input_tokens": 2, "output_tokens": 1}},
            }
        )
        result = translator.result()
        self.assertEqual(result["usage"]["server_tool_use"]["web_search_requests"], 1)
        self.assertEqual(result["choices"][0]["message"]["annotations"], [citation])

    def test_search_lifecycle_survives_as_hosted_tool_events(self):
        """The visible span of a provider-run search, not just its aftermath.

        Timing is the whole point: a client that only learns a search happened
        once the answer arrives has nothing to show for the wait.
        """
        decoder = CodexDecoder(ReasoningCache())
        phases = []
        for event in [
            {
                "type": "response.output_item.added",
                "item": {"type": "web_search_call", "id": "ws_1"},
            },
            {"type": "response.web_search_call.in_progress", "item_id": "ws_1"},
            {"type": "response.web_search_call.searching", "item_id": "ws_1"},
            {"type": "response.web_search_call.completed", "item_id": "ws_1"},
            # Responses says a search finished twice; a client must not see
            # two completions, and usage must not count two searches.
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                },
            },
        ]:
            for decoded in decoder.decode(event):
                self.assertIsInstance(decoded, HostedToolEvent)
                self.assertEqual((decoded.tool, decoded.id), ("web_search", "ws_1"))
                phases.append(decoded.phase)
        self.assertEqual(phases, ["started", "searching", "completed"])
        self.assertEqual(len(decoder.web_searches), 1)
        # A hosted search is not a round the client owes an answer to.
        finished = decoder.finish()
        self.assertFalse(any(isinstance(e, ToolCallStart) for e in finished))
        self.assertEqual(
            [e.reason for e in finished if isinstance(e, Finish)], ["end_turn"]
        )

    def test_independent_searches_are_tracked_apart(self):
        decoder = CodexDecoder(ReasoningCache())
        for search_id in ("ws_1", "ws_2"):
            decoder.decode(
                {
                    "type": "response.output_item.added",
                    "item": {"type": "web_search_call", "id": search_id},
                }
            )
        events = decoder.decode(
            {"type": "response.web_search_call.searching", "item_id": "ws_2"}
        )
        self.assertEqual([(e.id, e.phase) for e in events], [("ws_2", "searching")])
        self.assertEqual(len(decoder.web_searches), 2)

    def test_a_failed_search_item_reports_failure(self):
        decoder = CodexDecoder(ReasoningCache())
        events = decoder.decode(
            {
                "type": "response.output_item.done",
                "item": {"type": "web_search_call", "id": "ws_1", "status": "failed"},
            }
        )
        self.assertEqual([(e.phase, e.id) for e in events], [("failed", "ws_1")])

    def test_completed_output_can_supply_web_search_count(self):
        translator = ChunkEncoder("acme-gpt-1", CodexDecoder(ReasoningCache()))
        translator.feed(
            {
                "type": "response.completed",
                "response": {
                    "output": [{"type": "web_search_call", "id": "ws_1"}],
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
            }
        )
        self.assertEqual(
            translator.result()["usage"]["server_tool_use"]["web_search_requests"],
            1,
        )


def _jwt(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


class _FakeProc:
    """Stands in for the app-server child: a pipe in, a scripted pipe out."""

    def __init__(self, replies=None, alive=True):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.replies = replies or {}
        self._alive = alive
        self.terminated = False

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout=None):
        return 0


def _server(proc):
    """An AppServer wired to a fake child, without spawning anything."""
    server = AppServer.__new__(AppServer)
    server._proc = proc
    server._pending = {}
    server._lock = threading.Lock()
    server._next_id = 1
    server._model_contexts = None
    return server


class AppServerProtocolTest(unittest.TestCase):
    """The JSONL bridge: request ids, errors, timeouts, and a dead child."""

    def _answer(self, server, message):
        """Deliver one reply as the reader thread would."""
        for _ in range(200):
            pending = server._pending.get(message.get("id"))
            if pending:
                pending.put(message)
                return
            time.sleep(0.005)
        raise AssertionError("no request was waiting")

    def test_a_call_writes_one_request_and_returns_its_result(self):
        server = _server(_FakeProc())
        threading.Thread(
            target=self._answer,
            args=(server, {"id": 1, "result": {"ok": True}}),
            daemon=True,
        ).start()
        self.assertEqual(server.call("ping", {"x": 1}), {"ok": True})
        sent = json.loads(server._proc.stdin.getvalue().strip())
        self.assertEqual(sent, {"method": "ping", "id": 1, "params": {"x": 1}})

    def test_an_error_reply_becomes_an_rpc_error_naming_the_method(self):
        server = _server(_FakeProc())
        threading.Thread(
            target=self._answer,
            args=(server, {"id": 1, "error": {"message": "nope"}}),
            daemon=True,
        ).start()
        with self.assertRaisesRegex(RpcError, "ping: nope"):
            server.call("ping")

    def test_a_silent_child_times_out_instead_of_hanging(self):
        server = _server(_FakeProc())
        with self.assertRaisesRegex(RpcError, "slow timed out"):
            server.call("slow", timeout=0)
        # The waiter is cleaned up, so a later reply cannot be mistaken for it.
        self.assertEqual(server._pending, {})

    def test_a_dead_child_is_reported_rather_than_written_to(self):
        server = _server(_FakeProc(alive=False))
        with self.assertRaisesRegex(RpcError, "not running"):
            server.call("ping")

    def test_a_non_object_result_is_still_a_mapping(self):
        server = _server(_FakeProc())
        threading.Thread(
            target=self._answer, args=(server, {"id": 1, "result": 7}), daemon=True
        ).start()
        self.assertEqual(server.call("ping"), {"value": 7})

    def test_the_reader_answers_a_server_request_it_cannot_serve(self):
        # The app-server may call the client; an unanswered request would
        # stall it, so unknown methods are refused explicitly.
        proc = _FakeProc()
        proc.stdout = io.StringIO(
            json.dumps({"id": 9, "method": "client/doThing"}) + "\n"
        )
        server = _server(proc)
        server._read()
        self.assertEqual(
            json.loads(proc.stdin.getvalue().strip()),
            {
                "id": 9,
                "error": {"code": -32601, "message": "unsupported client method"},
            },
        )

    def test_a_stopped_child_releases_every_waiting_call(self):
        # Without this a caller waits out its whole timeout for a reply that
        # can never come.
        proc = _FakeProc()
        server = _server(proc)
        import queue as queue_module

        waiter: queue_module.Queue = queue_module.Queue(maxsize=1)
        server._pending[1] = waiter
        server._read()
        self.assertIn("stopped", waiter.get_nowait()["error"]["message"])

    def test_garbage_on_the_pipe_is_skipped(self):
        proc = _FakeProc()
        proc.stdout = io.StringIO("not json\n" + json.dumps({"id": 1}) + "\n")
        server = _server(proc)
        server._read()  # must not raise


class AppServerTokenTest(unittest.TestCase):
    """Credentials come from Codex's own auth.json; the proxy only reads it."""

    def _server_with_auth(self, auth):
        directory = tempfile.mkdtemp()
        server = _server(_FakeProc())
        server.auth_path = pathlib.Path(directory) / "auth.json"
        if auth is not None:
            server.auth_path.write_text(json.dumps(auth))
        return server

    def test_a_live_token_is_used_without_a_refresh(self):
        access = _jwt({"exp": time.time() + 3600})
        server = self._server_with_auth(
            {"tokens": {"access_token": access, "account_id": "acct"}}
        )
        self.assertEqual(server.token(), (access, "acct"))
        self.assertEqual(server._proc.stdin.getvalue(), "", "no refresh was needed")

    def test_the_account_id_falls_back_to_the_token_claim(self):
        access = _jwt(
            {
                "exp": time.time() + 3600,
                "https://api.openai.com/auth.chatgpt_account_id": "from-claim",
            }
        )
        server = self._server_with_auth({"tokens": {"access_token": access}})
        self.assertEqual(server.token()[1], "from-claim")

    def test_signed_out_is_a_readable_error(self):
        server = self._server_with_auth(None)
        server.call = lambda *a, **k: {}
        with self.assertRaisesRegex(RpcError, "not signed in"):
            server.token()

    def test_a_token_without_an_account_is_refused(self):
        server = self._server_with_auth(
            {"tokens": {"access_token": _jwt({"exp": time.time() + 3600})}}
        )
        with self.assertRaisesRegex(RpcError, "account id is missing"):
            server.token()

    def test_an_expiring_token_is_refreshed_before_it_is_used(self):
        stale = _jwt({"exp": time.time() + 30})
        fresh = _jwt({"exp": time.time() + 3600})
        server = self._server_with_auth(
            {"tokens": {"access_token": stale, "account_id": "acct"}}
        )

        def refresh(method, params=None, timeout=30):
            server.auth_path.write_text(
                json.dumps({"tokens": {"access_token": fresh, "account_id": "acct"}})
            )
            return {}

        server.call = refresh
        self.assertEqual(server.token()[0], fresh)

    def test_an_unreadable_payload_is_empty_rather_than_fatal(self):
        self.assertEqual(_jwt_payload("not.a.jwt"), {})
        self.assertEqual(_jwt_payload(""), {})


if __name__ == "__main__":
    unittest.main()
