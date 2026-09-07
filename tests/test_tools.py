"""Tool contracts through every ingress and provider renderer."""

import copy
import json
import unittest

from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.anthropic.ingress import parse as messages
from llm_local_proxy.dialects.openai.ingress import parse as chat
from llm_local_proxy.dialects.openai.responses_egress import ResponseEncoder
from llm_local_proxy.dialects.openai.responses_ingress import parse as responses
from llm_local_proxy.errors import RequestError
from llm_local_proxy.ir import FunctionTool, ToolUse, Turn, WebSearchTool
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as claude
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build as codex
from llm_local_proxy.providers.reasoning import ReasoningCache
from llm_local_proxy.tools import parse_function, render_function

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}},
    "required": ["path"],
    "additionalProperties": False,
}


def request(dialect, fields=None, parallel=None):
    tool = {"name": "read_file", "description": "Read one file.", **(fields or {})}
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "read it"}],
        "max_tokens": 128,
    }
    if dialect == "messages":
        tool.setdefault("input_schema", SCHEMA)
        body["tools"] = [tool]
        body["tool_choice"] = {"type": "auto"}
        if parallel is not None:
            body["tool_choice"]["disable_parallel_tool_use"] = parallel
        return messages(body)
    tool.setdefault("parameters", SCHEMA)
    body["parallel_tool_calls"] = parallel
    if dialect == "chat":
        body["tools"] = [{"type": "function", "function": tool}]
        return chat(body)
    return responses(
        {
            "model": "test",
            "input": "read it",
            "parallel_tool_calls": parallel,
            "tools": [{"type": "function", **tool}],
        }
    )


def render(provider, req):
    if provider == "claude":
        return claude(req, "test", max_output=4096)[0]
    return codex(req, ReasoningCache())[0]


class ToolContractTest(unittest.TestCase):
    def test_rich_tool_result_survives_native_replay(self):
        result = {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "is_error": False,
            "content": [
                {"type": "text", "text": "Screenshot"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                },
            ],
        }
        req = messages(
            {
                "model": "test",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": [result]}],
            }
        )
        self.assertEqual(render("claude", req)["messages"][0]["content"], [result])
        with self.assertRaises(RequestError):
            render("codex", req)

    def test_responses_preview_search_is_not_an_anthropic_native_tool(self):
        req = responses(
            {
                "model": "test",
                "input": "search",
                "tools": [{"type": "web_search_preview"}],
            }
        )
        tool = render("claude", req)["tools"][0]
        self.assertEqual(tool, {"type": "web_search_20250305", "name": "web_search"})
        req = responses(
            {
                "model": "test",
                "input": "search",
                "tools": [{"type": "web_search_preview", "search_context_size": "low"}],
            }
        )
        with self.assertRaises(RequestError):
            render("claude", req)

    def test_web_search_does_not_guess_unknown_wire_formats(self):
        req = request("chat")
        req.tools = [
            WebSearchTool(native={"type": "web_search"}, source="another_format")
        ]
        for provider in ("claude", "codex"):
            with self.subTest(provider=provider), self.assertRaises(RequestError):
                render(provider, req)

    def test_strict_true_false_and_absent_survive_every_pairing(self):
        for dialect in ("messages", "chat", "responses"):
            for provider in ("claude", "codex"):
                for strict in (None, True, False):
                    with self.subTest(
                        dialect=dialect, provider=provider, strict=strict
                    ):
                        req = request(
                            dialect, {} if strict is None else {"strict": strict}
                        )
                        tool = render(provider, req)["tools"][0]
                        if strict is None:
                            self.assertNotIn("strict", tool)
                        else:
                            self.assertIs(tool["strict"], strict)
                        self.assertEqual(
                            tool[
                                "input_schema" if provider == "claude" else "parameters"
                            ],
                            SCHEMA,
                        )

    def test_parallel_setting_survives_every_pairing(self):
        for dialect in ("messages", "chat", "responses"):
            for provider in ("claude", "codex"):
                for parallel in (None, True, False):
                    with self.subTest(
                        dialect=dialect, provider=provider, parallel=parallel
                    ):
                        # Messages expresses the inverse setting.
                        value = (
                            not parallel
                            if dialect == "messages" and parallel is not None
                            else parallel
                        )
                        body = render(provider, request(dialect, parallel=value))
                        if provider == "claude":
                            if parallel is None:
                                self.assertNotIn(
                                    "disable_parallel_tool_use", body["tool_choice"]
                                )
                            else:
                                self.assertIs(
                                    body["tool_choice"]["disable_parallel_tool_use"],
                                    not parallel,
                                )
                        else:
                            self.assertIs(
                                body["parallel_tool_calls"],
                                True if parallel is None else parallel,
                            )

    def test_bad_tool_shapes_and_non_boolean_controls_are_rejected(self):
        for dialect in ("messages", "chat", "responses"):
            for fields in (
                {"strict": "false"},
                {"name": 42},
                {"name": " "},
                {"input_schema" if dialect == "messages" else "parameters": []},
            ):
                with (
                    self.subTest(dialect=dialect, fields=fields),
                    self.assertRaises(RequestError),
                ):
                    request(dialect, fields)
            with self.subTest(dialect=dialect), self.assertRaises(RequestError):
                request(dialect, parallel="false")

    def test_anthropic_metadata_is_preserved_natively_and_rejected_cross_format(self):
        options = {
            "input_examples": [{"path": "a.txt"}],
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "eager_input_streaming": True,
            "defer_loading": False,
        }
        req = request("messages", options)
        tool = render("claude", req)["tools"][0]
        for key, value in options.items():
            self.assertEqual(tool[key], value)
        with self.assertRaisesRegex(
            RequestError, "cannot faithfully represent anthropic"
        ):
            render("codex", req)
        # An upstream-body mutation must not corrupt replay or the caller's schema.
        tool["input_examples"][0]["path"] = "changed"
        self.assertEqual(
            render("claude", req)["tools"][0]["input_examples"],
            options["input_examples"],
        )

    def test_responses_metadata_and_response_echo_stay_identical(self):
        req = request("responses", {"strict": False, "defer_loading": True})
        body = render("codex", req)
        output = ResponseEncoder("test", CodexDecoder(ReasoningCache()), req).result()
        self.assertEqual(output["tools"], body["tools"])
        self.assertTrue(body["tools"][0]["defer_loading"])
        with self.assertRaises(RequestError):
            render("claude", req)

    def test_unsupported_anthropic_discovery_is_explicit(self):
        with self.assertRaisesRegex(RequestError, "tool search"):
            render("claude", request("messages", {"defer_loading": True}))
        # Supplying input_schema must not disguise a provider-owned tool as a function.
        with self.assertRaisesRegex(RequestError, "unsupported server tool"):
            request("messages", {"type": "bash_20250124"})

    def test_unknown_options_are_not_silently_discarded(self):
        for dialect in ("messages", "chat", "responses"):
            req = request(dialect, {"future_option": {"enabled": True}})
            for provider in ("claude", "codex"):
                with self.subTest(dialect=dialect, provider=provider):
                    native = (dialect, provider) in {
                        ("messages", "claude"),
                        ("responses", "codex"),
                    }
                    if native:
                        self.assertEqual(
                            render(provider, req)["tools"][0]["future_option"],
                            {"enabled": True},
                        )
                    else:
                        with self.assertRaisesRegex(RequestError, "future_option"):
                            render(provider, req)

    def test_general_contract_needs_no_provider_registry(self):
        raw = {"name": "read_file", "schema": {}, "strict": False, "future_option": 3}
        original = copy.deepcopy(raw)
        tool = parse_function(raw, "another_wire_format", "schema")
        self.assertEqual(render_function(tool, "another_wire_format", "schema"), raw)
        raw["schema"]["changed"] = True
        self.assertEqual(
            render_function(tool, "another_wire_format", "schema"), original
        )
        portable = FunctionTool("read_file", SCHEMA, strict=True)
        self.assertEqual(
            render_function(portable, "another_wire_format", "schema")["schema"], SCHEMA
        )

    def test_populated_history_arguments_survive_every_pairing(self):
        value = {
            "text": "Grüezi 雪",
            "enabled": True,
            "missing": None,
            "nested": {"values": [1, False, "a\\b"]},
        }
        wire = json.dumps(value, ensure_ascii=False, indent=2)
        for dialect in ("messages", "chat", "responses"):
            for provider in ("claude", "codex"):
                with self.subTest(dialect=dialect, provider=provider):
                    body = {"model": "test", "max_tokens": 128}
                    if dialect == "responses":
                        req = responses(
                            {
                                **body,
                                "input": [
                                    {"role": "user", "content": "run"},
                                    {
                                        "type": "function_call",
                                        "call_id": "call",
                                        "name": "echo",
                                        "arguments": wire,
                                    },
                                    {
                                        "type": "function_call_output",
                                        "call_id": "call",
                                        "output": "ok",
                                    },
                                ],
                            }
                        )
                    else:
                        assistant = (
                            {
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": "call",
                                        "name": "echo",
                                        "input": value,
                                    }
                                ]
                            }
                            if dialect == "messages"
                            else {
                                "tool_calls": [
                                    {
                                        "id": "call",
                                        "type": "function",
                                        "function": {"name": "echo", "arguments": wire},
                                    }
                                ]
                            }
                        )
                        req = (messages if dialect == "messages" else chat)(
                            {
                                **body,
                                "messages": [
                                    {"role": "user", "content": "run"},
                                    {"role": "assistant", **assistant},
                                ],
                            }
                        )
                    original = copy.deepcopy(req)
                    rendered = render(provider, req)
                    if provider == "claude":
                        actual = rendered["messages"][1]["content"][0]["input"]
                    else:
                        call = next(
                            x
                            for x in rendered["input"]
                            if x.get("type") == "function_call"
                        )
                        actual = json.loads(call["arguments"])
                        if dialect != "messages":
                            self.assertEqual(call["arguments"], wire)
                    self.assertEqual(actual, value)
                    self.assertEqual(req, original)

    def test_bad_history_arguments_are_a_request_error(self):
        for value in ('{"path":', '["a"]', "null", 3):
            req = request("chat")
            req.turns.append(Turn("assistant", [ToolUse("call", "read", value)]))
            for provider in ("claude", "codex"):
                with (
                    self.subTest(value=value, provider=provider),
                    self.assertRaisesRegex(RequestError, "tool call arguments"),
                ):
                    render(provider, req)

    def test_bad_upstream_arguments_never_become_an_empty_or_array_call(self):
        for value in ('{"path":', '["a"]', "null"):
            for provider in ("claude", "codex"):
                with self.subTest(value=value, provider=provider):
                    decoder = (
                        ClaudeDecoder()
                        if provider == "claude"
                        else CodexDecoder(ReasoningCache())
                    )
                    output = MessageEncoder("test", decoder)
                    if provider == "codex":
                        events = [
                            {
                                "type": "response.output_item.done",
                                "item": {
                                    "type": "function_call",
                                    "call_id": "call",
                                    "name": "read",
                                    "arguments": value,
                                },
                            }
                        ]
                    else:
                        events = [
                            {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": "call",
                                    "name": "read",
                                    "input": {},
                                },
                            },
                            {
                                "type": "content_block_delta",
                                "index": 0,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": value,
                                },
                            },
                            {"type": "content_block_stop", "index": 0},
                        ]
                    with self.assertRaises(ValueError) as caught:
                        for event in events:
                            output.feed(event)
                        output.result()
                    self.assertNotIsInstance(caught.exception, RequestError)
                    self.assertFalse(
                        any(block["type"] == "tool_use" for block in output.blocks)
                    )

    def test_valid_parameterless_tool_calls_remain_supported(self):
        for value in ("", "{}"):
            output = MessageEncoder("test", CodexDecoder(ReasoningCache()))
            output.feed(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call",
                        "name": "ping",
                        "arguments": value,
                    },
                }
            )
            self.assertEqual(output.result()["content"][0]["input"], {})
