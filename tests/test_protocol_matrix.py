"""One executable contract for the four downstream/upstream pairings.

Each lane covers reasoning, a tool call followed by its result, and web-search
configuration. Streaming/output-shape and usage details remain pinned by the
dialect/provider tests and byte-level goldens.
"""

from __future__ import annotations

import json
import unittest

import claude_events

from llm_local_proxy.dialects.anthropic.egress import MessageEncoder
from llm_local_proxy.dialects.anthropic.ingress import parse as anthropic
from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.ingress import parse as chat
from llm_local_proxy.dialects.openai.responses_egress import ResponseEncoder
from llm_local_proxy.dialects.openai.responses_ingress import parse as responses
from llm_local_proxy.errors import RequestError
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as to_claude
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build as to_codex
from llm_local_proxy.providers.codex.thinking import ENVELOPE_PREFIX
from llm_local_proxy.providers.reasoning import ReasoningCache
from llm_local_proxy.tools import flatten, qualified_name

CLAUDE_THINKING = {
    "type": "thinking",
    "thinking": "Checked.",
    "signature": "claude-signature",
}
CODEX_REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Checked."}],
    "encrypted_content": "codex-encrypted",
}
CALL = {"id": "call_1", "name": "read", "arguments": "{}"}


def _cache_claude_turn(cache: ReasoningCache) -> None:
    encoder = ChunkEncoder("claude-test", ClaudeDecoder(cache))
    events = [
        *claude_events.thinking(0, "Checked.", "claude-signature"),
        *claude_events.tool_use(1, CALL["id"], CALL["name"], "{}"),
        claude_events.stop("tool_use"),
    ]
    for event in events:
        encoder.feed(event)
    encoder.result()


def _cache_codex_turn(cache: ReasoningCache) -> None:
    encoder = ChunkEncoder("gpt-test", CodexDecoder(cache))
    encoder.feed({"type": "response.output_item.done", "item": CODEX_REASONING})
    encoder.feed(
        {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": CALL["id"], **CALL},
        }
    )
    encoder.result()


def _anthropic_history(thinking: dict) -> list[dict]:
    return [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "content": [
                thinking,
                {
                    "type": "tool_use",
                    "id": CALL["id"],
                    "name": CALL["name"],
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": CALL["id"],
                    "content": "ok",
                }
            ],
        },
    ]


def _chat_history() -> list[dict]:
    return [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": CALL["id"],
                    "type": "function",
                    "function": {"name": CALL["name"], "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": CALL["id"], "content": "ok"},
    ]


#: Populated arguments: nested values, booleans, nulls and Unicode, so a
#: serialization bug cannot hide behind `{}`.
ARGS = '{"path":"/tmp/ä","opts":{"depth":2,"follow":false,"tags":["a","ß"],"x":null}}'


def _claude_tool_turn() -> list[dict]:
    return [
        *claude_events.thinking(0, "Checked.", "claude-signature"),
        *claude_events.tool_use(1, CALL["id"], "read", ARGS),
        claude_events.stop("tool_use"),
    ]


def _codex_tool_turn() -> list[dict]:
    return [
        {"type": "response.output_item.done", "item": CODEX_REASONING},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": CALL["id"],
                "name": "read",
                "arguments": ARGS,
            },
        },
    ]


def _chat_replay(result: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "read"},
            result["choices"][0]["message"],
            {"role": "tool", "tool_call_id": CALL["id"], "content": "ok"},
        ]
    }


def _responses_replay(result: dict) -> dict:
    return {
        "store": False,
        "input": [
            {"type": "message", "role": "user", "content": "read"},
            *result["output"],
            {"type": "function_call_output", "call_id": CALL["id"], "output": "ok"},
        ],
    }


def _messages_replay(result: dict) -> dict:
    return {
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": "read"},
            {"role": "assistant", "content": result["content"]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": CALL["id"], "content": "ok"}
                ],
            },
        ],
    }


#: (dialect, result encoder, ingress, replay body from that dialect's result).
#: Every format's record of a finished hosted search.
SEARCH_RECORDS = {"server_tool_use", "web_search_tool_result", "web_search_call"}

DIALECTS = (
    ("chat", ChunkEncoder, chat, _chat_replay),
    ("responses", ResponseEncoder, responses, _responses_replay),
    ("messages", MessageEncoder, anthropic, _messages_replay),
)


class ProtocolMatrixTest(unittest.TestCase):
    def test_every_pairing_replays_its_own_tool_turn(self):
        """Upstream turn -> IR -> client format -> IR -> next upstream request.

        Covers all six dialect/provider lanes: the call id, name and populated
        arguments, the tool result, and the provider's signed reasoning must
        all survive the full round trip through the intermediate representation.
        """
        for dialect, encoder_type, ingress, replay in DIALECTS:
            for provider in ("claude", "codex"):
                with self.subTest(dialect=dialect, provider=provider):
                    cache = ReasoningCache()
                    if provider == "claude":
                        model, events = "claude-test", _claude_tool_turn()
                        decoder = ClaudeDecoder(cache)
                    else:
                        model, events = "gpt-test", _codex_tool_turn()
                        decoder = CodexDecoder(cache)
                    encoder = encoder_type(model, decoder)
                    for event in events:
                        encoder.feed(event)
                    request = ingress({"model": model, **replay(encoder.result())})
                    if provider == "claude":
                        self._assert_claude_replay(
                            to_claude(request, model, 32768, reasoning_cache=cache)[0]
                        )
                    else:
                        self._assert_codex_replay(to_codex(request, cache)[0])

    def _assert_claude_replay(self, upstream: dict) -> None:
        assistant = upstream["messages"][1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(
            assistant["content"],
            [
                CLAUDE_THINKING,
                {
                    "type": "tool_use",
                    "id": CALL["id"],
                    "name": "read",
                    "input": json.loads(ARGS),
                },
            ],
        )
        self.assertEqual(
            upstream["messages"][2]["content"],
            [{"type": "tool_result", "tool_use_id": CALL["id"], "content": "ok"}],
        )

    def _assert_codex_replay(self, upstream: dict) -> None:
        items = upstream["input"]
        kinds = [item.get("type") for item in items]
        self.assertEqual(
            kinds[1:], ["reasoning", "function_call", "function_call_output"]
        )
        self.assertEqual(items[1]["encrypted_content"], "codex-encrypted")
        call = items[2]
        self.assertEqual((call["call_id"], call["name"]), (CALL["id"], "read"))
        self.assertEqual(json.loads(call["arguments"]), json.loads(ARGS))
        self.assertEqual(
            items[3],
            {"type": "function_call_output", "call_id": CALL["id"], "output": "ok"},
        )

    def test_every_pairing_continues_after_a_hosted_search(self):
        """A search the upstream ran must not break the client's next turn.

        The client echoes the search records it was shown. They replay
        verbatim to the upstream that speaks their format and are omitted
        elsewhere; cited answer text reaches both.
        """
        claude_turn = [
            *claude_events.server_block(
                0,
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_1",
                    "name": "web_search",
                    "input": {"query": "news"},
                },
            ),
            *claude_events.server_block(
                1,
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_1",
                    "content": [{"type": "web_search_result", "url": "https://a"}],
                },
            ),
            *claude_events.text(2, "Answer."),
            claude_events.stop("end_turn"),
        ]
        codex_turn = [
            {
                "type": "response.output_item.added",
                "item": {"type": "web_search_call", "id": "ws_1"},
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                },
            },
            {"type": "response.output_text.delta", "delta": "Answer."},
            {
                "type": "response.output_text.annotation.added",
                "annotation": {
                    "type": "url_citation",
                    "url": "https://a",
                    "title": "A",
                },
            },
        ]
        follow_up = {
            "chat": lambda result: {
                "messages": [
                    {"role": "user", "content": "news?"},
                    result["choices"][0]["message"],
                    {"role": "user", "content": "more"},
                ]
            },
            "responses": lambda result: {
                "store": False,
                "input": [
                    {"type": "message", "role": "user", "content": "news?"},
                    *result["output"],
                    {"type": "message", "role": "user", "content": "more"},
                ],
            },
            "messages": lambda result: {
                "max_tokens": 4096,
                "messages": [
                    {"role": "user", "content": "news?"},
                    {"role": "assistant", "content": result["content"]},
                    {"role": "user", "content": "more"},
                ],
            },
        }
        native = {("messages", "claude"), ("responses", "codex")}
        for dialect, encoder_type, ingress, _ in DIALECTS:
            for provider in ("claude", "codex"):
                with self.subTest(dialect=dialect, provider=provider):
                    cache = ReasoningCache()
                    if provider == "claude":
                        decoder, events = ClaudeDecoder(cache), claude_turn
                    else:
                        decoder, events = CodexDecoder(cache), codex_turn
                    encoder = encoder_type("m", decoder)
                    for event in events:
                        encoder.feed(event)
                    body = {"model": "m", **follow_up[dialect](encoder.result())}
                    request = ingress(body)
                    if provider == "claude":
                        turn = to_claude(request, "m", 32768)[0]["messages"][1][
                            "content"
                        ]
                        kinds = [block["type"] for block in turn]
                        text = "".join(b.get("text", "") for b in turn)
                    else:
                        items = to_codex(request, cache)[0]["input"]
                        kinds = [item.get("type") for item in items]
                        text = json.dumps(items)
                    self.assertIn("Answer.", text)
                    records = sorted(set(kinds) & SEARCH_RECORDS)
                    expected = {"claude": ["server_tool_use", "web_search_tool_result"]}
                    if (dialect, provider) in native:
                        self.assertEqual(
                            records, expected.get(provider, ["web_search_call"])
                        )
                    else:
                        self.assertEqual(records, [])

    def test_namespaced_calls_round_trip_through_responses(self):
        """Codex CLI groups MCP tools in namespaces; calls must find their way home.

        Codex speaks namespaces natively. Claude does not, so the tools are
        flattened under one qualified name (hashed when it would pass 64
        characters) and the call is mapped back before the client sees it.
        """
        namespace = "mcp__codex_apps__codex_document_control"
        member = "_get_document_tool_schemas"
        tool = {
            "type": "namespace",
            "name": namespace,
            "description": "Document sessions",
            "tools": [
                {
                    "type": "function",
                    "name": member,
                    "description": "Schemas",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        qualified = qualified_name(namespace, member)
        self.assertLessEqual(len(qualified), 64)
        for provider in ("claude", "codex"):
            with self.subTest(provider=provider):
                cache = ReasoningCache()
                body = {"model": "m", "store": False, "tools": [tool]}
                request = responses({**body, "input": "read"})
                if provider == "claude":
                    upstream = to_claude(request, "m", 32768)[0]
                    self.assertEqual(
                        [t["name"] for t in upstream["tools"]], [qualified]
                    )
                    decoder = ClaudeDecoder(cache, flatten(request.tools)[1])
                    events = [
                        *claude_events.tool_use(0, CALL["id"], qualified, ARGS),
                        claude_events.stop("tool_use"),
                    ]
                else:
                    self.assertEqual(to_codex(request, cache)[0]["tools"], [tool])
                    decoder = CodexDecoder(cache)
                    events = [
                        {
                            "type": "response.output_item.done",
                            "item": {
                                "type": "function_call",
                                "call_id": CALL["id"],
                                "namespace": namespace,
                                "name": member,
                                "arguments": ARGS,
                            },
                        }
                    ]
                encoder = ResponseEncoder("m", decoder, request)
                for event in events:
                    encoder.feed(event)
                output = encoder.result()["output"]
                call = next(item for item in output if item["type"] == "function_call")
                self.assertEqual((call["namespace"], call["name"]), (namespace, member))

                replay = responses(
                    {
                        **body,
                        "input": [
                            {"type": "message", "role": "user", "content": "read"},
                            *output,
                            {
                                "type": "function_call_output",
                                "call_id": CALL["id"],
                                "output": "ok",
                            },
                        ],
                    }
                )
                if provider == "claude":
                    sent = to_claude(replay, "m", 32768, reasoning_cache=cache)[0]
                    use = sent["messages"][1]["content"][-1]
                    self.assertEqual(
                        (use["name"], use["input"]), (qualified, json.loads(ARGS))
                    )
                else:
                    sent = to_codex(replay, cache)[0]["input"]
                    use = next(
                        item for item in sent if item.get("type") == "function_call"
                    )
                    self.assertEqual(
                        (use["namespace"], use["name"]), (namespace, member)
                    )

    def test_openai_to_claude(self):
        cache = ReasoningCache()
        _cache_claude_turn(cache)
        request = chat(
            {
                "model": "claude-test",
                "messages": _chat_history(),
                "web_search_options": {},
                "reasoning_effort": "high",
            }
        )
        upstream, betas = to_claude(
            request, "claude-test", max_output=32768, reasoning_cache=cache
        )
        self.assertEqual(upstream["messages"][1]["content"][0], CLAUDE_THINKING)
        self.assertEqual(upstream["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(upstream["output_config"], {"effort": "high"})
        self.assertEqual(upstream["thinking"]["display"], "summarized")
        self.assertEqual(upstream["tools"][0]["type"], "web_search_20250305")
        self.assertIn("web-search-2025-03-05", betas)

    def test_openai_to_codex(self):
        cache = ReasoningCache()
        _cache_codex_turn(cache)
        request = responses(
            {
                "model": "gpt-test",
                "input": [
                    {"type": "message", "role": "user", "content": "read"},
                    {
                        "type": "function_call",
                        "call_id": CALL["id"],
                        "name": CALL["name"],
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": CALL["id"],
                        "output": "ok",
                    },
                ],
                "tools": [{"type": "web_search", "search_context_size": "low"}],
                "reasoning": {"effort": "high", "summary": "auto"},
            }
        )
        upstream, _ = to_codex(request, cache)
        self.assertEqual(upstream["input"][1], CODEX_REASONING)
        self.assertEqual(upstream["input"][3]["type"], "function_call_output")
        self.assertEqual(upstream["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertEqual(
            upstream["tools"], [{"type": "web_search", "search_context_size": "low"}]
        )
        self.assertEqual(upstream["include"], ["reasoning.encrypted_content"])

    def test_anthropic_to_claude(self):
        request = anthropic(
            {
                "model": "claude-test",
                "max_tokens": 4096,
                "messages": _anthropic_history(CLAUDE_THINKING),
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 3,
                    }
                ],
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "high"},
            }
        )
        upstream, _ = to_claude(request, "claude-test", max_output=32768)
        self.assertEqual(upstream["messages"][1]["content"][0], CLAUDE_THINKING)
        self.assertEqual(upstream["messages"][2]["content"][0]["type"], "tool_result")
        self.assertEqual(upstream["tools"][0]["max_uses"], 3)
        self.assertEqual(upstream["thinking"]["type"], "adaptive")
        self.assertEqual(upstream["output_config"], {"effort": "high"})

    def test_anthropic_to_codex(self):
        encoder = MessageEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        encoder.feed(
            {"type": "response.reasoning_summary_text.delta", "delta": "Checked."}
        )
        encoder.feed({"type": "response.output_item.done", "item": CODEX_REASONING})
        encoder.feed(
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "call_id": CALL["id"], **CALL},
            }
        )
        assistant = encoder.result()["content"]
        signature = assistant[0]["signature"]
        self.assertTrue(signature.startswith(ENVELOPE_PREFIX))

        request = anthropic(
            {
                "model": "gpt-test",
                "max_tokens": 4096,
                "messages": _anthropic_history(assistant[0]),
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": "high"},
            }
        )
        upstream, _ = to_codex(request, ReasoningCache())
        self.assertEqual(upstream["input"][1], CODEX_REASONING)
        self.assertEqual(upstream["input"][3]["type"], "function_call_output")
        self.assertEqual(upstream["tools"], [{"type": "web_search"}])
        self.assertEqual(upstream["reasoning"], {"effort": "high", "summary": "auto"})

        lossy = (
            (
                {"thinking": {"type": "enabled", "budget_tokens": 2048}},
                "thinking budget",
            ),
            ({"thinking": {"type": "disabled"}}, "reasoning is disabled"),
            (
                {
                    "tools": [
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "blocked_domains": ["example.com"],
                        }
                    ]
                },
                "web_search options: blocked_domains",
            ),
        )
        base = {
            "model": "gpt-test",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": "hi"}],
        }
        for fields, message in lossy:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(RequestError, message),
            ):
                to_codex(anthropic({**base, **fields}), ReasoningCache())

        for signature, message in (
            ("foreign-signature", "signed thinking"),
            (ENVELOPE_PREFIX + "damaged", "malformed"),
        ):
            with (
                self.subTest(signature=signature),
                self.assertRaisesRegex(RequestError, message),
            ):
                foreign = dict(CLAUDE_THINKING, signature=signature)
                to_codex(
                    anthropic({**base, "messages": _anthropic_history(foreign)}),
                    ReasoningCache(),
                )


if __name__ == "__main__":
    unittest.main()
