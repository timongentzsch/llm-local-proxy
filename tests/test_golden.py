"""Byte-level regression gate for the dialect/provider refactor.

Every case below pins what the proxy produces *today* for a fixed upstream
stream or a fixed downstream request body. The refactor described in
docs/architecture.md rewrites the request builders (R3) and both translators
(R4); these files are the evidence that it changed nothing observable.

Regenerate deliberately, never to make a red test green:

    LLM_PROXY_RECORD=1 uv run python -m unittest tests.test_golden

Then read `git diff tests/golden/` line by line. A diff here is a change in
what a client receives.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from llm_local_proxy.dialects.openai.egress import ChunkEncoder
from llm_local_proxy.dialects.openai.ingress import parse
from llm_local_proxy.protocol import ReasoningCache
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as build_claude_request
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build as build_codex_request

GOLDEN = Path(__file__).parent / "golden"
RECORD = os.environ.get("LLM_PROXY_RECORD") == "1"

# Streams are frozen so ids and timestamps never enter a golden file.
FIXED_ID = "chatcmpl-0000000000000000000000000000000f"
FIXED_CREATED = 1700000000


# --- upstream event fixtures ------------------------------------------------

CODEX_USAGE = {
    "input_tokens": 120,
    "output_tokens": 34,
    "total_tokens": 154,
    "input_tokens_details": {"cached_tokens": 64},
    "output_tokens_details": {"reasoning_tokens": 12},
}

CODEX_STREAMS: dict[str, list[dict]] = {
    "text": [
        {"type": "response.output_text.delta", "delta": "Hello"},
        {"type": "response.output_text.delta", "delta": ", world"},
        {
            "type": "response.completed",
            "response": {"output": [], "usage": CODEX_USAGE},
        },
    ],
    "reasoning_and_tool_call": [
        {"type": "response.reasoning_summary_text.delta", "delta": "Weighing "},
        {"type": "response.reasoning_summary_text.delta", "delta": "options."},
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Weighing options."}],
                "encrypted_content": "ENCRYPTED-BLOB",
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city":"Berlin"}',
            },
        },
        {
            "type": "response.completed",
            "response": {"output": [], "usage": CODEX_USAGE},
        },
    ],
    "two_tool_calls": [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "alpha",
                "arguments": "{}",
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_2",
                "name": "beta",
                "arguments": '{"x":1}',
            },
        },
        {
            "type": "response.completed",
            "response": {"output": [], "usage": CODEX_USAGE},
        },
    ],
    "web_search_and_citation": [
        {
            "type": "response.output_item.added",
            "item": {"type": "web_search_call", "id": "ws_1"},
        },
        {
            "type": "response.output_text.annotation.added",
            "annotation": {
                "type": "url_citation",
                "url": "https://example.com/a",
                "title": "Example A",
                "start_index": 0,
                "end_index": 5,
            },
        },
        {"type": "response.output_text.delta", "delta": "Cited"},
        {
            "type": "response.output_text.annotation.added",
            "annotation": {
                "type": "url_citation",
                "url": "https://example.com/a",
                "title": "Example A",
                "start_index": 0,
                "end_index": 5,
            },
        },
        {
            "type": "response.completed",
            "response": {"output": [], "usage": CODEX_USAGE},
        },
    ],
    "items_only_on_completed": [
        {
            "type": "response.completed",
            "response": {
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_9",
                        "summary": [],
                        "encrypted_content": "ENC-9",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_9",
                        "name": "late",
                        "arguments": '{"ok":true}',
                    },
                ],
                "usage": CODEX_USAGE,
            },
        }
    ],
}

CLAUDE_START = {
    "type": "message_start",
    "message": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 90,
            "cache_read_input_tokens": 30,
            "cache_creation_input_tokens": 10,
            "output_tokens": 1,
        },
    },
}

CLAUDE_END_USAGE = {
    "output_tokens": 42,
    "cache_read_input_tokens": 30,
    "cache_creation_input_tokens": 10,
    "output_tokens_details": {"thinking_tokens": 15},
}

CLAUDE_STREAMS: dict[str, list[dict]] = {
    "text": [
        CLAUDE_START,
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
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": ", world"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": CLAUDE_END_USAGE,
        },
        {"type": "message_stop"},
    ],
    "thinking_and_tool_use": [
        CLAUDE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me check"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "SIG-1"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '"Berlin"}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": CLAUDE_END_USAGE,
        },
        {"type": "message_stop"},
    ],
    "tool_use_without_arguments": [
        CLAUDE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_2",
                "name": "now",
                "input": {},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": CLAUDE_END_USAGE,
        },
        {"type": "message_stop"},
    ],
    "redacted_thinking": [
        CLAUDE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "redacted_thinking", "data": "RED-"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "redacted_thinking_delta", "data": "TAIL"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Done"},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": CLAUDE_END_USAGE,
        },
        {"type": "message_stop"},
    ],
    "web_search_and_citation": [
        CLAUDE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [],
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Per the source"},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "url": "https://example.com/b",
                    "title": "Example B",
                    "cited_text": "quoted",
                },
            },
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": CLAUDE_END_USAGE,
        },
        {"type": "message_stop"},
    ],
    "stopped_at_max_tokens": [
        CLAUDE_START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Truncated"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
            "usage": CLAUDE_END_USAGE,
        },
        {"type": "message_stop"},
    ],
}


# --- downstream request fixtures --------------------------------------------

SIMPLE_BODY = {
    "model": "gpt-5.6-sol",
    "messages": [{"role": "user", "content": "Hello"}],
}

CONVERSATION_BODY = {
    "model": "gpt-5.6-sol",
    "messages": [
        {"role": "system", "content": "Be terse."},
        {"role": "developer", "content": "Prefer SI units."},
        {"role": "user", "content": "Weather in Berlin?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Berlin"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "17C"},
        {"role": "user", "content": "Thanks"},
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Look up weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "reasoning_effort": "high",
}

IMAGE_BODY = {
    "model": "gpt-5.6-sol",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ],
}

WEB_SEARCH_BODY = {
    "model": "gpt-5.6-sol",
    "messages": [{"role": "user", "content": "Latest news?"}],
    "tools": [
        {
            "type": "openrouter:web_search",
            "parameters": {"search_context_size": "high"},
        }
    ],
}

CLAUDE_BODY = {
    "model": "claude-sonnet-5",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1024,
}

CLAUDE_FULL_BODY = {
    "model": "claude-sonnet-5",
    "messages": CONVERSATION_BODY["messages"],
    "tools": CONVERSATION_BODY["tools"],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    "temperature": 0.5,
    "top_p": 0.9,
    "max_tokens": 2048,
}

CODEX_REQUESTS = {
    "simple": SIMPLE_BODY,
    "conversation_with_tools": CONVERSATION_BODY,
    "image_input": IMAGE_BODY,
    "web_search_tool": WEB_SEARCH_BODY,
}

CLAUDE_REQUESTS = {
    "simple": CLAUDE_BODY,
    "conversation_with_tools": CLAUDE_FULL_BODY,
    "image_input": {**IMAGE_BODY, "model": "claude-sonnet-5", "max_tokens": 512},
}


def _freeze(translator):
    translator.id = FIXED_ID
    translator.created = FIXED_CREATED
    return translator


def _run_stream(translator, events: list[dict]) -> dict:
    """Replay a stream exactly as server._chat does, capturing every chunk."""
    chunks = [translator.start()]
    for event in events:
        chunks.extend(translator.feed(event))
    chunks.extend(translator.finish())
    return {"stream": chunks}


def _run_result(translator, events: list[dict]) -> dict:
    for event in events:
        translator.feed(event)
    return {"result": translator.result()}


def build_all() -> dict[str, dict]:
    """Every golden, keyed by file name. Pure: no network, no clock, no uuid."""
    out: dict[str, dict] = {}
    for name, events in CODEX_STREAMS.items():
        out[f"codex_stream_{name}"] = _run_stream(
            _freeze(ChunkEncoder("gpt-5.6-sol", CodexDecoder(ReasoningCache()))), events
        )
        out[f"codex_result_{name}"] = _run_result(
            _freeze(ChunkEncoder("gpt-5.6-sol", CodexDecoder(ReasoningCache()))), events
        )
    for name, events in CLAUDE_STREAMS.items():
        out[f"claude_stream_{name}"] = _run_stream(
            _freeze(ChunkEncoder("claude-sonnet-5", ClaudeDecoder(ReasoningCache()))),
            events,
        )
        out[f"claude_result_{name}"] = _run_result(
            _freeze(ChunkEncoder("claude-sonnet-5", ClaudeDecoder(ReasoningCache()))),
            events,
        )
    for name, body in CODEX_REQUESTS.items():
        request, session = build_codex_request(parse(body), ReasoningCache())
        out[f"codex_request_{name}"] = {"request": request, "session": session}
    for name, body in CLAUDE_REQUESTS.items():
        request, betas = build_claude_request(
            parse(body), body["model"], reasoning_cache=ReasoningCache()
        )
        out[f"claude_request_{name}"] = {"request": request, "betas": betas}
    return out


class GoldenTest(unittest.TestCase):
    """Fails when observable output drifts. Not a unit test: a contract."""

    def test_goldens(self) -> None:
        produced = build_all()
        GOLDEN.mkdir(exist_ok=True)
        if RECORD:
            for name in sorted(set(produced) | {p.stem for p in GOLDEN.glob("*.json")}):
                path = GOLDEN / f"{name}.json"
                if name not in produced:
                    path.unlink()
                    continue
                path.write_text(
                    json.dumps(produced[name], indent=2, ensure_ascii=False) + "\n"
                )
            self.skipTest(f"recorded {len(produced)} goldens")
        missing = sorted(n for n in produced if not (GOLDEN / f"{n}.json").exists())
        self.assertEqual(missing, [], "unrecorded goldens; run with LLM_PROXY_RECORD=1")
        stale = sorted(p.stem for p in GOLDEN.glob("*.json") if p.stem not in produced)
        self.assertEqual(stale, [], "orphaned goldens; run with LLM_PROXY_RECORD=1")
        for name, value in sorted(produced.items()):
            with self.subTest(golden=name):
                expected = json.loads((GOLDEN / f"{name}.json").read_text())
                self.assertEqual(expected, value)


if __name__ == "__main__":
    unittest.main()
