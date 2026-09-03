"""Stateless OpenAI Responses downstream protocol contracts."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import unittest

from llm_local_proxy.dialects.openai.responses_egress import ResponseEncoder
from llm_local_proxy.dialects.openai.responses_ingress import parse
from llm_local_proxy.errors import RequestError
from llm_local_proxy.ir import NativeItem
from llm_local_proxy.providers.claude.events import ClaudeDecoder
from llm_local_proxy.providers.claude.request import build as _build_claude
from llm_local_proxy.providers.claude.thinking import (
    ENVELOPE_PREFIX,
    Outcome,
    pack,
    unpack,
)
from llm_local_proxy.providers.codex.events import CodexDecoder
from llm_local_proxy.providers.codex.request import build as build_codex
from llm_local_proxy.providers.reasoning import ReasoningCache

REASONING = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [{"type": "summary_text", "text": "Checked."}],
    "encrypted_content": "opaque-secret",
}


def build_claude(request, model, **kwargs):
    """Provider rendering with a catalog-reported test model limit."""
    kwargs.setdefault("max_output", 32768)
    return _build_claude(request, model, **kwargs)


class ResponsesIngressTest(unittest.TestCase):
    def test_rejects_server_side_state(self):
        base = {"model": "gpt-test", "input": "hi"}
        with self.assertRaisesRegex(RequestError, "store"):
            parse({**base, "store": True})
        with self.assertRaisesRegex(RequestError, "previous_response_id"):
            parse({**base, "previous_response_id": "resp_old"})

    def test_codex_reasoning_and_tool_history_round_trip_verbatim(self):
        body = {
            "model": "gpt-test",
            "instructions": "Be concise.",
            "input": [
                {"type": "message", "role": "user", "content": "read it"},
                REASONING,
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"a"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "hello",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "strict": True,
                    "parameters": {"type": "object"},
                }
            ],
            "reasoning": {"effort": "high", "summary": "auto"},
            "store": False,
        }
        upstream, _ = build_codex(parse(body), ReasoningCache())
        self.assertEqual(upstream["input"][1], REASONING)
        self.assertEqual(upstream["input"][2]["call_id"], "call_1")
        self.assertEqual(upstream["input"][3]["type"], "function_call_output")
        self.assertFalse(upstream["store"])
        self.assertEqual(upstream["include"], ["reasoning.encrypted_content"])
        self.assertTrue(upstream["tools"][0]["strict"])

    def test_native_items_and_structured_outputs_pass_through_to_codex(self):
        custom_call = {
            "type": "custom_tool_call",
            "id": "ct_1",
            "call_id": "call_1",
            "name": "shell",
            "input": "pwd",
            "status": "completed",
        }
        custom_output = {
            "type": "custom_tool_call_output",
            "call_id": "call_1",
            "output": "project",
        }
        structured_output = {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": [{"type": "input_text", "text": "hello"}],
        }
        request = parse(
            {
                "model": "gpt-test",
                "input": [
                    {"type": "message", "role": "user", "content": "run it"},
                    custom_call,
                    custom_output,
                    structured_output,
                ],
                "tools": [
                    {
                        "type": "custom",
                        "name": "shell",
                        "format": {"type": "text"},
                    }
                ],
            }
        )
        upstream, _ = build_codex(request, ReasoningCache())
        self.assertEqual(
            upstream["input"][1:], [custom_call, custom_output, structured_output]
        )
        self.assertEqual(upstream["tools"][0]["format"], {"type": "text"})

    def test_claude_rejects_unrepresentable_native_items(self):
        request = parse(
            {
                "model": "claude-test",
                "input": [
                    {"type": "message", "role": "user", "content": "run it"},
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_1",
                        "name": "shell",
                        "input": "pwd",
                    },
                ],
            }
        )
        with self.assertRaisesRegex(RequestError, "custom_tool_call"):
            build_claude(request, "claude-test")

    def test_claude_recovers_signed_thinking_from_responses_history(self):
        # Stateless: the client carries the block back, this process need not
        # remember it, and no cache is supplied here.
        signed = {"type": "thinking", "thinking": "Checked.", "signature": "SIG"}
        request = parse(
            {
                "model": "claude-test",
                "input": [
                    {"type": "message", "role": "user", "content": "read it"},
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "Checked."}],
                        "encrypted_content": pack(signed, 0),
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "hello",
                    },
                ],
                "reasoning": {"effort": "low"},
            }
        )
        upstream, _ = build_claude(request, "claude-test")
        self.assertEqual(upstream["output_config"], {"effort": "low"})
        assistant = upstream["messages"][1]["content"]
        self.assertEqual(assistant[0], signed)
        self.assertEqual(assistant[1]["type"], "tool_use")
        self.assertEqual(
            len([block for block in assistant if block["type"] == "thinking"]), 1
        )

    def test_claude_maps_a_requested_summary_to_visible_thinking(self):
        request = parse(
            {
                "model": "claude-test",
                "input": "think",
                "reasoning": {"effort": "high", "summary": "auto"},
            }
        )
        # Explicit reasoning activates adaptive thinking even if an incomplete
        # live catalog entry failed to advertise the capability.
        upstream, _ = build_claude(request, "claude-test")
        self.assertEqual(
            upstream["thinking"], {"type": "adaptive", "display": "summarized"}
        )


class ResponsesEgressTest(unittest.TestCase):
    def test_codex_stream_has_native_order_and_opaque_reasoning(self):
        encoder = ResponseEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        events = [encoder.start()]
        for upstream in (
            {"type": "response.reasoning_summary_text.delta", "delta": "Checked."},
            {"type": "response.output_item.done", "item": REASONING},
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{}",
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "output": [],
                    "usage": {"input_tokens": 2, "output_tokens": 3},
                },
            },
        ):
            events.extend(encoder.feed(upstream))
        events.extend(encoder.finish())
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[0], "response.created")
        self.assertLess(
            kinds.index("response.output_item.added"),
            kinds.index("response.reasoning_summary_text.delta"),
        )
        self.assertIn("response.function_call_arguments.done", kinds)
        self.assertEqual(kinds[-1], "response.completed")
        completed = events[-1]["response"]
        self.assertEqual(completed["output"][0]["encrypted_content"], "opaque-secret")
        self.assertEqual(completed["output"][1]["call_id"], "call_1")
        self.assertEqual(completed["parallel_tool_calls"], True)
        self.assertEqual(completed["tool_choice"], "auto")
        self.assertEqual(completed["tools"], [])
        self.assertEqual(
            completed["usage"]["input_tokens_details"]["cache_write_tokens"], 0
        )
        done = next(
            event
            for event in events
            if event["type"] == "response.function_call_arguments.done"
        )
        self.assertEqual(done["name"], "read_file")

    def test_empty_reasoning_summary_remains_exactly_empty(self):
        encoder = ResponseEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        encoder.feed(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_empty",
                    "summary": [],
                    "encrypted_content": "opaque",
                },
            }
        )
        self.assertEqual(encoder.result()["output"][0]["summary"], [])

    def test_completed_only_reasoning_summary_is_not_duplicated(self):
        encoder = ResponseEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        encoder.feed(
            {
                "type": "response.completed",
                "response": {"output": [REASONING], "usage": {}},
            }
        )
        reasoning = [
            item for item in encoder.result()["output"] if item["type"] == "reasoning"
        ]
        self.assertEqual(reasoning, [REASONING])

    def test_native_output_item_and_stream_error_are_preserved(self):
        decoder = CodexDecoder(ReasoningCache())
        encoder = ResponseEncoder("gpt-test", decoder)
        native = {
            "type": "custom_tool_call",
            "id": "ct_1",
            "call_id": "call_1",
            "name": "shell",
            "input": "pwd",
            "status": "completed",
        }
        events = encoder.feed({"type": "response.output_item.done", "item": native})
        self.assertIsInstance(
            decoder.decode(
                {"type": "response.output_item.done", "item": {**native, "id": "ct_2"}}
            )[0],
            NativeItem,
        )
        self.assertEqual(events[-1]["item"], native)
        failure = encoder.error("boom")
        self.assertEqual(failure["type"], "error")
        self.assertEqual(failure["message"], "boom")
        self.assertIn("sequence_number", failure)

    def test_hosted_search_keeps_its_item_open_across_the_wait(self):
        """`added` and `done` must not arrive together: the gap is the search.

        The `NativeItem` path emits both at once, which is why a hosted search
        gets its own encoder branch rather than reusing it.
        """
        encoder = ResponseEncoder("gpt-test", CodexDecoder(ReasoningCache()))
        added = encoder.feed(
            {
                "type": "response.output_item.added",
                "item": {"type": "web_search_call", "id": "ws_1"},
            }
        )
        self.assertEqual(
            [event["type"] for event in added], ["response.output_item.added"]
        )
        self.assertEqual(
            added[0]["item"],
            {"type": "web_search_call", "id": "ws_1", "status": "in_progress"},
        )
        searching = encoder.feed(
            {"type": "response.web_search_call.searching", "item_id": "ws_1"}
        )
        self.assertEqual(
            [(e["type"], e["item_id"]) for e in searching],
            [("response.web_search_call.searching", "ws_1")],
        )
        completed = encoder.feed(
            {"type": "response.web_search_call.completed", "item_id": "ws_1"}
        )
        self.assertEqual(
            [event["type"] for event in completed],
            ["response.web_search_call.completed", "response.output_item.done"],
        )
        # A repeat of the same terminal state adds nothing to the stream.
        self.assertEqual(
            encoder.feed(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                    },
                }
            ),
            [],
        )
        result = encoder.result()
        self.assertEqual(
            result["output"],
            [{"type": "web_search_call", "id": "ws_1", "status": "completed"}],
        )
        # Never a call the client is expected to run and answer.
        self.assertEqual(result["status"], "completed")

    def test_claude_signed_thinking_is_exposed_as_replayable_item(self):
        encoder = ResponseEncoder("claude-test", ClaudeDecoder())
        for event in (
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Checked."},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "signature"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        ):
            encoder.feed(event)
        result = encoder.result()
        self.assertEqual(result["output"][0]["summary"][0]["text"], "Checked.")
        recovered = unpack(result["output"][0]["encrypted_content"])
        self.assertIs(recovered.outcome, Outcome.OK)
        self.assertEqual(
            recovered.block,
            {"type": "thinking", "thinking": "Checked.", "signature": "signature"},
        )


def _claude_stream(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    """The Claude SSE events that produce these content blocks."""
    events: list[dict[str, object]] = []
    for index, block in enumerate(blocks):
        kind = block["type"]
        if kind == "thinking":
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
            )
            if block["thinking"]:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": block["thinking"],
                        },
                    }
                )
            events.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "signature_delta",
                        "signature": block["signature"],
                    },
                }
            )
        elif kind == "redacted_thinking":
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": block,
                }
            )
        else:
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "f"},
                }
            )
            events.append(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                }
            )
        events.append({"type": "content_block_stop", "index": index})
    events.append({"type": "message_delta", "delta": {"stop_reason": "tool_use"}})
    return events


class ClaudeThinkingRoundTripTest(unittest.TestCase):
    """What Claude signed must come back byte for byte, or not at all."""

    def _round_trip(self, blocks, cache=None):
        encoder = ResponseEncoder("claude-test", ClaudeDecoder(cache))
        for event in _claude_stream(blocks):
            encoder.feed(event)
        output = encoder.result()["output"]
        request = parse(
            {
                "model": "claude-test",
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "go"}],
                    },
                    *output,
                    {
                        "type": "function_call_output",
                        "call_id": "toolu_1",
                        "output": "done",
                    },
                ],
            }
        )
        upstream, _ = build_claude(request, "claude-test", reasoning_cache=cache)
        return upstream["messages"][1]["content"]

    def test_signed_thinking_survives_a_stateless_client(self):
        block = {"type": "thinking", "thinking": "Checked.", "signature": "SIG"}
        replayed = self._round_trip([block, {"type": "tool_use"}])
        self.assertEqual(replayed[0], block)
        self.assertEqual(replayed[1]["type"], "tool_use")

    def test_thinking_withheld_by_the_upstream_is_not_replayed(self):
        # The subscription edge signs reasoning it never streams: the block
        # arrives with a signature and no text. The signature covers what
        # Claude wrote, so sending the empty string back is the alteration
        # upstream refuses -- every stored envelope in a real history was one
        # of these. It carries nothing replayable, so the turn goes back
        # without thinking, which Claude accepts.
        block = {"type": "thinking", "thinking": "", "signature": "SIG"}
        replayed = self._round_trip([block, {"type": "tool_use"}])
        self.assertEqual([b["type"] for b in replayed], ["tool_use"])

    def test_redacted_thinking_stays_redacted(self):
        block = {"type": "redacted_thinking", "data": "OPAQUE"}
        replayed = self._round_trip([block, {"type": "tool_use"}])
        self.assertEqual(replayed[0], block)

    def test_thinking_between_tool_calls_keeps_its_place(self):
        # Claude interleaves thinking with the calls it precedes; grouping the
        # blocks by kind would hand back a turn it never produced.
        blocks = [
            {"type": "thinking", "thinking": "first", "signature": "S1"},
            {"type": "tool_use"},
            {"type": "thinking", "thinking": "second", "signature": "S2"},
        ]
        replayed = self._round_trip(blocks)
        self.assertEqual(
            [block["type"] for block in replayed],
            ["thinking", "tool_use", "thinking"],
        )
        self.assertEqual(replayed[0]["thinking"], "first")
        self.assertEqual(replayed[2]["thinking"], "second")

    def test_stored_signature_only_envelope_does_not_wedge_a_history(self):
        # Histories written before the withheld text was understood hold these
        # by the hundred: a valid envelope whose block can never be accepted.
        # Refusing them on the way out is what lets such a session continue,
        # since an append-only client cannot go back and remove them.
        withheld = pack({"type": "thinking", "thinking": "", "signature": "S"}, 0)
        request = parse(
            {
                "model": "claude-test",
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "go"}],
                    },
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [],
                        "encrypted_content": withheld,
                    },
                    {
                        "type": "function_call",
                        "call_id": "toolu_1",
                        "name": "f",
                        "arguments": "{}",
                    },
                ],
            }
        )
        with contextlib.redirect_stderr(io.StringIO()):
            upstream, _ = build_claude(request, "claude-test")
        self.assertEqual(
            [block["type"] for block in upstream["messages"][1]["content"]],
            ["tool_use"],
        )

    def test_interleaved_thinking_is_not_regrouped_by_a_cache_hit(self):
        # The decoder fills the cache as it streams the turn, so a hit is the
        # ordinary case rather than the exception. The cache keeps the signed
        # blocks without their positions, so answering from it would lift both
        # ahead of the call they surround -- a turn Claude never produced, and
        # one it rejects as modified. The client's own order is authoritative
        # whenever its envelopes are complete.
        cache = ReasoningCache()
        replayed = self._round_trip(
            [
                {"type": "thinking", "thinking": "first", "signature": "S1"},
                {"type": "tool_use"},
                {"type": "thinking", "thinking": "second", "signature": "S2"},
            ],
            cache=cache,
        )
        self.assertEqual(
            [block["type"] for block in replayed],
            ["thinking", "tool_use", "thinking"],
        )
        self.assertEqual(replayed[0]["thinking"], "first")
        self.assertEqual(replayed[2]["thinking"], "second")

    def test_dropped_or_reordered_signed_blocks_are_refused(self):
        signed = [
            {"type": "thinking", "thinking": "first", "signature": "S1"},
            {"type": "thinking", "thinking": "second", "signature": "S2"},
        ]

        def history(items):
            return parse(
                {
                    "model": "claude-test",
                    "store": False,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "go"}],
                        },
                        *items,
                        {
                            "type": "function_call",
                            "call_id": "toolu_1",
                            "name": "f",
                            "arguments": "{}",
                        },
                        {
                            "type": "function_call_output",
                            "call_id": "toolu_1",
                            "output": "done",
                        },
                    ],
                }
            )

        def item(block, ordinal):
            return {
                "type": "reasoning",
                "id": f"rs_{ordinal}",
                "summary": [],
                "encrypted_content": pack(block, ordinal),
            }

        items = [item(block, n) for n, block in enumerate(signed)]
        # The client kept only the second of two signed blocks.
        with self.assertRaisesRegex(RequestError, "out of order or incomplete"):
            build_claude(history([items[1]]), "claude-test")
        # The client swapped them.
        with self.assertRaisesRegex(RequestError, "out of order or incomplete"):
            build_claude(history(list(reversed(items))), "claude-test")
        # An intact pair replays untouched.
        upstream, _ = build_claude(history(items), "claude-test")
        self.assertEqual(upstream["messages"][1]["content"][:2], signed)
        # The cache says how many blocks the turn had, never what order they
        # sat in: it holds them without positions, and replaying from it would
        # group thinking ahead of the calls it was interleaved with. So a turn
        # it shows to be short loses its thinking rather than being rebuilt.
        cache = ReasoningCache()
        cache.put(["toolu_1"], signed)
        upstream, _ = build_claude(
            history([items[1]]), "claude-test", reasoning_cache=cache
        )
        self.assertEqual(
            [block["type"] for block in upstream["messages"][1]["content"]],
            ["tool_use"],
        )
        # A history holding one envelope this build reads and one it does not
        # -- an older version, or another upstream's blob -- must not send the
        # half that survived: a fraction of a signed turn is altered, while a
        # turn with no thinking is merely thinner and is accepted.
        mixed = history([items[0], {**items[1], "encrypted_content": "gAAAAABq-other"}])
        upstream, _ = build_claude(mixed, "claude-test")
        self.assertEqual(
            [block["type"] for block in upstream["messages"][1]["content"]],
            ["tool_use"],
        )
        # Dropping the *trailing* block leaves ordinals that still read 0..n-1,
        # so nothing in the envelopes says the turn is short. Only the cache's
        # count does, and it is what keeps the survivor from going up as a turn
        # Claude never signed.
        upstream, _ = build_claude(
            history([items[0]]), "claude-test", reasoning_cache=cache
        )
        self.assertEqual(
            [block["type"] for block in upstream["messages"][1]["content"]],
            ["tool_use"],
        )

    def test_unreadable_envelope_is_dropped_rather_than_stranding_the_client(self):
        # A client keeps its history append-only: refusing an item it cannot
        # repair would fail every later turn of that session the same way.
        # Claude accepts a turn with no thinking, so dropping is a recovery.
        def history(encrypted):
            return parse(
                {
                    "model": "claude-test",
                    "store": False,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "go"}],
                        },
                        {
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [],
                            "encrypted_content": encrypted,
                        },
                        {
                            "type": "function_call",
                            "call_id": "toolu_1",
                            "name": "f",
                            "arguments": "{}",
                        },
                    ],
                }
            )

        good = pack({"type": "thinking", "thinking": "t", "signature": "S"}, 0)
        # A signed block missing the field that signs it is not a block.
        blocks = {
            "whose envelope arrived damaged": (
                good[:-4],
                pack({"type": "thinking", "thinking": "t"}, 0),
            ),
            "whose envelope an unsupported version wrote": (
                "llpv9-claude-thinking:whatever",
                # v1 packed the bare block, without its ordinal.
                ENVELOPE_PREFIX.replace("v2", "v1")
                + base64.urlsafe_b64encode(
                    json.dumps(
                        {"type": "thinking", "thinking": "", "signature": "S"}
                    ).encode()
                ).decode(),
            ),
            "this proxy did not write and cannot replay": ("", None),
        }
        for reason, encrypted in blocks.items():
            for value in encrypted:
                with self.subTest(reason=reason, encrypted=value):
                    warnings = io.StringIO()
                    with contextlib.redirect_stderr(warnings):
                        upstream, _ = build_claude(history(value), "claude-test")
                    self.assertEqual(
                        [b["type"] for b in upstream["messages"][1]["content"]],
                        ["tool_use"],
                    )
                    self.assertIn(
                        f"dropped 1 reasoning item {reason}", warnings.getvalue()
                    )

    def test_losing_one_signed_block_costs_the_whole_turns_thinking(self):
        # Dropping what cannot be read is only safe wholesale: a turn Claude
        # signed, minus one of its blocks, is not the turn Claude signed.
        signed = [
            {"type": "thinking", "thinking": "first", "signature": "S1"},
            {"type": "thinking", "thinking": "second", "signature": "S2"},
        ]
        items = [
            {
                "type": "reasoning",
                "id": f"rs_{n}",
                "summary": [],
                "encrypted_content": pack(block, n),
            }
            for n, block in enumerate(signed)
        ]
        items[0]["encrypted_content"] = items[0]["encrypted_content"][:-4]
        request = parse(
            {
                "model": "claude-test",
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "go"}],
                    },
                    *items,
                ],
            }
        )
        # Losing one block of a signed turn costs the whole turn's thinking:
        # the survivor alone is a turn Claude did not sign, and refusing here
        # instead would strand an append-only history at that turn for good.
        # Reordering, which loses nothing, is still refused above.
        with contextlib.redirect_stderr(io.StringIO()):
            upstream, _ = build_claude(request, "claude-test")
        kinds = [
            block["type"]
            for message in upstream["messages"]
            for block in message["content"]
        ]
        self.assertNotIn("thinking", kinds)
        # Nothing but thinking was in that turn, so the turn itself goes: an
        # assistant message with no content is rejected upstream.
        self.assertEqual([m["role"] for m in upstream["messages"]], ["user"])

    def test_foreign_reasoning_blob_is_dropped_not_reshaped(self):
        request = parse(
            {
                "model": "claude-test",
                "store": False,
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "go"}],
                    },
                    {
                        "type": "reasoning",
                        "id": "rs_old",
                        "summary": [{"type": "summary_text", "text": ""}],
                        "encrypted_content": "a-signature-from-another-proxy",
                    },
                    {
                        "type": "function_call",
                        "call_id": "toolu_1",
                        "name": "f",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "toolu_1",
                        "output": "done",
                    },
                ],
            }
        )
        upstream, _ = build_claude(request, "claude-test")
        self.assertEqual(
            [block["type"] for block in upstream["messages"][1]["content"]],
            ["tool_use"],
        )
        # The originals are still known by tool call id when this process saw
        # them, and are preferred over dropping.
        cache = ReasoningCache()
        original = {"type": "thinking", "thinking": "Checked.", "signature": "SIG"}
        cache.put(["toolu_1"], [original])
        upstream, _ = build_claude(request, "claude-test", reasoning_cache=cache)
        self.assertEqual(upstream["messages"][1]["content"][0], original)


if __name__ == "__main__":
    unittest.main()
