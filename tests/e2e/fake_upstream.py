#!/usr/bin/env python3
"""Canned Anthropic and ChatGPT upstreams for the end-to-end comparison.

Serves just enough of both subscription edges for a proxy to run a request
through, and logs every request it receives so the two implementations can be
compared on what they *send* as well as on what their clients receive.

    fake_upstream.py PORT LOG

The scenario is chosen by the last user text of the request: "tool" answers
with reasoning and a tool call, "limited" with a 429, anything else with text.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG_LOCK = threading.Lock()

CLAUDE_MODELS = {
    "data": [
        {
            "id": "claude-test",
            "display_name": "Claude Test",
            "created_at": "2026-02-03T04:05:06Z",
            "max_tokens": 64000,
            "max_input_tokens": 200000,
            "capabilities": {
                "image_input": {"supported": True},
                "effort": {
                    "low": {"supported": True},
                    "high": {"supported": True},
                    "max": {"supported": False},
                },
                "thinking": {
                    "types": {
                        "adaptive": {"supported": True},
                        "enabled": {"supported": True},
                    }
                },
            },
        },
        {"id": "claude-plain", "display_name": "", "max_tokens": 8192},
    ]
}

USAGE = {
    "five_hour": {"utilization": 12.5, "resets_at": "2026-10-05T12:00:00Z"},
    "seven_day": {"utilization": 40, "resets_at": "2026-10-09T00:00:00Z"},
    "limits": [
        {
            "kind": "weekly_scoped",
            "percent": 7,
            "resets_at": "2026-10-09T00:00:00Z",
            "scope": {"model": {"display_name": "Opus"}},
        }
    ],
}

CLAUDE_USAGE = {
    "input_tokens": 20,
    "cache_read_input_tokens": 100,
    "cache_creation_input_tokens": 5,
    "output_tokens": 1,
}


def claude_text():
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_up",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "usage": CLAUDE_USAGE,
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Grüße, "},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "pong"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ]


def claude_long():
    """A two-second answer in 200 pieces, for the benchmark."""
    events = claude_text()
    delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "word "},
    }
    return events[:2] + [delta] * 200 + events[4:]


def claude_tool():
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_up",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [],
                "usage": CLAUDE_USAGE,
            },
        },
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
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "text_delta", "text": "Looking it up."},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_up1",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '"Berlin"}'},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 30},
        },
        {"type": "message_stop"},
    ]


CODEX_USAGE = {
    "input_tokens": 120,
    "output_tokens": 34,
    "total_tokens": 154,
    "input_tokens_details": {"cached_tokens": 64},
    "output_tokens_details": {"reasoning_tokens": 12},
}


def codex_text():
    message = {
        "type": "message",
        "id": "msg_up",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Grüße, pong", "annotations": []}],
    }
    return [
        {"type": "response.created", "response": {"id": "resp_up"}},
        {"type": "response.output_text.delta", "delta": "Grüße, "},
        {"type": "response.output_text.delta", "delta": "pong"},
        {"type": "response.output_item.done", "item": message},
        {
            "type": "response.completed",
            "response": {"output": [message], "usage": CODEX_USAGE},
        },
    ]


def codex_tool():
    reasoning = {
        "type": "reasoning",
        "id": "rs_up1",
        "summary": [{"type": "summary_text", "text": "Checking the weather"}],
        "encrypted_content": "ENC-1",
    }
    call = {
        "type": "function_call",
        "id": "fc_up1",
        "call_id": "call_up1",
        "name": "get_weather",
        "arguments": '{"city":"Berlin"}',
    }
    return [
        {"type": "response.created", "response": {"id": "resp_up"}},
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "rs_up1",
            "delta": "Checking the weather",
        },
        {"type": "response.output_item.done", "item": reasoning},
        {"type": "response.output_item.done", "item": call},
        {
            "type": "response.completed",
            "response": {"output": [reasoning, call], "usage": CODEX_USAGE},
        },
    ]


def last_user_text(value) -> str:
    """The text that selects a scenario, from either upstream's body shape."""
    found = ""

    def walk(node):
        nonlocal found
        if isinstance(node, dict):
            if node.get("role") == "user":
                content = node.get("content")
                if isinstance(content, str):
                    found = content
                else:
                    for part in content or []:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            found = part["text"]
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value.get("messages") or value.get("input") or [])
    return found


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    log_path = ""
    refreshed = False

    def log_message(self, *args):
        pass

    def _record(self, body):
        keep = (
            "authorization",
            "anthropic-version",
            "anthropic-beta",
            "x-app",
            "accept",
            "content-type",
            "chatgpt-account-id",
            "originator",
        )
        entry = {
            "method": self.command,
            "path": self.path,
            "headers": {
                k.lower(): v for k, v in self.headers.items() if k.lower() in keep
            },
            "has_request_id": "x-client-request-id" in self.headers,
            "user_agent": self.headers.get("User-Agent", "").split("/")[0],
            "body": body,
        }
        with LOG_LOCK, open(self.log_path, "a") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, events, named):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for event in events:
            frame = f"event: {event['type']}\n" if named else ""
            frame += f"data: {json.dumps(event)}\n\n"
            self.wfile.write(frame.encode())
            self.wfile.flush()
            if len(events) > 100:
                time.sleep(0.01)

    def do_GET(self):
        self._record(None)
        if self.headers.get("Authorization") == "Bearer at-stale":
            return self._json(401, {"error": {"message": "token expired"}})
        if self.path.startswith("/v1/models"):
            return self._json(200, CLAUDE_MODELS)
        if self.path == "/api/oauth/usage":
            return self._json(200, USAGE)
        if self.path == "/api/oauth/profile":
            return self._json(
                200,
                {
                    "account": {"email": "claude@example.com"},
                    "organization": {"rate_limit_tier": "default_claude_max_5x"},
                },
            )
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size)
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        self._record(body)
        if self.path == "/v1/oauth/token":
            return self._json(
                200,
                {
                    "access_token": "at-fresh",
                    "refresh_token": "rt-fresh",
                    "expires_in": 3600,
                    "scope": "user:profile user:inference",
                },
            )
        if self.headers.get("Authorization") == "Bearer at-stale":
            return self._json(401, {"error": {"message": "token expired"}})
        scenario = last_user_text(body)
        if self.path == "/v1/messages/count_tokens":
            return self._json(200, {"input_tokens": 42})
        if self.path == "/v1/messages":
            if "limited" in scenario:
                return self._json(
                    429,
                    {
                        "type": "error",
                        "error": {"type": "rate_limit_error", "message": "slow down"},
                    },
                )
            if body.get("max_tokens") == 0:
                return self._json(
                    200,
                    {
                        "id": "msg_up",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "stop_reason": "max_tokens",
                        "usage": CLAUDE_USAGE,
                    },
                )
            if "long" in scenario:
                return self._sse(claude_long(), named=True)
            events = claude_tool() if "tool" in scenario else claude_text()
            return self._sse(events, named=True)
        if self.path == "/backend-api/codex/responses":
            if (body.get("reasoning") or {}).get("effort") == "__probe__":
                return self._json(
                    400,
                    {
                        "error": {
                            "message": "Invalid value: '__probe__'. Supported values "
                            "are: 'low', 'medium', and 'high'."
                        }
                    },
                )
            if "limited" in scenario:
                return self._json(
                    429,
                    {"error": {"type": "usage_limit_reached", "message": "slow down"}},
                )
            events = codex_tool() if "tool" in scenario else codex_text()
            return self._sse(events, named=True)
        self._json(404, {"error": {"message": "not found"}})


def main() -> None:
    port, Handler.log_path = int(sys.argv[1]), sys.argv[2]
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
