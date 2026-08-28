"""The Anthropic Messages dialect.

Specified by specs/anthropic-openapi.json plus the streaming prose at
docs.anthropic.com/en/api/messages-streaming.md, which is where the named
SSE events, the ping keepalive and the error frame are defined — the schema
covers none of the three.

Mounted under a prefix because /v1/models means something different here
than in Chat Completions, and the two shapes must not collide.
"""

from __future__ import annotations

from typing import Any

from ..base import Dialect
from .egress import MessageEncoder
from .ingress import parse, parse_count

#: From the ErrorType enum in specs/; anything unmapped is api_error.
ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    402: "billing_error",
    403: "permission_error",
    404: "not_found_error",
    408: "timeout_error",
    429: "rate_limit_error",
    504: "timeout_error",
    529: "overloaded_error",
}


def _error(status: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        # Required by the schema even when the proxy has no id of its own.
        "request_id": None,
        "error": {"type": ERROR_TYPES.get(status, "api_error"), "message": message},
    }


def _catalog(models: list[dict[str, Any]]) -> dict[str, Any]:
    data = [
        {
            "type": "model",
            "id": model["id"],
            "display_name": model.get("name") or model["id"],
            "created_at": "1970-01-01T00:00:00Z",
            # Anthropic's own field for the window. A client that sizes its
            # context against the catalog gets the same number on both mounts.
            **(
                {"max_input_tokens": model["context_length"]}
                if model.get("context_length")
                else {}
            ),
        }
        for model in models
    ]
    return {
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    }


ANTHROPIC = Dialect(
    name="anthropic",
    prefix="/anthropic",
    # A client appends /v1/messages to this itself.
    base_path="/anthropic",
    chat_route="/v1/messages",
    parse=parse,
    encode=MessageEncoder,
    catalog=_catalog,
    event_name=lambda data: data.get("type"),
    error=_error,
    keepalive=b'event: ping\ndata: {"type":"ping"}\n\n',
    # No [DONE] sentinel; message_stop ends the stream.
    terminator=None,
    count_route="/v1/messages/count_tokens",
    parse_count=parse_count,
)
