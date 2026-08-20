"""The OpenAI Chat Completions dialect.

Specified by ``specs/openai-openapi.yaml`` (``createChatCompletion``). This is
the proxy's default dialect and owns the bare ``/v1`` paths.
"""

from __future__ import annotations

from typing import Any

from ..base import Dialect
from .ingress import parse


def _error(status: int, message: str) -> dict[str, Any]:
    # The status travels in the HTTP status line; Chat Completions carries
    # only a message and a type in the body.
    return {"error": {"message": message, "type": "proxy_error"}}


OPENAI = Dialect(
    name="openai",
    prefix="",
    chat_route="/v1/chat/completions",
    parse=parse,
    auth_header="Authorization",
    auth_scheme="bearer",
    error=_error,
    # A comment frame: ignored by every SSE client, costs no tokens.
    keepalive=b": keepalive\n\n",
    terminator=b"data: [DONE]\n\n",
)
