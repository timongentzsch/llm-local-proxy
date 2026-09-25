"""The OpenAI dialect: Chat Completions and stateless Responses.

Specified by openai/openai-openapi (``createChatCompletion``,
``createResponse``). This is the proxy's default dialect and owns the bare
``/v1`` paths.
"""

from __future__ import annotations

from typing import Any

from ..base import Dialect, Route
from .egress import ChunkEncoder
from .ingress import parse
from .responses_egress import ResponseEncoder
from .responses_ingress import parse as parse_responses


def _error(status: int, message: str) -> dict[str, Any]:
    # The status travels in the HTTP status line; Chat Completions carries
    # only a message and a type in the body.
    return {"error": {"message": message, "type": "proxy_error"}}


def _catalog(models: list[dict[str, Any]]) -> dict[str, Any]:
    return {"object": "list", "data": models}


OPENAI = Dialect(
    name="openai",
    prefix="/openai",
    base_path="/openai/v1",
    routes={
        "/v1/chat/completions": Route(
            parse, lambda model, decoder, request: ChunkEncoder(model, decoder)
        ),
        "/v1/responses": Route(parse_responses, ResponseEncoder, named=True),
    },
    catalog=_catalog,
    error=_error,
    # A comment frame: ignored by every SSE client, costs no tokens.
    keepalive=b": keepalive\n\n",
)
