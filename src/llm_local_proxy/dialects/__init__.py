"""The dialect registry.

Order matters only in that the default dialect (prefix "") must be tried last,
since its empty prefix matches every path.
"""

from __future__ import annotations

from .base import Dialect, Frame
from .openai import OPENAI

DIALECTS: tuple[Dialect, ...] = (OPENAI,)

__all__ = ["DIALECTS", "OPENAI", "Dialect", "Frame", "resolve"]


def resolve(path: str) -> tuple[Dialect, str]:
    """Split a request path into the dialect serving it and the rest.

    A prefixed dialect claims both "/anthropic/v1/messages" and, for clients
    that normalise away the trailing slash, "/anthropic" itself.
    """
    for dialect in DIALECTS:
        if not dialect.prefix:
            continue
        if path == dialect.prefix:
            return dialect, "/"
        if path.startswith(dialect.prefix + "/"):
            return dialect, path[len(dialect.prefix) :]
    return OPENAI, path
