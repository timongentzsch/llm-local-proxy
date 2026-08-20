"""The dialect registry.

Every dialect is mounted under its own prefix, so adding one can never change
what an existing route means. `DEFAULT` additionally answers on the bare,
unprefixed paths: those predate the prefixes and stay valid.

Adding a dialect is one package plus one entry here.
"""

from __future__ import annotations

from .anthropic import ANTHROPIC
from .base import Dialect, Frame
from .openai import OPENAI

DIALECTS: tuple[Dialect, ...] = (OPENAI, ANTHROPIC)

#: Serves unprefixed paths, so /v1/chat/completions keeps working exactly as
#: it did before /openai existed.
DEFAULT = OPENAI

__all__ = ["ANTHROPIC", "DEFAULT", "DIALECTS", "OPENAI", "Dialect", "Frame", "resolve"]


def resolve(path: str) -> tuple[Dialect, str]:
    """Split a request path into the dialect serving it and the rest.

    A dialect claims both "/openai/v1/models" and, for clients that normalise
    away the trailing slash, "/openai" itself.
    """
    for dialect in DIALECTS:
        if path == dialect.prefix:
            return dialect, "/"
        if path.startswith(dialect.prefix + "/"):
            return dialect, path[len(dialect.prefix) :]
    return DEFAULT, path
