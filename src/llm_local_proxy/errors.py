"""Errors that cross layer boundaries."""

from __future__ import annotations


class ProviderError(RuntimeError):
    """A provider failure safe to expose through a downstream error envelope."""

    status = 502


class RequestError(ValueError):
    """A downstream request the proxy will not serve.

    Raised by a dialect's ingress when a body is malformed, and by a
    provider's renderer when the request is well formed but asks for
    something that upstream cannot do. Both surface as the dialect's own
    400 envelope.
    """
