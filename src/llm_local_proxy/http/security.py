"""Loopback hardening, independent of dialect and provider.

The proxy holds live subscription credentials, so it refuses requests that a
browser on another origin could have forged, and requests addressed to a host
name it does not serve (DNS rebinding).
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from urllib.parse import urlparse

from ..dialects import Dialect

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def request_host(headers: Mapping[str, str]) -> tuple[str, int]:
    value = headers.get("Host", "")
    if value.startswith("["):
        host = value[1 : value.find("]")] if "]" in value else ""
        suffix = value[value.find("]") + 1 :] if "]" in value else ""
        port = suffix[1:] if suffix.startswith(":") else ""
    else:
        host, separator, port = value.rpartition(":")
        if not separator:
            host, port = value, ""
    try:
        return host, int(port) if port else 80
    except ValueError:
        return "", 0


def valid_host(headers: Mapping[str, str], configured: str) -> bool:
    host, _ = request_host(headers)
    return host.casefold() in LOOPBACK | {configured}


def same_origin(headers: Mapping[str, str]) -> bool:
    origin = headers.get("Origin")
    if not origin:
        return True
    parsed = urlparse(origin)
    _, port = request_host(headers)
    return parsed.hostname in LOOPBACK and (parsed.port or 80) == port


def authorized(headers: Mapping[str, str], dialect: Dialect, api_key: str) -> bool:
    """Whether the request carries the proxy's local key.

    Each dialect names the header its clients already send, so an Anthropic
    SDK authenticating with x-api-key needs no proxy-specific configuration.
    """
    if not api_key:
        return True
    value = headers.get(dialect.auth_header, "")
    if dialect.auth_scheme:
        scheme, _, value = value.partition(" ")
        if scheme.casefold() != dialect.auth_scheme:
            return False
    return hmac.compare_digest(value.encode("utf-8"), api_key.encode("utf-8"))
