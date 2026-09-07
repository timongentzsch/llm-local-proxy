"""Loopback hardening, independent of dialect and provider.

The proxy holds live subscription credentials, so it refuses requests that a
browser on another origin could have forged, and requests addressed to a host
name it does not serve (DNS rebinding).
"""

from __future__ import annotations

import hmac
from collections.abc import Mapping
from urllib.parse import urlparse

LOOPBACK = {"127.0.0.1", "::1", "localhost"}

#: Headers that may carry the proxy's own key, with the scheme inside each.
#: Every mount accepts all of them: refusing one only yields a confusing 401.
CREDENTIALS = (("Authorization", "bearer"), ("x-api-key", ""))


def request_host(headers: Mapping[str, str]) -> tuple[str, int]:
    try:
        parsed = urlparse("//" + headers.get("Host", ""))
        suffix = parsed.netloc.partition("]")[2]
        if suffix and not suffix.startswith(":"):
            return "", 0
        if (
            parsed.username is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            return "", 0
        return parsed.hostname or "", parsed.port or 80
    except ValueError:
        return "", 0


def valid_host(headers: Mapping[str, str], configured: str) -> bool:
    host, _ = request_host(headers)
    return host.casefold() in LOOPBACK | {configured}


def same_origin(headers: Mapping[str, str]) -> bool:
    origin = headers.get("Origin")
    if not origin:
        return True
    try:
        parsed = urlparse(origin)
        _, port = request_host(headers)
        return (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in LOOPBACK
            and (parsed.port or (443 if parsed.scheme == "https" else 80)) == port
        )
    except ValueError:
        return False


def authorized(headers: Mapping[str, str], api_key: str) -> bool:
    """Whether the request carries the local key, in any accepted form."""
    if not api_key:
        return True
    expected = api_key.encode("utf-8")
    for header, scheme in CREDENTIALS:
        value = headers.get(header, "")
        if not value:
            continue
        if scheme:
            sent, _, value = value.partition(" ")
            if sent.casefold() != scheme:
                continue
        if hmac.compare_digest(value.encode("utf-8"), expected):
            return True
    return False
