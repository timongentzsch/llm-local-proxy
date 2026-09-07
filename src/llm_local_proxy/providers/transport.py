"""HTTP plumbing shared by every provider transport.

Only what is provably identical lives here. The 401-retry envelopes in the
two upstreams look alike but differ in URL, headers, token source, error
mapping and whether they capture rate-limit headers, so they stay where they
are: a shared helper would need four injected callbacks to save ten lines.
Provider usage parsers share their request lifecycle accounting in ledger.py.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from typing import Any


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A subscription endpoint that redirects is a failure, not a hop."""

    def redirect_request(self, _req, _fp, code, _msg, headers, _newurl):
        return None


def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(NoRedirect)


def read_events(
    response: Any, terminal_events: set[str] | None = None
) -> Iterator[dict[str, Any]]:
    """Decode complete SSE frames and reject an upstream that ends mid-response.

    Multiple data lines form one JSON payload. A final unterminated frame is
    incomplete, even when its bytes happen to form valid JSON.
    """
    data: list[str] = []
    terminal = False
    try:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
            elif not line and data:
                payload = "\n".join(data)
                data.clear()
                if payload == "[DONE]":
                    break
                if not payload:
                    continue
                event = json.loads(payload)
                if not isinstance(event, dict):
                    raise ValueError("upstream SSE payload must be an object")
                terminal |= event.get("type") in (terminal_events or ())
                yield event
        if terminal_events and not terminal:
            raise RuntimeError("upstream stream ended before its terminal event")
    finally:
        response.close()
