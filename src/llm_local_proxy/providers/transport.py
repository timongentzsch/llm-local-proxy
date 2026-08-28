"""HTTP plumbing shared by every provider transport.

Only what is provably identical lives here. The 401-retry envelopes in the
two upstreams look alike but differ in URL, headers, token source, error
mapping and whether they capture rate-limit headers, so they stay where they
are: a shared helper would need four injected callbacks to save ten lines.
The usage tallies look even more alike and are the least mergeable of all —
Codex records once at response.completed, Claude accumulates across events
and commits at message_stop.
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


def read_events(response: Any) -> Iterator[dict[str, Any]]:
    """Yield the JSON objects carried by an SSE body.

    Event names and comments are ignored: both upstreams put everything the
    proxy needs in the data payload, which names its own type.
    """
    try:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if isinstance(event, dict):
                yield event
    finally:
        response.close()
