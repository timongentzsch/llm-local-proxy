#!/usr/bin/env python3
"""Run the Python reference proxy against `LLM_PROXY_TEST_UPSTREAM`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llm_local_proxy.http.server import main
from llm_local_proxy.providers.claude import auth, upstream
from llm_local_proxy.providers.codex import upstream as codex

BASE = os.environ["LLM_PROXY_TEST_UPSTREAM"].rstrip("/")


def local(url: str) -> str:
    return BASE + "/" + url.split("/", 3)[3]


for module, names in (
    (upstream, ("MESSAGES_URL", "COUNT_TOKENS_URL", "MODELS_URL", "USAGE_URL")),
    (auth, ("TOKEN_URL", "PROFILE_URL")),
    (codex, ("RESPONSES_URL",)),
):
    for name in names:
        setattr(module, name, local(getattr(module, name)))

# Bound when the method was defined, so the module constant alone is not enough.
upstream.ClaudeUpstream._open.__defaults__ = (upstream.MESSAGES_URL,)
assert all(
    BASE in str(value)
    for function in (upstream.ClaudeUpstream._open, upstream.ClaudeUpstream._get)
    for value in (function.__defaults__ or ())
    if isinstance(value, str) and value.startswith("http")
), "an upstream URL still points at the real service"

if __name__ == "__main__":
    main()
