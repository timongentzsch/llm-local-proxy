"""Undocumented details of the Claude subscription edge.

Nothing here is published or supported. It was established by observing the
real Claude Code client and may break without notice. Keep it out of the
request and event modules, which implement the public Messages API and are
checkable against docs/specs.md.
"""

from __future__ import annotations

# First system block of every real Claude Code request. Without it the same
# credentials get 429 while the real CLI succeeds; live A/B, 2026-08-19.
CLAUDE_CODE_SYSTEM_MARKER = (
    "You are a Claude agent, built on Anthropic's Claude Agent SDK."
)
