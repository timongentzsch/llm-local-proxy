"""Undocumented details of the Claude subscription edge.

Nothing here is published or supported. It was established by observing the
real Claude Code client and may break without notice. Keep it out of the
request and event modules, which implement the public Messages API and are
checkable against specs/PINNED.md.
"""

from __future__ import annotations

# First system block of every real Claude Code request. The subscription
# edge bills calls carrying this marker against the Claude Code usage pool;
# the same headers and token without it get 429 "rate limited" while the
# real CLI succeeds. Verified 2026-08-19 by live A/B on a Pro account.
CLAUDE_CODE_SYSTEM_MARKER = (
    "You are a Claude agent, built on Anthropic's Claude Agent SDK."
)
