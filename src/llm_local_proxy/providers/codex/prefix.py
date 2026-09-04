"""Why a Codex request missed the upstream prompt cache.

Codex caches prefixes implicitly: the discount covers the longest byte-stable
head of ``instructions``, ``tools`` and ``input`` that an earlier request under
the same ``prompt_cache_key`` already carried, rounded down to 128 tokens.
Nothing in the response says where that head ended, so a history that merely
grew and one whose second item was rewritten -- a refreshed timestamp in the
first turn, reasoning replayed from an evicted cache -- are indistinguishable
from the usage numbers alone.

This probe keeps the previous request per cache key, names the first item that
changed, and notices when one conversation arrives under two keys. It is off
unless ``LLM_PROXY_PREFIX_DEBUG`` names a file (``-`` for stderr), and it
records labels, hashes and lengths only unless ``LLM_PROXY_PREFIX_DEBUG_BODIES``
is ``1``, because the bodies are the user's own prompts.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ENABLE_VAR = "LLM_PROXY_PREFIX_DEBUG"
BODIES_VAR = "LLM_PROXY_PREFIX_DEBUG_BODIES"
#: Conversations remembered; an evicted key costs one ``first`` line, no more.
LIMIT = 64
#: Characters of a diverging item shown when body capture is enabled.
EXCERPT = 240


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Unit:
    """One indivisible piece of the cacheable head, in wire order."""

    label: str
    digest: str
    size: int
    text: str = ""


def units(body: Mapping[str, Any], keep_text: bool = False) -> list[Unit]:
    """Split a Responses body into the pieces a prefix match can end between."""

    values: list[tuple[str, Any]] = [("instructions", body.get("instructions", ""))]
    if body.get("tools"):
        # Tools are part of the cached prefix, so a reordered list breaks it.
        values.append(("tools", body["tools"]))
    for item in body.get("input") or []:
        label = "item"
        if isinstance(item, dict):
            label = str(item.get("type") or item.get("role") or "item")
        values.append((label, item))
    result: list[Unit] = []
    for label, value in values:
        text = _text(value)
        digest = hashlib.sha256(text.encode()).hexdigest()[:12]
        result.append(Unit(label, digest, len(text), text if keep_text else ""))
    return result


@dataclass(frozen=True)
class Report:
    """What the previous request under this key shares with the current one."""

    key: str
    kind: str
    items: int
    stable: int
    stable_chars: int
    total_chars: int
    label: str
    churn: str = ""
    before: str = ""
    after: str = ""

    def line(self) -> str:
        share = 100 * self.stable_chars / self.total_chars if self.total_chars else 0
        parts = [
            "codex-prefix",
            f"ts={int(time.time())}",
            f"key={self.key}",
            f"kind={self.kind}",
            f"items={self.items}",
            f"stable={self.stable}",
            f"stable_chars={self.stable_chars}/{self.total_chars}",
            f"stable_share={share:.0f}%",
            f"at={self.label}",
        ]
        if self.churn:
            parts.append(f"churn={self.churn}")
        if self.before or self.after:
            parts.append(f"before={self.before!r}")
            parts.append(f"after={self.after!r}")
        return " ".join(parts)


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _appender(path: Path) -> Callable[[str], None]:
    def write(line: str) -> None:
        # Reopened per line: this runs at request rate at most, and a long-lived
        # handle would outlive rotation of a file the operator owns.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return write


class PrefixProbe:
    """Disabled unless the environment names a sink; then one line per request."""

    def __init__(
        self,
        sink: Callable[[str], None] | None = None,
        bodies: bool = False,
        limit: int = LIMIT,
    ):
        self._sink = sink
        self._bodies = bodies
        self._limit = limit
        self._seen: OrderedDict[str, list[Unit]] = OrderedDict()
        self._roots: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._sink is not None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> PrefixProbe:
        values = os.environ if env is None else env
        target = str(values.get(ENABLE_VAR, "")).strip()
        if not target:
            return cls()
        bodies = str(values.get(BODIES_VAR, "")).strip() == "1"
        if target == "-":
            return cls(_stderr, bodies)
        return cls(_appender(Path(target).expanduser()), bodies)

    def record(self, key: str, body: Mapping[str, Any]) -> Report | None:
        """Compare this body with the last one under ``key`` and log the result."""

        if self._sink is None:
            return None
        current = units(body, keep_text=self._bodies)
        name = key or "<none>"
        # The fallback cache key hashes instructions plus the first user turn, so
        # that pair names the conversation. Tools sit between them in the body
        # and are shared by every conversation one client opens, so a root that
        # counted them would call two unrelated histories the same.
        first = 2 if len(current) > 1 and current[1].label == "tools" else 1
        root = current[0].digest + ":"
        if len(current) > first:
            root += current[first].digest
        with self._lock:
            previous = self._seen.get(name)
            self._seen[name] = current
            self._seen.move_to_end(name)
            while len(self._seen) > self._limit:
                self._seen.popitem(last=False)
            churned = self._roots.get(root)
            churn = churned if churned is not None and churned != name else ""
            self._roots[root] = name
            self._roots.move_to_end(root)
            while len(self._roots) > self._limit:
                self._roots.popitem(last=False)
        report = self._compare(name, previous, current, churn)
        try:
            self._sink(report.line())
        except OSError:
            # A diagnostic must never fail a request; stop trying after the
            # first refusal rather than paying for it on every call.
            self._sink = None
        return report

    def _compare(
        self,
        key: str,
        previous: list[Unit] | None,
        current: list[Unit],
        churn: str,
    ) -> Report:
        total = sum(unit.size for unit in current)
        if previous is None:
            return Report(key, "first", len(current), 0, 0, total, "-", churn)
        stable = 0
        while (
            stable < len(previous)
            and stable < len(current)
            and previous[stable].digest == current[stable].digest
        ):
            stable += 1
        if stable == len(previous):
            kind = "appended" if len(current) > stable else "identical"
        elif stable == len(current):
            kind = "shortened"
        else:
            kind = "changed"
        label = f"{stable}:{current[stable].label}" if stable < len(current) else "end"
        before = after = ""
        if self._bodies and kind == "changed":
            before = previous[stable].text[:EXCERPT]
            after = current[stable].text[:EXCERPT]
        return Report(
            key,
            kind,
            len(current),
            stable,
            sum(unit.size for unit in current[:stable]),
            total,
            label,
            churn,
            before,
            after,
        )
