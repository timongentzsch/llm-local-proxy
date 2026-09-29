"""Named API keys: who is calling, for attribution and a reduced dashboard.

The master key stays in ``config.toml``. Named keys live next to it in
``keys.json``, private to its owner like the config, and are kept readable so
the dashboard can hand a key out again.
"""

from __future__ import annotations

import hmac
import json
import re
import secrets
import threading
from pathlib import Path

from .atomic import atomic_write_json
from .errors import RequestError

MASTER = "master"
NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,31}")


class KeyStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def all(self) -> dict[str, str]:
        """Every name with its key."""
        with self._lock:
            return self._read()

    def add(self, name: str) -> str:
        if not NAME.fullmatch(name) or name == MASTER:
            raise RequestError(
                "key name must be 1-32 lowercase letters, digits, '.', '_' or '-'"
                f" and not {MASTER!r}"
            )
        with self._lock:
            keys = self._read()
            if name in keys:
                raise RequestError(f"key already exists: {name}")
            keys[name] = "llp_" + secrets.token_urlsafe(32)
            atomic_write_json(self.path, {"keys": keys})
            return keys[name]

    def remove(self, name: str) -> None:
        with self._lock:
            keys = self._read()
            if keys.pop(name, None) is None:
                raise RequestError(f"unknown key: {name}")
            atomic_write_json(self.path, {"keys": keys})

    def identify(self, token: str) -> str | None:
        """The name whose key this is; every key is compared in constant time."""
        found = None
        for name, key in self.all().items():
            if hmac.compare_digest(token.encode(), key.encode()):
                found = name
        return found

    def _read(self) -> dict[str, str]:
        try:
            value = json.loads(self.path.read_text())
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid key registry: {self.path}") from error
        keys = value.get("keys") if isinstance(value, dict) else None
        if not isinstance(keys, dict):
            raise TypeError(f"invalid key registry: {self.path}")
        return {str(name): str(key) for name, key in keys.items()}
