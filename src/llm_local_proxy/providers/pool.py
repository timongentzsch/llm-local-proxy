"""Shared account selection and rate-limit failover for subscription providers."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from ..atomic import atomic_write_json
from ..errors import RequestError
from .auth import Auth

T = TypeVar("T")
E = TypeVar("E")

RATE_LIMIT_COOLDOWN_SECONDS = 300


@dataclass(frozen=True)
class Account(Generic[T]):
    id: str
    auth: Auth
    client: T


class AccountPool(Generic[T]):
    """Select signed-in accounts and retry a request before its first event.

    A downstream session hashes to a stable starting account, which preserves
    upstream prompt-cache locality. Requests without a session round-robin.
    A 429 cools that account locally and advances to the next login. Once an
    event has been yielded, retrying would duplicate output, so errors pass
    through unchanged.
    """

    def __init__(self, accounts: Sequence[Account[T]]):
        self._accounts = tuple(accounts)
        self._cursor = 0
        self._cooldown: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def accounts(self) -> tuple[Account[T], ...]:
        with self._lock:
            return self._accounts

    def add(self, account: Account[T]) -> None:
        with self._lock:
            if any(item.id == account.id for item in self._accounts):
                raise RequestError(f"account already exists: {account.id}")
            self._accounts += (account,)

    def remove(self, account_id: str) -> Account[T]:
        with self._lock:
            account = next(
                (item for item in self._accounts if item.id == account_id), None
            )
            if account is None:
                raise RequestError(f"unknown account: {account_id}")
            self._accounts = tuple(
                item for item in self._accounts if item.id != account_id
            )
            self._cooldown.pop(account_id, None)
            return account

    def get(self, account_id: str) -> Account[T]:
        for account in self.accounts:
            if account.id == account_id:
                return account
        raise RequestError(f"unknown account: {account_id}")

    def require_no_unsigned(self) -> None:
        """Refuse another slot while an existing one still needs a login."""

        for account in self.accounts:
            try:
                signed_in = account.auth.signed_in()
            except (OSError, RuntimeError, ValueError):
                signed_in = False
            if not signed_in:
                raise RequestError(
                    "sign in or remove the existing unsigned account first"
                )

    def candidates(self, session: str = "") -> tuple[Account[T], ...]:
        signed_in = []
        for account in self.accounts:
            try:
                if account.auth.signed_in():
                    signed_in.append(account)
            except (OSError, RuntimeError, ValueError):
                continue
        if not signed_in:
            return ()
        now = time.time()
        with self._lock:
            ready = [
                account
                for account in signed_in
                if self._cooldown.get(account.id, 0) <= now
            ]
            choices = ready or signed_in
            if session:
                digest = hashlib.sha256(session.encode()).digest()
                start = int.from_bytes(digest[:8], "big") % len(choices)
            else:
                start = self._cursor % len(choices)
                self._cursor += 1
        return tuple(choices[start:] + choices[:start])

    def mark_rate_limited(self, account_id: str) -> None:
        with self._lock:
            self._cooldown[account_id] = time.time() + RATE_LIMIT_COOLDOWN_SECONDS

    def stream(
        self,
        session: str,
        create: Callable[[Account[T]], Iterator[E]],
        no_account: Callable[[], Exception],
    ) -> Iterator[E]:
        """Try each candidate on 429, but only before output has begun."""

        candidates = self.candidates(session)
        if not candidates:
            raise no_account()
        last: Exception | None = None
        for account in candidates:
            started = False
            try:
                for event in create(account):
                    started = True
                    yield event
                return
            except Exception as error:
                if started or getattr(error, "status", None) != 429:
                    raise
                self.mark_rate_limited(account.id)
                last = error
        assert last is not None
        raise last

    def call(
        self,
        session: str,
        invoke: Callable[[Account[T]], E],
        no_account: Callable[[], Exception],
    ) -> E:
        """Non-streaming equivalent used by token counting and usage probes."""

        candidates = self.candidates(session)
        if not candidates:
            raise no_account()
        last: Exception | None = None
        for account in candidates:
            try:
                return invoke(account)
            except Exception as error:
                if getattr(error, "status", None) != 429:
                    raise
                self.mark_rate_limited(account.id)
                last = error
        assert last is not None
        raise last


def account_file(directory: Path, provider: str, account_id: str, name: str) -> Path:
    """Return the canonical private state path for one provider account."""

    return directory / "accounts" / provider / account_id / f"{name}.json"


class AccountStore:
    """Persistent, uncapped slot ids shared by every pooled provider."""

    def __init__(self, directory: Path, provider: str, initial: Sequence[str] = ()):
        self.path = directory / "accounts" / provider / "slots.json"
        self._lock = threading.Lock()
        ids = tuple(dict.fromkeys(initial))
        if not self.path.exists() and ids:
            if any(not item.isdigit() or int(item) < 1 for item in ids):
                raise ValueError("initial account ids must be positive integers")
            self._write(ids)

    def ids(self) -> tuple[str, ...]:
        with self._lock:
            return self._read()

    def add(self) -> str:
        with self._lock:
            ids = self._read()
            used = {int(item) for item in ids}
            value = 1
            while value in used:
                value += 1
            account_id = str(value)
            self._write(ids + (account_id,))
            return account_id

    def remove(self, account_id: str) -> None:
        with self._lock:
            ids = self._read()
            if account_id not in ids:
                raise RequestError(f"unknown account: {account_id}")
            self._write(tuple(item for item in ids if item != account_id))

    def _read(self) -> tuple[str, ...]:
        try:
            value = json.loads(self.path.read_text())
        except FileNotFoundError:
            return ()
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid account registry: {self.path}") from error
        items = value.get("accounts") if isinstance(value, dict) else None
        if not isinstance(items, list):
            raise TypeError(f"invalid account registry: {self.path}")
        ids = tuple(str(item) for item in items)
        if len(set(ids)) != len(ids) or any(
            not item.isdigit() or int(item) < 1 for item in ids
        ):
            raise ValueError(f"invalid account registry: {self.path}")
        return ids

    def _write(self, ids: tuple[str, ...]) -> None:
        atomic_write_json(self.path, {"accounts": list(ids)})


def remove_account_state(path: Path) -> None:
    """Delete one validated slot directory after it leaves the registry."""

    if path.exists():
        shutil.rmtree(path)


def stored_account_ids(directory: Path, filename: str) -> tuple[str, ...]:
    """Find canonical credential stores when creating the first registry."""

    if not directory.exists():
        return ()
    ids = {
        item.parent.name
        for item in directory.glob(f"*/{filename}")
        if item.is_file() and item.parent.name.isdigit() and int(item.parent.name) > 0
    }
    return tuple(sorted(ids, key=int))
