"""Shared account selection and rate-limit failover for subscription providers."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Generic, TypeVar

from ..atomic import atomic_write_json
from ..errors import ProviderError, RequestError
from ..status import AccountStatus, ProviderStatus
from ..streaming import closing_iterator
from .auth import Auth
from .base import Provider, ProviderContext
from .reasoning import ReasoningCache

T = TypeVar("T")
E = TypeVar("E")

RATE_LIMIT_COOLDOWN_SECONDS = 300
AUTH_FAILURE_COOLDOWN_SECONDS = 60
CATALOG_TTL_SECONDS = 60
#: What a failing account or catalog degrades to instead of a whole-page error.
DEGRADES = (ProviderError, OSError, ValueError)


@dataclass(frozen=True)
class Account(Generic[T]):
    id: str
    auth: Auth
    client: T


class AccountPool(Generic[T]):
    """Select signed-in accounts and retry a request before its first event.

    A downstream session hashes to a stable starting account, which preserves
    upstream prompt-cache locality. Requests without a session round-robin.
    Rate limits and confirmed unusable credentials cool that account locally
    and advance to the next login. Once an event has been yielded, retrying
    would duplicate output, so errors pass through unchanged.
    """

    def __init__(self, accounts: Sequence[Account[T]]):
        self._accounts = tuple(accounts)
        self._cursor = 0
        self._cooldown: dict[str, float] = {}
        self._account_errors: dict[str, str] = {}
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
            self._account_errors.pop(account_id, None)
            return account

    def get(self, account_id: str) -> Account[T]:
        for account in self.accounts:
            if account.id == account_id:
                return account
        raise RequestError(f"unknown account: {account_id}")

    def require_no_unsigned(self) -> None:
        """Refuse another slot while an existing one still needs a login."""

        for account in self.accounts:
            if not _signed_in(account.auth):
                raise RequestError(
                    "sign in or remove the existing unsigned account first"
                )

    def candidates(self, session: str | None = None) -> tuple[Account[T], ...]:
        signed_in = [account for account in self.accounts if _signed_in(account.auth)]
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

    def account_error(self, account_id: str) -> str:
        """Latest terminal authentication failure observed for one account."""

        with self._lock:
            return self._account_errors.get(account_id, "")

    def clear_account_error(self, account_id: str) -> None:
        with self._lock:
            self._account_errors.pop(account_id, None)
            self._cooldown.pop(account_id, None)

    def _mark_account_error(self, account_id: str, error: Exception) -> None:
        with self._lock:
            self._account_errors[account_id] = str(error) or "authentication failed"
            self._cooldown[account_id] = time.time() + AUTH_FAILURE_COOLDOWN_SECONDS

    def stream(
        self,
        session: str | None,
        create: Callable[[Account[T]], Iterator[E]],
        no_account: Callable[[], Exception],
        retry_if: Callable[[Exception], bool] | None = None,
    ) -> Iterator[E]:
        """Fail over on rate limits or unusable auth before output begins."""

        candidates = self.candidates(session)
        if not candidates:
            raise no_account()
        last: Exception | None = None
        for account in candidates:
            started = False
            try:
                with closing_iterator(create(account)) as events:
                    for event in events:
                        if not started:
                            started = True
                            self.clear_account_error(account.id)
                        yield event
                if not started:
                    self.clear_account_error(account.id)
                return
            except Exception as error:
                unavailable = getattr(error, "account_unavailable", False)
                retryable = (
                    getattr(error, "status", None) == 429
                    or unavailable
                    or (retry_if is not None and retry_if(error))
                )
                if started or not retryable:
                    raise
                if unavailable:
                    self._mark_account_error(account.id, error)
                elif getattr(error, "status", None) == 429:
                    self.mark_rate_limited(account.id)
                last = error
        assert last is not None
        raise last

    def call(
        self,
        session: str | None,
        invoke: Callable[[Account[T]], E],
        no_account: Callable[[], Exception],
        retry_if: Callable[[Exception], bool] | None = None,
    ) -> E:
        """Non-streaming equivalent used by token counting and usage probes."""

        def create(account: Account[T]) -> Iterator[E]:
            yield invoke(account)

        with closing_iterator(
            self.stream(session, create, no_account, retry_if)
        ) as events:
            return next(events)


def account_file(directory: Path, provider: str, account_id: str, name: str) -> Path:
    """Return the canonical private state path for one provider account."""

    return directory / "accounts" / provider / account_id / f"{name}.json"


class AccountStore:
    """Persistent, uncapped slot ids shared by every pooled provider."""

    def __init__(self, directory: Path, provider: str):
        self.path = directory / "accounts" / provider / "slots.json"
        self._lock = threading.Lock()

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


def account_id(body: dict[str, Any]) -> str:
    value = body.get("account")
    if not isinstance(value, str) or not value:
        raise RequestError("account is required")
    return value


class PooledProvider(Generic[T]):
    """What every subscription provider shares: slots, logins, catalog, status.

    A subclass supplies how to build one account, read the catalog through
    it, and describe it; the slot lifecycle, catalog cache and the
    "reauthentication required" overlay are identical for every upstream.
    """

    name = ""
    retry_if: Callable[[Exception], bool] | None = None

    def __init__(self, context: ProviderContext):
        self.context = context
        self.store = AccountStore(context.directory, self.name)
        self.pool = AccountPool([self.new_account(i) for i in self.store.ids()])
        self.cache = ReasoningCache()
        self._catalog: tuple[float, list[dict[str, Any]]] | None = None
        self._lock = threading.Lock()
        self._accounts_lock = threading.Lock()

    # -- supplied by each provider -------------------------------------------

    def new_account(self, slot: str) -> Account[T]:
        raise NotImplementedError

    def fetch_catalog(self, account: Account[T]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def account_status(self, account: Account[T]) -> AccountStatus:
        raise NotImplementedError

    def no_account(self) -> ProviderError:
        raise NotImplementedError

    def state_dirs(self, slot: str) -> list[Path]:
        return [self.context.directory / "accounts" / self.name / slot]

    def closed(self, account: Account[T]) -> None:
        """Release what a removed slot holds beyond its files."""

    # -- shared --------------------------------------------------------------

    def provider(self, **fields: Any) -> Provider:
        routes = {
            "login": self.login,
            "logout": self.logout,
            "accounts": self.manage_accounts,
            **fields.pop("routes", {}),
        }
        return Provider(
            name=self.name,
            status=self.status,
            forget=self.forget,
            routes=routes,
            **fields,
        )

    def signed_in(self) -> bool:
        return any(_signed_in(account.auth) for account in self.pool.accounts)

    def login(self, body: dict[str, Any]) -> dict[str, Any]:
        slot = account_id(body)
        return {**self.pool.get(slot).auth.login_start(), "account": slot}

    def logout(self, body: dict[str, Any]) -> dict[str, Any]:
        self.pool.get(account_id(body)).auth.logout()
        self.forget()
        return {"ok": True}

    def manage_accounts(self, body: dict[str, Any]) -> dict[str, Any]:
        action = body.get("action")
        if action == "add":
            with self._accounts_lock:
                self.pool.require_no_unsigned()
                slot = self.store.add()
                try:
                    self.pool.add(self.new_account(slot))
                except Exception:
                    self.store.remove(slot)
                    for path in self.state_dirs(slot):
                        remove_account_state(path)
                    raise
        elif action == "remove":
            slot = account_id(body)
            with self._accounts_lock:
                account = self.pool.get(slot)
                if account.auth.signed_in():
                    raise RequestError("sign out before removing this account")
                self.store.remove(slot)
                self.pool.remove(slot)
                self.closed(account)
                for path in self.state_dirs(slot):
                    remove_account_state(path)
        else:
            raise RequestError("action must be add or remove")
        self.forget()
        return {"ok": True, "account": slot}

    def forget(self) -> None:
        with self._lock:
            self._catalog = None

    def _live_catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._catalog
        if cached and time.time() - cached[0] < CATALOG_TTL_SECONDS:
            return cached[1]
        try:
            items = self.pool.call(
                None, self.fetch_catalog, self.no_account, self.retry_if
            )
        except ProviderError:
            items = []
        with self._lock:
            self._catalog = (time.time(), items)
        return items

    def status(self) -> ProviderStatus:
        accounts = []
        for account in self.pool.accounts:
            try:
                value = self.account_status(account)
            except DEGRADES as error:
                value = AccountStatus(error=str(error) or "unavailable")
            observed = self.pool.account_error(account.id)
            if observed and value.signed_in:
                value = replace(
                    value,
                    signed_in=False,
                    error=f"reauthentication required: {observed}",
                )
            accounts.append(replace(value, id=account.id))
        return ProviderStatus(
            signed_in=any(account.signed_in for account in accounts),
            accounts=tuple(accounts),
        )


def _signed_in(auth: Auth) -> bool:
    try:
        return auth.signed_in()
    except (OSError, RuntimeError, ValueError):
        return False
