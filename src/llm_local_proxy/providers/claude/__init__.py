"""The Claude provider: an Anthropic subscription over the Claude Code edge."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from ...errors import RequestError
from ...ir import ChatRequest
from ...status import ProviderStatus, account_status
from ..auth import MultiAuth
from ..base import Provider, ProviderContext
from ..catalog import match_model
from ..pool import (
    Account,
    AccountPool,
    AccountStore,
    account_file,
    remove_account_state,
    stored_account_ids,
)
from ..reasoning import ReasoningCache
from .auth import ClaudeAuth, ClaudeAuthError
from .catalog import model_info
from .events import ClaudeDecoder
from .request import build
from .upstream import ClaudeUpstream, ClaudeUpstreamError

CATALOG_TTL_SECONDS = 60
#: The request fields /v1/messages/count_tokens accepts, per the pinned spec.
COUNTED_FIELDS = (
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "cache_control",
)


class Claude:
    def __init__(self, context: ProviderContext):
        self.context = context
        self.invalidate = context.invalidate
        state_root = context.directory / "accounts" / "claude"
        self.store = AccountStore(
            context.directory,
            "claude",
            stored_account_ids(state_root, "credentials.json"),
        )
        self.pool = AccountPool(
            [self._new_account(account_id) for account_id in self.store.ids()]
        )
        self.auth = MultiAuth(
            lambda: tuple((account.id, account.auth) for account in self.pool.accounts)
        )
        self.cache = ReasoningCache()
        self._catalog: tuple[float, list[dict[str, Any]]] | None = None
        self._lock = threading.Lock()
        self._accounts_lock = threading.Lock()

    def _new_account(self, account_id: str) -> Account[ClaudeUpstream]:
        directory = self.context.directory
        auth = ClaudeAuth(account_file(directory, "claude", account_id, "credentials"))
        upstream = ClaudeUpstream(
            auth,
            self.context.config.request_timeout,
            usage_path=account_file(directory, "claude", account_id, "usage"),
            tokens_path=account_file(directory, "claude", account_id, "tokens"),
        )
        return Account(account_id, auth, upstream)

    def manage_accounts(self, body: dict[str, Any]) -> dict[str, Any]:
        action = body.get("action")
        if action == "add":
            with self._accounts_lock:
                self.pool.require_no_unsigned()
                account_id = self.store.add()
                try:
                    self.pool.add(self._new_account(account_id))
                except Exception:
                    self.store.remove(account_id)
                    raise
        elif action == "remove":
            account_id = body.get("account")
            if not isinstance(account_id, str) or not account_id:
                raise RequestError("account is required")
            with self._accounts_lock:
                account = self.pool.get(account_id)
                if account.auth.signed_in():
                    raise RequestError("sign out before removing this account")
                self.store.remove(account_id)
                self.pool.remove(account_id)
                remove_account_state(
                    self.context.directory / "accounts" / "claude" / account_id
                )
        else:
            raise RequestError("action must be add or remove")
        self.forget()
        self.invalidate()
        return {"ok": True, "account": account_id}

    def _request(
        self, canonical: str, request: ChatRequest
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if not self.auth.signed_in():
            raise ClaudeAuthError(
                "not signed in to Claude; use the sign in button on the status page"
            )
        efforts = self._capability(canonical, "reasoning_efforts")
        body, betas = build(
            request,
            canonical,
            max_output=self._capability(canonical, "max_output_tokens"),
            thinking=self._capability(canonical, "thinking"),
            reasoning_efforts=efforts if isinstance(efforts, list) else None,
            reasoning_cache=self.cache,
        )
        return body, tuple(betas)

    def chat(
        self, canonical: str, request: ChatRequest
    ) -> tuple[Iterator[dict[str, Any]], ClaudeDecoder]:
        body, betas = self._request(canonical, request)
        events = self.pool.stream(
            request.session,
            lambda account: account.client.events(body, betas),
            lambda: ClaudeAuthError(
                "not signed in to Claude; use the sign in button on the status page"
            ),
        )
        return events, ClaudeDecoder(self.cache)

    def match(self, model: str) -> str | None:
        return (
            match_model(model, self._live_catalog()) if self.auth.signed_in() else None
        )

    def count_tokens(self, canonical: str, request: ChatRequest) -> dict[str, Any]:
        body, betas = self._request(canonical, request)
        # Its schema accepts only prompt fields; the rest are rejected.
        counted = {key: body[key] for key in COUNTED_FIELDS if key in body}
        return self.pool.call(
            request.session,
            lambda account: account.client.count_tokens(counted, betas),
            lambda: ClaudeAuthError(
                "not signed in to Claude; use the sign in button on the status page"
            ),
        )

    def models(self) -> list[dict[str, Any]]:
        return (
            [model_info(item) for item in self._live_catalog()]
            if self.auth.signed_in()
            else []
        )

    def status(self) -> ProviderStatus:
        accounts = []
        for account in self.pool.accounts:
            try:
                account.auth.hydrate_profile()
                value = replace(
                    account.auth.status(),
                    limits=account.client.usage.limits(),
                    tokens=account.client.ledger.windows(),
                    updated_at=account.client.usage.updated_at(),
                )
            except (ClaudeAuthError, ClaudeUpstreamError, OSError, ValueError) as error:
                value = ProviderStatus(error=str(error) or "unavailable")
            accounts.append(account_status(account.id, value))
        aggregate = self.auth.status()
        return replace(aggregate, accounts=tuple(accounts))

    def finish_login(self, body: dict[str, Any]) -> dict[str, Any]:
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RequestError("code is required")
        account = body.get("account")
        if not isinstance(account, str) or not account:
            raise RequestError("account is required")
        result = self.auth.finish(account, code)
        self.forget()
        self.invalidate()
        return result

    def usage(self, body: dict[str, Any]) -> dict[str, Any]:
        models = self._live_catalog()
        if not models:
            raise ClaudeUpstreamError(502, "Claude model catalog is empty")
        model = min(models, key=lambda item: int(item.get("max_output_tokens") or 0))
        account_id = body.get("account", "")
        if not isinstance(account_id, str):
            raise RequestError("account must be a string")
        if account_id:
            account = self.pool.get(account_id)
            if not account.auth.signed_in():
                raise ClaudeAuthError(f"Claude account {account_id} is not signed in")
            usage = account.client.ping_usage(str(model["id"]))
            return {"account": account_id, "usage": usage}
        usage = {}
        for account in self.pool.accounts:
            if account.auth.signed_in():
                usage[account.id] = account.client.ping_usage(str(model["id"]))
        return {"usage": usage}

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
                "catalog",
                lambda account: account.client.models(),
                lambda: ClaudeAuthError("not signed in to Claude"),
            )
        except (ClaudeAuthError, ClaudeUpstreamError):
            items = []
        with self._lock:
            self._catalog = (time.time(), items)
        return items

    def _capability(self, model: str, key: str) -> Any:
        if not self.auth.signed_in():
            return None
        for item in self._live_catalog():
            if item.get("id") == model:
                return item.get(key)
        return None


def create(context: ProviderContext) -> Provider:
    claude = Claude(context)
    return Provider(
        name="claude",
        auth=claude.auth,
        login_flow="paste_code",
        match=claude.match,
        chat=claude.chat,
        models=claude.models,
        status=claude.status,
        routes={
            "accounts": claude.manage_accounts,
            "code": claude.finish_login,
            "usage": claude.usage,
        },
        count_tokens=claude.count_tokens,
    )
