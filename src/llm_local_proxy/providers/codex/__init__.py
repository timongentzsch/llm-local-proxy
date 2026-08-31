"""The Codex provider: a ChatGPT subscription driven through codex app-server."""

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
from .app_server import AppServer, RpcError
from .auth import CodexAuth
from .catalog import model_info
from .events import CodexDecoder
from .request import build
from .upstream import Upstream, UpstreamError

CATALOG_TTL_SECONDS = 60


class Codex:
    def __init__(self, context: ProviderContext):
        self.context = context
        self.store = AccountStore(
            context.directory,
            "codex",
            stored_account_ids(context.config.codex_home / "accounts", "auth.json"),
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

    def _new_account(self, account_id: str) -> Account[Upstream]:
        config = self.context.config
        home = config.codex_home / "accounts" / account_id
        app = AppServer(config.codex_binary, home)
        upstream = Upstream(
            app,
            config.request_timeout,
            tokens_path=account_file(
                self.context.directory, "codex", account_id, "tokens"
            ),
        )
        return Account(account_id, CodexAuth(app), upstream)

    def manage_accounts(self, body: dict[str, Any]) -> dict[str, Any]:
        action = body.get("action")
        if action == "add":
            account_id = self.store.add()
            try:
                self.pool.add(self._new_account(account_id))
            except Exception:
                self.store.remove(account_id)
                remove_account_state(
                    self.context.config.codex_home / "accounts" / account_id
                )
                raise
        elif action == "remove":
            account_id = body.get("account")
            if not isinstance(account_id, str) or not account_id:
                raise RequestError("account is required")
            account = self.pool.get(account_id)
            if account.auth.signed_in():
                raise RequestError("sign out before removing this account")
            self.store.remove(account_id)
            self.pool.remove(account_id)
            account.client.app.close()
            remove_account_state(
                self.context.directory / "accounts" / "codex" / account_id
            )
            remove_account_state(
                self.context.config.codex_home / "accounts" / account_id
            )
        else:
            raise RequestError("action must be add or remove")
        self.forget()
        self.context.invalidate()
        return {"ok": True, "account": account_id}

    def forget(self) -> None:
        with self._lock:
            self._catalog = None

    def match(self, model: str) -> str | None:
        return match_model(model, self._live_catalog())

    def chat(
        self, canonical: str, request: ChatRequest
    ) -> tuple[Iterator[dict[str, Any]], CodexDecoder]:
        model = next(
            (item for item in self._live_catalog() if item.get("id") == canonical), {}
        )
        efforts = model.get("supported_reasoning_efforts")
        body, _ = build(
            request,
            self.cache,
            reasoning_efforts=efforts if isinstance(efforts, list) else None,
        )
        events = self.pool.stream(
            request.session,
            lambda account: account.client.events(body),
            lambda: UpstreamError(
                401,
                "not signed in to Codex; use the sign in button on the status page",
            ),
        )
        return events, CodexDecoder(self.cache)

    def models(self) -> list[dict[str, Any]]:
        return self._live_catalog()

    def _live_catalog(self) -> list[dict[str, Any]]:
        with self._lock:
            cached = self._catalog
        if cached and time.time() - cached[0] < CATALOG_TTL_SECONDS:
            return cached[1]
        candidates = self.pool.candidates("catalog")
        if not candidates:
            return []
        upstream = candidates[0].client
        app = upstream.app
        result = app.call("model/list", {"limit": 100, "includeHidden": False})
        contexts = app.model_contexts()
        items = [item for item in result.get("data", []) if isinstance(item, dict)]
        first_model = next(
            (
                str(item.get("model") or item.get("id"))
                for item in items
                if item.get("model") or item.get("id")
            ),
            "",
        )
        transport_efforts = (
            upstream.reasoning_efforts(first_model) if first_model else None
        )
        models = []
        for item in items:
            model = model_info(item, contexts, transport_efforts)
            if model:
                models.append(model)
        with self._lock:
            self._catalog = (time.time(), models)
        return models

    def status(self) -> ProviderStatus:
        accounts = []
        signed_in = 0
        for account in self.pool.accounts:
            try:
                value = replace(
                    account.auth.status(), tokens=account.client.ledger.windows()
                )
            except (RpcError, OSError, ValueError) as error:
                value = ProviderStatus(error=str(error) or "unavailable")
            signed_in += value.signed_in
            accounts.append(account_status(account.id, value))
        return ProviderStatus(
            signed_in=bool(signed_in),
            account=f"{signed_in} of {len(accounts)} accounts signed in",
            accounts=tuple(accounts),
        )

    def healthy(self) -> bool:
        return all(account.client.app.alive() for account in self.pool.accounts)

    def close(self) -> None:
        for account in self.pool.accounts:
            account.client.app.close()


def create(context: ProviderContext) -> Provider:
    codex = Codex(context)
    return Provider(
        name="codex",
        auth=codex.auth,
        login_flow="device_code",
        match=codex.match,
        chat=codex.chat,
        models=codex.models,
        status=codex.status,
        routes={"accounts": codex.manage_accounts},
        healthy=codex.healthy,
        close=codex.close,
    )
