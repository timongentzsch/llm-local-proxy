"""The Codex provider: a ChatGPT subscription driven through codex app-server."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...ir import ChatRequest
from ...status import AccountStatus
from ..base import Provider, ProviderContext
from ..catalog import match_model
from ..pool import Account, PooledProvider, account_file
from .app_server import AppServer, RpcError
from .auth import CodexAuth
from .catalog import model_info
from .events import CodexDecoder
from .request import build
from .upstream import Upstream, UpstreamError


class Codex(PooledProvider[Upstream]):
    name = "codex"

    @staticmethod
    def retry_if(error: Exception) -> bool:
        return isinstance(error, RpcError)

    def new_account(self, slot: str) -> Account[Upstream]:
        config = self.context.config
        app = AppServer(config.codex_binary, config.codex_home / "accounts" / slot)
        upstream = Upstream(
            app,
            config.request_timeout,
            tokens_path=account_file(self.context.directory, "codex", slot, "tokens"),
        )
        return Account(slot, CodexAuth(app), upstream)

    def state_dirs(self, slot: str) -> list[Path]:
        return [
            *super().state_dirs(slot),
            self.context.config.codex_home / "accounts" / slot,
        ]

    def closed(self, account: Account[Upstream]) -> None:
        account.client.app.close()

    def fetch_catalog(self, account: Account[Upstream]) -> list[dict[str, Any]]:
        upstream = account.client
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
        return models

    def account_status(self, account: Account[Upstream]) -> AccountStatus:
        return replace(account.auth.status(), tokens=account.client.ledger.windows())

    def no_account(self) -> UpstreamError:
        return UpstreamError(
            401, "not signed in to Codex; use the sign in button on the status page"
        )

    def match(self, model: str) -> str | None:
        return match_model(model, self._live_catalog())

    def chat(
        self, canonical: str, request: ChatRequest
    ) -> tuple[Iterator[dict[str, Any]], CodexDecoder]:
        model = next(
            (item for item in self._live_catalog() if item.get("id") == canonical), {}
        )
        efforts = model.get("supported_reasoning_efforts")
        body, cache_key = build(
            request,
            self.cache,
            reasoning_efforts=efforts if isinstance(efforts, list) else None,
        )
        # Without a session, the cache key, which is derived when the client
        # names none: each account has its own upstream cache, and round-robin
        # would hand every turn of one conversation a different one.
        events = self.pool.stream(
            request.session or cache_key,
            lambda account: account.client.events(body),
            self.no_account,
        )
        return events, CodexDecoder(self.cache)

    def healthy(self) -> bool:
        return all(account.client.app.alive() for account in self.pool.accounts)

    def close(self) -> None:
        for account in self.pool.accounts:
            account.client.app.close()


def create(context: ProviderContext) -> Provider:
    codex = Codex(context)
    return codex.provider(
        match=codex.match,
        chat=codex.chat,
        models=codex._live_catalog,
        healthy=codex.healthy,
        close=codex.close,
    )
