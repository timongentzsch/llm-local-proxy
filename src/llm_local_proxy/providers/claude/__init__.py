"""The Claude provider: an Anthropic subscription over the Claude Code edge."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from ...errors import RequestError
from ...ir import ChatRequest
from ...status import AccountStatus
from ...tools import flatten
from ..base import Provider, ProviderContext
from ..catalog import match_model
from ..limits import LimitsStore
from ..pool import Account, PooledProvider, account_file, account_id
from .auth import ClaudeAuth, ClaudeAuthError
from .catalog import model_info
from .events import ClaudeDecoder
from .request import build
from .upstream import ClaudeUpstream

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


class Claude(PooledProvider[ClaudeUpstream]):
    name = "claude"

    def new_account(self, slot: str) -> Account[ClaudeUpstream]:
        directory = self.context.directory
        auth = ClaudeAuth(account_file(directory, "claude", slot, "credentials"))
        upstream = ClaudeUpstream(
            auth,
            self.context.config.request_timeout,
            tokens_path=account_file(directory, "claude", slot, "tokens"),
        )
        return Account(slot, auth, upstream)

    def fetch_catalog(self, account: Account[ClaudeUpstream]) -> list[dict[str, Any]]:
        return account.client.models()

    def account_status(self, account: Account[ClaudeUpstream]) -> AccountStatus:
        account.auth.hydrate_profile()
        status = account.auth.status()
        # A login awaiting reauthentication would only fail the read again.
        if not status.signed_in or self.pool.account_error(account.id):
            return status
        limits, updated_at = account.client.limits.current()
        return replace(
            status,
            limits=limits,
            tokens=account.client.ledger.windows(),
            updated_at=updated_at,
        )

    def limits(self, account: Account[ClaudeUpstream]) -> LimitsStore:
        return account.client.limits

    def no_account(self) -> ClaudeAuthError:
        return ClaudeAuthError(
            "not signed in to Claude; use the sign in button on the status page"
        )

    def _request(
        self, canonical: str, request: ChatRequest
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if not self.signed_in():
            raise self.no_account()
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
            self.no_account,
        )
        return events, ClaudeDecoder(self.cache, flatten(request.tools)[1])

    def match(self, model: str) -> str | None:
        return match_model(model, self._live_catalog()) if self.signed_in() else None

    def count_tokens(self, canonical: str, request: ChatRequest) -> dict[str, Any]:
        body, betas = self._request(canonical, request)
        # Its schema accepts only prompt fields; the rest are rejected.
        counted = {key: body[key] for key in COUNTED_FIELDS if key in body}
        return self.pool.call(
            request.session,
            lambda account: account.client.count_tokens(counted, betas),
            self.no_account,
        )

    def models(self) -> list[dict[str, Any]]:
        return (
            [model_info(item) for item in self._live_catalog()]
            if self.signed_in()
            else []
        )

    def finish_login(self, body: dict[str, Any]) -> dict[str, Any]:
        code = body.get("code")
        if not isinstance(code, str) or not code.strip():
            raise RequestError("code is required")
        account = account_id(body)
        result = self.pool.get(account).auth.finish(code)
        # The slot may now hold a different login; its old bars are not ours.
        self.pool.get(account).client.limits.clear()
        self.pool.clear_account_error(account)
        self.forget()
        return result

    def _capability(self, model: str, key: str) -> Any:
        if not self.signed_in():
            return None
        for item in self._live_catalog():
            if item.get("id") == model:
                return item.get(key)
        return None


def create(context: ProviderContext) -> Provider:
    claude = Claude(context)
    return claude.provider(
        match=claude.match,
        chat=claude.chat,
        models=claude.models,
        routes={"code": claude.finish_login},
        count_tokens=claude.count_tokens,
    )
