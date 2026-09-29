"""A provider with no upstream, for tests that must not depend on a real one.

Its "upstream events" are already the shared response vocabulary (``ir``
stream events), so a test scripts exactly what the dialects receive and any
difference in the output comes from the dialect or the server alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from types import SimpleNamespace
from typing import Any

from llm_local_proxy.ir import ChatRequest, Finish, StreamEvent, TextDelta, Usage
from llm_local_proxy.providers import Provider
from llm_local_proxy.providers.catalog import match_model
from llm_local_proxy.providers.pool import Account, AccountPool
from llm_local_proxy.status import AccountStatus, ProviderStatus

#: A plain answer: one text delta, usage, and a normal stop.
REPLY: tuple[StreamEvent, ...] = (
    TextDelta("Hello"),
    Usage(prompt=11, completion=3, total=14),
    Finish("end_turn"),
)


class Decoder:
    """Passes the scripted events through unchanged."""

    def decode(self, event: Any) -> list[StreamEvent]:
        return [event]

    def finish(self) -> list[StreamEvent]:
        return []


def model(model_id: str, **fields: Any) -> dict[str, Any]:
    """A catalog entry in the listing shape every provider returns."""
    return {"id": model_id, "name": model_id, "object": "model", **fields}


def provider(
    name: str = "mock",
    models: Iterable[str] = ("mock-1",),
    reply: Callable[[str, ChatRequest], Iterable[StreamEvent]] | None = None,
    **fields: Any,
) -> Provider:
    """A registry entry whose chat replays ``reply`` (``REPLY`` by default).

    ``reply`` receives the canonical model and the parsed request, so one mock
    can script several outcomes by model name, or raise to simulate a failure.
    """
    catalog = [model(item) for item in models]

    def chat(canonical: str, request: ChatRequest) -> tuple[Iterator[Any], Decoder]:
        events = reply(canonical, request) if reply else REPLY
        return iter(list(events)), Decoder()

    return Provider(
        **{
            "name": name,
            "match": lambda requested: match_model(requested, catalog),
            "chat": chat,
            "models": lambda: catalog,
            "status": lambda: ProviderStatus(signed_in=True),
            "routes": {},
            **fields,
        }
    )


class FakeAuth:
    """A signed-in login with a fixed account line."""

    def __init__(self, account: str):
        self.account = account

    def signed_in(self) -> bool:
        return True

    def hydrate_profile(self) -> None:
        pass

    def status(self) -> AccountStatus:
        return AccountStatus(signed_in=True, account=self.account)


class FakeClient:
    """An upstream client whose catalog read answers ``result`` (or raises it)."""

    def __init__(self, result: Any):
        self.result = result
        self.calls = 0
        self.usage_reads = 0
        self.ledger = SimpleNamespace(windows=dict)
        self.limits = SimpleNamespace(current=self.read_usage)

    def read_usage(self, wait: bool = True) -> tuple[tuple, None]:
        self.usage_reads += 1
        return (), None

    def models(self) -> Any:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def two_accounts(first: Any, second: Any) -> AccountPool:
    """A pool of two signed-in accounts around the given clients."""
    return AccountPool(
        [Account("1", FakeAuth("one"), first), Account("2", FakeAuth("two"), second)]
    )
