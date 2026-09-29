from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_local_proxy.errors import RequestError
from llm_local_proxy.providers.base import ProviderContext
from llm_local_proxy.providers.limits import LimitsStore
from llm_local_proxy.providers.pool import (
    AUTH_FAILURE_COOLDOWN_SECONDS,
    RATE_LIMIT_COOLDOWN_SECONDS,
    SESSION_TTL_SECONDS,
    Account,
    AccountPool,
    AccountStore,
    PooledProvider,
)
from llm_local_proxy.status import AccountStatus, Limit


class _Auth:
    def __init__(self, signed_in=True):
        self.value = signed_in

    def signed_in(self):
        return self.value

    def login_start(self):
        return {"url": "https://example.test/login"}

    def logout(self):
        self.value = False

    def status(self):
        return AccountStatus(signed_in=self.value)


class _Error(RuntimeError):
    def __init__(self, status, *, account_unavailable=False):
        super().__init__(f"status {status}")
        self.status = status
        self.account_unavailable = account_unavailable


class AccountPoolTest(unittest.TestCase):
    @staticmethod
    def pool():
        return AccountPool([Account("1", _Auth(), "one"), Account("2", _Auth(), "two")])

    def test_sessionless_requests_round_robin(self):
        pool = self.pool()
        self.assertEqual(pool.candidates()[0].id, "1")
        self.assertEqual(pool.candidates()[0].id, "2")

    def test_a_session_has_a_stable_account(self):
        pool = self.pool()
        first = pool.candidates("session-42")[0].id
        self.assertEqual(pool.candidates("session-42")[0].id, first)

    def test_cooling_one_account_moves_only_its_own_sessions(self):
        pool = AccountPool([Account(i, _Auth(), i) for i in ("1", "2", "3")])
        sessions = [f"session-{n}" for n in range(200)]
        before = {s: pool.candidates(s)[0].id for s in sessions}
        pool.mark_rate_limited("2")
        after = {s: pool.candidates(s)[0].id for s in sessions}

        moved = {s for s in sessions if before[s] != after[s]}
        self.assertEqual(moved, {s for s in sessions if before[s] == "2"})
        self.assertTrue(moved)

    def test_a_session_stays_on_the_account_it_failed_over_to(self):
        pool = self.pool()
        session = next(
            f"s{n}" for n in range(100) if pool.candidates(f"s{n}")[0].id == "1"
        )

        def events(account):
            if account.id == "1":
                raise _Error(429)
            yield account.client

        self.assertEqual(list(pool.stream(session, events, RuntimeError)), ["two"])
        later = time.time() + RATE_LIMIT_COOLDOWN_SECONDS + 1
        with patch("llm_local_proxy.providers.pool.time.time", return_value=later):
            self.assertEqual(pool.candidates(session)[0].id, "2")
        # Once its cache would be gone, the session returns to its own account.
        much_later = later + SESSION_TTL_SECONDS
        with patch("llm_local_proxy.providers.pool.time.time", return_value=much_later):
            self.assertEqual(pool.candidates(session)[0].id, "1")

    def test_remembered_sessions_are_bounded_oldest_first(self):
        pool = self.pool()
        with patch("llm_local_proxy.providers.pool.SESSION_LIMIT", 2):
            for session in ("a", "b", "c"):
                pool._remember(session, "1")
        self.assertEqual(list(pool._sessions), ["b", "c"])
        pool.remove("1")
        self.assertEqual(list(pool._sessions), [])

    def test_new_sessions_skip_a_nearly_full_account_but_pinned_ones_stay(self):
        usage = {"1": 95.0, "2": 10.0}
        pool = AccountPool(
            [Account("1", _Auth(), "one"), Account("2", _Auth(), "two")],
            usage=lambda account: usage[account.id],
        )
        sessions = [f"s{n}" for n in range(50)]
        self.assertEqual({pool.candidates(s)[0].id for s in sessions}, {"2"})
        self.assertEqual(pool.candidates()[0].id, "2")
        self.assertEqual(pool.candidates()[0].id, "2")

        pool._remember("pinned", "1")
        self.assertEqual(pool.candidates("pinned")[0].id, "1")
        # Still a failover target, just last.
        self.assertEqual([a.id for a in pool.candidates("s0")], ["2", "1"])

    def test_routing_ignores_the_soft_limit_when_every_account_is_full(self):
        pool = AccountPool(
            [Account("1", _Auth(), "one"), Account("2", _Auth(), "two")],
            usage=lambda account: 99.0,
        )
        starts = {pool.candidates(f"s{n}")[0].id for n in range(50)}
        self.assertEqual(starts, {"1", "2"})

    def test_unknown_usage_counts_as_room(self):
        pool = AccountPool(
            [Account("1", _Auth(), "one"), Account("2", _Auth(), "two")],
            usage=lambda account: None,
        )
        self.assertFalse(pool.draining(pool.get("1")))
        self.assertEqual(pool.candidates()[0].id, "1")

    def test_signed_out_accounts_are_not_candidates(self):
        pool = AccountPool(
            [Account("1", _Auth(False), "one"), Account("2", _Auth(), "two")]
        )
        self.assertEqual([account.id for account in pool.candidates("catalog")], ["2"])

    def test_429_before_output_fails_over_and_cools_the_account(self):
        pool = self.pool()
        calls = []

        def events(account):
            calls.append(account.id)
            if account.id == "1":
                raise _Error(429)
            yield account.client

        self.assertEqual(list(pool.stream("", events, RuntimeError)), ["two"])
        self.assertEqual(calls, ["1", "2"])
        self.assertEqual(pool.candidates()[0].id, "2")

    def test_an_error_after_output_is_never_retried(self):
        pool = self.pool()
        calls = []

        def events(account):
            calls.append(account.id)
            yield "started"
            raise _Error(429)

        stream = pool.stream("", events, RuntimeError)
        self.assertEqual(next(stream), "started")
        with self.assertRaises(_Error):
            next(stream)
        self.assertEqual(calls, ["1"])

    def test_non_rate_limit_errors_are_not_retried(self):
        pool = self.pool()
        calls = []

        def invoke(account):
            calls.append(account.id)
            raise _Error(401)

        with self.assertRaises(_Error):
            pool.call("", invoke, RuntimeError)
        self.assertEqual(calls, ["1"])

    def test_unavailable_account_fails_over_before_output_and_is_reported(self):
        pool = self.pool()
        calls = []

        def events(account):
            calls.append(account.id)
            if account.id == "1":
                raise _Error(400, account_unavailable=True)
            yield account.client

        self.assertEqual(list(pool.stream("", events, RuntimeError)), ["two"])
        self.assertEqual(calls, ["1", "2"])
        self.assertEqual(pool.account_error("1"), "status 400")
        self.assertEqual(pool.account_error("2"), "")
        self.assertEqual(pool.candidates()[0].id, "2")

    def test_discovery_rotates_and_clears_a_recovered_account(self):
        pool = self.pool()
        starts = []
        stale = True

        def discover(account):
            nonlocal stale
            starts.append(account.id)
            if account.id == "1" and stale:
                raise _Error(401, account_unavailable=True)
            return account.client

        self.assertEqual(pool.call(None, discover, RuntimeError), "two")
        self.assertEqual(pool.account_error("1"), "status 401")
        self.assertEqual(pool.call(None, discover, RuntimeError), "two")
        stale = False
        after_cooldown = time.time() + AUTH_FAILURE_COOLDOWN_SECONDS
        with patch(
            "llm_local_proxy.providers.pool.time.time", return_value=after_cooldown
        ):
            self.assertEqual(pool.call(None, discover, RuntimeError), "one")
        self.assertEqual(starts, ["1", "2", "2", "1"])
        self.assertEqual(pool.account_error("1"), "")

    def test_accounts_can_be_added_and_removed_live(self):
        pool = AccountPool([])
        pool.add(Account("1", _Auth(), "one"))
        self.assertEqual(pool.get("1").client, "one")
        self.assertEqual(pool.remove("1").client, "one")
        self.assertEqual(pool.accounts, ())

    def test_only_one_unsigned_slot_is_allowed(self):
        pool = AccountPool([Account("1", _Auth(False), "one")])
        with self.assertRaisesRegex(RequestError, "existing unsigned account"):
            pool.require_no_unsigned()
        pool.get("1").auth.value = True
        pool.require_no_unsigned()


class AccountStoreTest(unittest.TestCase):
    def test_slots_are_persistent_reusable_and_uncapped(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AccountStore(Path(directory), "claude")
            self.assertEqual(store.ids(), ())
            self.assertEqual(
                [store.add() for _ in range(12)], [str(i) for i in range(1, 13)]
            )
            store.remove("2")
            self.assertEqual(store.add(), "2")
            self.assertEqual(AccountStore(Path(directory), "claude").ids()[-1], "2")


class PooledProviderTest(unittest.TestCase):
    class Fake(PooledProvider[str]):
        name = "fake"

        def new_account(self, slot):
            return Account(slot, _Auth(False), slot)

    def provider(self, directory):
        context = ProviderContext(config=None, directory=Path(directory))
        return self.Fake(context)

    def test_slots_logins_and_logouts_target_one_account(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = self.provider(directory)
            routes = fake.provider(match=None, chat=None, models=list).routes
            self.assertEqual(routes["accounts"]({"action": "add"})["account"], "1")
            with self.assertRaisesRegex(RequestError, "existing unsigned account"):
                routes["accounts"]({"action": "add"})
            self.assertEqual(routes["login"]({"account": "1"})["account"], "1")
            for route in ("login", "logout"):
                with self.assertRaisesRegex(RequestError, "account is required"):
                    routes[route]({})
            fake.pool.get("1").auth.value = True
            with self.assertRaisesRegex(RequestError, "sign out before removing"):
                routes["accounts"]({"action": "remove", "account": "1"})
            fake._catalog = (time.time(), [{"id": "cached"}])
            routes["logout"]({"account": "1"})
            self.assertFalse(fake.pool.get("1").auth.signed_in())
            self.assertIsNone(fake._catalog)
            routes["accounts"]({"action": "remove", "account": "1"})
            self.assertEqual(self.provider(directory).store.ids(), ())

    def test_status_marks_a_nearly_full_account_as_draining(self):
        stores = {}

        class Limited(self.Fake):
            def new_account(self, slot):
                return Account(slot, _Auth(), slot)

            def account_status(self, account):
                return account.auth.status()

            def limits(self, account):
                return stores[account.id]

        with tempfile.TemporaryDirectory() as directory:
            limited = Limited(ProviderContext(config=None, directory=Path(directory)))
            for slot, percent in (("1", 95.0), ("2", 20.0)):
                limited.store.add()
                limited.pool.add(limited.new_account(slot))
                store = LimitsStore(slot, lambda p=percent: (Limit("5 hour", p),))
                store.current()
                stores[slot] = store
            accounts = limited.status().accounts
        self.assertEqual([a.draining for a in accounts], [True, False])


if __name__ == "__main__":
    unittest.main()
