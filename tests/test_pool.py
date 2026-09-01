from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_local_proxy.errors import RequestError
from llm_local_proxy.providers.auth import MultiAuth
from llm_local_proxy.providers.pool import (
    AUTH_FAILURE_COOLDOWN_SECONDS,
    Account,
    AccountPool,
    AccountStore,
    stored_account_ids,
)
from llm_local_proxy.status import ProviderStatus


class _Auth:
    def __init__(self, signed_in=True):
        self.value = signed_in

    def signed_in(self):
        return self.value

    def login_start(self, account_id=""):
        return {"url": "https://example.test/login"}

    def logout(self, account_id=""):
        self.value = False

    def status(self):
        return ProviderStatus(signed_in=self.value)


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

        self.assertEqual(pool.discover(discover, RuntimeError), "two")
        self.assertEqual(pool.account_error("1"), "status 401")
        self.assertEqual(pool.discover(discover, RuntimeError), "two")
        stale = False
        after_cooldown = time.time() + AUTH_FAILURE_COOLDOWN_SECONDS
        with patch(
            "llm_local_proxy.providers.pool.time.time", return_value=after_cooldown
        ):
            self.assertEqual(pool.discover(discover, RuntimeError), "one")
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

    def test_canonical_credentials_seed_the_first_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "accounts" / "claude"
            credential = root / "7" / "credentials.json"
            credential.parent.mkdir(parents=True)
            credential.write_text("{}")
            ids = stored_account_ids(root, "credentials.json")
            self.assertEqual(ids, ("7",))
            self.assertEqual(AccountStore(Path(directory), "claude", ids).ids(), ("7",))


class MultiAuthTest(unittest.TestCase):
    def test_login_and_logout_target_one_slot(self):
        first, second = _Auth(), _Auth(False)
        accounts = (("1", first), ("2", second))
        auth = MultiAuth(lambda: accounts)
        self.assertEqual(auth.login_start("2")["account"], "2")
        auth.logout("1")
        self.assertFalse(first.signed_in())
        self.assertFalse(second.signed_in())

    def test_login_without_a_target_is_rejected(self):
        auth = MultiAuth(lambda: (("1", _Auth()), ("2", _Auth(False))))
        with self.assertRaisesRegex(RequestError, "unknown account"):
            auth.login_start()


if __name__ == "__main__":
    unittest.main()
