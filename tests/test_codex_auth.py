import unittest

from llm_local_proxy.app_server import RpcError
from llm_local_proxy.codex_auth import CodexAuth


class _App:
    def __init__(self, account=None, rate_limits=None, login=None, fail=()):
        self.responses = {
            "account/read": {"account": account} if account else {},
            "account/rateLimits/read": rate_limits or {},
            "account/login/start": login or {},
        }
        self.fail = set(fail)
        self.calls = []

    def call(self, method, params=None):
        self.calls.append((method, params))
        if method in self.fail:
            raise RpcError(f"{method} failed")
        return self.responses[method]


class CodexAuthTest(unittest.TestCase):
    def test_signed_out_status_is_the_empty_card(self):
        status = CodexAuth(_App()).status()
        self.assertFalse(status.signed_in)
        self.assertEqual(status.account, "")
        self.assertEqual(status.limits, ())
        self.assertIsNone(status.updated_at)

    def test_status_normalizes_windows_into_sorted_limits(self):
        app = _App(
            account={
                "type": "chatgpt",
                "email": "user@example.com",
                "planType": "demo",
            },
            rate_limits={
                "rateLimitsByLimitId": {
                    "fake_limit": {
                        "limitId": "fake_limit",
                        "limitName": "Fake Limit",
                        "primary": {
                            "usedPercent": 16,
                            "windowDurationMins": 300,
                            "resetsAt": 1787234107,
                        },
                        "secondary": {
                            "usedPercent": 4,
                            "windowDurationMins": 10080,
                            "resetsAt": 1787820907,
                        },
                    },
                    "unnamed": {
                        "limitId": "unnamed",
                        "limitName": None,
                        "primary": {"usedPercent": 1, "windowDurationMins": 120},
                        "secondary": None,
                    },
                }
            },
        )
        status = CodexAuth(app).status()
        self.assertTrue(status.signed_in)
        self.assertEqual(status.account, "user@example.com · demo")
        self.assertIsNotNone(status.updated_at)
        self.assertEqual(
            [(limit.label, limit.used_percent) for limit in status.limits],
            [
                ("unnamed · 2 hour", 1.0),
                ("Fake Limit · 5 hour", 16.0),
                ("Fake Limit · weekly", 4.0),
            ],
        )
        self.assertEqual(status.limits[1].resets_at, 1787234107)

    def test_status_survives_a_rate_limit_read_failure(self):
        app = _App(
            account={"email": "user@example.com"}, fail={"account/rateLimits/read"}
        )
        status = CodexAuth(app).status()
        self.assertTrue(status.signed_in)
        self.assertEqual(status.limits, ())

    def test_login_start_is_normalized_to_url_and_code(self):
        app = _App(
            login={
                "verificationUrl": "https://example.test/device",
                "userCode": "AB-CD",
            }
        )
        self.assertEqual(
            CodexAuth(app).login_start(),
            {"url": "https://example.test/device", "code": "AB-CD"},
        )


if __name__ == "__main__":
    unittest.main()
