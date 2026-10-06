#!/usr/bin/env python3
"""The ChatGPT login behind the Codex provider.

    tests/e2e/codex_login.py target/release/llm-local-proxy

Starts the proxy with an expired Codex token and checks that it refreshes the
token before the first request, stores the new pair in the CLI's `auth.json`
layout, and sends the refreshed token upstream.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import HERE, Client, free_port, jwt, prepare, text, wait_for


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="llp-native-"))
    port, upstream_port = free_port(), free_port()
    config = prepare(root, port)
    auth_path = root / "codex" / "accounts" / "1" / "auth.json"
    auth = json.loads(auth_path.read_text())
    auth["tokens"]["access_token"] = jwt({"exp": int(time.time()) - 60})
    auth["unknown_field"] = "kept"
    auth_path.write_text(json.dumps(auth))
    log = root / "upstream.jsonl"
    upstream = subprocess.Popen(
        [sys.executable, str(HERE / "fake_upstream.py"), str(upstream_port), str(log)]
    )
    wait_for(upstream_port, upstream, "fake upstream")
    proxy = subprocess.Popen(
        [str(Path(sys.argv[1]).resolve()), "--config", str(config)],
        env={
            **os.environ,
            "LLM_PROXY_TEST_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    failures = []

    def check(name: str, ok: bool, detail: object = "") -> None:
        print(("PASS " if ok else "FAIL ") + name, detail if not ok else "")
        if not ok:
            failures.append(name)

    try:
        wait_for(port, proxy, "proxy")
        client = Client(port)
        answer = client.call("POST", *text("gpt-test", "say pong", False)[0])
        check(
            "a request with an expired token succeeds", answer["status"] == 200, answer
        )

        sent = [json.loads(line) for line in log.read_text().splitlines()]
        refreshes = [e for e in sent if e["path"] == "/oauth/token"]
        check(
            "the token was refreshed exactly once", len(refreshes) == 1, len(refreshes)
        )
        check(
            "the refresh is the CLI's JSON grant",
            refreshes
            and refreshes[0]["body"]
            == {
                "grant_type": "refresh_token",
                "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
                "refresh_token": "rt-codex-1",
            },
            refreshes,
        )
        refreshed = jwt({"exp": 4102444800, "refreshed_from": "rt-codex-1"})
        calls = [e for e in sent if e["path"] == "/backend-api/codex/responses"]
        check(
            "upstream calls carry the refreshed token and the account id",
            calls
            and all(
                e["headers"]["authorization"] == "Bearer " + refreshed
                and e["headers"]["chatgpt-account-id"] == "acct-e2e"
                for e in calls
            ),
            [e["headers"] for e in calls][:1],
        )
        stored = json.loads(auth_path.read_text())
        check(
            "the new pair is stored in the CLI layout",
            stored["tokens"]["access_token"] == refreshed
            and stored["tokens"]["refresh_token"] == "rt-codex-2"
            and stored["tokens"]["account_id"] == "acct-e2e"
            and stored["tokens"]["id_token"] == auth["tokens"]["id_token"]
            and stored["last_refresh"].endswith("Z")
            and stored["unknown_field"] == "kept",
            {k: v for k, v in stored.items() if k != "tokens"},
        )
        check("the file stays private", (auth_path.stat().st_mode & 0o077) == 0)

        status = client.call("GET", "/api/status")["body"]
        codex = next(p for p in status["providers"] if p["name"] == "codex")
        check(
            "the status card shows the account and its bars",
            codex["accounts"][0]["account"] == "codex@example.com · pro"
            and len(codex["accounts"][0]["limits"]) == 3,
            codex["accounts"][0],
        )
        client.call("POST", "/api/codex/logout", {"account": "1"})
        sent = [json.loads(line) for line in log.read_text().splitlines()]
        revoked = [e["body"] for e in sent if e["path"] == "/oauth/revoke"]
        check(
            "signing out revokes the refresh token and removes the file",
            revoked
            == [
                {
                    "token": "rt-codex-2",
                    "token_type_hint": "refresh_token",
                    "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
                }
            ]
            and not auth_path.exists(),
            revoked,
        )
    finally:
        proxy.terminate()
        upstream.kill()
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
