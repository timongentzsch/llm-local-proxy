#!/usr/bin/env python3
"""A stand-in for the `codex` binary: its app-server and `debug models`.

Signed in exactly when `$CODEX_HOME/auth.json` exists, like the real one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AUTH = Path(os.environ.get("CODEX_HOME", ".")) / "auth.json"

MODELS = [
    {
        "id": "gpt-test",
        "model": "gpt-test",
        "displayName": "GPT Test",
        "supportedReasoningEfforts": [
            {"reasoningEffort": "low"},
            {"reasoningEffort": "medium"},
            {"reasoningEffort": "high"},
            {"reasoningEffort": "xhigh"},
        ],
        "defaultReasoningEffort": "medium",
        "inputModalities": ["text", "image"],
        "isDefault": True,
    },
    {"id": "gpt-plain", "displayName": None},
]

LIMITS = {
    "rateLimits": {"limitId": "codex"},
    "rateLimitsByLimitId": {
        "codex": {
            "limitId": "codex",
            "limitName": "Codex",
            "primary": {
                "usedPercent": 16,
                "windowDurationMins": 300,
                "resetsAt": 1787234107,
            },
            "secondary": {"usedPercent": 4, "windowDurationMins": 10080},
        },
        "spark": {
            "limitId": "spark",
            "limitName": "Spark",
            "primary": {"usedPercent": 1, "windowDurationMins": 10080},
            "secondary": None,
        },
    },
}


def answer(method: str):
    if method == "account/read":
        account = (
            {"type": "chatgpt", "email": "codex@example.com", "planType": "pro"}
            if AUTH.exists()
            else None
        )
        return {"account": account, "requiresOpenaiAuth": True}
    if method == "account/rateLimits/read":
        return LIMITS
    if method == "model/list":
        return {"data": MODELS, "nextCursor": None}
    if method == "account/login/start":
        return {
            "loginId": "login-1",
            "verificationUrl": "https://auth.example/device",
            "userCode": "ABCD-1234",
        }
    if method == "account/logout":
        AUTH.unlink(missing_ok=True)
        return {}
    return {}


def main() -> None:
    if sys.argv[1:3] == ["debug", "models"]:
        print(json.dumps({"models": [{"slug": "gpt-test", "context_window": 400000}]}))
        return
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if "id" in message:
            result = answer(message.get("method", ""))
            sys.stdout.write(json.dumps({"id": message["id"], "result": result}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
