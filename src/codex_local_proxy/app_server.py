from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


class RpcError(RuntimeError):
    pass


def _jwt_payload(token: str) -> dict[str, Any]:
    try:
        raw = token.split(".")[1]
        raw += "=" * (-len(raw) % 4)
        value = json.loads(base64.urlsafe_b64decode(raw))
        return value if isinstance(value, dict) else {}
    except (IndexError, ValueError, json.JSONDecodeError):
        return {}


class AppServer:
    """Small synchronous JSONL client; Codex owns OAuth and token refresh."""

    def __init__(self, binary: str, codex_home: Path):
        codex_home.mkdir(mode=0o700, parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CODEX_HOME"] = str(codex_home)
        self.auth_path = codex_home / "auth.json"
        self._pending: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._proc = subprocess.Popen(
            [
                binary,
                "app-server",
                "-c",
                'cli_auth_credentials_store="file"',
                "--stdio",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._read, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self.call(
            "initialize",
            {
                "clientInfo": {
                    "name": "codex_local_proxy",
                    "title": "Codex Local Proxy",
                    "version": "0.1.0",
                }
            },
        )
        self.notify("initialized", {})

    def _send(self, message: dict[str, Any]) -> None:
        if not self._proc.stdin or self._proc.poll() is not None:
            raise RpcError("codex app-server is not running")
        with self._lock:
            self._proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
            self._proc.stdin.flush()

    def call(
        self, method: str, params: dict[str, Any] | None = None, timeout: int = 30
    ) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            reply: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._pending[request_id] = reply
        try:
            self._send({"method": method, "id": request_id, "params": params or {}})
            message = reply.get(timeout=timeout)
        except queue.Empty as error:
            raise RpcError(f"{method} timed out") from error
        finally:
            self._pending.pop(request_id, None)
        if "error" in message:
            error = message["error"]
            detail = (
                error.get("message", str(error)) if isinstance(error, dict) else error
            )
            raise RpcError(f"{method}: {detail}")
        result = message.get("result", {})
        return result if isinstance(result, dict) else {"value": result}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"method": method, "params": params})

    def _read(self) -> None:
        assert self._proc.stdout
        for line in self._proc.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            request_id = message.get("id")
            pending = self._pending.get(request_id)
            if pending:
                pending.put(message)
            elif request_id is not None and message.get("method"):
                self._send(
                    {
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": "unsupported client method",
                        },
                    }
                )
        failure = {"error": {"message": "codex app-server stopped"}}
        for pending in list(self._pending.values()):
            try:
                pending.put_nowait(failure)
            except queue.Full:
                pass

    def _drain_stderr(self) -> None:
        assert self._proc.stderr
        for _ in self._proc.stderr:
            pass

    def _auth(self) -> dict[str, Any]:
        for _ in range(3):
            try:
                data = json.loads(self.auth_path.read_text())
                return data if isinstance(data, dict) else {}
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(0.02)
        return {}

    def token(self, force_refresh: bool = False) -> tuple[str, str]:
        auth = self._auth()
        tokens = auth.get("tokens", {}) if isinstance(auth.get("tokens"), dict) else {}
        access = str(tokens.get("access_token", ""))
        expiry = int(_jwt_payload(access).get("exp", 0) or 0)
        if force_refresh or not access or expiry <= time.time() + 120:
            self.call("account/read", {"refreshToken": True})
            auth = self._auth()
            tokens = (
                auth.get("tokens", {}) if isinstance(auth.get("tokens"), dict) else {}
            )
            access = str(tokens.get("access_token", ""))
        if not access:
            raise RpcError("not signed in; open the proxy status page")
        account_id = str(tokens.get("account_id", ""))
        if not account_id:
            claims = _jwt_payload(access)
            account_id = str(
                claims.get("https://api.openai.com/auth.chatgpt_account_id", "")
            )
        if not account_id:
            raise RpcError("ChatGPT account id is missing")
        return access, account_id

    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
