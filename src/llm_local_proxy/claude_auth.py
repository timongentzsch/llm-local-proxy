"""Claude subscription OAuth: the proxy keeps its own token pair.

Unlike the Codex upstream there is no binary that owns the login. The proxy
performs the same authorization-code + PKCE flow as the Claude Code CLI
(against the same client id) and stores the resulting token pair in a private
file next to the config. Because the proxy refreshes its own refresh token,
it does not contend with the token pairs held by Claude Code on other
machines of the same subscription.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json
from .auth import Auth
from .status import ProviderStatus

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://platform.claude.com/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
MANUAL_REDIRECT_URL = "https://platform.claude.com/oauth/code/callback"
SCOPE = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers"
OAUTH_BETA = "oauth-2025-04-20"
REFRESH_SKEW_SECONDS = 120
USER_AGENT = "llm-local-proxy/0.1.0"


class ClaudeAuthError(RuntimeError):
    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


class ClaudeAuth(Auth):
    """The proxy's own Claude subscription login, independent of Claude Code."""

    def __init__(self, path: Path, timeout: int = 30):
        self.path = path
        self.timeout = timeout
        self._lock = threading.Lock()
        self._verifier = ""
        self._state = ""

    # -- private store ----------------------------------------------------

    def _read(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _write(self, value: dict[str, Any]) -> None:
        atomic_write_json(self.path, value)

    def _login_path(self) -> Path:
        return self.path.with_name(self.path.name + ".login")

    def _read_login(self) -> dict[str, Any]:
        try:
            value = json.loads(self._login_path().read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_login(self, value: dict[str, Any]) -> None:
        atomic_write_json(self._login_path(), value)

    def _clear_login(self) -> None:
        self._login_path().unlink(missing_ok=True)

    # -- login flow ---------------------------------------------------------

    def login_start(self) -> dict[str, Any]:
        verifier = _b64url(secrets.token_bytes(32))
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        state = secrets.token_urlsafe(24)
        query = urllib.parse.urlencode(
            {
                # The CLI sends code=true for the manual (paste the code) flow.
                "code": "true",
                "client_id": CLIENT_ID,
                "response_type": "code",
                "redirect_uri": MANUAL_REDIRECT_URL,
                "scope": SCOPE,
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        with self._lock:
            self._verifier = verifier
            self._state = state
            self._write_login({"verifier": verifier, "state": state})
        return {"url": f"{AUTHORIZE_URL}?{query}"}

    def finish(self, code: str) -> dict[str, Any]:
        value = self._extract_code(code)
        with self._lock:
            login = self._read_login()
            verifier = str(login.get("verifier") or self._verifier)
            if not verifier:
                raise ClaudeAuthError("start a Claude login first", 400)
            expected_state = str(login.get("state") or self._state)
            pasted_state = self._extract_state(code)
            if pasted_state and expected_state and pasted_state != expected_state:
                raise ClaudeAuthError("state does not match this login", 400)
            # The token endpoint wants a JSON body, and the authorization_code
            # grant rejects one without `state` ("Invalid request format");
            # both verified against the CLI 2.1.235 source and a live probe.
            state = str(login.get("state") or self._state or pasted_state)
            token = self._token_request(
                {
                    "grant_type": "authorization_code",
                    "code": value,
                    "redirect_uri": MANUAL_REDIRECT_URL,
                    "client_id": CLIENT_ID,
                    "code_verifier": verifier,
                    "state": state,
                }
            )
            if not token.get("access_token"):
                raise ClaudeAuthError(
                    "Claude OAuth response is missing access_token", 502
                )
            self._write(token)
            self._verifier = ""
            self._state = ""
            self._clear_login()
            return self.status().payload()

    @staticmethod
    def _extract_state(value: str) -> str:
        if "state=" in value:
            query = value.partition("?")[2] if "?" in value else value
            for pair in query.split("&"):
                if pair.startswith("state="):
                    return pair[6:].strip()
            return ""
        if "#" in value and "code=" not in value:
            return value.partition("#")[2].strip()
        return ""

    @staticmethod
    def _extract_code(value: str) -> str:
        code = value.strip()
        if "code=" in code:
            query = code.partition("?")[2] if "?" in code else code
            for pair in query.split("&"):
                if pair.startswith("code="):
                    return pair[5:].strip()
        if "#" in code:
            code = code.partition("#")[0].strip()
        if not code:
            raise ClaudeAuthError("code is required", 400)
        return code

    def logout(self) -> None:
        with self._lock:
            self._verifier = ""
            self._state = ""
            self._clear_login()
        self.path.unlink(missing_ok=True)

    # -- token access -------------------------------------------------------

    def signed_in(self) -> bool:
        value = self._read()
        return bool(value and value.get("access_token"))

    def status(self) -> ProviderStatus:
        value = self._read()
        if not value or not value.get("access_token"):
            return ProviderStatus()
        return ProviderStatus(
            signed_in=True,
            account=str(value.get("subscription_type") or "claude"),
        )

    def access_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            value = self._read()
            if not value or not value.get("access_token"):
                raise ClaudeAuthError(
                    "not signed in to Claude; use the sign in button on the status page"
                )
            expires_at = int(value.get("expires_at", 0) or 0)
            stale = expires_at <= time.time() + REFRESH_SKEW_SECONDS
            if (force_refresh or stale) and value.get("refresh_token"):
                value = self._refresh(value)
            return str(value["access_token"])

    def _refresh(self, value: dict[str, Any]) -> dict[str, Any]:
        scopes = value.get("scopes")
        scope = " ".join(str(item) for item in scopes) if scopes else SCOPE
        token = self._token_request(
            {
                "grant_type": "refresh_token",
                "refresh_token": str(value["refresh_token"]),
                "client_id": CLIENT_ID,
                "scope": scope,
            }
        )
        if not token.get("access_token"):
            raise ClaudeAuthError("Claude token refresh is missing access_token", 502)
        merged = {**value, **token}
        self._write(merged)
        return merged

    def _token_request(self, fields: dict[str, str]) -> dict[str, Any]:
        # The CLI posts JSON (not form data) to the token endpoint; the
        # authorization_code grant answers form bodies with 400
        # "Invalid request format".
        payload = json.dumps(fields, separators=(",", ":")).encode()
        request = urllib.request.Request(
            TOKEN_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            raise ClaudeAuthError(
                f"Claude OAuth failed: {_error_message(error)}", 400
            ) from error
        except urllib.error.URLError as error:
            raise ClaudeAuthError(
                f"Claude OAuth unreachable: {error.reason}", 502
            ) from error
        return _normalize(raw)


def _error_message(error: urllib.error.HTTPError) -> str:
    raw = error.read().decode("utf-8", "replace")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw or str(error.reason)
    if isinstance(value, dict):
        error = value.get("error")
        if isinstance(error, dict):
            return str(
                error.get("message")
                or error.get("error_description")
                or error.get("type")
                or raw
            )
        return str(
            value.get("error_description")
            or value.get("message")
            or value.get("error")
            or raw
        )
    return raw


def _normalize(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    token: dict[str, Any] = {}
    access = value.get("access_token")
    refresh = value.get("refresh_token")
    if access:
        token["access_token"] = str(access)
    if refresh:
        token["refresh_token"] = str(refresh)
    expires_in = value.get("expires_in")
    if isinstance(expires_in, (int, float)) and expires_in > 0:
        token["expires_at"] = int(time.time() + expires_in)
    scope = value.get("scope")
    if isinstance(scope, str) and scope:
        token["scopes"] = scope.split()
    subscription = value.get("subscriptionType") or value.get("subscription_type")
    if subscription:
        token["subscription_type"] = str(subscription)
    return token
