#!/usr/bin/env python3
"""Run the Python reference and the Rust port side by side and diff them.

Each implementation gets its own config directory, its own fake upstream and
the same scripted client. Three things must then agree: what clients received,
what was sent upstream, and what was left on disk.

    tests/e2e/compare.py --rust path/to/llm-local-proxy [-v] [--only WORD]

Ids, timestamps and generated secrets are normalised; everything else is
compared exactly.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY = "e2e-master-key-0123456789abcdefghij"

TOOL_CHAT = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]
TOOL_RESPONSES = [{"type": "function", **TOOL_CHAT[0]["function"]}]
TOOL_MESSAGES = [
    {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": TOOL_CHAT[0]["function"]["parameters"],
    }
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def jwt(claims: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"header.{body}.signature"


def write(path: Path, value, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    path.chmod(mode)


def prepare(root: Path, port: int) -> Path:
    """A config directory with one signed-in account per provider."""
    config = root / "config"
    codex_home = root / "codex"
    write(
        config / "config.toml",
        f'host = "127.0.0.1"\nport = {port}\napi_key = "{KEY}"\n'
        f'codex_home = "{codex_home}"\ncodex_binary = "{HERE / "fake_codex.py"}"\n'
        "request_timeout = 30\n",
    )
    for provider in ("claude", "codex"):
        write(config / "accounts" / provider / "slots.json", {"accounts": ["1"]})
    # Expired, so the first use has to refresh and store the new pair.
    write(
        config / "accounts" / "claude" / "1" / "credentials.json",
        {
            "access_token": "at-stale",
            "refresh_token": "rt-old",
            "expires_at": int(time.time()) - 60,
            "scopes": ["user:profile", "user:inference"],
        },
    )
    access = jwt({"exp": 4102444800})
    identity = jwt(
        {
            "email": "codex@example.com",
            "https://api.openai.com/auth": {"chatgpt_plan_type": "pro"},
        }
    )
    write(
        codex_home / "accounts" / "1" / "auth.json",
        {
            "tokens": {
                "id_token": identity,
                "access_token": access,
                "refresh_token": "rt-codex-1",
                "account_id": "acct-e2e",
            }
        },
    )
    return config / "config.toml"


def wait_for(port: int, process: subprocess.Popen, what: str) -> None:
    for _ in range(200):
        if process.poll() is not None:
            raise SystemExit(f"{what} exited with {process.returncode}")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.05)
    raise SystemExit(f"{what} did not start listening on {port}")


class Client:
    def __init__(self, port: int):
        self.port = port

    def call(self, method, path, body=None, headers=None, key=KEY, host=None):
        sent = {"Host": host or f"127.0.0.1:{self.port}"}
        if key:
            sent["Authorization"] = f"Bearer {key}"
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            sent["Content-Type"] = "application/json"
        sent.update(headers or {})
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        connection.request(method, path, body=data, headers=sent)
        response = connection.getresponse()
        raw = response.read().decode("utf-8", "replace")
        content_type = response.getheader("Content-Type", "")
        connection.close()
        return {
            "status": response.status,
            "content_type": content_type,
            "body": parse(raw, content_type),
        }


def parse(raw: str, content_type: str):
    """JSON as JSON and a stream as its frames, so escaping does not matter."""
    if content_type.startswith("text/event-stream"):
        frames = []
        for block in raw.split("\n\n"):
            lines = block.split("\n")
            event = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
            data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
            if data is None:
                if block.strip():
                    frames.append({"raw": block})
                continue
            frames.append(
                {"event": event, "data": data if data == "[DONE]" else json.loads(data)}
            )
        return frames
    if content_type.startswith("application/json"):
        return json.loads(raw)
    if content_type.startswith("text/html"):
        return f"<html of {len(raw)} bytes>"
    return raw


def text(model: str, prompt: str, stream: bool):
    """The same question in each of the three dialects."""
    return [
        (
            "/openai/v1/chat/completions",
            {
                "model": model,
                "stream": stream,
                "max_tokens": 64,
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": prompt},
                ],
                "tools": TOOL_CHAT,
            },
        ),
        (
            "/openai/v1/responses",
            {
                "model": model,
                "stream": stream,
                "instructions": "Be terse.",
                "input": prompt,
                "tools": TOOL_RESPONSES,
                "reasoning": {"effort": "low", "summary": "auto"},
            },
        ),
        (
            "/anthropic/v1/messages",
            {
                "model": model,
                "stream": stream,
                "max_tokens": 64,
                "system": "Be terse.",
                "messages": [{"role": "user", "content": prompt}],
                "tools": TOOL_MESSAGES,
            },
        ),
    ]


def script(client: Client):
    """Every request of the comparison, as (label, response)."""
    get, post = (
        lambda path, **kw: client.call("GET", path, **kw),
        lambda path, body, **kw: client.call("POST", path, body, **kw),
    )
    yield "healthz", get("/healthz", key=None)
    yield "dashboard", get("/", key=None)
    yield "favicon", get("/favicon.ico", key=None)
    # First: the stale Claude token is refreshed on the way.
    yield "models openai", get("/openai/v1/models")
    yield "models anthropic", get("/anthropic/v1/models")
    yield "models bare", get("/v1/models?q=gpt")
    yield "models filter", get("/openai/v1/models?q=Claude%20Test&refresh=1")
    yield "models count", get("/v1/models/count")
    yield "status", get("/api/status")
    yield "me", get("/api/me")

    for model in ("claude-test", "gpt-test"):
        for stream in (False, True):
            for prompt in ("say pong", "use a tool"):
                for path, body in text(model, prompt, stream):
                    kind = "stream" if stream else "plain"
                    yield f"{model} {path} {kind} {prompt!r}", post(path, body)
    # The upstream refuses: a status for a plain request, a frame for a stream.
    for model in ("claude-test", "gpt-test"):
        for stream in (False, True):
            path, body = text(model, "limited", stream)[0]
            yield f"{model} limited stream={stream}", post(path, body)
            path, body = text(model, "limited", stream)[2]
            yield f"{model} limited messages stream={stream}", post(path, body)

    # A tool round, replayed: signed reasoning must survive every dialect.
    yield (
        "chat tool round",
        post(
            "/v1/chat/completions",
            {
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "use a tool"},
                    {
                        "role": "assistant",
                        "content": "Looking it up.",
                        "tool_calls": [
                            {
                                "id": "toolu_up1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"Berlin"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "toolu_up1", "content": "17C"},
                ],
                "tools": TOOL_CHAT,
            },
        ),
    )
    yield (
        "messages tool round on codex",
        post(
            "/anthropic/v1/messages",
            {
                "model": "gpt-test",
                "max_tokens": 64,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "use a tool"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_up1",
                                "name": "get_weather",
                                "input": {"city": "Berlin"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_up1",
                                "content": "17C",
                            }
                        ],
                    },
                ],
                "tools": TOOL_MESSAGES,
            },
            headers={"X-Claude-Code-Session-Id": "session-e2e"},
        ),
    )
    yield (
        "prewarm",
        post(
            "/anthropic/v1/messages",
            {
                "model": "claude-test",
                "max_tokens": 0,
                "messages": [{"role": "user", "content": "warm"}],
            },
        ),
    )

    count = {"model": "claude-test", "messages": [{"role": "user", "content": "hi"}]}
    yield "count claude", post("/anthropic/v1/messages/count_tokens", count)
    yield (
        "count codex",
        post("/anthropic/v1/messages/count_tokens", {**count, "model": "gpt-test"}),
    )

    # Refusals.
    simple = text("claude-test", "say pong", False)
    yield "no key openai", post(*simple[0], key=None)
    yield "no key anthropic", post(*simple[2], key=None)
    yield "wrong key", post(*simple[0], key="x" * 40)
    yield (
        "x-api-key",
        post(*simple[2], key=None, headers={"x-api-key": KEY}),
    )
    yield "bad host", get("/v1/models", host="evil.example")
    yield "bad host post", post(*simple[0], host="evil.example:80")
    yield (
        "bad origin",
        post(*simple[0], headers={"Origin": "http://evil.example"}),
    )
    yield (
        "own origin",
        post(*simple[0], headers={"Origin": f"http://127.0.0.1:{client.port}"}),
    )
    yield "not json", post("/v1/chat/completions", b"{nope")
    yield "not an object", post("/v1/chat/completions", [1])
    yield "empty body", client.call("POST", "/v1/chat/completions")
    yield "unknown model", post(simple[0][0], {**simple[0][1], "model": "nope"})
    yield "unknown route", post("/v1/nope", {})
    yield "unknown get", get("/v1/nope")
    yield "unknown api", post("/api/nope/login", {})
    yield "unknown provider route", post("/api/claude/nope", {})
    yield (
        "unsupported parameter",
        post(simple[0][0], {**simple[0][1], "logit_bias": {"1": 1}}),
    )
    yield (
        "stateful responses",
        post("/v1/responses", {"model": "gpt-test", "input": "hi", "store": True}),
    )
    yield (
        "missing max_tokens",
        post("/anthropic/v1/messages", {"model": "claude-test", "messages": []}),
    )

    # Named keys.
    added = post("/api/keys", {"action": "add", "name": "alice"})
    yield "key add", added
    named = (added["body"] or {}).get("key", "")
    yield "key add twice", post("/api/keys", {"action": "add", "name": "alice"})
    yield "key bad name", post("/api/keys", {"action": "add", "name": "Master!"})
    yield "keys", get("/api/keys")
    yield "named key calls a model", post(*simple[0], key=named)
    yield "named key me", get("/api/me", key=named)
    yield "named key status", get("/api/status", key=named)
    yield (
        "named key manages keys",
        post("/api/keys", {"action": "add", "name": "bob"}, key=named),
    )
    yield "usage by key", get("/api/keys")
    yield "key remove", post("/api/keys", {"action": "remove", "name": "alice"})
    yield "key remove twice", post("/api/keys", {"action": "remove", "name": "alice"})
    yield "removed key", post(*simple[0], key=named)

    # Accounts.
    yield "claude login", post("/api/claude/login", {"account": "1"})
    yield "claude login no account", post("/api/claude/login", {})
    yield "claude code missing", post("/api/claude/code", {"account": "1"})
    yield "codex login", post("/api/codex/login", {"account": "1"})
    yield "add slot", post("/api/claude/accounts", {"action": "add"})
    yield "add slot twice", post("/api/claude/accounts", {"action": "add"})
    yield (
        "remove signed in",
        post("/api/claude/accounts", {"action": "remove", "account": "1"}),
    )
    yield (
        "remove slot",
        post("/api/claude/accounts", {"action": "remove", "account": "2"}),
    )
    yield "bad action", post("/api/claude/accounts", {"action": "nope"})
    yield "status after", get("/api/status")
    yield "codex logout", post("/api/codex/logout", {"account": "1"})
    time.sleep(2.5)
    yield "codex models after logout", get("/v1/models?q=gpt&refresh=1")
    yield "codex request after logout", post(*text("gpt-test", "say pong", False)[0])
    yield "claude logout", post("/api/claude/logout", {"account": "1"})
    yield "models after logout", get("/v1/models?refresh=1")
    yield "status signed out", get("/api/status")


VOLATILE = [
    (
        re.compile(r"\b(chatcmpl-|resp_|msg_|rs_|fc_|ws_|toolu_)[0-9a-f]{24,32}\b"),
        r"\1<id>",
    ),
    (re.compile(r"llp_[A-Za-z0-9_-]{40,}"), "llp_<key>"),
    (re.compile(r"(state|code_challenge)=[A-Za-z0-9_-]+"), r"\1=<random>"),
    (re.compile(r"127\.0\.0\.1:\d+"), "127.0.0.1:<port>"),
    (re.compile(r"/tmp/[^\"' ]*"), "<tmp>"),
]
VOLATILE_KEYS = {
    "created",
    "created_at",
    "updated_at",
    "expires_at",
    "ts",
    "completed_at",
}


def normalise(value):
    if isinstance(value, dict):
        return {
            key: "<time>"
            if key in VOLATILE_KEYS and isinstance(item, (int, float))
            else normalise(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [normalise(item) for item in value]
    if isinstance(value, str):
        for pattern, replacement in VOLATILE:
            value = pattern.sub(replacement, value)
    return value


def run(name: str, command: list[str], env: dict, verbose: bool):
    """One implementation through the whole script."""
    root = Path(tempfile.mkdtemp(prefix=f"llp-e2e-{name}-"))
    port, upstream_port = free_port(), free_port()
    config = prepare(root, port)
    log = root / "upstream.jsonl"
    upstream = subprocess.Popen(
        [sys.executable, str(HERE / "fake_upstream.py"), str(upstream_port), str(log)]
    )
    wait_for(upstream_port, upstream, "fake upstream")
    output = open(root / "proxy.log", "w")  # noqa: SIM115 - closed in finally
    proxy = subprocess.Popen(
        [*command, "--config", str(config)],
        env={
            **os.environ,
            **env,
            "LLM_PROXY_TEST_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        },
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for(port, proxy, name)
        results = {}
        for label, response in script(Client(port)):
            results[label] = normalise(response)
            if verbose:
                print(f"  {name}: {label} -> {response['status']}")
        proxy.terminate()
        proxy.wait(timeout=10)
        sent = [json.loads(line) for line in log.read_text().splitlines()]
        disk = {}
        for path in sorted((root / "config").rglob("*.json")):
            relative = str(path.relative_to(root))
            disk[relative] = normalise(json.loads(path.read_text()))
        return results, normalise(sent), disk, root
    finally:
        for process in (proxy, upstream):
            if process.poll() is None:
                process.kill()
        output.close()


def diff(label: str, want, got, problems: list[str], path: str = "") -> None:
    if type(want) is not type(got):
        problems.append(f"{label}{path}: {want!r} != {got!r}"[:400])
    elif isinstance(want, dict):
        for key in want.keys() | got.keys():
            if key not in want:
                problems.append(
                    f"{label}{path}.{key}: only the port has {got[key]!r}"[:400]
                )
            elif key not in got:
                problems.append(
                    f"{label}{path}.{key}: the port lacks {want[key]!r}"[:400]
                )
            else:
                diff(label, want[key], got[key], problems, f"{path}.{key}")
    elif isinstance(want, list):
        if len(want) != len(got):
            problems.append(
                f"{label}{path}: {len(want)} items != {len(got)} items\n"
                f"    reference: {json.dumps(want, ensure_ascii=False)[:600]}\n"
                f"    port:      {json.dumps(got, ensure_ascii=False)[:600]}"
            )
        else:
            for index, (a, b) in enumerate(zip(want, got)):
                diff(label, a, b, problems, f"{path}[{index}]")
    elif want != got:
        problems.append(f"{label}{path}: {want!r} != {got!r}"[:400])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rust", required=True, help="the port's binary")
    parser.add_argument("--only", help="compare only labels containing this")
    parser.add_argument("--keep", action="store_true", help="keep the scratch dirs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    reference = run(
        "python", [sys.executable, str(HERE / "reference.py")], {}, args.verbose
    )
    port = run("rust", [str(Path(args.rust).resolve())], {}, args.verbose)

    problems: list[str] = []
    for label in reference[0]:
        if args.only and args.only not in label:
            continue
        diff(f"client: {label}", reference[0][label], port[0].get(label), problems)
    if not args.only:
        # The same requests, in the order each proxy chose to send them.
        key = lambda entry: json.dumps(entry, sort_keys=True)

        # The reference asks codex app-server for these; the port asks the
        # service itself, so only the port's log has them.
        native = (
            "/backend-api/codex/models",
            "/backend-api/wham/usage",
            "/api/accounts/deviceauth/",
            "/oauth/revoke",
        )

        def sent(entries):
            entries = [e for e in entries if not e["path"].startswith(native)]
            # The port asks the Codex transport for its effort enum once an
            # hour rather than at every catalog refresh, so how often the
            # probe was sent is the one thing allowed to differ.
            probes = {key(e) for e in entries if "__probe__" in key(e)}
            rest = [key(e) for e in entries if "__probe__" not in key(e)]
            return [json.loads(item) for item in sorted(rest) + sorted(probes)]

        diff("upstream", sent(reference[1]), sent(port[1]), problems)
        diff("disk", reference[2], port[2], problems)

    for problem in problems:
        print(problem)
    print(
        f"{len(reference[0])} client exchanges, {len(reference[1])} upstream requests "
        f"({len(port[1])} from the port), {len(reference[2])} files: "
        + ("identical" if not problems else f"{len(problems)} differences")
    )
    if args.keep or problems:
        print(f"reference: {reference[3]}\nport:      {port[3]}")
    else:
        for root in (reference[3], port[3]):
            shutil.rmtree(root, ignore_errors=True)
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
