#!/usr/bin/env python3
"""Record what the Python translation core does, as language-neutral cases.

Runs the unit test suite with the pure translation entry points wrapped, and
writes every distinct call as JSON: the request bodies each ingress saw and
what both providers rendered from them, and every stream an encoder shaped,
step by step. `rust/tests/conformance.rs` replays the files against the Rust
implementation, so the two cannot drift without a visible diff.

    PYTHONPATH=src python3 scripts/record-conformance.py

Ids and clocks are part of the cases rather than of the implementation: each
step lists the uuids it drew, and the replay hands back the same ones.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import itertools
import json
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from llm_local_proxy.dialects.anthropic import egress as messages_egress  # noqa: E402
from llm_local_proxy.dialects.anthropic import ingress as messages_ingress  # noqa: E402
from llm_local_proxy.dialects.openai import egress as chat_egress  # noqa: E402
from llm_local_proxy.dialects.openai import ingress as chat_ingress  # noqa: E402
from llm_local_proxy.dialects.openai import responses_egress  # noqa: E402
from llm_local_proxy.dialects.openai import responses_ingress  # noqa: E402
from llm_local_proxy.errors import ProviderError, RequestError  # noqa: E402
from llm_local_proxy.providers.claude import events as claude_events  # noqa: E402
from llm_local_proxy.providers.claude import request as claude_request  # noqa: E402
from llm_local_proxy.providers.codex import events as codex_events  # noqa: E402
from llm_local_proxy.providers.codex import request as codex_request  # noqa: E402
from llm_local_proxy.providers.reasoning import ReasoningCache  # noqa: E402
from llm_local_proxy.tools import flatten  # noqa: E402

OUT = ROOT / "tests" / "conformance"

# -- deterministic ids --------------------------------------------------------

_real_uuid4 = uuid.uuid4
_counter = itertools.count(1)
#: The innermost recorded step; the uuids it draws are part of its case.
_drawing: list[list[int]] = []


def _uuid4() -> uuid.UUID:
    value = next(_counter)
    for drawn in _drawing:
        drawn.append(value)
    return uuid.UUID(int=value)


uuid.uuid4 = _uuid4


@contextlib.contextmanager
def drawing():
    drawn: list[int] = []
    _drawing.append(drawn)
    try:
        yield drawn
    finally:
        _drawing.pop()


def failure(error: BaseException) -> dict:
    """An error as the replay compares it: its class of failure and message."""
    if isinstance(error, RequestError):
        return {"kind": "request", "message": str(error)}
    if isinstance(error, ProviderError):
        return {"kind": "provider", "status": error.status, "message": str(error)}
    if type(error) in (ValueError, RuntimeError):
        return {"kind": "upstream", "message": str(error)}
    # A crash in Python (TypeError, KeyError...): the port must fail too, but
    # its wording is its own.
    return {"kind": "crash", "message": f"{type(error).__name__}: {error}"}


def dump(value):
    """The IR as JSON: a dataclass is its fields under its class name."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            field.name: dump(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
        return {"type": type(value).__name__, **fields}
    if isinstance(value, (list, tuple)):
        return [dump(item) for item in value]
    return value


def jsonable(value) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


# -- requests ----------------------------------------------------------------

PARSERS = {
    "chat": (chat_ingress, "parse"),
    "responses": (responses_ingress, "parse"),
    "messages": (messages_ingress, "parse"),
    "messages_count": (messages_ingress, "parse_count"),
}
_parse = {name: getattr(module, attr) for name, (module, attr) in PARSERS.items()}
_build_codex = codex_request.build
_build_claude = claude_request.build

#: (dialect, body json, session) in first-seen order.
bodies: dict[tuple[str, str, str], None] = {}
#: Builder calls the tests made with options the defaults do not cover.
builds: dict[str, dict] = {}


def _wrap_parse(dialect: str):
    original = _parse[dialect]

    def parse(body, session=""):
        key = None
        if isinstance(body, dict) and isinstance(session, str) and jsonable(body):
            key = (dialect, json.dumps(body, ensure_ascii=False), session)
            bodies.setdefault(key)
        request = original(body, session)
        if key is not None:
            request._origin = key  # noqa: SLF001
        return request

    return parse


def _origin(request):
    """The body a request was parsed from, if nothing has changed it since."""
    key = getattr(request, "_origin", None)
    if key is None:
        return None
    dialect, body, session = key
    try:
        fresh = _parse[dialect](json.loads(body), session)
    except Exception:  # noqa: BLE001
        return None
    return key if fresh == request else None


def _cache_state(cache) -> list | None:
    if cache is None:
        return None
    items = [[key, value] for key, value in cache._items.items()]  # noqa: SLF001
    return items if jsonable(items) else None


def _efforts(value):
    return None if value is None else sorted(str(item) for item in value)


def _record_build(provider: str, request, options: dict, cache, run) -> None:
    key = _origin(request)
    state = _cache_state(cache)
    if key is None or not jsonable(options) or (cache is not None and state is None):
        return run()
    case = {
        "provider": provider,
        "dialect": key[0],
        "body": json.loads(key[1]),
        "session": key[2],
        "options": options,
        "cache": state,
        "ir": dump(request),
    }
    with drawing() as drawn:
        try:
            result = run()
            case["expect"] = {"ok": list(copy.deepcopy(result))}
        except Exception as error:
            case["expect"] = {"error": failure(error)}
            raise
        finally:
            case["uuids"] = list(drawn)
            if jsonable(case):
                builds.setdefault(json.dumps(case, sort_keys=True), case)
    return result


def build_codex(request, cache, reasoning_efforts=None):
    options = {"reasoning_efforts": _efforts(reasoning_efforts)}
    return _record_build(
        "codex",
        request,
        options,
        cache,
        lambda: _build_codex(request, cache, reasoning_efforts),
    )


def build_claude(
    request,
    model,
    max_output=None,
    thinking=None,
    reasoning_efforts=None,
    reasoning_cache=None,
):
    options = {
        "model": model,
        "max_output": max_output,
        "thinking": thinking,
        "reasoning_efforts": _efforts(reasoning_efforts),
    }
    return _record_build(
        "claude",
        request,
        options,
        reasoning_cache,
        lambda: _build_claude(
            request, model, max_output, thinking, reasoning_efforts, reasoning_cache
        ),
    )


for _dialect, (_module, _attr) in PARSERS.items():
    setattr(_module, _attr, _wrap_parse(_dialect))
codex_request.build = build_codex
claude_request.build = build_claude


def _attempt(run) -> dict:
    with drawing() as drawn, contextlib.redirect_stderr(io.StringIO()):
        try:
            outcome = {"ok": list(run())}
        except Exception as error:  # noqa: BLE001
            outcome = {"error": failure(error)}
    outcome["uuids"] = list(drawn)
    return outcome


def request_cases() -> list[dict]:
    """Every body the suite parsed, through both providers with defaults."""
    cases = []
    for dialect, text, session in bodies:
        body = json.loads(text)
        case = {"dialect": dialect, "body": body, "session": session}
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                _parse[dialect](copy.deepcopy(body), session)
        except Exception as error:  # noqa: BLE001
            case["parse"] = {"error": failure(error)}
            cases.append(case)
            continue
        case["parse"] = {"ok": True}
        case["ir"] = dump(_parse[dialect](copy.deepcopy(body), session))

        def fresh():
            return _parse[dialect](copy.deepcopy(body), session)

        model = body.get("model") if isinstance(body.get("model"), str) else ""
        case["codex"] = _attempt(lambda: _build_codex(fresh(), ReasoningCache()))
        case["claude"] = _attempt(
            lambda: _build_claude(fresh(), model, reasoning_cache=ReasoningCache())
        )
        case["claude_max_output"] = _attempt(
            lambda: _build_claude(
                fresh(), model, max_output=4096, reasoning_cache=ReasoningCache()
            )
        )
        # What a provider hands its decoder: flattened tool names to restore.
        try:
            case["names"] = {
                key: list(value) for key, value in flatten(fresh().tools)[1].items()
            }
        except Exception as error:  # noqa: BLE001
            case["names"] = {"error": failure(error)}
        cases.append(case)
    return cases


# -- streams -----------------------------------------------------------------

ENCODERS = {
    "chat": chat_egress.ChunkEncoder,
    "messages": messages_egress.MessageEncoder,
    "responses": responses_egress.ResponseEncoder,
}
DECODERS = {
    "codex": codex_events.CodexDecoder,
    "claude": claude_events.ClaudeDecoder,
}
streams: list[dict] = []
decoders: list[dict] = []
#: The encoder step being recorded; the events its decoder returns join it.
_encoding: list[dict] = []


def _decoded(op: str, events) -> None:
    if _encoding:
        _encoding[-1].setdefault("events", []).append(
            {"op": op, "events": dump(events)}
        )


def _wrap_decoder(kind: str, cls) -> None:
    init = cls.__init__

    def __init__(self, *args, **kwargs):
        with drawing() as drawn:
            init(self, *args, **kwargs)
        names = getattr(self, "names", {})
        self._case = {
            "kind": kind,
            "names": {key: list(value) for key, value in names.items()},
            "uuids": list(drawn),
            "steps": [],
        }
        decoders.append(self._case)

    cls.__init__ = __init__

    def wrap(name: str):
        original = getattr(cls, name)

        def method(self, *args):
            case = getattr(self, "_case", None)
            if case is None:
                return original(self, *args)
            step = {"op": name}
            if args:
                step["input"] = copy.deepcopy(args[0])
            with drawing() as drawn:
                try:
                    result = original(self, *args)
                    step["output"] = dump(result)
                except Exception as error:
                    step["error"] = failure(error)
                    raise
                finally:
                    step["uuids"] = list(drawn)
                    if _encoding and drawn:
                        # So an encoder can be replayed without its decoder.
                        _encoding[-1].setdefault("decoder_uuids", []).extend(drawn)
                    if jsonable(step):
                        case["steps"].append(step)
                    else:
                        case["broken"] = True
            _decoded(name, result)
            return result

        setattr(cls, name, method)

    wrap("decode")
    wrap("finish")


def _describe(decoder) -> dict | None:
    """A decoder the replay can construct itself, while it is still unused."""
    case = getattr(decoder, "_case", None)
    if case is None or type(decoder) not in DECODERS.values() or case["steps"]:
        return None
    return {"kind": case["kind"], "names": case["names"]}


def _tap(decoder) -> None:
    """Record what a decoder the replay cannot build (a test double) returns."""
    for name in ("decode", "finish"):
        original = getattr(decoder, name)

        def method(*args, _original=original, _name=name):
            result = _original(*args)
            _decoded(_name, result)
            return result

        setattr(decoder, name, method)


def _wrap_encoder(kind: str, cls) -> None:
    init = cls.__init__

    def __init__(self, model, decoder, request=None):
        with drawing() as drawn:
            if kind == "responses":
                init(self, model, decoder, request)
            else:
                init(self, model, decoder)
        described = _describe(decoder)
        if described is None and type(decoder) not in DECODERS.values():
            _tap(decoder)
        origin = _origin(request) if request is not None else None
        if request is not None and origin is None:
            return
        case = {
            "encoder": kind,
            "model": model,
            "decoder": described,
            "request": None
            if origin is None
            else {
                "dialect": origin[0],
                "body": json.loads(origin[1]),
                "session": origin[2],
                "ir": dump(request),
            },
            "uuids": list(drawn),
            "steps": [],
        }
        self._case = case
        self._recording = False
        streams.append(case)

    cls.__init__ = __init__

    def wrap(name: str):
        original = getattr(cls, name)

        def method(self, *args):
            case = getattr(self, "_case", None)
            # Nested: `finish` and `result` drain through the same paths.
            if case is None or case.get("broken") or self._recording:
                return original(self, *args)
            step = {"op": name}
            if args:
                if not jsonable(args[0]):
                    case["broken"] = True
                    return original(self, *args)
                step["input"] = copy.deepcopy(args[0])
            step["id"] = self.id
            if hasattr(self, "created"):
                step["created"] = self.created
            self._recording = True
            _encoding.append(step)
            with drawing() as drawn:
                try:
                    result = original(self, *args)
                    step["output"] = copy.deepcopy(result)
                except Exception as error:
                    step["error"] = failure(error)
                    raise
                finally:
                    _encoding.pop()
                    self._recording = False
                    step["uuids"] = list(drawn)
                    if jsonable(step):
                        case["steps"].append(step)
                    else:
                        case["broken"] = True
            return result

        setattr(cls, name, method)

    for name in ("start", "feed", "finish", "result", "error"):
        wrap(name)


for _kind, _cls in DECODERS.items():
    _wrap_decoder(_kind, _cls)
for _kind, _cls in ENCODERS.items():
    _wrap_encoder(_kind, _cls)


def distinct(cases: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for case in cases:
        if case.get("broken") or not case["steps"]:
            continue
        seen.setdefault(json.dumps(case, sort_keys=True), case)
    return list(seen.values())


# -- run ---------------------------------------------------------------------


def write(name: str, cases: list[dict]) -> None:
    path = OUT / name
    with path.open("w") as file:
        for case in cases:
            file.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(f"{path.relative_to(ROOT)}: {len(cases)} cases")


@contextlib.contextmanager
def _numbered_uuids():
    """The golden suite's own id patch, kept visible to the recorder.

    It replaces `uuid.uuid4` outright, which would hide those draws; restarting
    this recorder's counter yields the same ids and still records them.
    """
    global _counter
    previous, _counter = _counter, itertools.count(1)
    try:
        yield
    finally:
        _counter = previous


def main() -> None:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    sys.modules["matrix.test_golden"]._fixed_uuids = _numbered_uuids  # noqa: SLF001
    with contextlib.redirect_stderr(io.StringIO()) as log:
        result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    if not result.wasSuccessful():
        sys.stderr.write(log.getvalue())
        raise SystemExit("the suite must pass before its behaviour is recorded")
    OUT.mkdir(exist_ok=True)
    write("requests.jsonl", request_cases())
    write("builds.jsonl", list(builds.values()))
    write("streams.jsonl", distinct(streams))
    write("decoders.jsonl", distinct(decoders))


if __name__ == "__main__":
    main()
