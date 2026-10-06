#!/usr/bin/env python3
"""Measure the proxy against the fake upstream.

    tests/e2e/bench.py target/release/llm-local-proxy

Reports what the proxy itself costs: its memory, threads and CPU time, and
the latency it adds. The upstream is a local stand-in on the same machine, so
absolute latencies mean little; the difference between two builds does.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import HERE, KEY, free_port, prepare, wait_for

TICK = os.sysconf("SC_CLK_TCK")


def proc(pid: int) -> dict:
    status = Path(f"/proc/{pid}/status").read_text()
    field = lambda name: int(
        next(ln for ln in status.splitlines() if ln.startswith(name)).split()[1]
    )
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "rss_mb": field("VmRSS:") / 1024,
        "peak_mb": field("VmHWM:") / 1024,
        "threads": field("Threads:"),
        "cpu_s": (int(stat[11]) + int(stat[12])) / TICK,
    }


async def request(port: int, body: dict) -> tuple[float, float, int]:
    """(time to first body byte, total time, bytes) for one request."""
    data = json.dumps(body).encode()
    started = time.perf_counter()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        b"POST /anthropic/v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        b"Content-Type: application/json\r\nConnection: close\r\n"
        + f"Authorization: Bearer {KEY}\r\nContent-Length: {len(data)}\r\n\r\n".encode()
        + data
    )
    await writer.drain()
    await reader.readuntil(b"\r\n\r\n")
    first = await reader.read(1)
    ttfb = time.perf_counter() - started
    rest = await reader.read()
    writer.close()
    return ttfb, time.perf_counter() - started, len(first) + len(rest)


def body(prompt: str, stream: bool) -> dict:
    return {
        "model": "claude-test",
        "max_tokens": 64,
        "stream": stream,
        "messages": [{"role": "user", "content": prompt}],
    }


async def measure(port: int, pid: int) -> dict:
    out = {"idle": proc(pid)}
    for _ in range(20):
        await request(port, body("say pong", False))
    for name, stream in (("plain", False), ("stream", True)):
        before = proc(pid)["cpu_s"]
        times = [(await request(port, body("say pong", stream)))[1] for _ in range(300)]
        out[name] = {
            "p50_ms": statistics.median(times) * 1000,
            "p99_ms": sorted(times)[-3] * 1000,
            "cpu_ms_per_request": (proc(pid)["cpu_s"] - before) / 300 * 1000,
        }
    for count in (50, 200):
        before = proc(pid)
        started = time.perf_counter()
        results = await asyncio.gather(
            *(request(port, body("long answer", True)) for _ in range(count)),
            return_exceptions=True,
        )
        wall = time.perf_counter() - started
        good = [r for r in results if not isinstance(r, Exception)]
        during = proc(pid)
        out[f"{count} streams"] = {
            "completed": len(good),
            "wall_s": wall,
            "ttfb_p50_ms": statistics.median(r[0] for r in good) * 1000,
            "ttfb_max_ms": max(r[0] for r in good) * 1000,
            "cpu_s": during["cpu_s"] - before["cpu_s"],
            "peak_rss_mb": during["peak_mb"],
        }
    out["after"] = proc(pid)
    return out


def main() -> None:
    target = sys.argv[1]
    command = [str(Path(target).resolve())]
    root = Path(tempfile.mkdtemp(prefix="llp-bench-"))
    port, upstream_port = free_port(), free_port()
    config = prepare(root, port)
    upstream = subprocess.Popen(
        [sys.executable, str(HERE / "fake_upstream.py"), str(upstream_port), os.devnull]
    )
    wait_for(upstream_port, upstream, "fake upstream")
    started = time.perf_counter()
    proxy = subprocess.Popen(
        [*command, "--config", str(config)],
        env={
            **os.environ,
            "LLM_PROXY_TEST_UPSTREAM": f"http://127.0.0.1:{upstream_port}",
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for(port, proxy, target)
        startup = time.perf_counter() - started
        result = asyncio.run(measure(port, proxy.pid))
        result["startup_s"] = startup
        print(json.dumps(result, indent=2, default=lambda v: round(v, 3)))
    finally:
        proxy.terminate()
        upstream.kill()


if __name__ == "__main__":
    main()
