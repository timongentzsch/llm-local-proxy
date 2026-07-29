from __future__ import annotations

import ipaddress
import os
import secrets
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    api_key: str
    codex_home: Path
    codex_binary: str
    request_timeout: int
    path: Path

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{host}:{self.port}/v1"


def default_path() -> Path:
    root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return root / "codex-local-proxy" / "config.toml"


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _container_mode() -> bool:
    return (
        os.environ.get("CODEX_PROXY_CONTAINER") == "1" and Path("/.dockerenv").exists()
    )


def _write_default(path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    host = "0.0.0.0" if _container_mode() else "127.0.0.1"
    codex_home = os.environ.get("CODEX_HOME", "~/.codex")
    text = (
        f'host = "{host}"\n'
        "port = 8787\n"
        f'api_key = "{secrets.token_urlsafe(32)}"\n'
        f'codex_home = "{codex_home}"\n'
        'codex_binary = "codex"\n'
        "request_timeout = 600\n"
    )
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def load(path: Path | None = None) -> Config:
    path = (path or default_path()).expanduser()
    if not path.exists():
        _write_default(path)
    if path.stat().st_mode & 0o077:
        raise ValueError(f"config is not private; run: chmod 600 {path}")
    data = tomllib.loads(path.read_text())
    host = str(data.get("host", "127.0.0.1"))
    port = int(data.get("port", 8787))
    api_key = str(data.get("api_key", ""))
    if not _is_loopback(host) and not (_container_mode() and host in {"0.0.0.0", "::"}):
        raise ValueError("host must be a loopback address")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if api_key and len(api_key) < 24:
        raise ValueError("api_key must be empty or contain at least 24 characters")
    return Config(
        host=host,
        port=port,
        api_key=api_key,
        codex_home=Path(str(data.get("codex_home", "~/.codex"))).expanduser(),
        codex_binary=str(data.get("codex_binary", "codex")),
        request_timeout=max(1, int(data.get("request_timeout", 600))),
        path=path,
    )
