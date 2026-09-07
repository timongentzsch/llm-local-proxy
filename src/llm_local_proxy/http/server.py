"""Process entry point: bind the socket, serve, shut down cleanly."""

from __future__ import annotations

import argparse
import socket
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from ..config import load
from ..errors import ProviderError
from ..service import Service
from .handler import make_handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler):
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        super().__init__(address, handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local proxy for Codex and Claude subscriptions"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--show-config", action="store_true", help="print the config path and exit"
    )
    args = parser.parse_args()
    service = None
    try:
        config = load(args.config)
        if args.show_config:
            print(config.path)
            return
        service = Service(config)
        server = Server((config.host, config.port), make_handler(service))
    except (OSError, ValueError, ProviderError) as error:
        if service:
            service.close()
        raise SystemExit(f"llm-local-proxy: {error}") from error
    fragment = f"#key={quote(config.api_key)}" if config.api_key else ""
    print(f"LLM Local Proxy: {config.origin}/{fragment}")
    print(f"Config: {config.path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
