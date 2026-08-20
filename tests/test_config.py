import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_local_proxy.config import load


class ConfigTest(unittest.TestCase):
    def test_creates_secure_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            config = load(path)
            self.assertEqual(config.host, "127.0.0.1")
            self.assertGreaterEqual(len(config.api_key), 24)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_rejects_non_loopback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(
                'host="0.0.0.0"\nport=8787\napi_key="123456789012345678901234"\n'
            )
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "loopback"):
                load(path)

    def test_allows_explicit_no_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('host="127.0.0.1"\nport=8787\napi_key=""\n')
            path.chmod(0o600)
            self.assertEqual(load(path).api_key, "")

    def test_rejects_public_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text('host="127.0.0.1"\nport=8787\napi_key=""\n')
            path.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "not private"):
                load(path)

    def test_default_path_under_llm_local_proxy(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {"XDG_CONFIG_HOME": directory}),
        ):
            config = load()
            self.assertEqual(
                config.path, Path(directory) / "llm-local-proxy" / "config.toml"
            )

    def test_container_default_uses_internal_wildcard(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                "os.environ",
                {
                    "LLM_PROXY_CONTAINER": "1",
                    "CODEX_HOME": "/codex",
                    "XDG_CONFIG_HOME": directory,
                },
            ),
            patch("llm_local_proxy.config._container_mode", return_value=True),
        ):
            config = load()
            self.assertEqual(config.host, "0.0.0.0")
            self.assertEqual(config.codex_home, Path("/codex"))


if __name__ == "__main__":
    unittest.main()
