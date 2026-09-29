"""Named API keys: naming rules, persistence and identification."""

import os
import tempfile
import unittest
from pathlib import Path

from llm_local_proxy.errors import RequestError
from llm_local_proxy.keys import KeyStore


class KeyStoreTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "keys.json"
        self.keys = KeyStore(self.path)

    def test_keys_are_private_persistent_and_identified(self):
        key = self.keys.add("alice")
        self.assertTrue(key.startswith("llp_"))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(KeyStore(self.path).all(), {"alice": key})
        self.assertEqual(self.keys.identify(key), "alice")
        self.assertIsNone(self.keys.identify(key + "x"))
        self.keys.remove("alice")
        self.assertIsNone(self.keys.identify(key))

    def test_a_write_survives_a_crashed_writers_temp_file(self):
        (self.path.parent / f".keys.json.{os.getpid()}.tmp").write_text("partial")
        self.assertTrue(self.keys.add("alice"))

    def test_names_are_validated(self):
        self.keys.add("ci-bot.2")
        for name in ("", "Alice", "master", "a" * 33, "has space", "ci-bot.2"):
            with self.subTest(name=name), self.assertRaises(RequestError):
                self.keys.add(name)
        with self.assertRaises(RequestError):
            self.keys.remove("nobody")


if __name__ == "__main__":
    unittest.main()
