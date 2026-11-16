# -*- coding: utf-8 -*-
"""Offline tests for dashboard server access controls."""
from __future__ import annotations
import json
import os
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server  # noqa: E402


class HelperTests(unittest.TestCase):
    def test_public_static_whitelist(self):
        self.assertTrue(server.is_public_static(""))
        self.assertTrue(server.is_public_static("index.html"))
        self.assertTrue(server.is_public_static("signals_latest.csv"))
        self.assertTrue(server.is_public_static("signals_2026Q1.csv"))
        self.assertFalse(server.is_public_static("config.py"))
        self.assertFalse(server.is_public_static("server.py"))
        self.assertFalse(server.is_public_static("price_cache/prices_2026.csv"))
        self.assertFalse(server.is_public_static("GMAIL_APP_PASSWORD.txt"))
        self.assertFalse(server.is_public_static("POLYGON_API_KEY.txt"))
        self.assertFalse(server.is_public_static(".env"))
        self.assertFalse(server.is_public_static("signals_latest.csv.bak"))

    def test_path_traversal_rejected(self):
        self.assertIsNone(server.normalize_relpath("/../config.py"))
        self.assertIsNone(server.normalize_relpath("/foo/../../config.py"))
        self.assertIsNone(server.normalize_relpath("/.env"))
        self.assertIsNone(server.normalize_relpath("/.scanner_run_token"))
        self.assertEqual(server.normalize_relpath("/index.html"), "index.html")

    def test_blocked_names(self):
        self.assertTrue(server.is_blocked_name("config.py"))
        self.assertTrue(server.is_blocked_name("GMAIL_APP_PASSWORD.txt"))
        self.assertTrue(server.is_blocked_name("POLYGON_API_KEY.txt"))
        self.assertTrue(server.is_blocked_name(".scanner_run_token"))

    def test_authorize_requires_loopback_and_token(self):
        os.environ.pop("SCANNER_RUN_TOKEN", None)
        ok, status, _ = server.authorize_scan({}, "8.8.8.8")
        self.assertFalse(ok)
        self.assertEqual(status, 403)
        ok, status, _ = server.authorize_scan({}, "127.0.0.1")
        self.assertFalse(ok)
        self.assertEqual(status, 403)
        os.environ["SCANNER_RUN_TOKEN"] = "unit-test-token"
        try:
            ok, status, _ = server.authorize_scan({}, "127.0.0.1")
            self.assertFalse(ok)
            self.assertEqual(status, 401)
            headers = {"Authorization": "Bearer unit-test-token"}
            ok, status, _ = server.authorize_scan(headers, "127.0.0.1")
            self.assertTrue(ok)
            ok, status, _ = server.authorize_scan(headers, "10.0.0.2")
            self.assertFalse(ok)
            self.assertEqual(status, 403)
        finally:
            os.environ.pop("SCANNER_RUN_TOKEN", None)


class LiveServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server._ThreadingHTTPServer(("127.0.0.1", 0), server.KDJHandler)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _conn(self):
        return HTTPConnection("127.0.0.1", self.port, timeout=5)

    def test_get_index_ok(self):
        c = self._conn()
        c.request("GET", "/")
        r = c.getresponse()
        self.assertEqual(r.status, 200)
        body = r.read()
        self.assertIn(b"Trading Signals", body)
        c.close()

    def test_get_secrets_and_source_not_found(self):
        for path in (
            "/config.py",
            "/server.py",
            "/GMAIL_APP_PASSWORD.txt",
            "/POLYGON_API_KEY.txt",
            "/.env",
            "/.scanner_run_token",
            "/price_cache/prices_2026.csv",
            "/run_strategy.py",
        ):
            c = self._conn()
            c.request("GET", path)
            r = c.getresponse()
            # Read body only to drain; do not inspect secret material
            r.read()
            self.assertEqual(r.status, 404, path)
            c.close()

    def test_get_api_run_does_not_start_scan(self):
        started = {"n": 0}

        def _boom(*args, **kwargs):
            started["n"] += 1
            raise AssertionError("scan must not start on GET")

        c = self._conn()
        with mock.patch.object(server.KDJHandler, "handle_run_sse", _boom):
            c.request("GET", "/api/run")
            r = c.getresponse()
            body = r.read()
            self.assertEqual(r.status, 405)
            self.assertEqual(started["n"], 0)
            data = json.loads(body.decode("utf-8"))
            self.assertIn("method not allowed", data.get("error", "").lower())
        c.close()

    def test_post_api_run_without_token_rejected(self):
        started = {"n": 0}

        def _boom(*args, **kwargs):
            started["n"] += 1

        os.environ.pop("SCANNER_RUN_TOKEN", None)
        c = self._conn()
        with mock.patch.object(server.KDJHandler, "handle_run_sse", _boom):
            c.request("POST", "/api/run", body=b"", headers={"Content-Length": "0"})
            r = c.getresponse()
            r.read()
            self.assertIn(r.status, (401, 403))
            self.assertEqual(started["n"], 0)
        c.close()


if __name__ == "__main__":
    unittest.main()
