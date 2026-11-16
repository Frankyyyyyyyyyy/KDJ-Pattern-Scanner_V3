#!/usr/bin/env python3
"""
KDJ Pattern Scanner V3 — Lightweight Web Server
=================================================
Serves the local dashboard only. Defaults to loopback.

  GET  /                → dashboard page
  GET  /index.html      → dashboard page
  GET  /signals_latest.csv and /signals_YYYYQN.csv
  GET  /api/signals     → CSV data as JSON
  GET  /api/status      → {"running": true/false, ...}
  POST /api/run         → start a scan (loopback + SCANNER_RUN_TOKEN only)
  GET  /api/run         → 405 (GET cannot trigger a scan)

Usage:
  python server.py              # 127.0.0.1:8888
  python server.py --port 9000
  python server.py --host 127.0.0.1

A token is never printed. Set SCANNER_RUN_TOKEN in the environment, or put
it in a local `.scanner_run_token` file (not served, gitignored).
"""

import hmac
import http.server
import json
import os
import re
import subprocess
import sys
import threading
import posixpath
from urllib.parse import unquote, urlparse
import csv
import glob
import argparse
from datetime import datetime
from socketserver import ThreadingMixIn

# ─── State ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "bin", "python3")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = sys.executable  # fallback

run_lock = threading.Lock()
run_state = {
    "running": False,
    "last_run": None,
    "last_status": None,   # "success" | "error" | "cancelled"
    "last_signals": 0,
}

_SIGNAL_CSV_RE = re.compile(r"^signals_latest\.csv$|^signals_\d{4}Q[1-4]\.csv$")
_TOKEN_FILE = os.path.join(BASE_DIR, ".scanner_run_token")
_SECRET_NAME_RE = re.compile(
    r"(password|api_key|secret|token|\.env$|\.pem$|\.key$)",
    re.IGNORECASE,
)


def find_latest_csv():
    """Find the latest signals CSV file in the project root only."""
    pattern = os.path.join(BASE_DIR, "signals_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def read_csv_as_json():
    """Read latest CSV and return as JSON-friendly list of dicts."""
    csv_path = find_latest_csv()
    if not csv_path or not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        return [{"error": str(e)}]


def normalize_relpath(url_path):
    """Return a safe relative path inside BASE_DIR, or None if it escapes."""
    raw = unquote(url_path or "")
    # strip query (urlparse already does this if we pass .path)
    rel = posixpath.normpath(raw.lstrip("/"))
    if rel == ".":
        rel = ""
    if rel.startswith("..") or rel.startswith("/") or os.path.isabs(rel):
        return None
    if "\\" in rel or "\x00" in rel:
        return None
    # Reject hidden / vcs / env paths early
    parts = [p for p in rel.split("/") if p]
    if any(p.startswith(".") for p in parts):
        return None
    return rel


def is_public_static(rel):
    """Whitelist of files the dashboard may fetch."""
    if rel in ("", "index.html"):
        return True
    base = os.path.basename(rel)
    if rel != base:
        # no subdirectories are public
        return False
    return _SIGNAL_CSV_RE.match(base) is not None


def is_blocked_name(rel):
    name = os.path.basename(rel or "")
    if _SECRET_NAME_RE.search(name):
        return True
    if name in {"config.py", "GMAIL_APP_PASSWORD.txt", "POLYGON_API_KEY.txt",
                ".scanner_run_token", ".env"}:
        return True
    return False


def is_loopback_addr(addr):
    if not addr:
        return False
    return addr in ("127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1")


def get_run_token():
    """Load the local run token. Never log or return it to callers of HTTP errors."""
    env = os.environ.get("SCANNER_RUN_TOKEN")
    if isinstance(env, str) and env.strip():
        return env.strip()
    try:
        if os.path.isfile(_TOKEN_FILE):
            with open(_TOKEN_FILE, "r", encoding="utf-8") as f:
                val = f.read().strip()
            return val or None
    except Exception:
        return None
    return None


def extract_bearer_token(headers):
    auth = headers.get("Authorization") or headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # Explicit header alternative; never accept a query-string token
    hdr = headers.get("X-Scanner-Token") or headers.get("x-scanner-token") or ""
    return hdr.strip() or None


def authorize_scan(headers, client_addr):
    """
    POST /api/run requires loopback AND a configured shared token.
    Returns (ok: bool, http_status: int, public_message: str).
    public_message must never include the token.
    """
    if not is_loopback_addr(client_addr):
        return False, 403, "scan trigger is only allowed from localhost"
    token = get_run_token()
    if not token:
        return False, 403, "scan trigger is disabled until SCANNER_RUN_TOKEN is configured"
    provided = extract_bearer_token(headers)
    if not provided or not hmac.compare_digest(provided, token):
        return False, 401, "unauthorized"
    return True, 200, "ok"


class KDJHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for dashboard + API."""

    def log_message(self, format, *args):
        # Avoid echoing headers/tokens. Log only method and sanitized path.
        try:
            path = urlparse(self.path).path
        except Exception:
            path = "?"
        sys.stderr.write("%s %s %s\n" % (self.address_string(), self.command, path))

    def _client_addr(self):
        try:
            return self.client_address[0]
        except Exception:
            return ""

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

    def send_plain(self, status, message):
        body = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/signals":
            data = read_csv_as_json()
            self.send_json({"count": len(data), "signals": data})
            return

        if path == "/api/status":
            self.send_json(run_state)
            return

        if path == "/api/run":
            self.send_json(
                {"error": "method not allowed", "hint": "use POST /api/run from localhost with authorization"},
                status=405,
            )
            return

        if path == "/" or path == "/index.html":
            self.serve_dashboard()
            return

        rel = normalize_relpath(path)
        if rel is None or is_blocked_name(rel) or not is_public_static(rel):
            self.send_plain(404, "Not found")
            return

        self.serve_whitelisted_file(rel)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_plain(404, "Not found")
            return
        ok, status, msg = authorize_scan(self.headers, self._client_addr())
        if not ok:
            self.send_json({"error": msg}, status=status)
            return
        self.handle_run_sse()

    def do_PUT(self):
        self.send_plain(405, "Method not allowed")

    def do_DELETE(self):
        self.send_plain(405, "Method not allowed")

    def serve_dashboard(self):
        dashboard_path = os.path.join(BASE_DIR, "index.html")
        if not os.path.exists(dashboard_path):
            self.send_error(404, "index.html not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with open(dashboard_path, "rb") as f:
            self.wfile.write(f.read())

    def serve_whitelisted_file(self, rel):
        full = os.path.realpath(os.path.join(BASE_DIR, rel))
        base_real = os.path.realpath(BASE_DIR)
        if not (full == base_real or full.startswith(base_real + os.sep)):
            self.send_plain(404, "Not found")
            return
        if not os.path.isfile(full):
            self.send_plain(404, "Not found")
            return
        if rel.endswith(".csv"):
            content_type = "text/csv; charset=utf-8"
        else:
            content_type = "application/octet-stream"
        try:
            with open(full, "rb") as f:
                data = f.read()
        except Exception:
            self.send_plain(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_run_sse(self):
        """Execute run_strategy.py and stream output via SSE. POST + auth only."""
        if not run_lock.acquire(blocking=False):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps({
                "type": "error",
                "msg": "Already running, please wait..."
            }).encode() + b"\n\n")
            self.wfile.flush()
            return

        run_state["running"] = True
        run_state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            self._sse_send({"type": "start", "msg": "Starting scan..."})

            env = os.environ.copy()
            env["FORCE_RUN"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            # Do not inject or echo tokens into the child command line
            cmd = [VENV_PYTHON, os.path.join(BASE_DIR, "run_strategy.py")]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=BASE_DIR,
                env=env,
                bufsize=1,
                universal_newlines=True,
            )

            line_count = 0
            for line in proc.stdout:
                line = line.rstrip("\n")
                line_count += 1
                msg_type = "log"
                if "[ERROR]" in line or "CRITICAL" in line or "Traceback" in line:
                    msg_type = "error"
                elif "[WARNING]" in line:
                    msg_type = "warn"
                elif "Progress:" in line or "signals" in line.lower():
                    msg_type = "progress"
                try:
                    self._sse_send({"type": msg_type, "msg": line, "line": line_count})
                except (BrokenPipeError, ConnectionResetError):
                    proc.kill()
                    break

            proc.wait()
            exit_code = proc.returncode

            if exit_code == 0:
                run_state["last_status"] = "success"
                csv_path = find_latest_csv()
                if csv_path:
                    try:
                        with open(csv_path, "r", encoding="utf-8-sig") as f:
                            run_state["last_signals"] = sum(1 for _ in f) - 1
                    except Exception:
                        pass
                self._sse_send({
                    "type": "done",
                    "msg": "Scan complete!",
                    "signals": run_state["last_signals"],
                })
            else:
                run_state["last_status"] = "error"
                self._sse_send({"type": "error", "msg": "Scan failed"})

        except Exception:
            run_state["last_status"] = "error"
            try:
                self._sse_send({"type": "error", "msg": "Server error"})
            except Exception:
                pass
        finally:
            run_state["running"] = False
            run_lock.release()

    def _sse_send(self, data):
        payload = "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()


class _ThreadingHTTPServer(ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description="KDJ Scanner Dashboard Server")
    parser.add_argument("--port", type=int, default=8888, help="Port number (default: 8888)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1; do not use 0.0.0.0 unless you understand the risk)")
    args = parser.parse_args()

    server = _ThreadingHTTPServer((args.host, args.port), KDJHandler)
    token_ready = bool(get_run_token())
    print("KDJ Pattern Scanner V3 — Dashboard Server")
    print("  Listen:  http://%s:%s" % (args.host, args.port))
    print("  GET  /api/signals  — signal data (JSON)")
    print("  GET  /api/status   — server status")
    print("  POST /api/run      — run scanner (localhost + token only)")
    print("  GET  /api/run      — disabled")
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        print("  Warning: bound beyond loopback; static files stay whitelisted and POST still requires localhost + token.")
    if not token_ready:
        print("  Scan trigger is off (no SCANNER_RUN_TOKEN configured).")
    else:
        print("  Scan trigger token is configured (value not displayed).")
    print("  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
