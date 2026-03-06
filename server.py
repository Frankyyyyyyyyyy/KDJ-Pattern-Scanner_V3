#!/usr/bin/env python3
"""
KDJ Pattern Scanner V3 — Lightweight Web Server
=================================================
Serves index.html and provides:
  GET  /                → dashboard page
  GET  /api/signals     → CSV data as JSON
  GET  /api/run         → SSE stream (launches run_strategy.py, streams stdout/stderr)
  GET  /api/status      → {"running": true/false, "last_run": "..."}

Usage:
  python server.py              # default port 8888
  python server.py --port 9000  # custom port
"""

import http.server
import json
import os
import subprocess
import sys
import threading
import time
import csv
import io
import glob
import argparse
from datetime import datetime
from urllib.parse import urlparse

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


def find_latest_csv():
    """Find the latest signals CSV file."""
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


class KDJHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler for dashboard + API."""

    def log_message(self, format, *args):
        """Suppress default logging noise."""
        pass

    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # ─── Root: serve dashboard ────────────────────
        if path == "/" or path == "/index.html":
            self.serve_dashboard()
            return

        # ─── API: get signal data ─────────────────────
        if path == "/api/signals":
            data = read_csv_as_json()
            self.send_json({"count": len(data), "signals": data})
            return

        # ─── API: run strategy (SSE) ──────────────────
        if path == "/api/run":
            self.handle_run_sse()
            return

        # ─── API: status ──────────────────────────────
        if path == "/api/status":
            self.send_json(run_state)
            return

        # ─── Fallback: serve static files ─────────────
        self.directory = BASE_DIR
        super().do_GET()

    def serve_dashboard(self):
        dashboard_path = os.path.join(BASE_DIR, "index.html")
        if not os.path.exists(dashboard_path):
            self.send_error(404, "index.html not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        with open(dashboard_path, "rb") as f:
            self.wfile.write(f.read())

    def handle_run_sse(self):
        """Execute run_strategy.py and stream output via SSE."""
        # Prevent concurrent runs
        if not run_lock.acquire(blocking=False):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"data: " + json.dumps({
                "type": "error",
                "msg": "\u6b63\u5728\u8fd0\u884c\u4e2d\uff0c\u8bf7\u7b49\u5f85\u5f53\u524d\u4efb\u52a1\u5b8c\u6210 Already running, please wait..."
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
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Send start event
            self._sse_send({"type": "start", "msg": f"\u2501\u2501 \u542f\u52a8\u626b\u63cf Starting scan... [{run_state['last_run']}]"})

            # Build env with FORCE_RUN
            env = os.environ.copy()
            env["FORCE_RUN"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

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

            # Stream output line by line
            line_count = 0
            for line in proc.stdout:
                line = line.rstrip("\n")
                line_count += 1

                # Classify log lines
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
                # Count new signals
                csv_path = find_latest_csv()
                if csv_path:
                    try:
                        with open(csv_path, "r", encoding="utf-8-sig") as f:
                            run_state["last_signals"] = sum(1 for _ in f) - 1
                    except Exception:
                        pass
                self._sse_send({"type": "done", "msg": f"\u2714 \u626b\u63cf\u5b8c\u6210 Scan complete! \u5171 {run_state['last_signals']} \u6761\u4fe1\u53f7",
                                "signals": run_state["last_signals"]})
            else:
                run_state["last_status"] = "error"
                self._sse_send({"type": "error", "msg": f"\u2716 \u626b\u63cf\u5931\u8d25 Scan failed (exit code {exit_code})"})

        except Exception as e:
            run_state["last_status"] = "error"
            try:
                self._sse_send({"type": "error", "msg": f"\u670d\u52a1\u5668\u9519\u8bef Server error: {str(e)}"})
            except Exception:
                pass
        finally:
            run_state["running"] = False
            run_lock.release()

    def _sse_send(self, data):
        """Send an SSE event."""
        payload = "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()


def main():
    parser = argparse.ArgumentParser(description="KDJ Scanner Dashboard Server")
    parser.add_argument("--port", type=int, default=8888, help="Port number (default: 8888)")
    args = parser.parse_args()

    server = http.server.HTTPServer(("0.0.0.0", args.port), KDJHandler)
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║   KDJ Pattern Scanner V3 — Dashboard Server              ║
║                                                           ║
║   🌐 Open in browser:  http://localhost:{args.port}            ║
║   📡 API endpoints:                                       ║
║       GET /api/signals  — signal data (JSON)              ║
║       GET /api/run      — run scanner (SSE stream)        ║
║       GET /api/status   — server status                   ║
║                                                           ║
║   Press Ctrl+C to stop                                    ║
╚═══════════════════════════════════════════════════════════╝
""")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
