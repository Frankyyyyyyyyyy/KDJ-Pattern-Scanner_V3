#!/bin/bash
# ═══════════════════════════════════════════════════════
#  KDJ Pattern Scanner V3 — 一键启动 One-Click Launcher
# ═══════════════════════════════════════════════════════
#  Usage:
#    ./start_dashboard.sh          # 默认端口 default port 8888
#    ./start_dashboard.sh 9000     # 自定义端口 custom port
# ═══════════════════════════════════════════════════════

cd "$(dirname "$0")"

PORT=${1:-8888}

# Check if venv exists
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
else
    PYTHON="python3"
fi

echo ""
echo "  ╔═══════════════════════════════════════════════╗"
echo "  ║  KDJ Pattern Scanner V3 — Dashboard          ║"
echo "  ║  启动中 Starting on port $PORT ...             ║"
echo "  ╚═══════════════════════════════════════════════╝"
echo ""
echo "  浏览器打开 Open in browser:"
echo "  → http://localhost:$PORT"
echo ""
echo "  按 Ctrl+C 停止 Press Ctrl+C to stop"
echo ""

# Open browser automatically (macOS)
if command -v open &> /dev/null; then
    sleep 1 && open "http://localhost:$PORT" &
fi

# Start server
$PYTHON server.py --port $PORT
