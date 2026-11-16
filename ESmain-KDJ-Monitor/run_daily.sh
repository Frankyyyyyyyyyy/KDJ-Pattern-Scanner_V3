#!/usr/bin/env bash
# 每日 cron 入口（在你机器上）：
#   crontab -e
#   # 美东收盘后 30 分钟跑一次（北京时间次日 04:30）
#   30 4 * * 2-6 /path/to/ESmain-KDJ-Monitor/run_daily.sh >> /path/to/ESmain-KDJ-Monitor/logs/cron.log 2>&1

set -euo pipefail
cd "$(dirname "$0")"

# 激活虚拟环境（如有）
if [ -d ".venv" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

python monitor.py
