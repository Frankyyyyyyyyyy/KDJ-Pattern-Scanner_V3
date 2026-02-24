#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

FORCE_RUN_VAL="${FORCE_RUN:-0}"
SKIP_EMAIL_VAL="${SKIP_EMAIL:-0}"
LIMIT_TICKERS_VAL="${LIMIT_TICKERS:-}"

show_help() {
  cat <<'EOF'
Usage:
  ./run_daily.sh [--force] [--no-email] [--limit TICKERS]

Options:
  --force         强制运行（跳过交易日校验），等价于 FORCE_RUN=1
  --no-email      不发邮件，等价于 SKIP_EMAIL=1
  --limit         仅扫描指定ticker列表（逗号分隔），例如: --limit "AAPL,MSFT,QQQ"

Environment:
  POLYGON_API_KEY / POLYGON_API_KEY_FILE / POLYGON_API_KEY.txt 按 config.py 读取
  GMAIL_APP_PASSWORD / GMAIL_APP_PASSWORD_FILE / GMAIL_APP_PASSWORD.txt 按 config.py 读取
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --force)
      FORCE_RUN_VAL=1
      shift
      ;;
    --no-email)
      SKIP_EMAIL_VAL=1
      shift
      ;;
    --limit)
      LIMIT_TICKERS_VAL="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

export FORCE_RUN="$FORCE_RUN_VAL"
export SKIP_EMAIL="$SKIP_EMAIL_VAL"
if [[ -n "$LIMIT_TICKERS_VAL" ]]; then
  export LIMIT_TICKERS="$LIMIT_TICKERS_VAL"
fi

mkdir -p logs
TS="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="logs/run_${TS}.log"

PY_BIN="python"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PY_BIN="venv/bin/python"
fi

echo "[$(date +"%F %T")] Starting run_strategy.py" | tee -a "$LOG_FILE"
echo "ROOT_DIR=$ROOT_DIR" | tee -a "$LOG_FILE"
echo "FORCE_RUN=$FORCE_RUN" | tee -a "$LOG_FILE"
echo "SKIP_EMAIL=$SKIP_EMAIL" | tee -a "$LOG_FILE"
echo "LIMIT_TICKERS=${LIMIT_TICKERS:-}" | tee -a "$LOG_FILE"
echo "PY_BIN=$PY_BIN" | tee -a "$LOG_FILE"

"$PY_BIN" run_strategy.py 2>&1 | tee -a "$LOG_FILE"

echo "[$(date +"%F %T")] Done. Log: $LOG_FILE" | tee -a "$LOG_FILE"
