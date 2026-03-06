#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./tools/push_stock_list.sh [-m "commit message"] [--include-all] [--no-pull]

Options:
  -m, --message       Commit message (optional)
  --include-all       Also commit other changes (default: only stock_list.csv)
  --no-pull           Skip git pull --rebase --autostash (default: do pull if upstream exists)
EOF
}

message=""
include_all=0
do_pull=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--message)
      message="${2:-}"
      shift 2
      ;;
    --include-all)
      include_all=1
      shift
      ;;
    --no-pull)
      do_pull=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${root}" ]]; then
  echo "Error: not inside a git repository." >&2
  exit 1
fi
cd "${root}"

file="stock_list.csv"
if [[ ! -f "${file}" ]]; then
  echo "Error: ${file} not found at repo root." >&2
  exit 1
fi

if [[ "${include_all}" -eq 0 ]]; then
  if ! git diff --quiet -- . ':(exclude)stock_list.csv' || ! git diff --cached --quiet -- . ':(exclude)stock_list.csv'; then
    echo "Error: detected changes outside ${file}. Aborting to avoid pushing unrelated changes." >&2
    echo "Tip: use --include-all to commit everything, or stash other changes first." >&2
    exit 1
  fi
fi

if git diff --quiet -- "${file}" && git diff --cached --quiet -- "${file}"; then
  echo "No changes in ${file}. Nothing to push."
  exit 0
fi

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
if [[ "${do_pull}" -eq 1 && -n "${upstream}" ]]; then
  git pull --rebase --autostash
fi

if [[ "${include_all}" -eq 1 ]]; then
  git add -A
else
  git add "${file}"
fi

if [[ -z "${message}" ]]; then
  message="Update stock_list.csv ($(date '+%Y-%m-%d %H:%M:%S'))"
fi

git diff --cached --quiet || git commit -m "${message}"
git push
echo "Done."
