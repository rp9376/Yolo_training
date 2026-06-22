#!/usr/bin/env bash
# Launch the YOLO Training Studio web UI (bound to localhost).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python3"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"

if [ ! -x "$PY" ]; then
  echo "No virtualenv found. Run scripts/setup.sh first." >&2
  exit 1
fi

echo "============================================================"
echo "  YOLO Training Studio"
echo "  → http://${HOST}:${PORT}"
echo "  (bound to ${HOST}; for remote access use an SSH tunnel:"
echo "     ssh -L ${PORT}:localhost:${PORT} <user>@<server>)"
echo "============================================================"

exec "$PY" -m uvicorn backend.app:app --host "$HOST" --port "$PORT" --timeout-graceful-shutdown 3
