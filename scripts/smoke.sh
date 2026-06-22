#!/usr/bin/env bash
# HTTP smoke test: boot uvicorn, hit the key endpoints, shut down.
# Non-zero exit on any failure.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python3"
HOST="127.0.0.1"
PORT="${SMOKE_PORT:-8011}"
BASE="http://${HOST}:${PORT}"

if [ ! -x "$PY" ]; then
  echo "No virtualenv found. Run scripts/setup.sh first." >&2
  exit 1
fi

echo "==> Booting server on ${BASE}"
"$PY" -m uvicorn backend.app:app --host "$HOST" --port "$PORT" >/tmp/smoke_uvicorn.log 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for readiness (up to ~30s).
ready=0
for _ in $(seq 1 60); do
  if curl -fsS "${BASE}/api/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.5
done
if [ "$ready" -ne 1 ]; then
  echo "FAIL: server did not become ready"; cat /tmp/smoke_uvicorn.log; exit 1
fi

fail=0

check_json() {
  local path="$1"
  if curl -fsS "${BASE}${path}" | "$PY" -c "import sys,json; json.load(sys.stdin)" >/dev/null 2>&1; then
    echo "  OK   ${path}"
  else
    echo "  FAIL ${path}"; fail=1
  fi
}

check_html() {
  local body
  body="$(curl -fsS "${BASE}/" 2>/dev/null)"
  if echo "$body" | grep -qi "<html"; then
    echo "  OK   /  (HTML)"
  else
    echo "  FAIL /  (no HTML)"; fail=1
  fi
}

echo "==> Checking endpoints"
check_html
check_json "/api/health"
check_json "/api/hardware"
check_json "/api/datasets"
check_json "/api/models"
check_json "/api/queue"
check_json "/api/weights"

if [ "$fail" -ne 0 ]; then
  echo "==> SMOKE FAILED"; exit 1
fi
echo "==> SMOKE PASSED"
