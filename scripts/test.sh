#!/usr/bin/env bash
# Run the full test suite, including the real CPU end-to-end smoke test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python3"
if [ ! -x "$PY" ]; then
  echo "No virtualenv found. Run scripts/setup.sh first." >&2
  exit 1
fi

exec "$PY" -m pytest -q "$@"
