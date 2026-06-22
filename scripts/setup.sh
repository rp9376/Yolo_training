#!/usr/bin/env bash
# Idempotent bootstrap for YOLO Training Studio.
# Creates/reuses .venv, installs deps, seeds base weight, ensures runtime dirs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python3"

echo "==> YOLO Training Studio setup"

if [ -x "$PY" ]; then
  echo "==> Reusing existing virtualenv at .venv"
else
  echo "==> Creating virtualenv at .venv"
  python3 -m venv "$VENV"
fi

echo "==> Upgrading pip"
"$PY" -m pip install --upgrade pip >/dev/null

echo "==> Installing requirements (ultralytics/torch may take a while)"
"$PY" -m pip install -r "$ROOT/requirements.txt"

echo "==> Ensuring runtime directories"
mkdir -p "$ROOT/weights" "$ROOT/runs" "$ROOT/datasets"

# Seed a base weight so pretrained init + offline smoke don't need a download.
if [ ! -f "$ROOT/weights/yolo26n.pt" ] && [ -f "$ROOT/legacy/yolov26/yolo26n.pt" ]; then
  echo "==> Seeding weights/yolo26n.pt from legacy/"
  cp "$ROOT/legacy/yolov26/yolo26n.pt" "$ROOT/weights/yolo26n.pt"
fi

echo "==> Verifying imports"
"$PY" -c "import fastapi, uvicorn, psutil, yaml; print('  core deps OK')"
"$PY" -c "import ultralytics; print('  ultralytics', ultralytics.__version__)" || \
  echo "  (ultralytics not importable yet — the app still boots; training will fail until fixed)"

echo "==> Setup complete. Start the server with: scripts/run.sh"
