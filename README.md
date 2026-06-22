# YOLO Training Studio

A local **web UI** for training **YOLOv8** and **YOLOv26** object-detection
models on custom datasets. It wraps the existing ultralytics training pipeline
(the exact augmentations and fitness formula are preserved) with a dashboard for
hardware monitoring, dataset management, a training queue, a live training
monitor, and a model browser.

- **Backend:** Python + FastAPI, single process (reuses ultralytics directly).
- **Frontend:** vanilla JS (ES modules) + hand-written CSS — **no Node, no build
  step, no `node_modules`, zero third-party JS.** Charts are hand-drawn on a 2D
  canvas.
- **Access:** binds to `127.0.0.1` (single-user, no login).

## Quick start

```bash
scripts/setup.sh      # once: create/reuse .venv, install deps, seed a base weight
scripts/run.sh        # serve the UI at http://127.0.0.1:8000
```

Then open <http://localhost:8000>.

Working on a remote server? The UI binds to localhost only — use an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 <user>@<server>
# then browse to http://localhost:8000 on your laptop
```

(Change the bind with `HOST=0.0.0.0 PORT=8000 scripts/run.sh` if you really want
LAN access — there is no auth, so only do this on a trusted network.)

## Features

- **Dashboard** — live CPU / RAM / disk and one card per GPU (memory bar,
  utilization ring, temp, power), plus a rolling utilization chart. GPU info via
  `pynvml` with an `nvidia-smi` fallback; degrades gracefully with no GPU.
- **Datasets** — list classes + image counts; **upload a `.zip`** (extracted and
  validated, with zip-slip protection) or **register an existing server path**
  (symlink); re-validate; delete.
- **Queue builder** — pick family (v8/v26), size, init (pretrained/scratch),
  dataset, epochs (presets + custom), batch, image size, and device; reorder and
  remove pending tasks; persisted to `training_queue.json`.
- **Monitor** — start/stop the queue; per-task progress; a **live log stream**
  (SSE) and **live loss/mAP curves**. Survives page reloads and reattaches to an
  in-progress run.
- **Models** — browse finished runs with key metrics; a detail page with config,
  per-epoch curves, result plots (incl. confusion matrix), class list; download
  `best.pt` / `last.pt`; export to ONNX / TorchScript.

## Project layout

```
core/        framework-agnostic engine (no FastAPI; lazy-imports torch/ultralytics)
backend/     thin FastAPI layer (routers, SSE, queue-runner process manager)
static/      the entire frontend (index.html, css/, js/, assets/)
tests/       pytest suite (unit + API + real CPU end-to-end smoke)
scripts/     setup.sh  run.sh  test.sh  smoke.sh
legacy/      the original CLI scripts (yolov8/, yolov26/), kept for reference
datasets/    your datasets (git-ignored contents)
weights/     base .pt files (git-ignored; see weights/README.md)
runs/        training outputs (git-ignored)
```

The engine runs the **verbatim** ultralytics call with the project's standard
augmentations, names runs `{size}_e{epochs}_{timestamp}`, and selects the best
epoch by `fitness = 0.1·mAP50 + 0.9·mAP50-95`. Training executes in a separate
`python -m core.queue_runner` subprocess, so the API only reads state/logs and
the runner survives a backend restart (the backend reattaches via a PID file).

## Adding datasets & weights

- **Datasets:** export from Roboflow in YOLO format (a folder with `data.yaml`
  and `train/valid/test` `images/`+`labels/`). Upload the `.zip` in the UI, or
  drop the folder under `datasets/` and it is auto-discovered. See
  `Dataset_fetching.md`.
- **Weights:** drop base `.pt` files into `weights/` using the documented naming
  (`yolo26{n..x}.pt`, `yolov8{n..x}.pt`). See `weights/README.md`.

## Testing

```bash
scripts/test.sh       # full suite incl. a real 1-epoch CPU train (no GPU/network)
scripts/smoke.sh      # boot the server and curl the key endpoints
```

## Requirements

Python 3.12, a CUDA-capable GPU for real training (CPU works for tests).
`pip install -r requirements.txt` pulls FastAPI/uvicorn, psutil, `nvidia-ml-py`,
PyYAML, and ultralytics (which brings torch + torchvision). No Node required.
