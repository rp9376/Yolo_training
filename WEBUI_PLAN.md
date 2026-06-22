# YOLO Training Studio — Web UI Implementation Plan

> **Audience:** an executor (Claude Opus) who will build this end-to-end, test it, and
> debug until it works out of the box. This document is the single source of truth for
> scope, architecture, the API contract, the design system, and the test/acceptance gates.
>
> **Prime directive:** Do not declare the work done until `scripts/test.sh` is fully green
> **and** `scripts/smoke.sh` passes **and** every box in §12 (Acceptance Criteria) is
> checked. Run, observe, fix, repeat. You are expected to actually boot the server, hit the
> endpoints, and verify each page renders and behaves — not just write code.

---

## 1. Locked decisions

These were chosen with the project owner. Do not revisit them.

| Decision | Choice | Consequence |
|---|---|---|
| Backend | **Python + FastAPI**, single process | Reuses existing YOLO/ultralytics code directly |
| Frontend | **Vanilla JS (ES modules) + hand-written CSS**, *no framework, no build step* | No Node, no npm, no bundler, no `node_modules`. Served as static files by FastAPI |
| Third-party JS | **Zero.** Charts/gauges are hand-drawn with Canvas 2D in our own `charts.js` | Keeps dependency surface minimal (owner's explicit preference) |
| Project structure | **Refactor shared logic into `core/`**, move old CLI scripts to `legacy/` | One engine, no duplication; old scripts kept for reference |
| Access / auth | **Bind `127.0.0.1` only, no login** | Single-user local tool; LAN is a one-line change later |
| Datasets | **Zip upload + register existing server path** | Covers laptop-upload and already-on-server |
| Concurrency | **Queue runs one task at a time** (sequential) | Matches existing behavior + single GPU here; parallel multi-GPU is a non-goal |

**Owner preference to honor throughout: minimize dependencies and keep it simple.** When in
doubt, choose the option with fewer moving parts.

---

## 2. What exists today (starting point)

A terminal-driven YOLO training framework with two near-identical toolsets sharing one
`datasets/` folder.

```
yolov8/   train.py setup_queue.py run_queue.py validate.py extract_models.py cleanup.py run_queue_bg.sh  (+ yolo26n.pt)
yolov26/  (same set) + list_queue.py                                                                        (+ yolo26n.pt)
datasets/ place_dataset_dirs_here.txt   (no real datasets present)
README.md Dataset_fetching.md .gitignore
```

**Behavior to preserve in `core/`:**

- **Settings offered to the user:** dataset (auto-discovered = dirs under `datasets/` containing
  `data.yaml`), model size `n/s/m/l/x`, init mode (v26: pretrained `.pt` vs from-scratch `.yaml`;
  v8 historically always from-scratch), epochs (presets `5/50/100/200/300/500` + custom),
  batch (`-1` auto / `16/32/64/128`), image size (`640/1280`). Fixed: `patience=50`, `workers=8`,
  device = all GPUs (`"0"`, `"0,1,..."`, or `"cpu"`).
- **The exact training call** (must be reproduced verbatim, including augmentations):
  `YOLO(model_source).train(data, epochs, batch, imgsz, device, project, name, patience, workers,
  exist_ok=True, pretrained=<bool>, verbose=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=15.0,
  translate=0.1, scale=0.5, shear=0.0, perspective=0.0005, flipud=0.0, fliplr=0.5, mosaic=1.0,
  mixup=0.15, copy_paste=0.2)`.
- **Run naming:** `{size_letter}_e{epochs}_{YYYYmmdd_HHMMSS}` (e.g. `x_e300_20260616_120000`).
- **Queue file** `training_queue.json`: `{created, status, total_tasks, completed_tasks,
  failed_tasks, tasks:[...]}`; each task carries its full config + `status`.
- **Best-epoch / fitness** from `results.csv`: `fitness = 0.1*mAP50 + 0.9*mAP50-95`, best row wins.
- **Outputs:** `runs/.../<name>/weights/{best,last}.pt`, `results.csv`, `results.png`,
  `confusion_matrix.png`, `args.yaml`, PR/F1 curves, etc.

**Environment reality (drives "works out of the box"):**

- Python **3.12.3**, system interpreter. **No `.venv` yet.**
- Installed: `psutil 5.9.8`. **Not installed:** `ultralytics`, `torch`, `fastapi`, `pynvml`.
- GPU: **4× NVIDIA H100, 80 GB (lookup other details).
- Some datasets already present


---

## 3. Target structure

```
Yolo_training/
├── core/                      # framework-agnostic engine (no FastAPI imports here)
│   ├── __init__.py
│   ├── config.py              # paths, model families, presets, augmentation defaults
│   ├── hardware.py            # CPU/mem/disk/GPU snapshot (pynvml → nvidia-smi fallback)
│   ├── datasets.py            # discover / validate / upload-zip / register-path / delete
│   ├── weights.py             # discover base weights per family+size
│   ├── naming.py              # run-name + descriptive export-name helpers
│   ├── engine.py              # train_one(task): the verbatim YOLO.train call
│   ├── queue.py               # Queue model + JSON persistence (training_queue.json)
│   ├── queue_runner.py        # `python -m core.queue_runner`: runs queue, writes progress/log
│   └── models.py              # scan runs/, metadata, metric curves, artifacts, export
├── backend/                   # FastAPI layer (thin; delegates to core/)
│   ├── __init__.py
│   ├── app.py                 # app factory, static mount, routers, lifespan
│   ├── process.py             # queue-runner subprocess manager (start/stop/status/pid + reattach)
│   └── api/
│       ├── hardware.py  datasets.py  weights.py  queue.py  models.py  stream.py  health.py
├── static/                    # the entire frontend (no build)
│   ├── index.html
│   ├── css/styles.css
│   ├── assets/                # favicon, logo (inline SVG ok)
│   └── js/
│       ├── app.js             # bootstrap + hash router
│       ├── api.js             # fetch wrapper + error/toast handling
│       ├── charts.js          # Canvas 2D line charts + ring gauges (zero deps)
│       ├── ui.js              # DOM helpers (h(), escape(), toast(), modal(), confirm())
│       └── pages/
│           ├── dashboard.js   datasets.js  queue.js  monitor.js  models.js  modelDetail.js
├── tests/
│   ├── conftest.py
│   ├── make_synthetic_dataset.py
│   ├── test_hardware.py  test_datasets.py  test_weights.py  test_queue.py
│   ├── test_models.py    test_naming.py     test_api.py     test_e2e_smoke.py
├── scripts/
│   ├── setup.sh   run.sh   test.sh   smoke.sh
├── legacy/
│   ├── yolov8/…   yolov26/…           # the old scripts, moved as-is
├── datasets/                          # unchanged (gitignored contents)
├── weights/                           # created; base .pt live here (gitignored)
│   └── README.md
├── runs/                              # training outputs (gitignored)
├── requirements.txt
├── README.md                          # rewritten for the web UI
└── WEBUI_PLAN.md                      # this file
```

**Move, don't delete:** `git mv yolov8 legacy/yolov8 && git mv yolov26 legacy/yolov26`. Create
`weights/` and copy one existing base weight in: `cp legacy/yolov26/yolo26n.pt weights/yolo26n.pt`
(the engine and the offline smoke test must not require network downloads).

---

## 4. `core/` — the shared engine

`core/` must **not import FastAPI** and must **lazy-import `ultralytics`/`torch`** (only inside
training/export functions) so the API and dashboard boot even if the training stack is broken.

### 4.1 `config.py`
- `PROJECT_ROOT`, `DATASETS_DIR`, `WEIGHTS_DIR`, `RUNS_DIR`, `QUEUE_FILE`, `LOG_DIR`.
- `FAMILIES = {"yolov8": {...}, "yolov26": {...}}` with per-family: weight-file prefix
  (`yolov8` / `yolo26`), size list, display labels, default size.
- `EPOCH_PRESETS`, `BATCH_PRESETS`, `IMGSZ_PRESETS` (mirror legacy).
- `AUGMENTATION = {...}` — the exact dict from §2 so `engine.py` and any docs share one source.
- `DEFAULTS = {"patience": 50, "workers": 8}`.

### 4.2 `hardware.py`
`snapshot() -> dict` returning:
```jsonc
{
  "timestamp": 1718536800.0,
  "cpu":   {"percent": 12.5, "per_core": [..], "cores": 8, "freq_mhz": 2300, "load_avg": [..]},
  "memory":{"total": 0, "used": 0, "available": 0, "percent": 0.0},   // bytes
  "swap":  {"total": 0, "used": 0, "percent": 0.0},
  "disk":  {"path": "<runs partition>", "total": 0, "used": 0, "free": 0, "percent": 0.0},
  "gpus":  [{"index":0,"name":"...","mem_total":0,"mem_used":0,"mem_free":0,
             "util":0,"temp":0,"power":0.0,"power_limit":0.0}],
  "gpu_backend": "pynvml" | "nvidia-smi" | "none"
}
```
- CPU/mem/disk/swap via `psutil`.
- GPUs: try `pynvml` first; on any failure fall back to parsing
  `nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits`;
  if neither works, `gpus: []`, `gpu_backend: "none"`. **Never raise** — degrade gracefully.
- Cheap enough to call ~1×/sec. Do **not** import torch here.

### 4.3 `datasets.py`
- `discover() -> [DatasetInfo]`: dirs in `DATASETS_DIR` (follow symlinks) that contain `data.yaml`.
- `info(name)`: parse `data.yaml` → `nc`, `names[]`; count images in `train/valid/test` `images/`
  dirs; total size; `source` (`uploaded` vs `registered` symlink); `valid` + `issues[]`.
- `validate(path)`: `data.yaml` present & parseable; `train` and `val` image dirs resolve and are
  non-empty; `names`/`nc` consistent. Return structured issues, don't throw.
- `import_zip(file_stream, name=None)`: stream to a temp file, extract safely
  (**reject path traversal / zip-slip**, reject absolute/`..` members), auto-detect the dataset
  root (zip may contain `data.yaml` at top level or nested one level), move into
  `datasets/<slug>/`, validate, clean temp. Collisions → suffix `-2`, `-3`.
- `register_path(server_path, name=None)`: validate, then create a **symlink** `datasets/<slug> →
  server_path` (discovery already follows symlinks). If symlink unavailable, record in a small
  `datasets/.registry.json`.
- `delete(name)`: unlink symlink, or `rmtree` an uploaded dir. Confirm at API layer.
- `slugify(name)`: filesystem-safe, reuse the cleanup rules from legacy `extract_models.py`.

### 4.4 `weights.py`
- `available() -> {family: {size: path|None}}` scanning `WEIGHTS_DIR` for `yolov8{n..x}.pt` and
  `yolo26{n..x}.pt`. Drives the UI's pretrained-vs-scratch choice.
- `resolve(family, size, init)` → `(model_source, pretrained_bool)`: pretrained → abs `.pt` path;
  scratch → `"{prefix}{size}.yaml"`.

### 4.5 `naming.py`
- `run_name(size, epochs)` → `{size}_e{epochs}_{ts}` (legacy format).
- `export_name(meta)` → descriptive `.pt` name (port `make_model_name` from legacy
  `extract_models.py`) for nicer downloads.

### 4.6 `engine.py`
- `build_task(...) -> dict` — validates inputs, resolves model source/project/run-name/device,
  returns a complete **task dict** (schema in §6).
- `train_one(task, on_event=None) -> dict` — lazy-import ultralytics; run the **verbatim**
  `YOLO.train` call from §2; after training parse `results.csv` for `best_epoch`/`best_fitness`;
  return result summary. `device` resolution: `"auto"` → all GPUs or `"cpu"` if none.
- `export_model(run_dir, fmt) -> path` — lazy-import; `YOLO(best.pt).export(format=fmt)`.

### 4.7 `queue.py`
- Load/save `training_queue.json` (schema in §6) with an **atomic write** (temp file + `os.replace`).
- `add/get/update/remove/reorder/clear`. Computes `total/completed/failed` counts on save.
- Guard: while `status == "running"`, only **appending pending tasks** and **stop** are allowed;
  editing/removing/reordering existing tasks is rejected (enforced again at API layer).

### 4.8 `queue_runner.py` (`python -m core.queue_runner`)
- Standalone process the backend launches. Mirrors legacy `run_queue.py` but uses `engine.train_one`.
- Iterate pending tasks sequentially: set `running` → train → `completed`/`failed`, persist after
  every transition, write human log to `runs/queue_progress.log` and structured one-line JSON events
  to `runs/queue_events.log` (`{"ts","type":"status|epoch|log|done","task_id",...}`).
- Honor a stop signal (SIGINT/SIGTERM): mark the running task `canceled`, exit cleanly.
- 30 s GPU-cooldown between tasks (as legacy). On exit, queue `status = "completed"`.

### 4.9 `models.py`
- `list_models() -> [summary]`: scan `runs/<family>/detect/*/` with `weights/best.pt`. From
  `args.yaml` (model, epochs, imgsz, data, batch) + `results.csv` (best epoch/fitness, final
  mAP50, mAP50-95, precision, recall) + file sizes + mtime. Sort newest first.
- `detail(run_name)`: summary + full per-epoch metric series (for charts) + class names (from the
  run's dataset `data.yaml`) + artifact file list (PNGs in the run dir).
- `artifact_path(run_name, filename)`: safe-join, must stay inside the run dir (no traversal).
- `weights_path(run_name, which)`: `best`/`last`.
- `delete(run_name)`: rmtree the run dir (API confirms).

---

## 5. `backend/` — FastAPI

- `app.py`: app factory; mount `static/` at `/`; include routers under `/api`; `lifespan` reattaches
  to an already-running queue runner via the PID file. Serve `index.html` for `/` and unknown
  non-`/api` non-`/static` routes (so hash-routing/deep links work). Bind **127.0.0.1**.
- Keep CORS off (same-origin). No auth.
- `process.py` — single queue-runner subprocess manager:
  - `start()`: if PID file alive → 409. Else spawn `python -m core.queue_runner` (same interpreter,
    `cwd=PROJECT_ROOT`, `start_new_session=True`), append stdout/stderr to `runs/queue_output.log`,
    write `runs/.queue_runner.pid`. Returns status.
  - `stop()`: signal the process group (SIGTERM, then SIGKILL after grace), clear PID file.
  - `status()`: `{running, pid, running_task_id, ...}` derived from PID liveness + queue file.
  - Runner survives a backend restart; backend reattaches by reading PID + tailing logs.

### 5.1 REST + SSE contract (all under `/api`)

**Health**
- `GET /api/health` → `{ok, python, gpu_count, gpu_backend, torch?, cuda?, ultralytics?}`
  (torch/cuda/ultralytics resolved lazily; missing → `null`, never an error).

**Hardware**
- `GET /api/hardware` → §4.2 snapshot. (Frontend polls ~1 s; no SSE needed.)

**Datasets**
- `GET /api/datasets` → `[info]`
- `GET /api/datasets/{name}` → info
- `POST /api/datasets/upload` (multipart `file`, optional `name`) → created info. Stream to disk.
- `POST /api/datasets/register` `{path, name?}` → created info
- `POST /api/datasets/{name}/validate` → `{valid, issues[]}`
- `DELETE /api/datasets/{name}` → `{deleted:true}`

**Weights**
- `GET /api/weights` → `{family:{size: {available:bool, path?}}}`

**Queue**
- `GET /api/queue` → full queue file
- `POST /api/queue/tasks` `{family,size,init,dataset,epochs,batch,imgsz,device?}` → built task
  (server resolves model_source/name/project/device, validates dataset & weights)
- `PUT /api/queue/tasks/{id}` → edit a **pending** task (409 if running/locked)
- `DELETE /api/queue/tasks/{id}` → remove a **pending** task (409 if running)
- `POST /api/queue/reorder` `{order:[id,...]}` (409 if running)
- `POST /api/queue/clear` `{scope:"all|completed|pending"}`
- `POST /api/queue/start` → 202 `{started:true,pid}` | 409 if already running | 400 if no pending
- `POST /api/queue/stop` → `{stopped:true}`
- `GET /api/queue/status` → `{status,pid,running_task_id,counts,...}`
- `GET /api/queue/tasks/{id}/metrics` → parsed `results.csv` series for charts
- `GET /api/queue/stream` → **SSE**: `status`, `log`, `epoch` events (used by Monitor page)

**Models**
- `GET /api/models` → `[summary]`
- `GET /api/models/{run_name}` → detail
- `GET /api/models/{run_name}/artifact/{filename}` → image/file (for plots), safe-joined
- `GET /api/models/{run_name}/download?which=best|last` → `.pt`, `Content-Disposition` with a
  descriptive filename from `naming.export_name`
- `POST /api/models/{run_name}/export` `{format:"onnx|torchscript"}` → runs export (may be slow;
  acceptable to run synchronously with a clear loading state, or as a tracked job) → `{path}`
- `GET /api/models/{run_name}/download_export?format=onnx` → exported file
- `DELETE /api/models/{run_name}` → `{deleted:true}` (API confirms)

**SSE format:** `event: <type>\ndata: <json>\n\n`. Implement with Starlette `StreamingResponse`
(no extra dependency). The stream tails `runs/queue_events.log` and emits `status` heartbeats;
closes when the client disconnects.

---

## 6. Data schemas

**Task**
```jsonc
{
  "id": "uuid4",
  "family": "yolov26",            // or "yolov8"
  "size": "x",                    // n|s|m|l|x
  "model": "yolo26x",             // prefix+size
  "init": "pretrained",           // "pretrained" | "scratch"
  "model_source": "/abs/weights/yolo26x.pt",   // or "yolo26x.yaml"
  "pretrained": true,
  "dataset": "/abs/datasets/foo/data.yaml",
  "dataset_name": "foo",
  "epochs": 100, "batch": -1, "imgsz": 640,
  "device": "auto",               // "auto" | "cpu" | "0" | "0,1"
  "patience": 50, "workers": 8,
  "name": "x_e100_20260616_120000",
  "project": "/abs/runs/yolov26/detect",
  "status": "pending",            // pending|running|completed|failed|canceled
  "created_at": "...", "started_at": null, "finished_at": null,
  "duration": null, "best_epoch": null, "best_fitness": null,
  "error": null, "run_dir": null
}
```

**Queue file** `training_queue.json`
```jsonc
{
  "version": 2,
  "created": "...", "updated": "...",
  "status": "idle",               // idle|running|completed
  "runner_pid": null,
  "total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0,
  "tasks": [ /* Task */ ]
}
```
Stay backward-compatible enough to read a legacy v1 queue if present (best effort; not required).

---

## 7. Frontend (vanilla JS, no build)

`index.html` loads `<script type="module" src="/js/app.js">` and `css/styles.css`. Modern browsers
run ES modules natively — **no bundler**. Hash router: `#/dashboard #/datasets #/queue #/monitor
#/models #/models/<run_name>`. Each page module exports `render(root)` + `unmount()` (clears
intervals / closes SSE). Build DOM with a tiny `h()` helper or template strings; **always
`escape()` user-supplied strings** (dataset names, class names, log lines, errors) — this is the
main XSS surface.

### 7.1 Design system (dark, with popping accents)
CSS variables in `:root`:
```
--bg-0:#0b0e14  --bg-1:#121622  --bg-2:#1a2030  --elev:#212a3d  --border:#2a3346
--text:#e6e9f2  --muted:#8d97ad  --text-dim:#5c6680
--accent:#00e5c0           /* primary: electric teal — buttons, active nav, chart lines, focus */
--accent-2:#7c5cff         /* secondary: violet — secondary highlights, gradients */
--success:#3ddc84  --warn:#ffb020  --danger:#ff5470  --info:#3da5ff
--radius:12px  --shadow:0 6px 24px rgba(0,0,0,.45)  --glow:0 0 0 1px var(--accent),0 0 18px -4px var(--accent)
```
- Layout: fixed **left sidebar** (logo + icon/label nav, active item gets the accent bar + glow) and
  a **top bar** (page title + a compact global status: queue-running pulse, mini GPU mem/util chips).
- Surfaces are layered (`--bg-0` app → `--bg-1` panels → `--bg-2`/`--elev` cards). Accent used
  sparingly for **details**: active states, primary buttons, progress bars, chart strokes, focus
  rings, status pills, and a subtle glow on the running-task indicator. Status colors map to pills.
- Type: system stack (`ui-sans-serif, -apple-system, Segoe UI, Roboto, Inter, sans-serif`); logs in
  `ui-monospace, SFMono-Regular, Menlo, monospace`. No web-font download.
- Components: stat tiles, **GPU card** (name + memory bar + util ring gauge + temp/power), data
  tables with status pills, custom selects/number inputs, **drag-and-drop upload zone** with a
  progress bar, modal + confirm dialog, toast notifications, tabs, and a **log console**
  (monospace, auto-scroll, color-coded by level). Responsive down to a narrow window.

### 7.2 Pages
1. **Dashboard / Hardware** — polls `/api/hardware` ~1 s. Top stat tiles (CPU %, RAM, disk, GPU
   count). One **GPU card per device** with live memory bar, utilization ring, temp, power. A live
   line chart (`charts.js`) of CPU% and per-GPU util over a rolling ~60 s window. Banner if no GPU.
   Small "system" panel from `/api/health` (versions, CUDA available?).
2. **Datasets** — table (name, classes, train/valid/test counts, size, source, valid?). **Add
   dataset** modal with two tabs: *Upload .zip* (drag-drop + progress) and *Register server path*.
   Per-row: view classes, re-validate, delete (confirm). Inline validation errors.
3. **Queue builder** — form to add a task: family (v8/v26) → size → init (pretrained/scratch,
   disabled+labeled if no weight file) → dataset (from `/api/datasets`) → epochs (presets + custom)
   → batch (auto/16/32/64/128) → imgsz (640/1280) → device (auto/cpu/specific GPU). Live preview of
   the resulting run name. Queue list below with drag-reorder + remove (disabled while running) and
   "Clear completed / all". Big **Start queue** button (disabled if empty/running) and **Stop**.
4. **Monitor** — live view of the running queue. Per-task status pills + progress (epoch x/N from
   events). **Live log console** via `GET /api/queue/stream` (SSE). **Live training charts** (loss
   and mAP50/mAP50-95 vs epoch) from `/api/queue/tasks/{id}/metrics` polled while running. Must
   **survive page reload** and reattach to an in-progress run. Stop button with confirm.
5. **Models** — table of finished models (run, family/size, dataset, epochs, best epoch, mAP50,
   mAP50-95, size, date). Row → detail.
6. **Model detail** — header with key metrics; tabs: *Overview* (config + final/best metrics +
   class list), *Curves* (per-epoch loss/mAP from results.csv via `charts.js`), *Plots* (the run's
   PNGs: results.png, confusion_matrix.png, PR/F1 curves via the artifact endpoint), *Download*
   (best.pt / last.pt buttons; ONNX/TorchScript export → download). Delete run (confirm).

### 7.3 `charts.js` (zero-dependency)
Canvas 2D primitives only: `lineChart(canvas, series, opts)` (multi-series, axes, accent strokes,
rolling window) and `ringGauge(canvas, pct, label)`. Handle devicePixelRatio for crispness. This is
the only "charting" code — no third-party lib.

---

## 8. Environment & bootstrap ("out of the box")

`requirements.txt` (prefer the **minimal** set; plain `uvicorn`, not `[standard]`):
```
fastapi
uvicorn
python-multipart      # required for zip upload (multipart form)
psutil
nvidia-ml-py          # provides pynvml; nvidia-smi fallback covers its absence
pyyaml
ultralytics           # pulls torch + torchvision
# dev/test:
pytest
httpx                 # FastAPI TestClient
```

`scripts/setup.sh`: create `.venv` (`python3 -m venv .venv`), upgrade pip, `pip install -r
requirements.txt`, ensure `weights/` exists and seed `weights/yolo26n.pt` from `legacy/`, create
`runs/` and `datasets/`. Idempotent.

`scripts/run.sh`: activate `.venv`, `exec python -m uvicorn backend.app:app --host 127.0.0.1
--port 8000` (no `--reload` in normal use). Print the URL.

**Robustness rules that make it boot cleanly even on this machine:**
- The app starts and the **Dashboard, Datasets, Queue, Models** pages all work **even if torch/CUDA
  is broken** (lazy imports; `/api/health` reports what's missing instead of crashing).
- No real datasets / no `weights/*.pt` → empty states with guidance, never a 500.
- GPU listing works through `nvidia-smi` if `pynvml` import fails.

Update `.gitignore`: add `/runs/`, `/weights/*.pt`, `.venv/` (already), `runs/*.log`,
`runs/.queue_runner.pid`, `training_queue.json`. Keep `datasets/**` ignore as-is.

---

## 9. Testing strategy

> The owner specifically asked that the tool **perform tests and debug until everything works.**
> Treat the suite below as a gate, not a formality.

`scripts/test.sh` → `pytest -q`. All training-touching tests run on **CPU** with a tiny synthetic
dataset so they never need the GPU or network.

### 9.1 Synthetic fixtures — `tests/make_synthetic_dataset.py`
Generate a minimal valid YOLO dataset on the fly: e.g. 8 train + 4 valid 64×64 PNGs (solid shapes),
matching YOLO `.txt` labels, 2 classes, and a correct `data.yaml`. Also a helper to fabricate a
fake completed run dir (`args.yaml`, `results.csv` with a few epochs, `weights/best.pt` as a small
dummy file, a couple of PNGs) for model/metadata tests that don't need real training.

### 9.2 Unit tests (`core/`)
- **hardware**: `snapshot()` returns the full shape; works with pynvml mocked, with nvidia-smi
  fallback mocked, and with neither (`gpus:[]`, no exception).
- **datasets**: validate good vs broken datasets; `import_zip` on a generated zip (incl. nested
  root); **zip-slip** member is rejected; `register_path` creates a working symlink; `delete`.
- **weights**: discovery + `resolve()` for pretrained/scratch, present/absent.
- **queue**: add/get/update/remove/reorder/clear; atomic-write round-trip; running-lock guard;
  counts recomputed.
- **models**: scan the fabricated run → correct best-epoch/fitness (matches `0.1*mAP50+0.9*mAP50-95`),
  metadata, artifact list; `artifact_path` rejects traversal.
- **naming**: run-name + export-name formats.

### 9.3 API tests (`tests/test_api.py`, FastAPI `TestClient`)
- `/api/health`, `/api/hardware` shapes.
- Datasets: upload a generated zip → appears in `GET /api/datasets`; register a path; delete.
- Weights endpoint shape.
- Queue: build a task via `POST /api/queue/tasks`; reorder/remove; `clear`. For **start**, inject a
  fake/fast engine (monkeypatch `engine.train_one`) so the lifecycle (`start → running → completed`)
  is exercised in seconds without real training; assert status transitions and the runner PID flow.
- Models: list/detail/artifact/download against the fabricated run.
- SSE: `GET /api/queue/stream` yields at least one event then closes.
- Path-traversal attempts on artifact/download endpoints return 400/404.

### 9.4 Real end-to-end smoke (`tests/test_e2e_smoke.py`, marked `slow`)
The one test that runs the **actual** ultralytics pipeline: build a queue task (`yolo26n`,
**scratch yaml** so no download, `epochs=1` or `2`, `imgsz=320`, `batch=2`, `device="cpu"`) against
the synthetic dataset; run `core.queue_runner` (or `engine.train_one`) to completion; assert
`runs/.../weights/best.pt` and `results.csv` exist, queue `status == completed`, and the model shows
up in `list_models()`. Skip with a clear reason only if `ultralytics` import fails; **it must pass in
the final acceptance run.**

### 9.5 HTTP smoke (`scripts/smoke.sh`)
Boot uvicorn on a test port in the background; wait for readiness; `curl` `/` (expect the HTML),
`/api/health`, `/api/hardware`, `/api/datasets`, `/api/models`, `/api/queue` (expect 200 + sane
JSON); then shut the server down. Non-zero exit on any failure.

### 9.6 Manual UI verification (executor performs this, fixing as needed)
Boot `scripts/run.sh`, open `http://localhost:8000`, and confirm against §12. Use the synthetic
dataset and a 1-epoch CPU task to exercise the **full** flow live: upload/register a dataset → build
a queue task → start → watch the Monitor page stream logs + draw live curves → see it land on the
Models page → open detail → view plots → download `best.pt`. Fix anything that doesn't work and
re-verify. (A headless-browser check via Playwright is **optional** and only if it adds no required
dependency to the shipped tool; the HTTP smoke + this manual pass are the requirement.)

---

## 10. Execution phases (ordered, with checkpoints)

1. **Restructure** — `git mv` the two tool dirs into `legacy/`; create `core/ backend/ static/
   tests/ scripts/ weights/`; seed `weights/yolo26n.pt`; write `requirements.txt`; `scripts/setup.sh`
   then run it (creates `.venv`, installs deps). ✅ *Checkpoint:* `python -c "import fastapi,
   ultralytics, psutil, yaml"` succeeds in the venv (ultralytics may take a while / large torch).
2. **`core/` engine + unit tests** — implement config/hardware/datasets/weights/naming/queue/
   models, then `engine`/`queue_runner`. Write §9.1–9.2 as you go. ✅ *Checkpoint:* core unit tests
   green; `python -m core.hardware` (add a `__main__` that prints the snapshot) shows the real GPU.
3. **`backend/` API + process manager + API tests** — routers, `process.py`, SSE. ✅ *Checkpoint:*
   `test_api.py` green; `scripts/smoke.sh` passes.
4. **Frontend** — design system + `app.js/api.js/ui.js/charts.js` + the six pages. ✅ *Checkpoint:*
   every page renders with live/real data; dark theme + accents look professional.
5. **End-to-end** — run `tests/test_e2e_smoke.py` (real 1–2 epoch CPU train) and the full §9.6
   manual pass. ✅ *Checkpoint:* §12 fully checked.
6. **Docs** — rewrite `README.md` for the web UI (setup, run, features, where things live, how to add
   datasets/weights, the localhost/SSH-tunnel note); add `weights/README.md`. Keep
   `Dataset_fetching.md`.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
|
| `ultralytics`/`torch` install is large/slow | Allow setup to take time; gate training tests behind import availability; never block app boot on these imports |
| Long-running training blocks the API | Training runs in a **separate subprocess** (`core.queue_runner`); API only reads state/logs |
| Browser zip of thousands of files | Upload a **single .zip**; stream to disk; or **register a server path** (no upload) |
| Zip-slip / path traversal on upload & artifact serving | Sanitize every archive member and every served path; safe-join under the intended root; covered by tests |
| Queue edited mid-run / double-start | File-level running lock + PID file; mutating endpoints return 409 while running |
| Backend restart loses track of a running train | PID file + log tailing → reattach on `lifespan` startup |
| SSE/proxy buffering | Set no-cache/`X-Accel-Buffering: no` headers; frontend also polls status as a fallback |

---

## 12. Acceptance criteria (definition of done)

- [ ] One command (`scripts/setup.sh` once, then `scripts/run.sh`) serves the UI at
      `http://localhost:8000`, bound to `127.0.0.1`.
- [ ] **Hardware**: dashboard shows live CPU %, RAM, disk, and **one card per GPU** (name, mem
      used/total bar, utilization ring, temp, power) updating ~1 s; correct for 0/1/N GPUs; uses
      pynvml or nvidia-smi; live CPU/GPU line chart renders.
- [ ] **Datasets**: list shows classes + image counts; **upload a .zip** (extracted & validated);
      **register a server path**; delete; validation errors surfaced; empty state when none.
- [ ] **Queue builder**: add tasks choosing family (v8/v26), size, init (pretrained/scratch),
      epochs (presets + custom), batch (auto/16/32/64/128), imgsz (640/1280), device; reorder &
      remove pending; persists to `training_queue.json`.
- [ ] **Run queue**: start/stop; per-task live status; **live streaming log**; **live loss/mAP
      curves** for the running task; survives page reload; reattaches if a runner is already running.
- [ ] **Models**: list finished models with key metrics; detail page with config, metrics, per-epoch
      curves, result plots (incl. confusion matrix), and class list; **download best.pt/last.pt**;
      ONNX/TorchScript **export + download**.
- [ ] **Design**: dark theme with popping accent details, cohesive and professional; responsive.
- [ ] **No Node/build step**; only `pip install -r requirements.txt` + run.
- [ ] `scripts/test.sh` **fully green** (incl. the real CPU e2e smoke) and `scripts/smoke.sh` passes.
- [ ] Old CLI scripts preserved under `legacy/`; `README.md` updated for the web UI.

---

## 13. Notes for the executor

- **Reuse, don't reinvent** the training/metadata logic in `legacy/` — port it into `core/` keeping
  the exact `YOLO.train` arguments and the `0.1*mAP50 + 0.9*mAP50-95` fitness formula.
- Keep `core/` free of FastAPI and free of eager `torch`/`ultralytics` imports.
- Prefer clarity and few dependencies over cleverness — the owner explicitly wants this simple.
- Don't stop at "code compiles." **Boot it, click through every page with the synthetic dataset and
  a real 1–2 epoch CPU run, and fix until §12 is genuinely satisfied.**
```
