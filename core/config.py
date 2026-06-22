"""Central paths, model families, presets, and augmentation defaults.

Single source of truth shared by the engine, the API, and the docs. No heavy
imports here (no torch/ultralytics) — this module loads instantly.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# YOLO_STUDIO_ROOT lets tests (and alternate deployments) relocate the whole
# project tree to a temp directory without touching the real datasets/runs.
_root_env = os.environ.get("YOLO_STUDIO_ROOT")
PROJECT_ROOT = Path(_root_env).resolve() if _root_env else Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RUNS_DIR = PROJECT_ROOT / "runs"
QUEUE_FILE = PROJECT_ROOT / "training_queue.json"
LOG_DIR = RUNS_DIR  # queue logs/events/pid live under runs/

# Files written by the queue runner / process manager (all under RUNS_DIR).
PROGRESS_LOG = RUNS_DIR / "queue_progress.log"
EVENTS_LOG = RUNS_DIR / "queue_events.log"
OUTPUT_LOG = RUNS_DIR / "queue_output.log"
PID_FILE = RUNS_DIR / ".queue_runner.pid"


# --- Model families ----------------------------------------------------------
# `prefix` is the weight-file / model-id prefix. v8 weights are `yolov8{s}.pt`
# (model id `yolov8{s}`), v26 weights are `yolo26{s}.pt` (model id `yolo26{s}`).
SIZES = ["n", "s", "m", "l", "x"]

SIZE_LABELS = {
    "n": "Nano",
    "s": "Small",
    "m": "Medium",
    "l": "Large",
    "x": "XLarge",
}

FAMILIES = {
    "yolov8": {
        "label": "YOLOv8",
        "prefix": "yolov8",
        "sizes": SIZES,
        "default_size": "x",
    },
    "yolov26": {
        "label": "YOLOv26",
        "prefix": "yolo26",
        "sizes": SIZES,
        "default_size": "x",
    },
}


def family_prefix(family: str) -> str:
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family!r}")
    return FAMILIES[family]["prefix"]


# --- UI presets (mirror the legacy CLI) --------------------------------------
EPOCH_PRESETS = [5, 50, 100, 200, 300, 500]
BATCH_PRESETS = [-1, 16, 32, 64, 128]   # -1 == auto
IMGSZ_PRESETS = [640, 1280]


# --- Training defaults & augmentation (verbatim from the legacy scripts) -----
DEFAULTS = {"patience": 50, "workers": 8}

# The exact augmentation dict that must be reproduced in every training call.
AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.15,
    "copy_paste": 0.2,
}

# Image extensions used when counting dataset images.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def ensure_dirs() -> None:
    """Create the runtime directories if missing. Safe to call repeatedly."""
    for d in (DATASETS_DIR, WEIGHTS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def runs_project_dir(family: str) -> Path:
    """Where a family's runs are written: runs/<family>/detect."""
    return RUNS_DIR / family / "detect"
