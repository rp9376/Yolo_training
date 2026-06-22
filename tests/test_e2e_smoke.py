"""Real end-to-end smoke test: actual ultralytics training on CPU.

Builds a yolo26n *from-scratch* task (no download), 1 epoch, tiny imgsz, on a
synthetic dataset, runs it through the real queue-runner subprocess, and checks
the outputs. Skipped only if ultralytics can't be imported.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import make_synthetic_dataset as synth
from core import config, engine, models
from core import queue as q

try:
    import ultralytics  # noqa: F401
    HAVE_ULTRALYTICS = True
except Exception:
    HAVE_ULTRALYTICS = False


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_ULTRALYTICS, reason="ultralytics not importable")
def test_real_cpu_training(root, monkeypatch):
    monkeypatch.delenv("YOLO_STUDIO_FAKE_TRAIN", raising=False)
    from backend import process

    synth.make_dataset(config.DATASETS_DIR / "synth", n_train=8, n_val=4)
    task = engine.build_task("yolov26", "n", "scratch", "synth",
                             epochs=1, batch=2, imgsz=64, device="cpu")
    q.add(task)

    assert process.start()["started"]

    completed = False
    deadline = time.time() + 600
    try:
        while time.time() < deadline:
            st = process.status()
            if not st["running"] and st["queue_status"] == "completed":
                completed = True
                break
            time.sleep(1.0)
    finally:
        process.stop()

    data = q.load()
    t = data["tasks"][0]
    assert completed, f"training did not finish (status={t.get('status')}, " \
                      f"error={t.get('error')})"
    assert t["status"] == "completed", t.get("error")

    run_dir = Path(t["run_dir"])
    assert (run_dir / "weights" / "best.pt").exists()
    assert (run_dir / "results.csv").exists()

    listed = models.list_models()
    assert any(m["run_name"] == task["name"] for m in listed)
