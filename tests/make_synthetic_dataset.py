"""Generate tiny synthetic YOLO datasets and fake completed runs for tests.

No network, no GPU. Uses Pillow (already a dependency of ultralytics).
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

CLASS_NAMES = ["shape_a", "shape_b"]


def _make_image(path: Path, cls: int, size: int = 64) -> tuple[float, float, float, float]:
    """Draw a solid background with one centered rectangle; return its yolo bbox."""
    bg = (30, 30, 30) if cls == 0 else (60, 20, 20)
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    # Centered box occupying ~40% of the frame.
    w = h = size * 0.4
    cx = cy = size / 2
    x0, y0 = cx - w / 2, cy - h / 2
    x1, y1 = cx + w / 2, cy + h / 2
    color = (200, 80, 80) if cls == 0 else (80, 200, 120)
    draw.rectangle([x0, y0, x1, y1], fill=color)
    img.save(path)
    # Normalized yolo bbox: cx, cy, w, h.
    return (cx / size, cy / size, w / size, h / size)


def _make_split(root: Path, split_folder: str, n: int, size: int) -> None:
    img_dir = root / split_folder / "images"
    lbl_dir = root / split_folder / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cls = i % 2
        stem = f"{split_folder}_{i:03d}"
        cx, cy, w, h = _make_image(img_dir / f"{stem}.png", cls, size)
        (lbl_dir / f"{stem}.txt").write_text(f"{cls} {cx} {cy} {w} {h}\n")


def make_dataset(root: Path, n_train: int = 8, n_val: int = 4, n_test: int = 2,
                 size: int = 64) -> Path:
    """Create a minimal valid YOLO dataset rooted at ``root``. Returns ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    _make_split(root, "train", n_train, size)
    _make_split(root, "valid", n_val, size)
    if n_test:
        _make_split(root, "test", n_test, size)

    names = "[" + ", ".join(f"'{n}'" for n in CLASS_NAMES) + "]"
    (root / "data.yaml").write_text(
        f"path: {root.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {names}\n"
    )
    return root


_FAKE_RESULTS_ROWS = [
    # epoch, mAP50, mAP50-95, precision, recall -> fitness 0.1*mAP50+0.9*mAP50-95
    (1, 0.20, 0.10, 0.30, 0.25),   # fitness 0.11
    (2, 0.60, 0.40, 0.70, 0.65),   # fitness 0.42  <-- best
    (3, 0.50, 0.30, 0.66, 0.60),   # fitness 0.32
]
FAKE_BEST_EPOCH = 2
FAKE_BEST_FITNESS = round(0.1 * 0.60 + 0.9 * 0.40, 4)  # 0.42


def make_fake_run(runs_root: Path, family: str = "yolov26",
                  name: str = "n_e3_20260101_000000",
                  data_yaml: str = "/tmp/data.yaml",
                  model: str = "yolo26n.yaml", imgsz: int = 640) -> Path:
    """Fabricate a completed run dir (no training). Returns the run directory."""
    run_dir = Path(runs_root) / family / "detect" / name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    # Dummy weight files.
    (run_dir / "weights" / "best.pt").write_bytes(b"PK\x03\x04 dummy-best")
    (run_dir / "weights" / "last.pt").write_bytes(b"PK\x03\x04 dummy-last")

    # args.yaml
    (run_dir / "args.yaml").write_text(
        f"task: detect\nmode: train\nmodel: {model}\n"
        f"data: {data_yaml}\nepochs: 3\nimgsz: {imgsz}\nbatch: 16\n"
        f"project: {run_dir.parent}\nname: {name}\npretrained: false\n"
    )

    # results.csv (header matches real ultralytics output)
    header = ("epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
              "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
              "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n")
    lines = [header]
    for (ep, m50, m5095, p, r) in _FAKE_RESULTS_ROWS:
        lines.append(f"{ep},{ep*10.0},1.0,1.0,1.0,{p},{r},{m50},{m5095},1.0,1.0,1.0\n")
    (run_dir / "results.csv").write_text("".join(lines))

    # A couple of tiny PNG artifacts.
    for art in ("results.png", "confusion_matrix.png", "BoxPR_curve.png"):
        Image.new("RGB", (16, 16), (20, 30, 40)).save(run_dir / art)

    return run_dir


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/synthetic_ds")
    make_dataset(out)
    print("Created synthetic dataset at", out)
