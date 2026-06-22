"""Trained-model discovery, metadata, metric curves, artifacts, export.

Scans ``runs/<family>/detect/<name>/`` directories. Lazy-imports ultralytics
only inside :func:`export_model`.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import yaml

from .config import FAMILIES, RUNS_DIR, SIZES

# Metric columns of interest (kept in this order for the frontend).
_SERIES_COLS = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
]


def _model_size(model_str: str) -> str:
    s = str(model_str).lower()
    for size in SIZES:
        if f"yolo26{size}" in s or f"yolov8{size}" in s:
            return size
    for size in SIZES:
        if s.endswith(f"{size}.pt") or s.endswith(f"{size}.yaml"):
            return size
    return "?"


def _family_of(run_dir: Path) -> str:
    # runs/<family>/detect/<name>
    try:
        return run_dir.parent.parent.name
    except Exception:
        return "yolov26"


def _read_csv(results_csv: Path) -> list[dict]:
    rows = []
    try:
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({(k.strip() if k else k): (v.strip() if isinstance(v, str) else v)
                             for k, v in row.items()})
    except Exception:
        return []
    return rows


def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _best_and_final(rows: list[dict]) -> dict:
    best_fitness = 0.0
    best_epoch = 0
    total_epochs = 0
    final = {}
    for row in rows:
        map50 = _fnum(row.get("metrics/mAP50(B)")) or 0.0
        map5095 = _fnum(row.get("metrics/mAP50-95(B)")) or 0.0
        epoch = _fnum(row.get("epoch"))
        epoch = int(epoch) if epoch is not None else 0
        total_epochs = max(total_epochs, epoch)
        fitness = 0.1 * map50 + 0.9 * map5095
        if fitness > best_fitness:
            best_fitness = fitness
            best_epoch = epoch
    if rows:
        last = rows[-1]
        final = {
            "map50": _fnum(last.get("metrics/mAP50(B)")),
            "map50_95": _fnum(last.get("metrics/mAP50-95(B)")),
            "precision": _fnum(last.get("metrics/precision(B)")),
            "recall": _fnum(last.get("metrics/recall(B)")),
        }
    return {
        "best_epoch": best_epoch,
        "best_fitness": round(best_fitness, 4),
        "total_epochs": total_epochs,
        "final": final,
    }


def _summary(run_dir: Path) -> dict | None:
    best_pt = run_dir / "weights" / "best.pt"
    args_file = run_dir / "args.yaml"
    results_csv = run_dir / "results.csv"
    if not best_pt.exists():
        return None

    args = {}
    if args_file.exists():
        try:
            with open(args_file) as f:
                args = yaml.safe_load(f) or {}
        except Exception:
            args = {}

    rows = _read_csv(results_csv) if results_csv.exists() else []
    bf = _best_and_final(rows)

    family = _family_of(run_dir)
    from . import naming  # local import to avoid cycles at module load
    dataset_name = naming.clean_dataset_name(args.get("data", ""))

    try:
        mtime = best_pt.stat().st_mtime
    except OSError:
        mtime = 0.0
    try:
        size_bytes = best_pt.stat().st_size
    except OSError:
        size_bytes = 0

    return {
        "run_name": run_dir.name,
        "family": family,
        "size": _model_size(args.get("model", "")),
        "model": args.get("model", ""),
        "dataset": args.get("data", ""),
        "dataset_name": dataset_name,
        "epochs": args.get("epochs"),
        "imgsz": args.get("imgsz", 640),
        "batch": args.get("batch", "auto"),
        "best_epoch": bf["best_epoch"],
        "best_fitness": bf["best_fitness"],
        "total_epochs": bf["total_epochs"],
        "final": bf["final"],
        "size_bytes": size_bytes,
        "mtime": mtime,
        "run_dir": str(run_dir),
        "has_last": (run_dir / "weights" / "last.pt").exists(),
    }


def _all_run_dirs() -> list[Path]:
    dirs = []
    for family in FAMILIES:
        detect = RUNS_DIR / family / "detect"
        if not detect.is_dir():
            continue
        for d in detect.iterdir():
            try:
                if d.is_dir():
                    dirs.append(d)
            except OSError:
                continue
    return dirs


def list_models() -> list[dict]:
    out = [s for d in _all_run_dirs() if (s := _summary(d))]
    out.sort(key=lambda m: m["mtime"], reverse=True)
    return out


def find_run_dir(run_name: str) -> Path | None:
    for family in FAMILIES:
        cand = RUNS_DIR / family / "detect" / run_name
        if cand.is_dir():
            return cand
    return None


def metric_series(run_dir: Path) -> dict:
    rows = _read_csv(run_dir / "results.csv")
    series: dict[str, list] = {"epoch": []}
    for col in _SERIES_COLS:
        series[col] = []
    for row in rows:
        ep = _fnum(row.get("epoch"))
        series["epoch"].append(int(ep) if ep is not None else len(series["epoch"]) + 1)
        for col in _SERIES_COLS:
            series[col].append(_fnum(row.get(col)))
    return series


def _class_names(args: dict) -> list[str]:
    data_path = args.get("data")
    if not data_path:
        return []
    try:
        with open(data_path) as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names")
        if isinstance(names, dict):
            try:
                return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
            except Exception:
                return [str(v) for v in names.values()]
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
    except Exception:
        return []
    return []


def detail(run_name: str) -> dict:
    run_dir = find_run_dir(run_name)
    if run_dir is None:
        raise FileNotFoundError(run_name)
    summary = _summary(run_dir)
    if summary is None:
        raise FileNotFoundError(run_name)

    args = {}
    args_file = run_dir / "args.yaml"
    if args_file.exists():
        try:
            with open(args_file) as f:
                args = yaml.safe_load(f) or {}
        except Exception:
            args = {}

    # Image artifacts in the run dir (plots/curves/etc.).
    artifacts = sorted(
        f.name for f in run_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    summary["series"] = metric_series(run_dir)
    summary["class_names"] = _class_names(args)
    summary["artifacts"] = artifacts
    summary["args"] = args
    return summary


def artifact_path(run_name: str, filename: str) -> Path:
    """Safe-join an artifact path; must stay inside the (resolved) run dir."""
    run_dir = find_run_dir(run_name)
    if run_dir is None:
        raise FileNotFoundError(run_name)
    base = run_dir.resolve()
    target = (run_dir / filename).resolve()
    if base != target and base not in target.parents:
        raise ValueError("Path escapes run directory")
    if not target.is_file():
        raise FileNotFoundError(filename)
    return target


def weights_path(run_name: str, which: str = "best") -> Path:
    if which not in ("best", "last"):
        raise ValueError("which must be 'best' or 'last'")
    run_dir = find_run_dir(run_name)
    if run_dir is None:
        raise FileNotFoundError(run_name)
    pt = run_dir / "weights" / f"{which}.pt"
    if not pt.is_file():
        raise FileNotFoundError(f"{which}.pt")
    return pt


def export_model(run_name: str, fmt: str) -> Path:
    """Export best.pt to onnx/torchscript. Lazy-imports ultralytics."""
    if fmt not in ("onnx", "torchscript"):
        raise ValueError("format must be 'onnx' or 'torchscript'")
    best = weights_path(run_name, "best")
    from ultralytics import YOLO  # lazy import
    model = YOLO(str(best))
    out = model.export(format=fmt)
    return Path(out)


def exported_file(run_name: str, fmt: str) -> Path | None:
    """Locate a previously exported file for download."""
    run_dir = find_run_dir(run_name)
    if run_dir is None:
        return None
    weights = run_dir / "weights"
    ext = ".onnx" if fmt == "onnx" else ".torchscript"
    for cand in (weights / f"best{ext}", run_dir / f"best{ext}"):
        if cand.is_file():
            return cand
    matches = sorted(weights.glob(f"*{ext}")) if weights.is_dir() else []
    return matches[0] if matches else None


def delete(run_name: str) -> dict:
    run_dir = find_run_dir(run_name)
    if run_dir is None:
        raise FileNotFoundError(run_name)
    if run_dir.is_symlink():
        run_dir.unlink()
    else:
        shutil.rmtree(run_dir)
    return {"deleted": True, "run_name": run_name}
