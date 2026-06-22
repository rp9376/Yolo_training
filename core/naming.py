"""Run-name and descriptive export-name helpers (ported from legacy)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .config import FAMILIES


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_name(size: str, epochs: int, ts: str | None = None) -> str:
    """Legacy run-name format: ``{size}_e{epochs}_{YYYYmmdd_HHMMSS}``."""
    return f"{size}_e{epochs}_{ts or timestamp()}"


def clean_dataset_name(data_path: str | Path) -> str:
    """Port of legacy ``clean_dataset_name`` — readable, filesystem-safe token.

    Used only for *descriptive export filenames*, not for dataset directory
    slugs (see ``datasets.slugify``).
    """
    try:
        name = Path(str(data_path)).parent.name
        for suffix in [
            ".v1i.yolov8", ".v2i.yolov8", ".v3i.yolov8", ".v4i.yolov8",
            ".v5i.yolov8", ".v6i.yolov8", ".v7i.yolov8", ".yolov8",
            "roboflow-fast-model-augmented3x", " images",
        ]:
            name = name.replace(suffix, "")
        name = name.replace(" ", "_").replace("-", "_").replace(".", "_").lower()
        while "__" in name:
            name = name.replace("__", "_")
        return name.strip("_") or "unknown"
    except Exception:
        return "unknown"


def export_name(meta: dict, which: str = "best") -> str:
    """Descriptive ``.pt`` filename for nicer downloads.

    Generalises legacy ``make_model_name`` across families:
    ``{prefix}_{size}[_{imgsz}]_{dataset}[_f{fitness%}][_last].pt``
    """
    family = meta.get("family") or "yolov26"
    prefix = FAMILIES.get(family, FAMILIES["yolov26"])["prefix"]

    parts: list[str] = [str(meta.get("size") or "model")]

    imgsz = meta.get("imgsz") or 640
    try:
        if int(imgsz) != 640:
            parts.append(str(int(imgsz)))
    except (TypeError, ValueError):
        pass

    dataset = meta.get("dataset_name")
    if not dataset:
        dataset = clean_dataset_name(meta.get("dataset", ""))
    parts.append(str(dataset)[:20] or "unknown")

    fitness = meta.get("best_fitness") or 0.0
    try:
        fitness_pct = int(float(fitness) * 100)
        if fitness_pct > 0:
            parts.append(f"f{fitness_pct}")
    except (TypeError, ValueError):
        pass

    stem = prefix + "_" + "_".join(parts)
    if which == "last":
        stem += "_last"
    # Final safety pass for the filesystem.
    stem = re.sub(r"[^\w\-.]+", "_", stem)
    return stem + ".pt"
