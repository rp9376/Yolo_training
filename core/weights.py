"""Base-weight discovery and model-source resolution.

Drives the UI's pretrained-vs-scratch choice. Pretrained weights live in
``weights/`` as ``yolov8{n..x}.pt`` / ``yolo26{n..x}.pt``.
"""

from __future__ import annotations

from pathlib import Path

from .config import FAMILIES, SIZES, WEIGHTS_DIR, family_prefix


def available() -> dict:
    """Return ``{family: {size: {available: bool, path: str|None}}}``."""
    out: dict = {}
    for family, spec in FAMILIES.items():
        prefix = spec["prefix"]
        out[family] = {}
        for size in spec["sizes"]:
            pt = WEIGHTS_DIR / f"{prefix}{size}.pt"
            exists = pt.exists()
            out[family][size] = {
                "available": exists,
                "path": str(pt) if exists else None,
            }
    return out


def weight_path(family: str, size: str) -> Path:
    return WEIGHTS_DIR / f"{family_prefix(family)}{size}.pt"


def has_pretrained(family: str, size: str) -> bool:
    return weight_path(family, size).exists()


def resolve(family: str, size: str, init: str) -> tuple[str, bool]:
    """Resolve ``(model_source, pretrained_bool)`` for a family/size/init.

    - ``init == "pretrained"`` → absolute ``.pt`` path (must exist).
    - ``init == "scratch"``    → ``"{prefix}{size}.yaml"`` (ultralytics resolves it).
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family!r}")
    if size not in SIZES:
        raise ValueError(f"Unknown size: {size!r}")
    prefix = family_prefix(family)

    if init == "pretrained":
        pt = weight_path(family, size)
        if not pt.exists():
            raise ValueError(
                f"No pretrained weights for {family} {size} "
                f"(expected {pt.name} in weights/)"
            )
        return str(pt.resolve()), True
    if init == "scratch":
        return f"{prefix}{size}.yaml", False
    raise ValueError(f"Unknown init mode: {init!r} (expected 'pretrained' or 'scratch')")
