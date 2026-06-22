"""Health + version info. Resolves torch/cuda/ultralytics lazily; never errors."""

from __future__ import annotations

import platform

from fastapi import APIRouter

from core import hardware

router = APIRouter()


@router.get("/health")
def health() -> dict:
    gpus, backend = hardware._gpus()
    info = {
        "ok": True,
        "python": platform.python_version(),
        "gpu_count": len(gpus),
        "gpu_backend": backend,
        "torch": None,
        "cuda": None,
        "ultralytics": None,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info
