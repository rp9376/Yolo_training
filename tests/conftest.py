"""Pytest fixtures + isolated project root.

We point YOLO_STUDIO_ROOT at a temp directory *before* importing ``core`` so the
whole engine (datasets/weights/runs/queue file) operates inside a sandbox and
never touches the real project tree.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Establish the sandbox root before core is imported anywhere.
_ROOT = Path(tempfile.mkdtemp(prefix="yolo_studio_test_"))
os.environ["YOLO_STUDIO_ROOT"] = str(_ROOT)

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))            # so `import core`, `import backend` work
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import make_synthetic_dataset`

import pytest  # noqa: E402

from core import config  # noqa: E402
import make_synthetic_dataset as synth  # noqa: E402


def _reset_tree() -> None:
    for sub in ("datasets", "weights", "runs"):
        d = _ROOT / sub
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
    if config.QUEUE_FILE.exists():
        config.QUEUE_FILE.unlink()


@pytest.fixture(scope="session", autouse=True)
def _session_dirs():
    config.ensure_dirs()
    yield
    shutil.rmtree(_ROOT, ignore_errors=True)


@pytest.fixture
def root() -> Path:
    """A clean sandbox tree for a single test."""
    _reset_tree()
    return _ROOT


@pytest.fixture
def dataset(root: Path) -> Path:
    """A synthetic dataset installed under datasets/synth."""
    return synth.make_dataset(config.DATASETS_DIR / "synth")


@pytest.fixture
def fake_weight(root: Path):
    """Create a dummy pretrained weight (yolo26n.pt) in the weights dir."""
    pt = config.WEIGHTS_DIR / "yolo26n.pt"
    pt.write_bytes(b"PK\x03\x04 dummy-weight")
    return pt


@pytest.fixture
def fake_run(root: Path) -> Path:
    """A fabricated completed run under runs/yolov26/detect/."""
    ds = synth.make_dataset(config.DATASETS_DIR / "synth")
    return synth.make_fake_run(config.RUNS_DIR, family="yolov26",
                               name="n_e3_20260101_000000",
                               data_yaml=str(ds / "data.yaml"))
