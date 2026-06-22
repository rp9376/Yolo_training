import pytest

from core import models
from make_synthetic_dataset import FAKE_BEST_EPOCH, FAKE_BEST_FITNESS


def test_scan(fake_run):
    ms = models.list_models()
    assert len(ms) == 1
    m = ms[0]
    assert m["run_name"] == "n_e3_20260101_000000"
    assert m["family"] == "yolov26"
    assert m["size"] == "n"
    assert m["best_epoch"] == FAKE_BEST_EPOCH
    assert abs(m["best_fitness"] - FAKE_BEST_FITNESS) < 1e-6
    assert m["final"]["map50"] == 0.5  # last row


def test_detail(fake_run):
    d = models.detail("n_e3_20260101_000000")
    assert "results.png" in d["artifacts"]
    assert "confusion_matrix.png" in d["artifacts"]
    assert d["class_names"] == ["shape_a", "shape_b"]
    assert len(d["series"]["epoch"]) == 3
    assert d["series"]["metrics/mAP50(B)"][1] == 0.6


def test_artifact_path_ok(fake_run):
    p = models.artifact_path("n_e3_20260101_000000", "results.png")
    assert p.is_file()


def test_artifact_traversal_rejected(fake_run):
    with pytest.raises(ValueError):
        models.artifact_path("n_e3_20260101_000000", "../../../../etc/passwd")
    with pytest.raises(ValueError):
        models.artifact_path("n_e3_20260101_000000", "weights/../../../secret")


def test_weights_path(fake_run):
    assert models.weights_path("n_e3_20260101_000000", "best").is_file()
    assert models.weights_path("n_e3_20260101_000000", "last").is_file()
    with pytest.raises(ValueError):
        models.weights_path("n_e3_20260101_000000", "bogus")


def test_delete(fake_run):
    models.delete("n_e3_20260101_000000")
    assert models.find_run_dir("n_e3_20260101_000000") is None


def test_missing_run():
    with pytest.raises(FileNotFoundError):
        models.detail("does_not_exist")
