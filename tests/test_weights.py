import pytest

from core import weights


def test_available_absent(root):
    av = weights.available()
    assert av["yolov26"]["n"]["available"] is False
    assert av["yolov26"]["n"]["path"] is None
    assert set(av["yolov8"].keys()) == {"n", "s", "m", "l", "x"}


def test_available_present(root, fake_weight):
    av = weights.available()
    assert av["yolov26"]["n"]["available"] is True
    assert av["yolov26"]["n"]["path"].endswith("yolo26n.pt")


def test_resolve_scratch(root):
    src, pre = weights.resolve("yolov26", "x", "scratch")
    assert src == "yolo26x.yaml" and pre is False
    src8, pre8 = weights.resolve("yolov8", "s", "scratch")
    assert src8 == "yolov8s.yaml" and pre8 is False


def test_resolve_pretrained(root, fake_weight):
    src, pre = weights.resolve("yolov26", "n", "pretrained")
    assert src.endswith("yolo26n.pt") and pre is True


def test_resolve_pretrained_missing(root):
    with pytest.raises(ValueError):
        weights.resolve("yolov26", "n", "pretrained")


def test_resolve_bad_inputs(root):
    with pytest.raises(ValueError):
        weights.resolve("bogus", "n", "scratch")
    with pytest.raises(ValueError):
        weights.resolve("yolov26", "z", "scratch")
    with pytest.raises(ValueError):
        weights.resolve("yolov26", "n", "bogus")
