import io
import os
import zipfile
from pathlib import Path

import pytest

import make_synthetic_dataset as synth
from core import config, datasets


def _zip_dir(src_dir: Path, arc_prefix: str = "") -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for f in Path(src_dir).rglob("*"):
            if f.is_file():
                arc = os.path.join(arc_prefix, f.relative_to(src_dir).as_posix())
                zf.write(f, arc)
    buf.seek(0)
    return buf


def test_validate_good(dataset):
    v = datasets.validate(dataset)
    assert v["valid"], v["issues"]


def test_validate_missing_yaml(root, tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    v = datasets.validate(d)
    assert not v["valid"]
    assert any("data.yaml" in i for i in v["issues"])


def test_validate_empty_split(root, tmp_path):
    d = synth.make_dataset(tmp_path / "ds", n_train=0)
    # Remove train images to simulate an empty split.
    for f in (d / "train" / "images").iterdir():
        f.unlink()
    v = datasets.validate(d)
    assert not v["valid"]


def test_discover_and_info(dataset):
    items = datasets.discover()
    assert len(items) == 1
    info = items[0]
    assert info["nc"] == 2
    assert info["names"] == ["shape_a", "shape_b"]
    assert info["counts"]["train"] == 8
    assert info["counts"]["valid"] == 4
    assert info["counts"]["test"] == 2
    assert info["total_images"] == 14
    assert info["source"] == "uploaded"
    assert info["valid"]


def test_import_zip_toplevel(root, tmp_path):
    src = synth.make_dataset(tmp_path / "src")
    info = datasets.import_zip(_zip_dir(src), name="uploaded", filename="uploaded.zip")
    assert info["valid"]
    assert (config.DATASETS_DIR / info["name"] / "data.yaml").exists()
    assert info["counts"]["train"] == 8


def test_import_zip_nested_root(root, tmp_path):
    src = synth.make_dataset(tmp_path / "mydata")
    info = datasets.import_zip(_zip_dir(src, arc_prefix="mydata"))
    assert info["valid"]
    assert info["name"].startswith("mydata")


def test_import_zip_collision_suffix(root, tmp_path):
    src = synth.make_dataset(tmp_path / "src")
    first = datasets.import_zip(_zip_dir(src), name="dup")
    second = datasets.import_zip(_zip_dir(src), name="dup")
    assert first["name"] == "dup"
    assert second["name"] != "dup"
    assert second["name"].startswith("dup-")


def test_zip_slip_rejected(root):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("data.yaml", "nc: 1\nnames: ['a']\n")
        zf.writestr("../evil.txt", "pwned")
    buf.seek(0)
    with pytest.raises(ValueError):
        datasets.import_zip(buf, name="evil")
    # Nothing should have escaped the datasets dir.
    assert not (config.DATASETS_DIR.parent / "evil.txt").exists()


def test_not_a_zip(root):
    buf = io.BytesIO(b"this is not a zip file")
    with pytest.raises(ValueError):
        datasets.import_zip(buf, name="bad")


def test_register_path_symlink(root, tmp_path):
    src = synth.make_dataset(tmp_path / "external_ds")
    info = datasets.register_path(str(src), name="reg")
    entry = config.DATASETS_DIR / info["name"]
    assert entry.is_symlink()
    assert info["source"] == "registered"
    assert any(d["name"] == info["name"] for d in datasets.discover())


def test_register_invalid_path(root, tmp_path):
    with pytest.raises(ValueError):
        datasets.register_path(str(tmp_path / "does_not_exist"))


def test_delete_uploaded(root, tmp_path):
    src = synth.make_dataset(tmp_path / "src")
    info = datasets.import_zip(_zip_dir(src), name="todelete")
    datasets.delete(info["name"])
    assert not (config.DATASETS_DIR / info["name"]).exists()


def test_delete_registered_keeps_target(root, tmp_path):
    src = synth.make_dataset(tmp_path / "ext")
    info = datasets.register_path(str(src), name="reg2")
    datasets.delete(info["name"])
    assert not (config.DATASETS_DIR / info["name"]).exists()
    assert src.exists()  # the real data is preserved
