"""Dataset discovery, validation, zip-import, path-registration, deletion.

A dataset is a directory under ``datasets/`` containing a ``data.yaml``.
Discovery follows symlinks, so a "registered" dataset is just a symlink whose
target lives elsewhere on the server.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from .config import DATASETS_DIR, IMAGE_EXTS

REGISTRY_FILE = DATASETS_DIR / ".registry.json"


# --- helpers -----------------------------------------------------------------
def slugify(name: str) -> str:
    """Filesystem-safe, readable dataset directory name.

    Keeps letters, digits, ``-``, ``_`` and ``.``; collapses everything else to
    ``-``. Preserves Roboflow-style names like ``Military.v1i.yolo26``.
    """
    name = str(name).strip()
    name = re.sub(r"[^\w\-.]+", "-", name)
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-_.")
    return name or "dataset"


def _unique_dir(slug: str) -> Path:
    """Return a non-colliding ``datasets/<slug>`` path (suffix -2, -3, ...)."""
    candidate = DATASETS_DIR / slug
    if not candidate.exists() and not candidate.is_symlink():
        return candidate
    i = 2
    while True:
        candidate = DATASETS_DIR / f"{slug}-{i}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
        i += 1


def _load_yaml(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _names_list(data: dict) -> list[str]:
    names = data.get("names")
    if isinstance(names, dict):
        # YOLO sometimes stores {0: 'a', 1: 'b'}; keep index order.
        try:
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        except Exception:
            return [str(v) for v in names.values()]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return []


def _resolve_split_dir(ds_dir: Path, value) -> Path | None:
    """Find the actual images directory for a split.

    Tries (in order): the yaml value stripped of leading ``./`` / ``../`` joined
    under the dataset dir; the raw value joined under the dataset dir; an
    absolute value; and finally ``<ds_dir>/<split>/images`` style fallbacks done
    by the caller.
    """
    if not value:
        return None
    raw = str(value)
    candidates: list[Path] = []

    stripped = raw
    while stripped.startswith(("./", "../")):
        stripped = stripped[3:] if stripped.startswith("../") else stripped[2:]
    if stripped:
        candidates.append(ds_dir / stripped)
    candidates.append(ds_dir / raw)
    p = Path(raw)
    if p.is_absolute():
        candidates.append(p)

    for c in candidates:
        try:
            if c.is_dir():
                return c
        except OSError:
            continue
    return None


def _split_images_dir(ds_dir: Path, data: dict, split: str) -> Path | None:
    """Resolve a split's image directory, trying yaml then conventional dirs."""
    key = {"train": "train", "val": "val", "test": "test"}[split]
    found = _resolve_split_dir(ds_dir, data.get(key))
    if found:
        return found
    # Conventional Roboflow layout fallbacks.
    folder = {"train": "train", "val": "valid", "test": "test"}[split]
    for cand in (ds_dir / folder / "images", ds_dir / split / "images"):
        if cand.is_dir():
            return cand
    return None


def _count_images(d: Path | None) -> int:
    if not d or not d.is_dir():
        return 0
    n = 0
    try:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                n += 1
    except OSError:
        return 0
    return n


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


# --- public API --------------------------------------------------------------
def discover() -> list[dict]:
    """All datasets under DATASETS_DIR (following symlinks), newest-ish order."""
    if not DATASETS_DIR.exists():
        return []
    out = []
    for item in sorted(DATASETS_DIR.iterdir(), key=lambda p: p.name.lower()):
        try:
            if item.name.startswith("."):
                continue
            if item.is_dir() and (item / "data.yaml").exists():
                out.append(info(item.name))
        except OSError:
            continue
    return out


def _dataset_dir(name: str) -> Path:
    return DATASETS_DIR / name


def info(name: str) -> dict:
    """Full info for a single dataset directory name."""
    ds_dir = _dataset_dir(name)
    entry = ds_dir  # the symlink-or-dir entry under datasets/
    is_symlink = entry.is_symlink()
    yaml_path = ds_dir / "data.yaml"
    data = _load_yaml(yaml_path) or {}
    names = _names_list(data)
    nc = data.get("nc")
    if nc is None:
        nc = len(names)

    train_dir = _split_images_dir(ds_dir, data, "train")
    val_dir = _split_images_dir(ds_dir, data, "val")
    test_dir = _split_images_dir(ds_dir, data, "test")

    counts = {
        "train": _count_images(train_dir),
        "valid": _count_images(val_dir),
        "test": _count_images(test_dir),
    }

    v = validate(str(ds_dir))
    target = None
    if is_symlink:
        try:
            target = str(entry.resolve())
        except OSError:
            target = None

    return {
        "name": name,
        "path": str(ds_dir),
        "data_yaml": str(yaml_path),
        "nc": int(nc) if isinstance(nc, int) or (isinstance(nc, str) and str(nc).isdigit()) else len(names),
        "names": names,
        "counts": counts,
        "total_images": sum(counts.values()),
        "size_bytes": _dir_size(ds_dir),
        "source": "registered" if is_symlink else "uploaded",
        "target": target,
        "valid": v["valid"],
        "issues": v["issues"],
    }


def validate(path: str | Path) -> dict:
    """Structured validation. Never raises."""
    ds_dir = Path(path)
    issues: list[str] = []

    if not ds_dir.exists():
        return {"valid": False, "issues": [f"Path does not exist: {ds_dir}"]}

    yaml_path = ds_dir / "data.yaml"
    if not yaml_path.exists():
        return {"valid": False, "issues": ["Missing data.yaml"]}

    data = _load_yaml(yaml_path)
    if data is None:
        return {"valid": False, "issues": ["data.yaml is missing or not parseable"]}

    names = _names_list(data)
    nc = data.get("nc")
    if nc is None:
        issues.append("data.yaml has no 'nc'; inferred from names")
    elif names and int(nc) != len(names):
        issues.append(f"nc ({nc}) does not match number of names ({len(names)})")
    if not names:
        issues.append("data.yaml has no class names")

    train_dir = _split_images_dir(ds_dir, data, "train")
    val_dir = _split_images_dir(ds_dir, data, "val")
    if train_dir is None:
        issues.append("train images directory not found")
    elif _count_images(train_dir) == 0:
        issues.append("train images directory is empty")
    if val_dir is None:
        issues.append("validation images directory not found")
    elif _count_images(val_dir) == 0:
        issues.append("validation images directory is empty")

    return {"valid": len(issues) == 0, "issues": issues}


def _find_dataset_root(extract_dir: Path) -> Path | None:
    """Locate the directory containing data.yaml (top level or nested one level)."""
    if (extract_dir / "data.yaml").exists():
        return extract_dir
    # Nested one level (zip contains a single top folder).
    for child in sorted(extract_dir.iterdir()):
        if child.is_dir() and (child / "data.yaml").exists():
            return child
    # Last resort: search a little deeper.
    for yml in extract_dir.rglob("data.yaml"):
        return yml.parent
    return None


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract rejecting zip-slip / absolute / traversal members."""
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        name = member.filename
        # Reject absolute paths and parent traversal outright.
        if name.startswith(("/", "\\")) or ".." in Path(name).parts:
            raise ValueError(f"Unsafe path in archive: {name!r}")
        target = (dest / name).resolve()
        if dest_resolved != target and dest_resolved not in target.parents:
            raise ValueError(f"Unsafe path in archive (escapes root): {name!r}")
    zf.extractall(dest)


def import_zip(file_obj, name: str | None = None, filename: str | None = None) -> dict:
    """Stream a .zip to a temp file, extract safely, install under datasets/.

    ``file_obj`` is any binary read()-able stream. Returns the new dataset info.
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="ds_upload_", dir=str(DATASETS_DIR)))
    tmp_zip = tmpdir / "upload.zip"
    extract_dir = tmpdir / "extracted"
    extract_dir.mkdir()
    try:
        # Stream to disk in chunks (avoid loading the whole archive in memory).
        with open(tmp_zip, "wb") as out:
            shutil.copyfileobj(file_obj, out, length=1024 * 1024)

        if not zipfile.is_zipfile(tmp_zip):
            raise ValueError("Uploaded file is not a valid .zip archive")

        with zipfile.ZipFile(tmp_zip) as zf:
            _safe_extract(zf, extract_dir)

        root = _find_dataset_root(extract_dir)
        if root is None:
            raise ValueError("No data.yaml found in the archive")

        if not name:
            # Prefer the archive's folder name, else the uploaded filename stem.
            if root != extract_dir:
                name = root.name
            elif filename:
                name = Path(filename).stem
            else:
                name = "dataset"
        slug = slugify(name)
        dest = _unique_dir(slug)
        shutil.move(str(root), str(dest))

        result = info(dest.name)
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def register_path(server_path: str, name: str | None = None) -> dict:
    """Register an existing on-server dataset directory via a symlink."""
    src = Path(server_path).expanduser()
    if not src.is_absolute():
        src = (Path.cwd() / src)
    src = src.resolve()

    if not src.exists() or not src.is_dir():
        raise ValueError(f"Path is not a directory: {server_path}")
    v = validate(src)
    if not v["valid"]:
        raise ValueError("Dataset failed validation: " + "; ".join(v["issues"]))

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(name or src.name)
    dest = _unique_dir(slug)
    try:
        dest.symlink_to(src, target_is_directory=True)
    except (OSError, NotImplementedError):
        # Symlinks unavailable — record in a small registry file instead.
        _registry_add(dest.name, str(src))
        raise ValueError(
            "Symlinks are not available on this filesystem; recorded path in "
            ".registry.json but cannot serve it. Consider uploading a zip."
        )
    return info(dest.name)


def _registry_add(name: str, target: str) -> None:
    reg = {}
    if REGISTRY_FILE.exists():
        try:
            reg = json.loads(REGISTRY_FILE.read_text())
        except Exception:
            reg = {}
    reg[name] = target
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2))


def delete(name: str) -> dict:
    """Remove a dataset: unlink a symlink, or rmtree an uploaded directory."""
    ds_dir = _dataset_dir(name)
    if ds_dir.is_symlink():
        ds_dir.unlink()
    elif ds_dir.is_dir():
        shutil.rmtree(ds_dir)
    else:
        raise FileNotFoundError(f"No such dataset: {name}")
    # Clean any registry entry too.
    if REGISTRY_FILE.exists():
        try:
            reg = json.loads(REGISTRY_FILE.read_text())
            if name in reg:
                del reg[name]
                REGISTRY_FILE.write_text(json.dumps(reg, indent=2))
        except Exception:
            pass
    return {"deleted": True, "name": name}
