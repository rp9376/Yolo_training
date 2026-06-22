"""Dataset endpoints: list / info / upload-zip / register-path / validate / delete."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel

from core import datasets

router = APIRouter()


class RegisterRequest(BaseModel):
    path: str
    name: str | None = None


@router.get("/datasets")
def list_datasets() -> list[dict]:
    return datasets.discover()


@router.get("/datasets/{name}")
def dataset_info(name: str) -> dict:
    return datasets.info(name)


@router.post("/datasets/upload")
def upload_dataset(file: UploadFile = File(...), name: str | None = Form(None)) -> dict:
    return datasets.import_zip(file.file, name=name, filename=file.filename)


@router.post("/datasets/register")
def register_dataset(req: RegisterRequest) -> dict:
    return datasets.register_path(req.path, name=req.name)


@router.post("/datasets/{name}/validate")
def validate_dataset(name: str) -> dict:
    from core.config import DATASETS_DIR
    return datasets.validate(DATASETS_DIR / name)


@router.delete("/datasets/{name}")
def delete_dataset(name: str) -> dict:
    return datasets.delete(name)
