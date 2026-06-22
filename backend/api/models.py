"""Model endpoints: list / detail / artifact / download / export / delete."""

from __future__ import annotations

import mimetypes

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core import models, naming

router = APIRouter()


class ExportRequest(BaseModel):
    format: str = "onnx"


@router.get("/models")
def list_models() -> list[dict]:
    return models.list_models()


@router.get("/models/{run_name}")
def model_detail(run_name: str) -> dict:
    return models.detail(run_name)


@router.get("/models/{run_name}/artifact/{filename:path}")
def model_artifact(run_name: str, filename: str):
    path = models.artifact_path(run_name, filename)
    media = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(str(path), media_type=media)


@router.get("/models/{run_name}/download")
def download_weights(run_name: str, which: str = "best"):
    pt = models.weights_path(run_name, which)
    summary = models.detail(run_name)
    filename = naming.export_name(summary, which=which)
    return FileResponse(str(pt), media_type="application/octet-stream",
                        filename=filename)


@router.post("/models/{run_name}/export")
def export_model(run_name: str, req: ExportRequest) -> dict:
    path = models.export_model(run_name, req.format)
    return {"path": str(path), "format": req.format}


@router.get("/models/{run_name}/download_export")
def download_export(run_name: str, format: str = "onnx"):
    path = models.exported_file(run_name, format)
    if path is None:
        raise FileNotFoundError(f"No {format} export found for {run_name}")
    return FileResponse(str(path), media_type="application/octet-stream",
                        filename=path.name)


@router.delete("/models/{run_name}")
def delete_model(run_name: str) -> dict:
    return models.delete(run_name)
