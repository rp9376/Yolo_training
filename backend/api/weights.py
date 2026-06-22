"""Base-weight availability endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from core import weights

router = APIRouter()


@router.get("/weights")
def list_weights() -> dict:
    return weights.available()
