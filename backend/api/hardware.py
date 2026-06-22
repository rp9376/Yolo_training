"""Hardware snapshot endpoint (frontend polls ~1s)."""

from __future__ import annotations

from fastapi import APIRouter

from core import hardware as hw

router = APIRouter()


@router.get("/hardware")
def hardware_snapshot() -> dict:
    return hw.snapshot()
