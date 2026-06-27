"""
GET /health — Service health check.

Returns the current operational status and loaded model inventory.
"""

from __future__ import annotations

from typing import Dict, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    device: Optional[str] = None
    models_loaded: Dict[str, bool] = {}


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns the running status and model load state for all AI models.",
)
async def health_check(request: Request) -> HealthResponse:
    models_loaded: Dict[str, bool] = {}

    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        for name, model in pipeline._models.items():
            models_loaded[name] = model.is_loaded

    device = getattr(request.app.state, "device", None)

    return HealthResponse(
        status="running",
        device=device,
        models_loaded=models_loaded,
    )
