"""
POST /process — Submit a video processing job.

The request body is validated by Pydantic. The pipeline is executed
synchronously (long-running). For production at scale, move to a task
queue (Celery/RQ) and return a job ID instead.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from app.pipeline.pipeline import PipelineManager, ProcessingRequest, ProcessingResult
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ── Request / Response schemas ─────────────────────────────────────────────────

class ProcessRequestSchema(BaseModel):
    """
    Incoming JSON body for POST /process.

    Compatible with the Node.js backend payload.
    """

    videoPath: str = Field(..., description="Local file path or streaming URL.")
    stabilization: bool = Field(default=False, description="Enable RAFT video stabilization.")
    heavyRainRemoval: bool = Field(
        default=False, alias="heavyRainRemoval",
        description="Enable heavy rain / adverse weather removal.",
    )
    videoVisibility: bool = Field(
        default=False, alias="videoVisibility",
        description="Enable PromptIR visibility enhancement.",
    )
    objectDetection: bool = Field(
        default=False, alias="objectDetection",
        description="Enable YOLOv11n object detection with bounding boxes.",
    )

    model_config = {"populate_by_name": True}

    @field_validator("videoPath")
    @classmethod
    def video_path_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("videoPath must not be empty.")
        return v.strip()


class ProcessResponseSchema(BaseModel):
    """JSON response for POST /process."""

    status: str
    outputVideo: Optional[str] = None
    executionTime: str = ""
    logs: List[str] = []
    error: Optional[str] = None
    detectionSummary: Optional[dict] = None


# ── Dependency ─────────────────────────────────────────────────────────────────

def get_pipeline(request: Request) -> PipelineManager:
    """Retrieve the shared PipelineManager from app state."""
    pipeline: Optional[PipelineManager] = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is not initialised. The service may still be starting up.",
        )
    return pipeline


# ── Endpoint ───────────────────────────────────────────────────────────────────

@router.post(
    "/process",
    response_model=ProcessResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Process a video through enabled AI models",
    description=(
        "Submit a video URL or file path with a feature selection mask. "
        "The service runs only the enabled models in order: "
        "Stabilization → Heavy Rain Removal → Video Visibility → Object Detection."
    ),
)
async def process_video(
    body: ProcessRequestSchema,
    pipeline: PipelineManager = Depends(get_pipeline),
) -> ProcessResponseSchema:
    logger.info(
        "POST /process — videoPath=%s stabilization=%s heavyRainRemoval=%s "
        "videoVisibility=%s objectDetection=%s",
        body.videoPath,
        body.stabilization,
        body.heavyRainRemoval,
        body.videoVisibility,
        body.objectDetection,
    )

    processing_request = ProcessingRequest(
        video_path=body.videoPath,
        stabilization=body.stabilization,
        heavy_rain_removal=body.heavyRainRemoval,
        video_visibility=body.videoVisibility,
        object_detection=body.objectDetection,
    )

    result: ProcessingResult = pipeline.run(processing_request)

    if result.status == "failed":
        logger.error("Pipeline failed: %s", result.error)

    return ProcessResponseSchema(
        status=result.status,
        outputVideo=result.output_video,
        executionTime=result.execution_time,
        logs=result.logs,
        error=result.error,
        detectionSummary=result.detection_summary,
    )
