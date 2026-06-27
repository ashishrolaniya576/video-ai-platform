"""
AI Service entry point.

Startup sequence:
  1. Configure logging.
  2. Resolve torch device.
  3. Instantiate all AI models.
  4. Call load_model() on each model (weights loaded into memory once).
  5. Build PipelineManager with loaded models.
  6. Attach pipeline to app.state.
  7. Start serving requests.

Shutdown sequence:
  1. Call cleanup() on each model to release GPU memory.
  2. Log graceful shutdown.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.process import router as process_router
from app.config.settings import settings
from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.object_detection import ObjectDetectionModel
from app.models.stabilize import StabilizationModel
from app.models.video_visibility import VideoVisibilityModel
from app.pipeline.pipeline import PipelineManager
from app.utils.logger import get_logger, setup_logging

setup_logging(settings.log_level)
logger = get_logger(__name__)


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.

    Everything before `yield` runs on startup.
    Everything after `yield` runs on shutdown.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("AI Service starting up…")
    logger.info("=" * 60)

    device = settings.resolve_device()
    logger.info("Resolved compute device: %s", device.upper())
    app.state.device = device

    # Instantiate models
    models = {
        "stabilization": StabilizationModel(device=device),
        "heavy_rain_removal": HeavyRainRemovalModel(device=device),
        "video_visibility": VideoVisibilityModel(device=device),
        "object_detection": ObjectDetectionModel(device=device),
    }

    # Load all models into memory — weights are loaded exactly once here
    for feature_name, model in models.items():
        try:
            logger.info("Loading model: %s…", model.name)
            model.load_model()
            logger.info("Model loaded: %s ✓", model.name)
        except FileNotFoundError as exc:
            logger.warning(
                "Model '%s' weights not found — feature will be unavailable: %s",
                feature_name,
                exc,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load model '%s': %s — feature will be unavailable.",
                feature_name,
                exc,
            )

    # Build pipeline and attach to app state
    pipeline = PipelineManager(models=models)
    app.state.pipeline = pipeline

    logger.info("Pipeline manager ready with models: %s", list(models.keys()))
    logger.info("AI Service is ready to accept requests on port %d", settings.port)
    logger.info("=" * 60)

    yield  # Application is running

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("AI Service shutting down…")
    for feature_name, model in models.items():
        try:
            model.cleanup()
            logger.info("Model '%s' cleaned up.", feature_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error during cleanup of '%s': %s", feature_name, exc)

    logger.info("AI Service shutdown complete.")


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="VideoAI Processing Service",
        description=(
            "AI-powered video processing service providing stabilization, "
            "heavy rain removal, visibility enhancement, and object detection "
            "via RAFT, HeavyRainRemoval, PromptIR, and YOLOv11."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS — allow the Node.js backend to call this service
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production to the backend origin
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # Global exception handler — never let unhandled errors crash the server
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}"},
        )

    # Register routers
    app.include_router(health_router, prefix="", tags=["Health"])
    app.include_router(process_router, prefix="", tags=["Processing"])

    return app


app = create_app()


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,  # Never use reload=True with GPU models in memory
    )
