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

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio

from app.api.health import router as health_router
from app.api.process import router as process_router
from app.api.url_stream import router as url_stream_router
from app.streaming.webrtc import router as webrtc_router
from app.config.settings import settings
from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.distance_estimation import DistanceEstimationModel
from app.models.stabilize import StabilizationModel
from app.models.video_visibility import VideoVisibilityModel
from app.pipeline.pipeline import PipelineManager
from app.pipeline.live_pipeline import LivePipelineManager
from app.utils.logger import get_logger, setup_logging
from app.pipeline.model_manager import model_manager, BackendState
from app.dependencies import verify_backend_ready

setup_logging(settings.log_level)
logger = get_logger(__name__)


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager.
    Yields immediately so Uvicorn port binds instantly.
    Spawns background task for heavy model initialization.
    """
    logger.info("=" * 60)
    logger.info("AI Service starting up (Background Task mode)...")
    logger.info("=" * 60)

    # Fire and forget the heavy initialization
    asyncio.create_task(model_manager.initialize_background(app.state))

    yield  # Application is running and port is open

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("AI Service shutting down…")
    for feature_name, model in model_manager.models.items():
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
            "heavy rain removal, visibility enhancement, and distance estimation "
            "via RAFT, HeavyRainRemoval, PromptIR, and DistanceEstimation."
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
    # Health endpoint is available immediately
    app.include_router(health_router, prefix="", tags=["Health"])
    
    # Process and Stream endpoints require backend READY
    ready_dep = [Depends(verify_backend_ready)]
    app.include_router(process_router, prefix="", tags=["Processing"], dependencies=ready_dep)
    app.include_router(webrtc_router, prefix="", tags=["WebRTC"], dependencies=ready_dep)
    app.include_router(url_stream_router, prefix="", tags=["URLStream"], dependencies=ready_dep)

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
