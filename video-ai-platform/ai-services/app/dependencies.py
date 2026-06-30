from fastapi import HTTPException, status
from app.pipeline.model_manager import model_manager, BackendState

async def verify_backend_ready():
    """
    FastAPI Dependency to ensure the backend has finished initializing.
    If the startup background task is still running, requests to functional endpoints
    will cleanly fail with HTTP 503 instead of hanging or crashing.
    """
    if model_manager.state != BackendState.RUNNING:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Server Initializing",
                "state": model_manager.state.name,
                "progress": model_manager.startup_progress
            }
        )
