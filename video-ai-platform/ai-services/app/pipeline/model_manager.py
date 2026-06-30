import time
import asyncio
from enum import Enum
import torch
from typing import Dict, Optional, Any

from app.models.base import BaseModel
from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.distance_estimation import DistanceEstimationModel
from app.models.stabilize import StabilizationModel
from app.models.video_visibility import VideoVisibilityModel
from app.pipeline.pipeline import PipelineManager
from app.pipeline.live_pipeline import LivePipelineManager
from app.utils.logger import get_logger

logger = get_logger(__name__)

class BackendState(Enum):
    BOOTING = "BOOTING"
    LOADING_CONFIGURATION = "LOADING_CONFIGURATION"
    INITIALIZING_CUDA = "INITIALIZING_CUDA"
    LOADING_MODELS = "LOADING_MODELS"
    VALIDATING_MODELS = "VALIDATING_MODELS"
    WARMING_MODELS = "WARMING_MODELS"
    REGISTERING_PIPELINES = "REGISTERING_PIPELINES"
    REGISTERING_ROUTES = "REGISTERING_ROUTES"
    READY = "READY"
    RUNNING = "RUNNING"


class AppModelManager:
    """
    Encapsulates backend state and model initialization sequence.
    Handles startup asynchronously to prevent Uvicorn port blocking.
    """
    def __init__(self):
        self.state = BackendState.BOOTING
        self.models: Dict[str, BaseModel] = {}
        self.pipeline: Optional[PipelineManager] = None
        self.live_pipeline: Optional[LivePipelineManager] = None
        self.startup_progress = 0
        self.device = "cpu"
        self.error_log = []

    def transition(self, new_state: BackendState, progress: int = 0):
        self.state = new_state
        if progress > 0:
            self.startup_progress = progress
        logger.info(f"Backend State: {self.state.name} ({self.startup_progress}%)")

    async def initialize_background(self, app_state: Any):
        """The main initialization routine running in the background."""
        try:
            self.transition(BackendState.LOADING_CONFIGURATION, 10)
            from app.config.settings import settings
            self.device = settings.resolve_device()
            app_state.device = self.device
            
            # CUDA initialization
            self.transition(BackendState.INITIALIZING_CUDA, 20)
            if self.device == "cuda" and torch.cuda.is_available():
                # Force CUDA context creation
                torch.cuda.init()
            
            # Model Instantiation
            self.transition(BackendState.LOADING_MODELS, 30)
            self.models = {
                "stabilization": StabilizationModel(device=self.device),
                "heavy_rain_removal": HeavyRainRemovalModel(device=self.device),
                "video_visibility": VideoVisibilityModel(device=self.device),
                "distance_estimation": DistanceEstimationModel(device=self.device),
            }
            
            # Load Weights & Validate
            self.transition(BackendState.VALIDATING_MODELS, 40)
            load_progress = 40
            for name, model in self.models.items():
                try:
                    logger.info(f"Loading weights for {name}...")
                    # Run CPU-bound loading in a thread if load_model is sync, 
                    # but here we just call it. Since it's an asyncio background task,
                    # blocking the event loop is okay because FastAPI uses a threadpool for sync routes,
                    # but to be safe, we'll run blocking code in an executor.
                    await asyncio.to_thread(model.load_model)
                    
                    if not model.is_available:
                        logger.warning(f"Model {name} load failed: {model.unavailable_reason}")
                except Exception as e:
                    import traceback
                    logger.error(f"Failed to load model {name}: {e}\n{traceback.format_exc()}")
                    model.is_available = False
                    model.unavailable_reason = str(e)
                    self.error_log.append(f"{name} load error: {e}")
                load_progress += 10
                self.startup_progress = load_progress
                
            # Warmup
            self.transition(BackendState.WARMING_MODELS, 80)
            dummy_frame = __import__('numpy').zeros((540, 960, 3), dtype=__import__('numpy').uint8)
            
            for name, model in self.models.items():
                if model.is_available:
                    try:
                        logger.info(f"Warming up {name}...")
                        if name == "stabilization":
                            await asyncio.to_thread(model.process_frame_streaming, dummy_frame, 0)
                        else:
                            await asyncio.to_thread(model.process_frame, dummy_frame, 0)
                    except Exception as e:
                        logger.warning(f"Failed to warmup model {name}: {e}")
                        model.is_available = False
                        model.unavailable_reason = f"Warmup failed: {e}"
                        self.error_log.append(f"{name} warmup error: {e}")
            
            # Pipelines
            self.transition(BackendState.REGISTERING_PIPELINES, 90)
            self.pipeline = PipelineManager(models=self.models)
            self.live_pipeline = LivePipelineManager(models=self.models)
            app_state.pipeline = self.pipeline
            app_state.live_pipeline = self.live_pipeline
            
            # Finished
            self.transition(BackendState.READY, 99)
            # Give FastAPI a tiny breather before officially running
            await asyncio.sleep(0.1)
            self.transition(BackendState.RUNNING, 100)
            logger.info("=" * 60)
            logger.info("AI Service successfully initialized in background!")
            logger.info("=" * 60)
            
        except Exception as fatal_e:
            import traceback
            logger.critical(f"FATAL INITIALIZATION ERROR: {fatal_e}\n{traceback.format_exc()}")
            self.error_log.append(f"FATAL: {fatal_e}")
            # Do not crash the process; leave it in a broken state so health checks can report it.
            self.state = BackendState.BOOTING

# Global instance
model_manager = AppModelManager()
