import asyncio
import queue
import threading
import time
import uuid
import torch
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import httpx
import numpy as np

from app.models.base import BaseModel
from app.utils.logger import get_logger
from app.config.settings import settings

logger = get_logger(__name__)


class SessionState(Enum):
    INITIALIZING = "INITIALIZING"
    LOADING_MODELS = "LOADING_MODELS"
    READY = "READY"
    STREAMING = "STREAMING"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"
    TERMINATING = "TERMINATING"
    TERMINATED = "TERMINATED"

class SessionHealth(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    STALLED = "STALLED"
    FAILED = "FAILED"
    TERMINATING = "TERMINATING"

class GpuHealthState(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"

# Global tracking of GPU health
GLOBAL_GPU_STATE = GpuHealthState.HEALTHY

# Define valid state transitions
VALID_TRANSITIONS = {
    SessionState.INITIALIZING: [SessionState.LOADING_MODELS, SessionState.FAILED, SessionState.TERMINATING],
    SessionState.LOADING_MODELS: [SessionState.READY, SessionState.FAILED, SessionState.TERMINATING],
    SessionState.READY: [SessionState.STREAMING, SessionState.FAILED, SessionState.TERMINATING],
    SessionState.STREAMING: [SessionState.RECOVERING, SessionState.FAILED, SessionState.TERMINATING],
    SessionState.RECOVERING: [SessionState.STREAMING, SessionState.FAILED, SessionState.TERMINATING],
    SessionState.FAILED: [SessionState.TERMINATING],
    SessionState.TERMINATING: [SessionState.TERMINATED],
    SessionState.TERMINATED: []
}

class InvalidStateTransitionError(Exception):
    pass


class CudaErrorCategory(Enum):
    RECOVERABLE = "RECOVERABLE"
    FATAL = "FATAL"
    UNKNOWN = "UNKNOWN"


def classify_cuda_error(e: Exception) -> CudaErrorCategory:
    """Intelligently distinguish between recoverable OOMs and fatal context corruption."""
    if not isinstance(e, RuntimeError):
        return CudaErrorCategory.UNKNOWN
        
    error_msg = str(e).lower()
    
    # Recoverable: Out of Memory, allocation failures
    if "out of memory" in error_msg or "allocate" in error_msg or isinstance(e, torch.cuda.OutOfMemoryError):
        return CudaErrorCategory.RECOVERABLE
        
    # Fatal: Illegal memory access, device-side assert, driver failures
    if "illegal memory access" in error_msg or "device-side assert" in error_msg or "cuda error" in error_msg:
        return CudaErrorCategory.FATAL
        
    return CudaErrorCategory.UNKNOWN


class LiveSession:
    """Manages the state and queues for a single live streaming session."""
    def __init__(self, session_id: str, request: dict, models: Dict[str, BaseModel], model_order: List[str]):
        # Identifiers
        self.session_id = session_id
        self.session_uuid = str(uuid.uuid4())
        self.worker_uuid = str(uuid.uuid4())
        
        # State tracking
        self.current_state = SessionState.INITIALIZING
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.state_history: List[Dict[str, Any]] = [{
            "state": self.current_state,
            "timestamp": self.created_at,
            "reason": "Initialization"
        }]
        
        # Timing metrics
        self.last_successful_inference = 0.0
        self.last_frame_timestamp = 0.0
        self.recovery_attempts = 0
        
        self.request = request
        self.models = models
        self.model_order = model_order
        
        self.profiling = {
            "queue_wait": [],
            "inference": []
        }
        
        # Bounded queues to prevent memory explosion if inference is slower than capture
        self.frame_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=5)
        
        self.is_running = False
        self.worker_thread: Optional[threading.Thread] = None
        self.metrics_thread: Optional[threading.Thread] = None
        
        # Performance metrics
        self.frame_idx = 0
        self.last_inference_time = 0.0
        self.max_inference_time = 0.0
        self.start_time = time.time()
        self.processed_frames = 0
        self.dropped_frames = 0
        self.input_frames = 0
        
        self.transition_state(SessionState.LOADING_MODELS, "Loading requested models")
        # Build active models list
        self.active_models = self._build_pipeline()
        self.transition_state(SessionState.READY, "Pipeline built successfully")

    def transition_state(self, new_state: SessionState, reason: str = ""):
        """Enforce strict state transitions and log history."""
        if new_state not in VALID_TRANSITIONS.get(self.current_state, []):
            raise InvalidStateTransitionError(
                f"Invalid transition from {self.current_state.name} to {new_state.name}"
            )
            
        prev_state = self.current_state
        self.current_state = new_state
        self.updated_at = time.time()
        
        self.state_history.append({
            "state": self.current_state,
            "timestamp": self.updated_at,
            "reason": reason
        })
        
        logger.info(
            f"[Session: {self.session_uuid}] [Worker: {self.worker_uuid}] "
            f"State Transition: {prev_state.name} -> {self.current_state.name} "
            f"at {self.updated_at} | Reason: {reason}"
        )

    def _build_pipeline(self) -> List[tuple]:
        active: List[tuple] = []
        for feature_name in self.model_order:
            # Check if feature is enabled in request
            if not self.request.get(feature_name, False):
                continue

            model = self.models.get(feature_name)
            if model is None or not model.is_available:
                logger.warning(f"Feature '{feature_name}' requested but model unavailable.")
                continue
                
            active.append((feature_name, model))
            
        stage_names = " -> ".join(m.name for _, m in active)
        logger.info(f"Live Pipeline for {self.session_id}: Input -> {stage_names} -> Output")
        return active

    def get_health(self) -> SessionHealth:
        """Calculate health using multiple indicators."""
        if self.current_state in [SessionState.FAILED, SessionState.TERMINATED]:
            return SessionHealth.FAILED
        if self.current_state == SessionState.TERMINATING:
            return SessionHealth.TERMINATING
            
        qsize = self.frame_queue.qsize()
        now = time.time()
        
        time_since_last_inference = now - self.last_successful_inference
        if self.last_successful_inference == 0.0:
            time_since_last_inference = 0.0 # Hasn't started yet
            
        if time_since_last_inference > settings.watchdog_stall_timeout_seconds and self.is_running:
            return SessionHealth.STALLED
            
        if qsize >= settings.watchdog_queue_critical_threshold:
            return SessionHealth.WARNING
            
        return SessionHealth.HEALTHY

    def start(self):
        self.transition_state(SessionState.STREAMING, "Starting worker and metrics threads")
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()
        
        self.metrics_thread = threading.Thread(target=self._metrics_loop, daemon=True)
        self.metrics_thread.start()
        
        logger.info(f"[Worker: {self.worker_id}] Live session {self.session_id} started.")

    def recover(self) -> bool:
        """
        Attempt graceful recovery of a stalled session.
        Stops worker, flushes queues, clears session tensors, and restarts worker.
        """
        if self.recovery_attempts >= settings.watchdog_max_recovery_attempts:
            self.transition_state(SessionState.FAILED, f"Exceeded max recovery attempts ({self.recovery_attempts})")
            return False
            
        self.recovery_attempts += 1
        prev_worker_id = self.worker_uuid
        self.transition_state(SessionState.RECOVERING, f"Starting recovery attempt {self.recovery_attempts}")
        
        # 1. Signal shutdown to existing worker
        self.is_running = False
        
        # Unblock queues by pushing Nones
        try:
            self.frame_queue.put_nowait(None)
        except queue.Full:
            pass
            
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
            
        # 2. Flush Queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break
                
        # 3. Release temporary per-session GPU/CPU tensors
        for feature_name, model in self.active_models:
            if hasattr(model, "cleanup_session"):
                model.cleanup_session(self.session_uuid)
                
        # 4. Generate new worker ID and restart
        self.worker_uuid = str(uuid.uuid4())
        self.last_successful_inference = time.time()
        logger.info(f"[Session: {self.session_uuid}] Graceful recovery complete. Worker {prev_worker_id} -> {self.worker_uuid}")
        
        # 5. Restart streaming
        self.start()
        return True

    def stop(self):
        try:
            self.transition_state(SessionState.TERMINATING, "Stop requested")
        except InvalidStateTransitionError:
            # If already terminated or in a state that can't terminate gracefully, force it.
            pass
            
        self.is_running = False
        if self.worker_thread:
            # Unblock queue
            try:
                self.frame_queue.put(None, timeout=1)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=2.0)
            
        if self.metrics_thread:
            self.metrics_thread.join(timeout=2.0)
            
        try:
            self.transition_state(SessionState.TERMINATED, "Cleanup complete")
        except InvalidStateTransitionError:
            pass
            
        # Clear per-session model state buffers to prevent GPU/CPU memory leaks
        for feature_name, model in self.active_models:
            if hasattr(model, "cleanup_session"):
                model.cleanup_session(self.session_uuid)
                
        logger.info(f"[Worker: {self.worker_uuid}] Live session {self.session_id} stopped.")

    def _inference_loop(self):
        """Background thread that pulls frames from queue and runs AI models."""
        while self.is_running:
            try:
                # Wait for a frame
                wait_start = time.perf_counter()
                frame = self.frame_queue.get(timeout=0.5)
                queue_wait_time = time.perf_counter() - wait_start
                
                if frame is None:
                    break  # Stop signal
                
                self.last_frame_timestamp = time.time()
                start_time = time.perf_counter()
                
                # Process frame
                processed_frame = frame
                for feature_name, model in self.active_models:
                    if feature_name == "stabilization" and hasattr(model, "process_frame_streaming"):
                        processed_frame = model.process_frame_streaming(processed_frame, self.frame_idx, session_id=self.session_uuid)
                    else:
                        processed_frame = model.process_frame(processed_frame, self.frame_idx, request=self.request)
                
                self.frame_idx += 1
                self.processed_frames += 1
                self.last_successful_inference = time.time()
                
                inf_time = time.perf_counter() - start_time
                self.last_inference_time = inf_time
                self.max_inference_time = max(self.max_inference_time, inf_time)
                
                # Add to profiling (limit to last 100 frames)
                self.profiling["queue_wait"].append(queue_wait_time * 1000)
                self.profiling["inference"].append(inf_time * 1000)
                if len(self.profiling["queue_wait"]) > 100:
                    self.profiling["queue_wait"].pop(0)
                    self.profiling["inference"].pop(0)
                
                # Push to output (drop if full during normal operation, but shouldn't block)
                try:
                    self.output_queue.put_nowait(processed_frame)
                except queue.Full:
                    self.dropped_frames += 1
                    
            except Exception as e:
                global GLOBAL_GPU_STATE
                
                error_cat = classify_cuda_error(e)
                
                logger.error(
                    f"[CUDA Classifier] Session: {self.session_uuid} | Worker: {self.worker_uuid} | "
                    f"Error: {type(e).__name__} | Category: {error_cat.name} | Message: {e}"
                )
                
                # Update global metrics (if available)
                pipeline = getattr(self, "_pipeline_manager", None)
                if pipeline:
                    pipeline.total_cuda_errors += 1
                
                if error_cat == CudaErrorCategory.RECOVERABLE:
                    # OOM or allocation failure. We can try to recover.
                    logger.warning(f"[Session: {self.session_uuid}] Initiating staged recovery due to recoverable CUDA error.")
                    if self.is_running:
                        # Clear cache before recovery
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        success = self.recover()
                        if not success:
                            self.transition_state(SessionState.FAILED, f"Recovery from CUDA OOM failed: {e}")
                elif error_cat == CudaErrorCategory.FATAL:
                    # Context is corrupted. Mark server as DEGRADED.
                    logger.critical(f"[Session: {self.session_uuid}] FATAL CUDA ERROR detected. Marking server as DEGRADED.")
                    GLOBAL_GPU_STATE = GpuHealthState.DEGRADED
                    if self.is_running:
                        self.transition_state(SessionState.FAILED, f"Fatal GPU error: {e}")
                else:
                    # Normal exceptions
                    logger.error(f"[Worker: {self.worker_uuid}] Inference loop crashed: {e}")
                    if self.is_running:
                        self.transition_state(SessionState.FAILED, f"Worker crashed: {e}")
                        
            finally:
                self.is_running = False

    def _metrics_loop(self):
        """Periodically send metrics to Node.js backend."""
        # Get Node URL from settings or use default localhost
        node_url = "http://localhost:5000"
        
        last_check_time = time.time()
        last_processed_frames = 0
        
        while self.is_running:
            time.sleep(1.0)
            
            now = time.time()
            elapsed = now - last_check_time
            frames_since_last = self.processed_frames - last_processed_frames
            fps = frames_since_last / elapsed if elapsed > 0 else 0
            
            last_check_time = now
            last_processed_frames = self.processed_frames
            
            # GPU Info
            gpu_mem_mb = 0
            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            input_fps = (self.input_frames - last_processed_frames) / elapsed if elapsed > 0 else 0

            metrics = {
                "inferenceFps": round(fps, 1),
                "inputFps": round(input_fps, 1),
                "latencyMs": round(self.last_inference_time * 1000, 1),
                "maxLatencyMs": round(self.max_inference_time * 1000, 1),
                "inputQueue": self.frame_queue.qsize(),
                "outputQueue": self.output_queue.qsize(),
                "totalProcessed": self.processed_frames,
                "droppedFrames": self.dropped_frames,
                "gpuMemoryMb": round(gpu_mem_mb, 1),
                "state": self.current_state.name,
                "workerId": self.worker_uuid,
                "sessionUuid": self.session_uuid,
                "recoveryAttempts": self.recovery_attempts,
                "activeModels": [m.name for _, m in self.active_models]
            }
            
            # Reset max latency per interval to reflect current max
            self.max_inference_time = 0.0
            
            try:
                # Use a short timeout so we don't block the metrics thread
                with httpx.Client(timeout=2.0) as client:
                    client.post(
                        f"{node_url}/api/metrics",
                        json={"jobId": self.session_id, "metrics": metrics}
                    )
            except Exception as e:
                logger.debug(f"Failed to send metrics to backend: {e}")


class LivePipelineManager:
    """
    Manages multiple live video streams and routes them through the AI models.
    """
    def __init__(self, models: Dict[str, BaseModel]) -> None:
        self._models = models
        self.sessions: Dict[str, LiveSession] = {}
        
        # Start Watchdog
        self._watchdog_running = True
        self.total_recoveries = 0
        self.total_cuda_errors = 0
        
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True, name="WatchdogThread")
        self._watchdog_thread.start()
        logger.info("LivePipelineManager Watchdog initialized.")

    def _watchdog_loop(self):
        """Dedicated Watchdog thread to monitor all active sessions."""
        sleep_sec = settings.watchdog_monitor_interval_ms / 1000.0
        while self._watchdog_running:
            try:
                # Iterate over a copy of items to avoid dictionary mutation errors during iteration
                for session_id, session in list(self.sessions.items()):
                    health = session.get_health()
                    
                    if health == SessionHealth.STALLED:
                        logger.error(
                            f"[Watchdog] Session {session_id} STALLED. Initiating Recovery Manager..."
                        )
                        success = session.recover()
                        if success:
                            self.total_recoveries += 1
                        else:
                            logger.critical(f"[Watchdog] Session {session_id} recovery failed. Terminating.")
                            
                    elif health == SessionHealth.WARNING:
                        logger.warning(
                            f"[Watchdog] Session: {session.session_uuid} | Worker: {session.worker_uuid} | "
                            f"State: {session.current_state.name} | Health: {health.name} | "
                            f"QSize: {session.frame_queue.qsize()} | "
                            f"LastInf: {time.time() - session.last_successful_inference:.1f}s ago | "
                            f"Processed: {session.processed_frames}"
                        )
                        
            except Exception as e:
                logger.error(f"Watchdog internal error: {e}")
                
            time.sleep(sleep_sec)

    def shutdown(self):
        self._watchdog_running = False
        if self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)
            
        for s_id, session in list(self.sessions.items()):
            self.stop_session(s_id)

    def start_session(self, session_id: str, request: dict):
        if GLOBAL_GPU_STATE == GpuHealthState.DEGRADED:
            logger.warning(f"Rejecting new session {session_id} because server is in DEGRADED GPU state.")
            raise RuntimeError("Server is currently in DEGRADED GPU state. New live streams are rejected.")
            
        if session_id in self.sessions:
            logger.info(f"Session {session_id} already running.")
            return
            
        session = LiveSession(session_id, request, self._models, ["stabilization", "heavy_rain_removal", "video_visibility", "distance_estimation"])
        session._pipeline_manager = self
        self.sessions[session_id] = session
        session.start()

    def stop_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id].stop()
            del self.sessions[session_id]

    async def process_frame_async(self, session_id: str, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Called by the aiortc video track to enqueue a frame and dequeue a processed frame.
        """
        session = self.sessions.get(session_id)
        if not session:
            return frame

        session.input_frames += 1

        # Put new frame in queue
        if session.frame_queue.full():
            try:
                # Drop oldest frame to keep latency low
                session.frame_queue.get_nowait()
                session.dropped_frames += 1
            except queue.Empty:
                pass
        
        session.frame_queue.put(frame)

        # Retrieve a processed frame if available (don't block forever)
        try:
            # We await slightly to allow the background thread to run
            await asyncio.sleep(0.001)
            # Try to get the latest processed frame
            processed_frame = session.output_queue.get_nowait()
            return processed_frame
        except queue.Empty:
            # If no frame is ready yet, we return None (track will handle it)
            return None
        except Exception as e:
            logger.error(f"Error fetching output frame for session {session_id}: {e}")
            return None
