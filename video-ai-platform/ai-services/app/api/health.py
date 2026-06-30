"""
GET /health — Service health check.

Returns the current operational status and loaded model inventory.
"""

from __future__ import annotations

from typing import Dict, Optional, List, Any
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel
import psutil
import torch

from app.utils.logger import get_logger
from app.pipeline.live_pipeline import GpuHealthState
from app.pipeline.model_manager import model_manager, BackendState

logger = get_logger(__name__)

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    backend_state: str
    startup_progress: int
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

    for name, model in model_manager.models.items():
        models_loaded[name] = getattr(model, 'is_available', False)

    return HealthResponse(
        status="running",
        backend_state=model_manager.state.name,
        startup_progress=model_manager.startup_progress,
        device=model_manager.device,
        models_loaded=models_loaded,
    )


class ReadyResponse(BaseModel):
    ready: bool
    state: str
    progress: int
    remaining_models: List[str]
    message: Optional[str] = None


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Backend Readiness Check",
    description="Returns true only when the AI backend has finished loading weights, CUDA, and model warmups.",
)
async def get_ready_status() -> ReadyResponse:
    is_ready = model_manager.state == BackendState.RUNNING
    
    remaining = []
    if not is_ready:
        for name, model in model_manager.models.items():
            if not getattr(model, '_loaded', False):
                remaining.append(name)
                
    return ReadyResponse(
        ready=is_ready,
        state=model_manager.state.name,
        progress=model_manager.startup_progress,
        remaining_models=remaining,
        message="Backend is ready." if is_ready else "AI service is still initializing."
    )


# Store startup time for uptime calculation
START_TIME = time.time()

class LiveHealthResponse(BaseModel):
    serverStatus: str
    gpuStatus: str
    healthScore: int
    serverUptime: float
    cpuUsage: float
    ramUsage: float
    gpuUsage: float
    gpuMemoryMb: float
    modelStatus: Dict[str, bool]
    watchdogStatus: str
    recoveryStatus: str
    activeSessions: List[Dict[str, Any]]
    totalRecoveries: int
    totalCudaErrors: int
    workerCount: int
    threadCount: int

@router.get(
    "/health/live",
    response_model=LiveHealthResponse,
    summary="Live Streaming Health Check",
)
async def live_health_check(request: Request) -> LiveHealthResponse:
    live_pipeline = getattr(request.app.state, "live_pipeline", None)
    
    models_loaded: Dict[str, bool] = {}
    for name, model in model_manager.models.items():
        models_loaded[name] = getattr(model, 'is_available', False)
            
    # System stats
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage = psutil.virtual_memory().percent
    
    gpu_mem_mb = 0.0
    gpu_usage = 0.0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        # Note: accurate GPU usage % is hard without pynvml, using 0.0 as placeholder unless available
        
    # Active Sessions
    active_sessions = []
    total_dropped = 0
    total_qsize = 0
    total_recoveries = getattr(live_pipeline, 'total_recoveries', 0)
    total_cuda_errors = getattr(live_pipeline, 'total_cuda_errors', 0)
    
    avg_latency = 0.0
    count_latency = 0
    
    if live_pipeline:
        for sid, session in list(live_pipeline.sessions.items()):
            
            # Profiling aggregates
            q_wait = session.profiling["queue_wait"]
            inf = session.profiling["inference"]
            
            p95_inf = sorted(inf)[int(len(inf)*0.95)] if inf else 0.0
            avg_inf = sum(inf)/len(inf) if inf else 0.0
            
            if inf:
                avg_latency += avg_inf
                count_latency += 1
                
            total_dropped += session.dropped_frames
            total_qsize += session.frame_queue.qsize()
            
            active_sessions.append({
                "sessionUuid": session.session_uuid,
                "workerUuid": session.worker_uuid,
                "sessionState": session.current_state.name,
                "currentFps": session.processed_frames / (time.time() - session.created_at) if time.time() > session.created_at else 0.0,
                "averageFps": session.processed_frames / (time.time() - session.created_at) if time.time() > session.created_at else 0.0,
                "latency": round(session.last_inference_time * 1000, 1),
                "droppedFrames": session.dropped_frames,
                "queueSize": session.frame_queue.qsize(),
                "recoveryCount": session.recovery_attempts,
                "profiling": {
                    "avgQueueWaitMs": round(sum(q_wait)/len(q_wait), 1) if q_wait else 0.0,
                    "avgInferenceMs": round(avg_inf, 1),
                    "p95InferenceMs": round(p95_inf, 1),
                }
            })
            
    avg_latency_overall = avg_latency / count_latency if count_latency > 0 else 0.0
            
    # Calculate Health Score (0-100)
    # Start at 100
    # Penalty for DEGRADED GPU: -100
    # Penalty for WARNING GPU: -20
    # Penalty for high latency (> 100ms): - (latency-100)/10
    # Penalty for dropped frames: - dropped_frames * 2
    # Penalty for queue backup: - queue_size * 5
    # Penalty for recoveries: - recoveries * 10
    score = 100
    
    # Check global GPU state from model_manager (we removed the singleton)
    # But since it's dynamic, we just check CUDA directly
    gpu_status = GpuHealthState.HEALTHY.name
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
        except Exception:
            gpu_status = GpuHealthState.UNAVAILABLE.name
            score -= 100
        
    if avg_latency_overall > 100:
        score -= int((avg_latency_overall - 100) / 10)
        
    score -= (total_dropped * 2)
    score -= (total_qsize * 5)
    score -= (total_recoveries * 10)
    
    score = max(0, min(100, score))
    
    status = "OK" if score >= 80 else "DEGRADED" if score >= 40 else "CRITICAL"
            
    return LiveHealthResponse(
        serverStatus=status,
        gpuStatus=gpu_status,
        healthScore=score,
        serverUptime=time.time() - START_TIME,
        cpuUsage=cpu_usage,
        ramUsage=ram_usage,
        gpuUsage=gpu_usage,
        gpuMemoryMb=round(gpu_mem_mb, 1),
        modelStatus=models_loaded,
        watchdogStatus="ACTIVE" if getattr(live_pipeline, "_watchdog_running", False) else "INACTIVE",
        recoveryStatus="ENABLED" if settings.watchdog_max_recovery_attempts > 0 else "DISABLED",
        activeSessions=active_sessions,
        totalRecoveries=total_recoveries,
        totalCudaErrors=total_cuda_errors,
        workerCount=len(active_sessions),
        threadCount=len(active_sessions) * 2 + 1 # Workers + Metrics + Watchdog
    )
