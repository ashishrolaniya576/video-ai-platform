import asyncio
import os
import sys
import time
import queue
import numpy as np
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.stabilize import StabilizationModel
from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.video_visibility import VideoVisibilityModel
from app.models.distance_estimation import DistanceEstimationModel
from app.pipeline.live_pipeline import LivePipelineManager
from app.utils.logger import get_logger

logger = get_logger("profiler")

async def profile_pipeline():
    logger.info("Initializing Models...")
    
    device = "cuda"
    models = {
        "stabilization": StabilizationModel(device),
        "heavy_rain_removal": HeavyRainRemovalModel(device),
        "video_visibility": VideoVisibilityModel(device),
        "distance_estimation": DistanceEstimationModel(device),
    }

    # Load models
    for name, model in models.items():
        t0 = time.time()
        model.load_model()
        logger.info(f"{name} loaded in {time.time()-t0:.2f}s")
        
    pipeline = LivePipelineManager(models)
    
    # Enable all models
    request = {
        "stabilization": True,
        "heavy_rain_removal": True,
        "video_visibility": True,
        "distance_estimation": True,
    }
    
    session_id = "profile_session"
    pipeline.start_session(session_id, request)
    
    # Wait for session to start
    await asyncio.sleep(1.0)
    
    # Generate 50 dummy frames (1080p BGR)
    num_frames = 50
    dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    logger.info(f"Starting profiling with {num_frames} frames...")
    
    start_time = time.time()
    
    frames_processed = 0
    latencies = []
    
    for i in range(num_frames):
        push_t0 = time.time()
        # Push frame
        await pipeline.process_frame_async(session_id, dummy_frame)
        
        # In a real scenario, frames come at 30 FPS. Let's push as fast as we can to measure throughput,
        # but also read output.
        while True:
            # Try to get processed frame
            session = pipeline.sessions.get(session_id)
            if not session:
                break
                
            try:
                out = session.output_queue.get_nowait()
                latencies.append(time.time() - push_t0)
                frames_processed += 1
                break
            except queue.Empty:
                await asyncio.sleep(0.01)
                
    end_time = time.time()
    
    pipeline.stop_session(session_id)
    
    total_time = end_time - start_time
    fps = frames_processed / total_time if total_time > 0 else 0
    
    logger.info("--- PROFILING RESULTS ---")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Frames processed: {frames_processed}")
    logger.info(f"End-to-End Throughput: {fps:.2f} FPS")
    
    if latencies:
        logger.info(f"Average Latency: {np.mean(latencies)*1000:.2f} ms")
        logger.info(f"Min Latency: {np.min(latencies)*1000:.2f} ms")
        logger.info(f"Max Latency: {np.max(latencies)*1000:.2f} ms")
        logger.info(f"95th pctl Latency: {np.percentile(latencies, 95)*1000:.2f} ms")
        
    for name, model in models.items():
        model.cleanup()

if __name__ == "__main__":
    asyncio.run(profile_pipeline())
