import time
import torch
import cv2
import numpy as np
from app.config.settings import settings
from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.video_visibility import VideoVisibilityModel
from app.models.distance_estimation import DistanceEstimationModel
from app.models.stabilize import StabilizationModel

def profile_model(model_class, model_name):
    print(f"\n--- Profiling {model_name} ---")
    device = settings.resolve_device()
    model = model_class(device=device)
    
    t0 = time.perf_counter()
    model.load_model()
    t1 = time.perf_counter()
    print(f"[{model_name}] Model load time: {t1 - t0:.2f}s")
    
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Warmup
    model.process_frame(frame, 0)
    
    num_frames = 2
    t2 = time.perf_counter()
    for i in range(num_frames):
        model.process_frame(frame, i + 1)
    t3 = time.perf_counter()
    
    avg_time = (t3 - t2) / num_frames
    print(f"[{model_name}] Avg inference time: {avg_time:.3f}s/frame ({1/avg_time:.1f} FPS)")

if __name__ == "__main__":
    profile_model(HeavyRainRemovalModel, "Heavy Rain")
    profile_model(VideoVisibilityModel, "PromptIR")
    profile_model(DistanceEstimationModel, "Distance Estimation")
    profile_model(StabilizationModel, "Stabilization")
