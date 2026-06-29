import time
import torch
import numpy as np
import cv2

from app.models.distance_estimation import DistanceEstimationModel
from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.video_visibility import VideoVisibilityModel
from app.models.stabilize import StabilizationModel

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Benchmarking on:", device)
    
    # 1080p dummy frame
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    models = [
        DistanceEstimationModel(device),
        HeavyRainRemovalModel(device),
        VideoVisibilityModel(device),
        StabilizationModel(device)
    ]
    
    for model in models:
        print(f"\nLoading {model.name}...")
        model.load_model()
        print(f"Benchmarking {model.name}...")
        
        # Warmup
        try:
            if model.name == "VideoStabilization":
                for _ in range(2):
                    model._estimate_flow(frame, frame)
                
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    model._estimate_flow(frame, frame)
                if device == "cuda": torch.cuda.synchronize()
                t1 = time.time()
            else:
                for _ in range(2):
                    model.process_frame(frame, 0)
                
                if device == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    model.process_frame(frame, 0)
                if device == "cuda": torch.cuda.synchronize()
                t1 = time.time()
                
            ms_per_frame = ((t1 - t0) / 5.0) * 1000
            print(f"==> {model.name}: {ms_per_frame:.2f} ms/frame ({1000/ms_per_frame:.2f} FPS)")
        except Exception as e:
            print(f"Failed to benchmark {model.name}: {e}")
            
        model.cleanup()

if __name__ == '__main__':
    benchmark()
