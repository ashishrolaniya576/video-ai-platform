import sys
import traceback
from app.pipeline.pipeline import PipelineManager, ProcessingRequest
from app.models.distance_estimation import DistanceEstimationModel
from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.video_visibility import VideoVisibilityModel
from app.models.stabilize import StabilizationModel
import torch
import cv2
import numpy as np
import os

def create_dummy_video(path, frames=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for i in range(frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path

def main():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models = {
            "stabilization": StabilizationModel(device),
            "heavy_rain_removal": HeavyRainRemovalModel(device),
            "video_visibility": VideoVisibilityModel(device),
            "distance_estimation": DistanceEstimationModel(device),
        }
        
        manager = PipelineManager(models)
        
        video_path = "dummy_test_video.mp4"
        create_dummy_video(video_path, frames=2)
        
        req = ProcessingRequest(
            video_path=video_path,
            distance_estimation=True
        )
        
        print("Starting pipeline run...")
        res = manager.run(req)
        print("Result status:", res.status)
        if res.error:
            print("Pipeline error:", res.error)
            
    except Exception as e:
        print("FATAL UNCAUGHT EXCEPTION:")
        traceback.print_exc()

if __name__ == '__main__':
    with open("debug_output.log", "w") as f:
        sys.stdout = f
        sys.stderr = f
        main()
