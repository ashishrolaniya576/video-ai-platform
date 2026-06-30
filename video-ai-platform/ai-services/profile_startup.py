import time
import asyncio
from app.config.settings import settings
import torch

async def profile_startup():
    print("Starting Profile...")
    timings = {}
    
    t0 = time.time()
    device = settings.resolve_device()
    t1 = time.time()
    timings['configuration_load'] = t1 - t0
    
    # CUDA Init
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.init()
    t1 = time.time()
    timings['cuda_initialization'] = t1 - t0
    
    dummy_frame = __import__('numpy').zeros((540, 960, 3), dtype=__import__('numpy').uint8)

    # RAFT Load
    t0 = time.time()
    from app.models.stabilize import StabilizationModel
    raft = StabilizationModel(device=device)
    raft.load_model()
    t1 = time.time()
    timings['raft_load'] = t1 - t0
    
    # RAFT Warmup
    t0 = time.time()
    raft.process_frame_streaming(dummy_frame, 0)
    t1 = time.time()
    timings['raft_warmup'] = t1 - t0
    
    # HeavyRain Load
    t0 = time.time()
    from app.models.heavy_rain_remove import HeavyRainRemovalModel
    heavy_rain = HeavyRainRemovalModel(device=device)
    heavy_rain.load_model()
    t1 = time.time()
    timings['heavy_rain_load'] = t1 - t0
    
    # HeavyRain Warmup
    t0 = time.time()
    heavy_rain.process_frame(dummy_frame, 0)
    t1 = time.time()
    timings['heavy_rain_warmup'] = t1 - t0
    
    # PromptIR Load
    t0 = time.time()
    from app.models.video_visibility import VideoVisibilityModel
    prompt_ir = VideoVisibilityModel(device=device)
    prompt_ir.load_model()
    t1 = time.time()
    timings['prompt_ir_load'] = t1 - t0
    
    # PromptIR Warmup
    t0 = time.time()
    prompt_ir.process_frame(dummy_frame, 0)
    t1 = time.time()
    timings['prompt_ir_warmup'] = t1 - t0
    
    # Distance Load
    t0 = time.time()
    from app.models.distance_estimation import DistanceEstimationModel
    distance = DistanceEstimationModel(device=device)
    distance.load_model()
    t1 = time.time()
    timings['distance_load'] = t1 - t0
    
    # Distance Warmup
    t0 = time.time()
    distance.process_frame(dummy_frame, 0)
    t1 = time.time()
    timings['distance_warmup'] = t1 - t0
    
    # Pipeline creation
    t0 = time.time()
    from app.pipeline.pipeline import PipelineManager
    from app.pipeline.live_pipeline import LivePipelineManager
    models = {"stabilization": raft, "heavy_rain_removal": heavy_rain, "video_visibility": prompt_ir, "distance_estimation": distance}
    pipeline = PipelineManager(models=models)
    live_pipeline = LivePipelineManager(models=models)
    t1 = time.time()
    timings['pipeline_creation'] = t1 - t0
    
    import json
    print(json.dumps(timings, indent=2))

if __name__ == "__main__":
    asyncio.run(profile_startup())
