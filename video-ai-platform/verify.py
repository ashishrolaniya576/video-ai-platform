import sys
import os
import traceback

try:
    import torch
    import yaml
    
    sys.path.insert(0, "/home/ashish/video_player/video-ai-platform/distanceEstimation_d2")
    from model_utils import estModel
    
    device = torch.device('cpu')
    
    yaml_path = "/home/ashish/video_player/video-ai-platform/distanceEstimation_d2/data.yaml"
    with open(yaml_path, 'r') as f:
        contents = yaml.safe_load(f)
    num_classes = contents['nc']
    
    model_path = "/home/ashish/video_player/video-ai-platform/distanceEstimation_d2/best.pth"
    print(f"Loading {model_path} with num_classes={num_classes + 1}...")
    
    model = estModel(num_classes=num_classes + 1).to(device)
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    
    print("SUCCESS: Checkpoint loaded and state_dict matches!")
    
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
