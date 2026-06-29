import torch
import sys

try:
    print("Testing torch.load on 'best.pth'...")
    state1 = torch.load("/home/ashish/video_player/video-ai-platform/distanceEstimation_d2/best.pth", map_location="cpu")
    print("SUCCESS: best.pth loaded.")
except Exception as e:
    print(f"FAILED to load best.pth: {e}")

try:
    print("Testing torch.load on 'best' folder...")
    state2 = torch.load("/home/ashish/video_player/video-ai-platform/distanceEstimation_d2/best", map_location="cpu")
    print("SUCCESS: best/ loaded.")
except Exception as e:
    print(f"FAILED to load best/: {e}")

try:
    print("Testing torch.load on 'best/data.pkl'...")
    state3 = torch.load("/home/ashish/video_player/video-ai-platform/distanceEstimation_d2/best/data.pkl", map_location="cpu")
    print("SUCCESS: best/data.pkl loaded.")
except Exception as e:
    print(f"FAILED to load best/data.pkl: {e}")
