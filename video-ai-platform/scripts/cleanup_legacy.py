#!/usr/bin/env python3
"""
Final Cleanup Script

Removes obsolete YOLO files and unused training/notebook files from the 
distanceEstimation_d2 directory, retaining only the necessary inference assets.

Usage:
    python3 scripts/cleanup_legacy.py
"""

import os
import shutil
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    
    # 1. Purge legacy YOLO script
    yolo_script = root / "ai-services" / "app" / "models" / "object_detection.py"
    if yolo_script.exists():
        print(f"Removing legacy script: {yolo_script}")
        yolo_script.unlink()
        
    yolo_weights = root / "ai-services" / "models_weights" / "yolo11n.pt"
    if yolo_weights.exists():
        print(f"Removing legacy weights: {yolo_weights}")
        yolo_weights.unlink()

    # 2. Clean distanceEstimation_d2 directory (keep only best.pth and data.yaml)
    dist_dir = root / "distanceEstimation_d2"
    if dist_dir.exists() and dist_dir.is_dir():
        keep = {"best.pth", "data.yaml"}
        for item in dist_dir.iterdir():
            if item.name not in keep:
                print(f"Removing unused asset: {item}")
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                    
    print("Project cleanup complete.")

if __name__ == "__main__":
    main()
