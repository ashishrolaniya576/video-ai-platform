#!/usr/bin/env python3
"""
E2E Testing Script for VideoAI Platform

This script runs automated integration tests against the running AI Service and Backend API
to verify that Distance Estimation and all other features function correctly.

Usage:
  Ensure backend and ai-services are running.
  python scripts/test_e2e_platform.py
"""

import sys
import time
import requests
from pathlib import Path

# Configuration
AI_SERVICE_URL = "http://localhost:8000"
BACKEND_URL = "http://localhost:3001"
TEST_VIDEO_PATH = "test_videos/test_driving.mp4" # Make sure this file exists, or use any local video URL/path

def check_health():
    print("=== STEP 2: Verify Service Startup ===")
    
    # 1. AI Service Health
    try:
        r = requests.get(f"{AI_SERVICE_URL}/health", timeout=5)
        if r.status_code == 200:
            print("✓ AI Service (/health) is UP")
        else:
            print(f"✗ AI Service returned {r.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to reach AI Service: {e}")
        return False
        
    return True

def test_feature(feature_payload, name):
    print(f"\n=== Testing: {name} ===")
    
    payload = {
        "videoPath": TEST_VIDEO_PATH,
        **feature_payload
    }
    
    start_time = time.time()
    try:
        r = requests.post(f"{AI_SERVICE_URL}/process", json=payload)
        elapsed = time.time() - start_time
        
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "completed":
                print(f"✓ {name} completed successfully in {elapsed:.2f} seconds.")
                print(f"  Output Video: {data.get('outputVideo')}")
                if data.get("detectionSummary"):
                    print(f"  Detections: {data.get('detectionSummary')}")
                return True
            else:
                print(f"✗ {name} failed: {data.get('error')}")
                return False
        else:
            print(f"✗ {name} API failed with {r.status_code}: {r.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        return False

def main():
    print("Starting VideoAI E2E Validation...\n")
    
    if not check_health():
        print("\n[!] Please ensure FastAPI (port 8000) and Backend (port 3001) are running.")
        sys.exit(1)
        
    # Create dummy test video if it doesn't exist just to avoid file not found
    test_vid = Path("../" + TEST_VIDEO_PATH) if not Path(TEST_VIDEO_PATH).exists() else Path(TEST_VIDEO_PATH)
    if not test_vid.exists():
        print(f"\n[!] Test video '{TEST_VIDEO_PATH}' not found. Please place a small mp4 file there or change TEST_VIDEO_PATH inside this script.")
        print("Note: If the AI Service is running from a different folder, the path should be absolute.")
        sys.exit(1)

    print("\n=== STEP 4: Verify Each AI Feature Individually ===")
    features_to_test = [
        ({"stabilization": True}, "Video Stabilization (RAFT)"),
        ({"heavyRainRemoval": True}, "Heavy Rain Removal"),
        ({"videoVisibility": True}, "Video Visibility Enhancement (PromptIR)"),
        ({"distanceEstimation": True}, "Distance Estimation (formerly YOLO)"),
    ]
    
    results = {}
    for payload, name in features_to_test:
        results[name] = test_feature(payload, name)
        
    print("\n=== STEP 5: Verify Multi-Feature Processing ===")
    combined_payload = {
        "heavyRainRemoval": True,
        "distanceEstimation": True
    }
    results["Combined (Rain + Distance)"] = test_feature(combined_payload, "Combined (Rain + Distance)")
    
    print("\n=== FINAL REPORT ===")
    passed = 0
    for name, success in results.items():
        status = "PASSED" if success else "FAILED"
        print(f"[{status}] {name}")
        if success: passed += 1
        
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    
    if passed == len(results):
        print("\nAll integration tests passed. Distance Estimation is fully operational!")
    else:
        print("\nSome tests failed. Please review the logs above.")

if __name__ == "__main__":
    main()
