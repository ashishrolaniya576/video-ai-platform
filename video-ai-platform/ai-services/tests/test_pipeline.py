import pytest
import os
import cv2
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path

from app.pipeline.pipeline import PipelineManager, ProcessingRequest, ProcessingResult

# Dummy Model for testing
class DummyModel:
    def __init__(self, name):
        self.name = name
        self._loaded = True
        self._last_detection_summary = {"person": 5} if name == "DistanceEstimation" else {}

    def load_model(self):
        self._loaded = True

    def process_frame(self, frame, idx):
        return frame

    def cleanup(self):
        pass

    def compute_corrections(self, reader):
        pass


@pytest.fixture
def dummy_pipeline():
    models = {
        "stabilization": DummyModel("VideoStabilization"),
        "heavy_rain_removal": DummyModel("HeavyRainRemoval"),
        "video_visibility": DummyModel("VideoVisibility"),
        "distance_estimation": DummyModel("DistanceEstimation"),
    }
    return PipelineManager(models=models)


def create_dummy_video(path, frames=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for i in range(frames):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    return path


def test_invalid_video_path(dummy_pipeline):
    req = ProcessingRequest(video_path="nonexistent.mp4", stabilization=True)
    res = dummy_pipeline.run(req)
    assert res.status == "failed"
    assert "error" in res.error.lower()


def test_empty_video(dummy_pipeline, tmp_path):
    video_path = str(tmp_path / "empty.mp4")
    # create empty file
    with open(video_path, "wb") as f:
        pass
        
    req = ProcessingRequest(video_path=video_path, heavy_rain_removal=True)
    res = dummy_pipeline.run(req)
    assert res.status == "failed"
    assert "no decodable frames" in res.error.lower() or "validation or i/o error" in res.error.lower()


def test_zero_frame_video(dummy_pipeline, tmp_path):
    video_path = create_dummy_video(str(tmp_path / "zero.mp4"), frames=0)
    req = ProcessingRequest(video_path=video_path, video_visibility=True)
    res = dummy_pipeline.run(req)
    assert res.status == "failed"
    assert "no decodable frames" in res.error.lower()


def test_single_feature_stabilization(dummy_pipeline, tmp_path):
    video_path = create_dummy_video(str(tmp_path / "test1.mp4"), frames=5)
    req = ProcessingRequest(video_path=video_path, stabilization=True)
    res = dummy_pipeline.run(req)
    assert res.status == "completed", f"Failed: {res.error}"
    assert res.output_video is not None


def test_single_feature_distance_estimation(dummy_pipeline, tmp_path):
    video_path = create_dummy_video(str(tmp_path / "test2.mp4"), frames=5)
    req = ProcessingRequest(video_path=video_path, distance_estimation=True)
    res = dummy_pipeline.run(req)
    assert res.status == "completed"
    assert res.detection_summary is not None
    assert res.detection_summary["person"] == 5


def test_multi_feature_all(dummy_pipeline, tmp_path):
    video_path = create_dummy_video(str(tmp_path / "test3.mp4"), frames=5)
    req = ProcessingRequest(
        video_path=video_path,
        stabilization=True,
        heavy_rain_removal=True,
        video_visibility=True,
        distance_estimation=True
    )
    res = dummy_pipeline.run(req)
    assert res.status == "completed"
    assert res.detection_summary is not None
