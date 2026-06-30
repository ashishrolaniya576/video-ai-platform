"""
Distance Estimation Model (DistanceEstimation_d2)

Replaces YOLOv11 for the Object Detection stage.
Detects objects and estimates their distances.
"""

from __future__ import annotations

import time
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import yaml
import numpy as np
import torch

from app.config.settings import settings
from app.models.base import BaseModel
from app.utils.logger import get_logger
from app.models.distance_est_utils import estModel

logger = get_logger(__name__)


# ── Colour palette for bounding boxes (BGR) ──────────────────────────────────
_PALETTE = [
    (0, 114, 189),   (217, 83, 25),   (237, 177, 32),  (126, 47, 142),
    (119, 172, 48),  (77, 190, 238),  (162, 20, 47),   (76, 76, 76),
    (153, 153, 153), (255, 0, 0),     (255, 128, 0),   (191, 191, 0),
    (0, 255, 0),     (0, 0, 255),     (170, 0, 255),   (85, 85, 0),
    (85, 170, 0),    (85, 255, 0),    (170, 85, 0),    (170, 170, 0),
]


def _colour_for(class_id: int) -> tuple:
    return _PALETTE[class_id % len(_PALETTE)]


def read_yaml(yaml_path: str):
    with open(yaml_path, "r") as f:
        contents = yaml.safe_load(f)
    return contents["nc"], contents["names"]


def to_tensor_chw_uint(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR -> RGB -> CHW float32 [0,1]"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0


def scale_to_width_keep_aspect(image_bgr: np.ndarray, width: int) -> np.ndarray:
    if width is None:
        return image_bgr
    h, w = image_bgr.shape[:2]
    n_w = int(width)
    n_h = int((h / w) * n_w)
    return cv2.resize(image_bgr, (n_w, n_h), interpolation=cv2.INTER_AREA)


class DistanceEstimationModel(BaseModel):
    """
    Distance Estimation model wrapper.
    """

    name = "DistanceEstimation"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._model = None
        self._last_detection_summary: Dict[str, int] = {}
        self.int_cls = {}
        self.fov = 0.57
        self.image_size = 1920
        self.available = True
        self.unavailable_reason = ""

    @property
    def is_available(self) -> bool:
        return self.available

    # ── load_model ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        if self._loaded:
            logger.debug("%s: already loaded — skipping.", self.name)
            return

        logger.info("Loading Distance Estimation model…")

        weights_path = Path(settings.distance_weights_path).resolve()
        yaml_path = Path(settings.distance_yaml_path).resolve()
        
        # Phase 8 / Auto-download: If the checkpoint is missing, fetch it from Hugging Face
        if not weights_path.exists():
            logger.info("Distance Estimation checkpoint not found locally. Downloading from Hugging Face...")
            try:
                import shutil
                from huggingface_hub import hf_hub_download
                
                cached_path = hf_hub_download(
                    repo_id="ashish576/video-ai-platform-models",
                    filename="distance_estimation/best.pth"
                )
                
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_path, weights_path)
                logger.info(f"Successfully downloaded and saved Distance Estimation checkpoint to {weights_path}")
                
            except ImportError:
                self.available = False
                self.unavailable_reason = "huggingface_hub package is required to download the model but is not installed."
                logger.warning(self.unavailable_reason)
                return
            except Exception as exc:
                self.available = False
                self.unavailable_reason = (
                    "Distance Estimation checkpoint could not be loaded.\n"
                    f"Expected: {weights_path}\n"
                    f"Actual: The local file is missing and the Hugging Face download failed.\n"
                    f"Error: {exc}\n"
                    "Suggested action: Check your internet connection or download manually."
                )
                logger.warning(self.unavailable_reason)
                return

        # Check readability
        if not os.access(weights_path, os.R_OK):
            self.available = False
            self.unavailable_reason = f"Distance Estimation checkpoint could not be loaded.\nExpected: {weights_path}\nActual: Permission denied (unreadable file)."
            logger.warning(self.unavailable_reason)
            return
            
        if not yaml_path.exists():
            self.available = False
            self.unavailable_reason = f"Distance Estimation YAML configuration not found at {yaml_path}"
            logger.warning(self.unavailable_reason)
            return

        try:
            num_classes, class_names = read_yaml(str(yaml_path))
            cls_int = {c: i + 1 for i, c in enumerate(class_names)}
            cls_int['background'] = 0
            self.int_cls = {v: k for k, v in cls_int.items()}

            self._model = estModel(num_classes=num_classes + 1)
        except Exception as exc:
            self.available = False
            self.unavailable_reason = f"Distance Estimation initialization failed.\nActual: Could not parse YAML or construct model architecture: {exc}"
            logger.warning(self.unavailable_reason)
            return

        # Phase 7: Verify the checkpoint (Loadability)
        try:
            state = torch.load(str(weights_path), map_location=self._device)
        except Exception as exc:
            self.available = False
            self.unavailable_reason = (
                "Distance Estimation checkpoint could not be loaded.\n"
                f"Expected: {weights_path}\n"
                f"Actual: torch.load() failed. The file may be corrupted or incompatible: {exc}"
            )
            logger.warning(self.unavailable_reason)
            return

        # Phase 7: Verify the checkpoint (Architecture Compatibility)
        try:
            self._model.load_state_dict(state)
            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:
            self.available = False
            self.unavailable_reason = (
                "Distance Estimation checkpoint could not be loaded.\n"
                f"Expected: {weights_path}\n"
                f"Actual: Architecture mismatch. The state_dict keys do not match the estModel initialization: {exc}"
            )
            logger.warning(self.unavailable_reason)
            return

        self._loaded = True
        self.available = True
        logger.info("Distance Estimation Model Loaded")
        logger.info("%s: Model loaded successfully on %s.", self.name, self._device.upper())

    # ── process ───────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        **kwargs: object,
    ) -> np.ndarray:
        """
        Run inference on a single BGR frame and draw annotations.
        """
        self._assert_loaded()

        if frame_idx == 0:
            self._last_detection_summary.clear()
            logger.info("%s: Processing Frame %d", self.name, frame_idx)
        elif frame_idx % 25 == 0:
            logger.info("%s: Processing Frame %d", self.name, frame_idx)

        start_time = time.perf_counter()
        
        conf_thresh: float = float(kwargs.get("conf", settings.distance_confidence_threshold))

        # We must resize to training width (1920) before inference, then scale boxes back, 
        # OR just resize, infer, draw, and resize back to original.
        # But wait, resizing back to original is cleaner if the pipeline expects original size.
        original_h, original_w = frame.shape[:2]
        
        # We process using 1920 width
        scaled_frame = scale_to_width_keep_aspect(frame, self.image_size)
        scaled_h, scaled_w = scaled_frame.shape[:2]
        
        scale_x = original_w / scaled_w
        scale_y = original_h / scaled_h
        
        image_tensor = to_tensor_chw_uint(scaled_frame)
        inputs = {
            "image": image_tensor.to(self._device, non_blocking=True), 
            "fov": torch.tensor(float(self.fov), dtype=torch.float32, device=self._device)
        }

        # Prepare phase timing
        prepare_time = time.perf_counter() - start_time
        inference_start = time.perf_counter()

        with torch.inference_mode(), torch.autocast(device_type=self._device, enabled=self._device=="cuda"):
            output = self._model([inputs])[0]

        # Inference phase timing
        inference_time = time.perf_counter() - inference_start
        postprocess_start = time.perf_counter()

        boxes = output.get("boxes", torch.tensor([], device=self._device))
        labels = output.get("labels", torch.tensor([], device=self._device))
        scores = output.get("scores", torch.tensor([], device=self._device))
        distances = output.get("distances", torch.tensor([], device=self._device))

        if len(scores) > 0 and frame_idx % 25 == 0:
            logger.info(f"[DistanceEstimation] Frame {frame_idx} | Raw Scores Max: {scores.max().item():.3f} | Raw Detections: {len(scores)}")

        # Handle empty detections
        if len(scores) == 0:
            postprocess_time = time.perf_counter() - postprocess_start
            total_latency = time.perf_counter() - start_time
            fps = 1.0 / total_latency if total_latency > 0 else 0.0
            
            logger.info(
                f"[DistanceEstimation] Frame {frame_idx} | Detected Objects = 0 | "
                f"Prepare = {prepare_time*1000:.1f}ms | Inference = {inference_time*1000:.1f}ms | "
                f"Postprocess = {postprocess_time*1000:.1f}ms | Total = {total_latency*1000:.1f}ms | FPS = {fps:.1f}"
            )
            return frame

        # Threshold filter on GPU
        mask = scores > conf_thresh
        boxes_f = boxes[mask].cpu().numpy()
        labels_f = labels[mask].cpu().numpy()
        scores_f = scores[mask].cpu().numpy()
        distances_f = distances[mask].cpu().numpy()

        # Work directly on original frame for zero-copy memory pipeline
        annotated = frame

        for box, lbl, scr, dist in zip(boxes_f, labels_f, scores_f, distances_f):
            
            # Scale coordinates back to original frame size
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            cls_id = int(lbl)
            cls_name = self.int_cls.get(cls_id, str(cls_id))
            colour = _colour_for(cls_id)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            
            # Build label: "Person 0.85 25.43"
            label = f"{cls_name.capitalize()} {scr:.2f} {dist:.2f}m"
            
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            label_y1 = max(y1 - text_h - baseline - 4, 0)
            cv2.rectangle(
                annotated,
                (x1, label_y1),
                (x1 + text_w + 4, label_y1 + text_h + baseline + 4),
                colour,
                cv2.FILLED,
            )
            
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y1 + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            
            self._last_detection_summary[cls_name] = self._last_detection_summary.get(cls_name, 0) + 1
            
        postprocess_time = time.perf_counter() - postprocess_start
        total_latency = time.perf_counter() - start_time
        fps = 1.0 / total_latency if total_latency > 0 else 0.0
        
        logger.info(
            f"[DistanceEstimation] Frame {frame_idx} | Detected Objects = {len(boxes_f)} | "
            f"Prepare = {prepare_time*1000:.1f}ms | Inference = {inference_time*1000:.1f}ms | "
            f"Postprocess = {postprocess_time*1000:.1f}ms | Total = {total_latency*1000:.1f}ms | FPS = {fps:.1f}"
        )
        
        for cls_name, count in self._last_detection_summary.items():
            logger.info(f"  -> {cls_name}: {count}")

        return annotated

    # ── cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Release GPU memory and reset state."""
        if self._model is not None:
            del self._model
            self._model = None
        self._last_detection_summary = {}
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.debug("%s: CUDA cache cleared.", self.name)
            except Exception:
                pass
        self._loaded = False
        logger.info("%s: Model Closed and resources released.", self.name)
