"""
Object Detection Model (YOLOv11n — Ultralytics)

Detects common objects (people, vehicles, animals, traffic signs, etc.)
on every processed frame using pretrained COCO weights.

Architecture mirrors heavy_rain_remove.py and video_visibility.py exactly:
  - Subclasses BaseModel
  - Implements load_model(), process(), cleanup()
  - Loads weights once at startup, reuses for every request
  - Uses torch.no_grad() during inference
  - Auto-downloads weights if missing
  - Detects CUDA automatically, falls back to CPU gracefully
  - Stores per-class detection counts in self._last_detection_summary
    (pipeline.py reads this after process() returns)
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from app.config.settings import settings
from app.models.base import BaseModel
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ── Colour palette for bounding boxes (BGR) ──────────────────────────────────
# One colour per COCO class index (cycles if > 80 classes)
_PALETTE = [
    (0, 114, 189),   (217, 83, 25),   (237, 177, 32),  (126, 47, 142),
    (119, 172, 48),  (77, 190, 238),  (162, 20, 47),   (76, 76, 76),
    (153, 153, 153), (255, 0, 0),     (255, 128, 0),   (191, 191, 0),
    (0, 255, 0),     (0, 0, 255),     (170, 0, 255),   (85, 85, 0),
    (85, 170, 0),    (85, 255, 0),    (170, 85, 0),    (170, 170, 0),
]


def _colour_for(class_id: int) -> tuple:
    return _PALETTE[class_id % len(_PALETTE)]


def _download_if_missing(weights_path: Path) -> None:
    """
    Ensure the YOLO weights file exists.

    ultralytics.YOLO() auto-downloads to its cache when given just a name
    ('yolo11n.pt'). We redirect the download to our managed path by passing
    the full absolute path — ultralytics will download there if the file is
    absent.
    """
    if weights_path.exists():
        logger.debug("YOLO weights already present at %s", weights_path)
        return

    logger.info("Loading YOLOv11… (weights not found — will auto-download to %s)", weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # Importing here keeps startup fast when YOLO is not needed
    try:
        from ultralytics import YOLO  # type: ignore
        # Passing the full path triggers download to that exact location
        _ = YOLO(str(weights_path))
        logger.info("YOLO weights downloaded successfully to %s", weights_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download YOLOv11 weights to '{weights_path}': {exc}"
        ) from exc


class ObjectDetectionModel(BaseModel):
    """
    YOLOv11n object detection model.

    Detects COCO objects on every frame, draws bounding boxes + labels,
    and accumulates per-class detection counts accessible via
    self._last_detection_summary after each process() call.
    """

    name = "ObjectDetection"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._model = None                          # ultralytics.YOLO instance
        self._last_detection_summary: Dict[str, int] = {}

    # ── load_model ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        if self._loaded:
            logger.debug("%s: already loaded — skipping.", self.name)
            return

        logger.info("Loading YOLOv11…")

        weights_path = Path(settings.yolo_weights_path).resolve()
        _download_if_missing(weights_path)

        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. "
                "Run: pip install ultralytics>=8.3.0"
            ) from exc

        logger.info("%s: loading weights from %s on %s", self.name, weights_path, self._device)

        try:
            self._model = YOLO(str(weights_path))
            # Move model to the target device
            self._model.to(self._device)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLOv11 weights from '{weights_path}': {exc}"
            ) from exc

        self._loaded = True
        logger.info("YOLO Model Loaded")
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

        Returns:
            annotated_frame: BGR frame with bounding boxes drawn.
        """
        self._assert_loaded()

        if frame_idx % 25 == 0 or frame_idx == 0:
            logger.info("%s: Processing Frame %d", self.name, frame_idx)

        conf_thresh: float = float(kwargs.get("conf", settings.yolo_confidence_threshold))
        iou_thresh: float = float(kwargs.get("iou", settings.yolo_iou_threshold))

        with torch.no_grad():
            results = self._model(
                frame,
                conf=conf_thresh,
                iou=iou_thresh,
                verbose=False,
                device=self._device,
            )

        # Work on a copy so we don't mutate the original
        annotated = frame.copy()

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            names = result.names  # {int: str}

            for box in boxes:
                # Coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf: float = float(box.conf[0].cpu())
                cls_id: int = int(box.cls[0].cpu())
                cls_name: str = names.get(cls_id, str(cls_id))

                colour = _colour_for(cls_id)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                # Build label: "Person 98%"
                label = f"{cls_name.capitalize()} {conf * 100:.0f}%"

                # Draw filled label background for readability
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

                # Draw label text in white
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

                # Accumulate for global summary
                self._last_detection_summary[cls_name] = self._last_detection_summary.get(cls_name, 0) + 1

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
            except Exception:  # noqa: BLE001
                pass
        self._loaded = False
        logger.info("%s: Model Closed and resources released.", self.name)


