"""
Pipeline Manager — orchestrates model execution for a single processing request.

Responsibilities:
  1. Validate the incoming request.
  2. Open the video source using VideoReader.
  3. Build an ordered list of enabled models (dynamic pipeline).
  4. Load all frames (buffered approach for models that need the full sequence).
  5. Execute models sequentially, passing frames from one stage to the next.
  6. Write the final frames via VideoWriter.
  7. Return a structured result.

The pipeline contains ZERO AI logic. It calls exactly:
    model.load_model()   — guaranteed already called at startup
    model.process(frames, fps)
    (cleanup is handled at shutdown, not per-request)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.config.settings import settings
from app.models.base import BaseModel
from app.streaming.reader import VideoReader
from app.streaming.writer import VideoWriter
from app.utils.logger import get_logger
from app.utils.video_utils import (
    build_output_path,
    cleanup_paths,
    format_duration,
    log_memory_usage,
    probe_video,
    validate_video_source,
)

logger = get_logger(__name__)


@dataclass
class ProcessingRequest:
    """Validated processing request."""

    video_path: str
    stabilization: bool = False
    heavy_rain_removal: bool = False
    video_visibility: bool = False
    object_detection: bool = False

    @property
    def has_any_feature(self) -> bool:
        return (
            self.stabilization
            or self.heavy_rain_removal
            or self.video_visibility
            or self.object_detection
        )


@dataclass
class ProcessingResult:
    """Result returned to the API layer."""

    status: str                                      # "completed" | "failed"
    output_video: Optional[str] = None              # Relative or absolute path to output
    execution_time: str = ""
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    detection_summary: Optional[Dict[str, int]] = None  # Per-class object counts


class PipelineManager:
    """
    Stateless pipeline orchestrator.

    One instance is created at startup and reused for every request.
    Models are injected as a dict keyed by feature name so the pipeline
    is fully decoupled from model implementations.
    """

    # Canonical order in which models are applied
    _MODEL_ORDER = ["stabilization", "heavy_rain_removal", "video_visibility", "object_detection"]

    def __init__(self, models: Dict[str, BaseModel]) -> None:
        """
        Args:
            models: Mapping of feature name → model instance.
                    All models must already be loaded (load_model() called).
        """
        self._models = models

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, request: ProcessingRequest) -> ProcessingResult:
        """
        Execute the full processing pipeline for one request.

        Never raises — all exceptions are caught and returned as a
        failed ProcessingResult so the FastAPI handler can return a
        clean HTTP 500 or 422 response.
        """
        start_time = time.perf_counter()
        logs: List[str] = []
        temp_paths: List[Path] = []

        def log(msg: str, level: str = "info") -> None:
            getattr(logger, level)(msg)
            logs.append(msg)

        try:
            # ── Validation ────────────────────────────────────────────────────
            self._validate(request, log)

            # ── Probe source ─────────────────────────────────────────────────
            metadata = probe_video(request.video_path)
            log(
                f"Video loaded: {request.video_path} | {metadata}"
            )

            # ── Build pipeline ────────────────────────────────────────────────
            active_models = self._build_pipeline(request, log)

            # ── Read frames ───────────────────────────────────────────────────
            log(f"Reading {metadata.frame_count} frames from source…")
            log_memory_usage("before_read")

            with VideoReader(request.video_path, buffer_size=settings.frame_buffer_size) as reader:
                frames: List[np.ndarray] = reader.read_all_frames()

            if not frames:
                raise RuntimeError("Video contains no readable frames.")

            log(f"Frame count: {len(frames)}")
            log_memory_usage("after_read")

            # Track the output resolution (may change if stabilization crops)
            current_fps = metadata.fps

            # ── Execute models ────────────────────────────────────────────────
            for feature_name, model in active_models:
                log(f"Stage: {model.name} — processing {len(frames)} frames…")
                stage_start = time.perf_counter()

                frames = model.process(frames, current_fps)

                stage_elapsed = time.perf_counter() - stage_start
                log(
                    f"Stage {model.name} complete — "
                    f"{len(frames)} frames out | {format_duration(stage_elapsed)}"
                )
                log_memory_usage(f"after_{feature_name}")

            # ── Collect detection summary (if object detection ran) ────────────
            detection_summary: Optional[Dict[str, int]] = None
            obj_model = self._models.get("object_detection")
            if obj_model is not None and obj_model.is_loaded:
                raw_summary = getattr(obj_model, "_last_detection_summary", None)
                if raw_summary:
                    detection_summary = dict(raw_summary)

            # ── Determine output dimensions ───────────────────────────────────
            if frames:
                out_h, out_w = frames[0].shape[:2]
            else:
                out_h, out_w = metadata.height, metadata.width

            # ── Write output ──────────────────────────────────────────────────
            output_path = build_output_path(request.video_path, settings.output_dir)
            log(f"Writing output video: {output_path}")

            with VideoWriter(output_path, fps=current_fps, resolution=(out_w, out_h)) as writer:
                writer.write_batch(frames)

            elapsed = time.perf_counter() - start_time
            exec_time_str = format_duration(elapsed)
            log(f"Processing completed in {exec_time_str} — saved to {output_path}")
            log_memory_usage("done")

            return ProcessingResult(
                status="completed",
                output_video=str(output_path),
                execution_time=exec_time_str,
                logs=logs,
                detection_summary=detection_summary,
            )

        except (ValueError, FileNotFoundError) as exc:
            elapsed = time.perf_counter() - start_time
            msg = f"Validation or I/O error: {exc}"
            log(msg, level="error")
            return ProcessingResult(
                status="failed",
                execution_time=format_duration(elapsed),
                logs=logs,
                error=msg,
            )

        except RuntimeError as exc:
            elapsed = time.perf_counter() - start_time
            msg = f"Runtime error during pipeline: {exc}"
            log(msg, level="error")

            # Check for CUDA OOM
            if "CUDA out of memory" in str(exc):
                log(
                    "CUDA out of memory — consider reducing tile size or "
                    "enabling CPU fallback via DEVICE=cpu in .env",
                    level="error",
                )

            return ProcessingResult(
                status="failed",
                execution_time=format_duration(elapsed),
                logs=logs,
                error=msg,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start_time
            msg = f"Unexpected error: {exc}"
            log(msg, level="error")
            logger.exception("Unexpected pipeline failure")
            return ProcessingResult(
                status="failed",
                execution_time=format_duration(elapsed),
                logs=logs,
                error=msg,
            )

        finally:
            cleanup_paths(*temp_paths)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate(request: ProcessingRequest, log) -> None:
        """Raise ValueError for invalid requests."""
        validate_video_source(request.video_path)

        if not request.has_any_feature:
            raise ValueError(
                "No processing features enabled. "
                "Set at least one of: stabilization, heavyRainRemoval, "
                "videoVisibility, objectDetection."
            )
        log("Request validated successfully.")

    def _build_pipeline(
        self,
        request: ProcessingRequest,
        log,
    ) -> List[tuple]:
        """
        Return an ordered list of (feature_name, model) pairs for enabled features.

        Only models for enabled features are included.
        """
        active: List[tuple] = []

        feature_map = {
            "stabilization": request.stabilization,
            "heavy_rain_removal": request.heavy_rain_removal,
            "video_visibility": request.video_visibility,
            "object_detection": request.object_detection,
        }

        for feature_name in self._MODEL_ORDER:
            if not feature_map.get(feature_name, False):
                continue

            model = self._models.get(feature_name)
            if model is None:
                raise RuntimeError(
                    f"Feature '{feature_name}' is enabled but no model is registered for it."
                )
            if not model.is_loaded:
                raise RuntimeError(
                    f"Model for '{feature_name}' is not loaded. "
                    "Ensure load_model() was called at startup."
                )

            active.append((feature_name, model))

        stage_names = " → ".join(m.name for _, m in active)
        log(f"Pipeline: Input → {stage_names} → Output")
        return active
