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
import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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
    distance_estimation: bool = False

    @property
    def has_any_feature(self) -> bool:
        return (
            self.stabilization
            or self.heavy_rain_removal
            or self.video_visibility
            or self.distance_estimation
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
    _MODEL_ORDER = ["stabilization", "heavy_rain_removal", "video_visibility", "distance_estimation"]

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
        active_models: List[tuple] = []
        detection_summary: Optional[Dict[str, int]] = None

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

            # ── Execute Pass 1: Stabilization (if enabled) ────────────────────
            # Stabilization needs to read all frames to compute trajectories.
            stabilizer = self._models.get("stabilization")
            is_stabilizing = stabilizer is not None and any(name == "stabilization" for name, _ in active_models)

            with VideoReader(request.video_path, buffer_size=settings.frame_buffer_size) as reader:
                current_fps = reader.fps
                out_h, out_w = reader.height, reader.width
                total_frames = reader.frame_count

                # Initialize progress
                from app.api.process import GLOBAL_PROGRESS
                GLOBAL_PROGRESS[request.video_path] = 10

                if is_stabilizing:
                    log(f"Stage: {stabilizer.name} — starting Pass 1 (Optical Flow)…")
                    stage_start = time.perf_counter()
                    stabilizer.compute_corrections(reader)
                    stage_elapsed = time.perf_counter() - stage_start
                    log(f"Stage {stabilizer.name} Pass 1 complete | {format_duration(stage_elapsed)}")
                    log_memory_usage("after_stabilization_pass1")
                    
                    # Update output dimensions if stabilization is cropping
                    # Crop ratio is stored in the model, but we can compute it from the first frame in pass 2
                    # For now, we'll wait until pass 2 to get the exact cropped shape.

                # ── Execute Pass 2: Streaming Pipeline ───────────────────────────
                log("Starting Pass 2: Streaming Frame Processing…")
                
                # Output path determination
                output_path = build_output_path(request.video_path, settings.output_dir)
                log(f"Writing output video: {output_path}")

                # We need to peek at the first frame to determine final cropped resolution if stabilization is active
                reader.seek(0)
                frame_generator = reader.frames()
                
                try:
                    ret_idx, first_frame = next(frame_generator)
                except StopIteration as e:
                    raise ValueError("Video source contains no decodable frames or is corrupted.") from e
                
                # Process the first frame through the pipeline to determine final resolution
                dry_frame = first_frame.copy()
                
                # Optional high-resolution downscaling
                if max(dry_frame.shape[1], dry_frame.shape[0]) > settings.max_resolution:
                    scale = settings.max_resolution / max(dry_frame.shape[1], dry_frame.shape[0])
                    new_w = int(dry_frame.shape[1] * scale)
                    new_h = int(dry_frame.shape[0] * scale)
                    dry_frame = cv2.resize(dry_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    log(f"Downscaling input frames to {new_w}x{new_h} to preserve performance.")

                for feature_name, model in active_models:
                    dry_frame = model.process_frame(dry_frame, 0)
                
                out_h, out_w = dry_frame.shape[:2]

                with VideoWriter(
                    output_path, 
                    fps=current_fps, 
                    resolution=(out_w, out_h), 
                    original_video_path=request.video_path
                ) as writer:
                    # Write the already processed first frame
                    writer.write(dry_frame)

                    # Continue streaming the remaining frames from the generator
                    for idx, frame in frame_generator:
                        if max(frame.shape[1], frame.shape[0]) > settings.max_resolution:
                            scale = settings.max_resolution / max(frame.shape[1], frame.shape[0])
                            new_w = int(frame.shape[1] * scale)
                            new_h = int(frame.shape[0] * scale)
                            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                        for feature_name, model in active_models:
                            frame = model.process_frame(frame, idx)
                        writer.write(frame)

                        if total_frames > 0 and idx % 10 == 0:
                            pct = 10 + int((idx / total_frames) * 90)
                            GLOBAL_PROGRESS[request.video_path] = min(100, pct)

                # ── Extract Optional Results ──────────────────────────────────────
                try:
                    if request.distance_estimation:
                        dist_model = self._models.get("distance_estimation")
                        if dist_model and hasattr(dist_model, "_last_detection_summary"):
                            detection_summary = dist_model._last_detection_summary.copy()
                except Exception as e:
                    log(f"Warning: Failed to extract detection summary: {e}", level="warning")

                # ── Validate Final Output ─────────────────────────────────────────
                from app.utils.video_utils import validate_output_video
                try:
                    log("Validating browser compatibility of output MP4...")
                    validate_output_video(output_path)
                except Exception as e:
                    log(f"Warning: Output validation failed: {e}", level="warning")

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
            # 5. Clean up temporary files. Models remain loaded in GPU (Singletons)
            cleanup_paths(*temp_paths)
            try:
                from app.api.process import GLOBAL_PROGRESS
                if request.video_path in GLOBAL_PROGRESS:
                    del GLOBAL_PROGRESS[request.video_path]
            except Exception:
                pass

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _validate(request: ProcessingRequest, log) -> None:
        """Raise ValueError for invalid requests."""
        validate_video_source(request.video_path)

        if not request.has_any_feature:
            raise ValueError(
                "No processing features enabled. "
                "Set at least one of: stabilization, heavyRainRemoval, "
                "videoVisibility, distanceEstimation."
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
            "distance_estimation": request.distance_estimation,
        }

        for feature_name in self._MODEL_ORDER:
            if not feature_map.get(feature_name, False):
                continue

            model = self._models.get(feature_name)
            if model is None:
                raise RuntimeError(
                    f"Feature '{feature_name}' is enabled but no model is registered for it."
                )
            if not model._loaded:
                log(f"Lazy loading model for {feature_name}...")
                model.load_model()

            active.append((feature_name, model))

        stage_names = " → ".join(m.name for _, m in active)
        log(f"Pipeline: Input → {stage_names} → Output")
        return active
