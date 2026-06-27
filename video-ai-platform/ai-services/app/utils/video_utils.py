"""
Video utility helpers — validation, metadata extraction, frame conversion,
temporary file management, and cleanup. No AI or pipeline logic here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable container for video properties."""

    width: int
    height: int
    fps: float
    frame_count: int
    codec: str
    duration_seconds: float
    source_path: str

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)

    def __str__(self) -> str:
        return (
            f"{self.width}x{self.height} @ {self.fps:.2f}fps | "
            f"{self.frame_count} frames | {self.duration_seconds:.1f}s | "
            f"codec={self.codec}"
        )


# ── Validation ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SUPPORTED_STREAM_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


def is_streaming_url(path: str) -> bool:
    """Return True if the path looks like a network stream URL."""
    lower = path.lower()
    return any(lower.startswith(p) for p in SUPPORTED_STREAM_PREFIXES)


def validate_video_source(source: str) -> None:
    """
    Validate that a video source is usable.

    Raises:
        ValueError: if the source is empty or a local file has unsupported extension.
        FileNotFoundError: if a local file path does not exist.
    """
    if not source or not source.strip():
        raise ValueError("Video source path or URL must not be empty.")

    if is_streaming_url(source):
        logger.debug("Video source is a network URL: %s", source)
        return

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {source}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def probe_video(source: str) -> VideoMetadata:
    """
    Open a video and extract its metadata without reading frames.

    Raises:
        RuntimeError: if OpenCV cannot open the source.
    """
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video source: '{source}'. "
                "The file may be corrupted, the format unsupported, or "
                "the stream unreachable."
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Decode the fourcc integer to a human-readable codec string
        codec = "".join([
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ]).strip("\x00") or "unknown"

        fps = fps if fps > 0 else 25.0
        frame_count = max(frame_count, 0)
        duration = frame_count / fps if fps > 0 else 0.0

        metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            codec=codec,
            duration_seconds=duration,
            source_path=source,
        )
        logger.debug("Probed video: %s", metadata)
        return metadata

    finally:
        cap.release()


# ── Frame conversion ──────────────────────────────────────────────────────────

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB frame back to OpenCV BGR."""
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def frame_to_float(frame: np.ndarray) -> np.ndarray:
    """Normalise a uint8 frame to float32 in [0, 1]."""
    return frame.astype(np.float32) / 255.0


def float_to_frame(frame: np.ndarray) -> np.ndarray:
    """Convert a float32 frame in [0, 1] back to uint8."""
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def resize_frame(
    frame: np.ndarray,
    width: int,
    height: int,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize a frame to the given dimensions."""
    return cv2.resize(frame, (width, height), interpolation=interpolation)


# ── Temporary file management ─────────────────────────────────────────────────

def make_temp_path(suffix: str = ".mp4", prefix: str = "tmp_", base_dir: Optional[Path] = None) -> Path:
    """
    Return a unique temporary file path (the file is NOT created).
    The caller is responsible for cleanup.
    """
    directory = base_dir or Path(tempfile.gettempdir())
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{prefix}{uuid.uuid4().hex}{suffix}"


def cleanup_path(path: Path) -> None:
    """Silently remove a file or directory tree if it exists."""
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not remove temporary path %s: %s", path, exc)


def cleanup_paths(*paths: Path) -> None:
    """Silently remove multiple files/directories."""
    for p in paths:
        cleanup_path(p)


# ── Memory & timing helpers ───────────────────────────────────────────────────

def log_memory_usage(label: str = "") -> None:
    """Log current process RSS memory usage (requires psutil if available)."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / 1_048_576
        logger.info("[memory] %s RSS=%.1f MB", label, rss_mb)
    except ImportError:
        pass  # psutil is optional


def format_duration(seconds: float) -> str:
    """Return a human-readable duration string like '1m 23.4s'."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes}m {remaining:.1f}s"


# ── Output path builder ───────────────────────────────────────────────────────

def build_output_path(source: str, output_dir: Path, suffix: str = "_processed") -> Path:
    """Derive a unique output file path from the source name."""
    if is_streaming_url(source):
        stem = f"stream_{uuid.uuid4().hex[:8]}"
    else:
        stem = Path(source).stem

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / f"{stem}{suffix}.mp4"

    # Avoid overwriting an existing file
    counter = 1
    while candidate.exists():
        candidate = output_dir / f"{stem}{suffix}_{counter}.mp4"
        counter += 1

    return candidate
