"""
VideoReader — frame-by-frame streaming reader for local files and network streams.

Never loads the entire video into RAM. Supports MP4, AVI, MOV,
RTSP, RTMP, HTTP Video, and HLS sources.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger
from app.utils.video_utils import VideoMetadata, probe_video, validate_video_source

logger = get_logger(__name__)


class VideoReader:
    """
    Context-manager wrapper around cv2.VideoCapture.

    Usage:
        with VideoReader("path/to/video.mp4") as reader:
            for idx, frame in reader.frames():
                process(frame)
    """

    def __init__(self, source: str, buffer_size: int = 32) -> None:
        """
        Args:
            source:      Local file path or streaming URL.
            buffer_size: Internal OpenCV capture buffer size hint.
        """
        self._source = source
        self._buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Validate and open the video source. Raises on failure."""
        validate_video_source(self._source)

        self._metadata = probe_video(self._source)
        logger.info(
            "Opening video: %s | %s",
            self._source,
            self._metadata,
        )

        self._cap = cv2.VideoCapture(self._source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source after validation: {self._source}")

        logger.info(
            "Video opened — %dx%d @ %.2f fps | %d frames",
            self._metadata.width,
            self._metadata.height,
            self._metadata.fps,
            self._metadata.frame_count,
        )

    def release(self) -> None:
        """Release the underlying VideoCapture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("VideoReader released: %s", self._source)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("VideoReader has not been opened yet. Call open() first.")
        return self._metadata

    @property
    def fps(self) -> float:
        return self.metadata.fps

    @property
    def width(self) -> int:
        return self.metadata.width

    @property
    def height(self) -> int:
        return self.metadata.height

    @property
    def frame_count(self) -> int:
        return self.metadata.frame_count

    # ── Frame iteration ───────────────────────────────────────────────────────

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Yield (frame_index, frame_bgr) tuples one at a time.

        Never accumulates frames in memory — always reads and yields one frame
        at a time so the caller controls when to move to the next.
        """
        if self._cap is None:
            raise RuntimeError("VideoReader is not open. Call open() or use as a context manager.")

        index = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            yield index, frame
            index += 1

        logger.info("Frame iteration complete — %d frames read from %s", index, self._source)

    def read_all_frames(self) -> list[np.ndarray]:
        """
        Read all frames into a list.

        WARNING: Only use this for short videos or when the stabilizer
        requires the full sequence up-front (e.g. trajectory smoothing).
        For long videos prefer the `frames()` generator.
        """
        if self._cap is None:
            raise RuntimeError("VideoReader is not open.")

        logger.warning(
            "read_all_frames() called — loading entire video into RAM for: %s",
            self._source,
        )
        # Rewind to the beginning
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        all_frames: list[np.ndarray] = []
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            all_frames.append(frame)

        logger.info("read_all_frames() loaded %d frames", len(all_frames))
        return all_frames

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame index."""
        if self._cap is None:
            raise RuntimeError("VideoReader is not open.")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
