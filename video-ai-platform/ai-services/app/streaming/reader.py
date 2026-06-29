"""
VideoReader — frame-by-frame streaming reader for local files and network streams.

Uses a background thread and a Queue to prefetch frames, overlapping I/O with GPU inference.
Never loads the entire video into RAM.
"""

from __future__ import annotations

import threading
import queue
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger
from app.utils.video_utils import VideoMetadata, probe_video, validate_video_source

logger = get_logger(__name__)


class VideoReader:
    """
    Context-manager wrapper around cv2.VideoCapture with asynchronous prefetching.

    Usage:
        with VideoReader("path/to/video.mp4") as reader:
            for idx, frame in reader.frames():
                process(frame)
    """

    def __init__(self, source: str, buffer_size: int = 32) -> None:
        """
        Args:
            source:      Local file path or streaming URL.
            buffer_size: Maximum frames to prefetch in the queue.
        """
        self._source = source
        self._buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
        # We handle our own buffering via threading, but this sets OpenCV's internal hint.
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
        
        # Start the prefetching thread
        self._start_thread()

    def _start_thread(self):
        self._stop_event.clear()
        
        # Clear any existing queue elements
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
                
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """Background thread loop to read frames from cv2 and push to the queue."""
        if self._cap is None:
            return
            
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                # Push None to signal EOF
                try:
                    self._queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
                break
            
            try:
                # Block if the queue is full, effectively backpressuring OpenCV
                self._queue.put(frame, timeout=1.0)
            except queue.Full:
                # If it times out, continue the loop (checking stop_event)
                if not self._stop_event.is_set():
                    try:
                        self._queue.put(frame, timeout=1.0)
                    except queue.Full:
                        pass
                        
    def release(self) -> None:
        """Release the underlying VideoCapture handle and stop the thread."""
        self._stop_event.set()
        
        # Drain the queue to unblock the thread if it's waiting to put
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
                
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            
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
        Pulls from the prefetching queue to overlap I/O with inference.
        """
        if self._cap is None:
            raise RuntimeError("VideoReader is not open. Call open() or use as a context manager.")

        index = 0
        while True:
            # Block until a frame is available or EOF
            frame = self._queue.get()
            if frame is None:
                break
            
            yield index, frame
            index += 1

        logger.info("Frame iteration complete — %d frames read from %s", index, self._source)

    def seek(self, frame_index: int) -> None:
        """Seek to a specific frame index and restart the prefetcher."""
        if self._cap is None:
            raise RuntimeError("VideoReader is not open.")
            
        self._stop_event.set()
        
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
                
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self._start_thread()
