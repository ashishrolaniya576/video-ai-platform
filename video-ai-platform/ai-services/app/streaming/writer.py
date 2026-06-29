"""
VideoWriter — writes processed frames to an MP4 output file.

Streams raw frames directly to an FFmpeg subprocess via stdin.
Uses a background thread and a Queue to decouple GPU inference from disk I/O.
This strictly avoids double-encoding and eliminates pipe backpressure stalls.
"""

from __future__ import annotations

import subprocess
import shutil
import threading
import queue
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoWriter:
    """
    Context-manager wrapper around FFmpeg raw streaming.

    Usage:
        with VideoWriter("output.mp4", fps=30.0, resolution=(1280, 720)) as writer:
            for frame in processed_frames:
                writer.write(frame)
    """

    def __init__(
        self,
        output_path: Path,
        fps: float,
        resolution: Tuple[int, int],
        codec: str = "libx264",
        original_video_path: Optional[str] = None,
        buffer_size: int = 64,
    ) -> None:
        """
        Args:
            output_path:  Destination MP4 file. Parent directories are created.
            fps:          Frames per second of the output video.
            resolution:   (width, height) tuple.
            codec:        FFmpeg output codec string. Defaults to 'libx264'.
            original_video_path: Path to the original video to extract audio from.
            buffer_size:  Size of the async frame writing queue.
        """
        self._final_output_path = Path(output_path)
        self._fps = fps
        self._resolution = resolution  # (width, height)
        self._codec = codec
        self._original_video_path = original_video_path
        
        self._proc: Optional[subprocess.Popen] = None
        self._frames_written: int = 0
        
        # Async writing components
        self._queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # If an exception occurred in the pipeline, we still release safely
        self.release()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the output file for writing via FFmpeg and start writer thread."""
        self._final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found in PATH! Required for fast video streaming.")

        width, height = self._resolution

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "bgr24",
            "-r", str(self._fps),
            "-i", "-"  # Read from stdin
        ]
        
        if self._original_video_path:
            cmd.extend(["-i", str(self._original_video_path)])
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k", "-shortest"])

        cmd.extend([
            "-c:v", self._codec,
            "-preset", "veryfast",  # Optimized for speed
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self._final_output_path)
        ])
        
        logger.info(
            "VideoWriter opened: %s | %.2f fps | %dx%d | codec=%s",
            self._final_output_path, self._fps, width, height, self._codec
        )
        
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Start background consumer thread
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self) -> None:
        """Background thread that consumes frames and pipes to FFmpeg."""
        if self._proc is None or self._proc.stdin is None:
            return
            
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    frame = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                    
                if frame is None:
                    # None is a sentinel indicating end of stream
                    break
                    
                self._proc.stdin.write(frame)
                self._frames_written += 1
                self._queue.task_done()
                
        except Exception as e:
            self._error = e
            logger.error("FFmpeg writer thread failed: %s", e)

    def release(self) -> None:
        """Flush the queue, stop the thread, and release the FFmpeg process."""
        self._stop_event.set()
        
        if self._thread is not None:
            # Inject sentinel to unblock the get() immediately if queue is empty
            try:
                self._queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            
            self._thread.join(timeout=5.0)
            self._thread = None
            
        if self._proc is not None:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
            self._proc.wait()
            self._proc = None
            logger.info(
                "FFmpeg VideoWriter released: %s | %d frames written",
                self._final_output_path,
                self._frames_written,
            )

    # ── Writing ───────────────────────────────────────────────────────────────

    def write(self, frame: np.ndarray) -> None:
        """Enqueue a single BGR frame for async writing."""
        if self._proc is None or self._thread is None:
            raise RuntimeError("VideoWriter is not open. Call open() first.")
            
        if self._error is not None:
            raise RuntimeError(f"VideoWriter background thread crashed: {self._error}")

        # Ensure the frame matches the declared resolution
        h, w = frame.shape[:2]
        if (w, h) != self._resolution:
            frame = cv2.resize(frame, self._resolution, interpolation=cv2.INTER_LINEAR)

        # Push raw bytes to the queue (blocks if queue is full, effectively backpressuring)
        self._queue.put(frame.tobytes())

    def write_batch(self, frames: list[np.ndarray]) -> None:
        """Enqueue a list of frames in order."""
        for frame in frames:
            self.write(frame)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def output_path(self) -> Path:
        return self._final_output_path
