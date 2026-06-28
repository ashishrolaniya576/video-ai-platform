"""
VideoWriter — writes processed frames to an MP4 output file.

Preserves FPS and resolution. Releases resources correctly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import subprocess
import shutil
import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger(__name__)

# Preferred codec order — tried in sequence until one works
_CODEC_CANDIDATES = ["mp4v", "avc1", "XVID"]


class VideoWriter:
    """
    Context-manager wrapper around cv2.VideoWriter.

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
        codec: str = "mp4v",
    ) -> None:
        """
        Args:
            output_path:  Destination MP4 file. Parent directories are created.
            fps:          Frames per second of the output video.
            resolution:   (width, height) tuple.
            codec:        FourCC codec string. Defaults to 'mp4v'.
        """
        self._final_output_path = Path(output_path)
        self._output_path = self._final_output_path.with_name(f".tmp_{self._final_output_path.name}")
        self._fps = fps
        self._resolution = resolution  # (width, height)
        self._codec = codec
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_written: int = 0

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "VideoWriter":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the output file for writing."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = self._try_open_with_codec(self._codec)
        if writer is None:
            # Try fallback codecs
            for fallback in _CODEC_CANDIDATES:
                if fallback == self._codec:
                    continue
                writer = self._try_open_with_codec(fallback)
                if writer is not None:
                    logger.warning(
                        "Primary codec '%s' failed; using fallback '%s'",
                        self._codec,
                        fallback,
                    )
                    self._codec = fallback
                    break

        if writer is None:
            raise RuntimeError(
                f"Could not open VideoWriter for {self._output_path}. "
                "No working codec found."
            )

        self._writer = writer
        logger.info(
            "VideoWriter opened: %s | %.2f fps | %dx%d | codec=%s",
            self._output_path,
            self._fps,
            self._resolution[0],
            self._resolution[1],
            self._codec,
        )

    def _try_open_with_codec(self, codec: str) -> Optional[cv2.VideoWriter]:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(
            str(self._output_path),
            fourcc,
            self._fps,
            self._resolution,  # (width, height)
        )
        if writer.isOpened():
            return writer
        writer.release()
        return None

    def release(self) -> None:
        """Flush and release the underlying VideoWriter and encode with FFmpeg."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info(
                "OpenCV VideoWriter released: %s | %d frames written",
                self._output_path,
                self._frames_written,
            )
            
            if not shutil.which("ffmpeg"):
                logger.error("FFmpeg not found in PATH! Please install ffmpeg (e.g. sudo apt install ffmpeg).")
                logger.warning("Falling back to raw OpenCV video. This may not be playable in browsers.")
                shutil.move(str(self._output_path), str(self._final_output_path))
                return

            logger.info("Encoding final video with FFmpeg for browser compatibility...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(self._output_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(self._final_output_path)
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                logger.info("FFmpeg encoding completed successfully: %s", self._final_output_path)
                if self._output_path.exists():
                    self._output_path.unlink()
            except subprocess.CalledProcessError as e:
                logger.error("FFmpeg encoding failed:\n%s", e.stderr.decode("utf-8", errors="ignore"))
                logger.warning("Falling back to raw OpenCV video.")
                shutil.move(str(self._output_path), str(self._final_output_path))

    # ── Writing ───────────────────────────────────────────────────────────────

    def write(self, frame: np.ndarray) -> None:
        """
        Write a single BGR frame.

        Args:
            frame: HxWx3 uint8 numpy array in BGR colour space.

        Raises:
            RuntimeError: if the writer was not opened.
        """
        if self._writer is None:
            raise RuntimeError("VideoWriter is not open. Call open() first.")

        # Ensure the frame matches the declared resolution
        h, w = frame.shape[:2]
        if (w, h) != self._resolution:
            frame = cv2.resize(frame, self._resolution, interpolation=cv2.INTER_LINEAR)

        # Ensure uint8 dtype
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        self._writer.write(frame)
        self._frames_written += 1

    def write_batch(self, frames: list[np.ndarray]) -> None:
        """Write a list of frames in order."""
        for frame in frames:
            self.write(frame)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def output_path(self) -> Path:
        return self._final_output_path
