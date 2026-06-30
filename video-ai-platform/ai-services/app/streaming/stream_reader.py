import threading
import queue
import time
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

from app.utils.logger import get_logger
from app.pipeline.live_pipeline import LivePipelineManager, LiveSession

logger = get_logger(__name__)


class BaseStreamReader(ABC):
    """
    Abstract base class for all live streaming input sources.
    """
    def __init__(self, session_id: str, pipeline_manager: LivePipelineManager):
        self.session_id = session_id
        self.pipeline_manager = pipeline_manager

    @abstractmethod
    def start(self) -> None:
        """Starts reading frames from the source and pushing to the pipeline."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the reader and cleans up resources."""
        pass


class URLStreamReader(BaseStreamReader):
    """
    Reads from network streams (RTSP, RTMP, HLS, MJPEG) using FFmpeg/OpenCV.
    Automatically reconnects on failure and pushes frames to the LiveSession frame queue.
    """
    def __init__(self, session_id: str, pipeline_manager: LivePipelineManager, url: str):
        super().__init__(session_id, pipeline_manager)
        self.url = url
        self._cap: Optional[cv2.VideoCapture] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.reconnect_delay = 2.0
        self.max_retries = 10

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name=f"URLReader-{self.session_id}")
        self._thread.start()
        logger.info(f"[URLStreamReader] Started for session {self.session_id}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info(f"[URLStreamReader] Stopped for session {self.session_id}")

    def _open_stream(self) -> bool:
        if self._cap:
            self._cap.release()
            
        stream_url = self.url
        if "youtube.com" in self.url or "youtu.be" in self.url:
            try:
                import yt_dlp
                ydl_opts = {'format': 'best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.url, download=False)
                    stream_url = info['url']
                    logger.info(f"[URLStreamReader] Extracted YouTube direct stream URL.")
            except ImportError:
                logger.error("[URLStreamReader] yt-dlp is not installed. YouTube URLs will fail.")
            except Exception as e:
                logger.error(f"[URLStreamReader] Failed to extract YouTube stream: {e}")
                return False

        # Use CAP_FFMPEG to enforce the FFmpeg backend, which is best for network streams
        self._cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Small buffer for low latency
        
        # Configure FFmpeg timeouts (if supported by the cv2 build, usually via env vars)
        return self._cap.isOpened()

    def _read_loop(self):
        retries = 0
        
        while not self._stop_event.is_set():
            if not self._cap or not self._cap.isOpened():
                logger.info(f"[URLStreamReader] Attempting connection to {self.url} (Try {retries+1})")
                success = self._open_stream()
                if not success:
                    retries += 1
                    if retries >= self.max_retries:
                        logger.error(f"[URLStreamReader] Max retries reached for {self.url}. Terminating reader.")
                        break
                    time.sleep(self.reconnect_delay)
                    continue
                else:
                    logger.info(f"[URLStreamReader] Successfully connected to {self.url}")
                    retries = 0

            # Read frame
            ret, frame = self._cap.read()
            if not ret:
                logger.warning(f"[URLStreamReader] Connection lost or stream ended for {self.url}")
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue

            # Push to the pipeline
            session: Optional[LiveSession] = self.pipeline_manager.sessions.get(self.session_id)
            if not session:
                # Session was removed from manager, stop reader
                logger.info(f"[URLStreamReader] Session {self.session_id} not found in manager. Stopping reader.")
                break

            session.input_frames += 1

            if session.frame_queue.full():
                try:
                    # Drop oldest frame to maintain realtime latency
                    session.frame_queue.get_nowait()
                    session.dropped_frames += 1
                except queue.Empty:
                    pass
            
            try:
                session.frame_queue.put(frame, timeout=0.1)
            except queue.Full:
                pass
                
        # Loop exited
        if self._cap:
            self._cap.release()
            self._cap = None
