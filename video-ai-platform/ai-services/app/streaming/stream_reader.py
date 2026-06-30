import threading
import queue
import time
import uuid
import os
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

# Force OpenCV FFmpeg backend to use TCP for RTSP streams to bypass UDP packet drops in Docker/WSL
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from app.utils.logger import get_logger
from app.pipeline.live_pipeline import LivePipelineManager, LiveSession, SessionState, SessionEvent

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

    def _open_stream(self, session: Optional[LiveSession]) -> bool:
        if self._cap:
            self._cap.release()
            
        stream_url = self.url
        
        # Translate localhost to host.docker.internal for containerized environments
        if "localhost" in stream_url or "127.0.0.1" in stream_url:
            stream_url = stream_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
            
        if "youtube.com" in self.url or "youtu.be" in self.url:
            # Skip yt-dlp logging to avoid state machine breakage. Handled dynamically.
            try:
                import yt_dlp
                ydl_opts = {'format': 'best', 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.url, download=False)
                    stream_url = info['url']
                    logger.info(f"[URLStreamReader] Extracted YouTube direct stream URL.")
            except ImportError:
                logger.error("[URLStreamReader] yt-dlp is not installed. YouTube URLs will fail.")
                if session:
                    self.pipeline_manager.publish_event(self.session_id, SessionEvent.ERROR_OCCURRED, "yt-dlp not installed")
                return False
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                logger.error(f"[URLStreamReader] Failed to extract YouTube stream: {e}\n{tb}")
                if session:
                    self.pipeline_manager.publish_event(self.session_id, SessionEvent.ERROR_OCCURRED, f"yt-dlp extraction failed: {e}")
                return False

        if session and session.current_state != SessionState.FAILED:
            self.pipeline_manager.publish_event(self.session_id, SessionEvent.STREAM_OPEN_START, "Opening stream via FFmpeg")
            
        # Use CAP_FFMPEG to enforce the FFmpeg backend, which is best for network streams
        self._cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 3) # Small buffer for low latency
        
        is_opened = self._cap.isOpened()
        if is_opened and session and session.current_state != SessionState.FAILED:
            self.pipeline_manager.publish_event(self.session_id, SessionEvent.STREAM_OPEN_SUCCESS, "FFmpeg Stream Opened")
            
        return is_opened

    def _read_loop(self):
        try:
            retries = 0
            
            while not self._stop_event.is_set():
                session: Optional[LiveSession] = self.pipeline_manager.sessions.get(self.session_id)
                if not session:
                    logger.info(f"[URLStreamReader] Session {self.session_id} not found in manager. Stopping reader.")
                    break
                    
                if not self._cap or not self._cap.isOpened():
                    logger.info(f"[URLStreamReader] Attempting connection to {self.url} (Try {retries+1})")
                    success = self._open_stream(session)
                    if not success:
                        retries += 1
                        if retries >= self.max_retries:
                            logger.error(f"[URLStreamReader] Max retries reached for {self.url}. Terminating reader.")
                            if session and session.current_state != SessionState.FAILED:
                                self.pipeline_manager.publish_event(self.session_id, SessionEvent.ERROR_OCCURRED, "Max retries reached during stream connection")
                            break
                        time.sleep(self.reconnect_delay)
                        continue
                    else:
                        logger.info(f"[URLStreamReader] Successfully connected to {self.url}")
                        retries = 0
                        if session and session.current_state == SessionState.STREAM_CONNECTED:
                            self.pipeline_manager.publish_event(self.session_id, SessionEvent.DECODER_START)

                # Read frame
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning(f"[URLStreamReader] Connection lost or stream ended for {self.url}")
                    self._cap.release()
                    self._cap = None
                    time.sleep(self.reconnect_delay)
                    continue

                if session.current_state == SessionState.DECODER_READY:
                    self.pipeline_manager.publish_event(self.session_id, SessionEvent.FRAME_DECODED, "First frame decoded")
                    self.pipeline_manager.publish_event(self.session_id, SessionEvent.PIPELINE_START)
                    self.pipeline_manager.publish_event(self.session_id, SessionEvent.WORKER_STARTING)

                session.input_frames += 1

                if session.frame_queue.full():
                    try:
                        # Drop oldest frame to maintain realtime latency
                        session.frame_queue.get_nowait()
                        session.dropped_frames += 1
                    except queue.Empty:
                        pass
                
                try:
                    frame_id = str(uuid.uuid4())
                    session.frame_queue.put((frame_id, frame), timeout=0.1)
                    
                    # First frame successfully enqueued
                except queue.Full:
                    pass
                    
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(
                f"=== FATAL READER THREAD CRASH ===\n"
                f"Exception: {type(e).__name__}\n"
                f"Message: {e}\n"
                f"URL: {self.url}\n"
                f"Traceback:\n{tb}\n"
                f"================================="
            )
            session = self.pipeline_manager.sessions.get(self.session_id)
            if session and session.current_state != SessionState.FAILED:
                self.pipeline_manager.publish_event(self.session_id, SessionEvent.ERROR_OCCURRED, f"Reader thread crashed: {e}")
        finally:
            # Loop exited or crashed
            if self._cap:
                self._cap.release()
                self._cap = None
