import asyncio
import queue
import uuid
import cv2
from typing import Optional

from fastapi import APIRouter, Request, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import numpy as np

from app.pipeline.live_pipeline import LivePipelineManager, SessionState
from app.streaming.stream_reader import URLStreamReader
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class URLStreamRequestSchema(BaseModel):
    url: str = Field(..., description="The RTSP, RTMP, HTTP, or HLS stream URL")
    stabilization: bool = False
    heavyRainRemoval: bool = False
    videoVisibility: bool = False
    distanceEstimation: bool = False


# Global registry for URL Stream Readers to ensure they are properly cleaned up
url_readers: dict[str, URLStreamReader] = {}


def get_live_pipeline(request: Request) -> LivePipelineManager:
    pipeline = getattr(request.app.state, "live_pipeline", None)
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Live Pipeline is not initialized."
        )
    return pipeline


@router.post("/live/start_url")
async def start_url_stream(body: URLStreamRequestSchema, request: Request):
    """
    Initializes a new Live Session for a URL stream, starts a background reader thread,
    and returns the session ID.
    """
    pipeline = get_live_pipeline(request)

    # Basic URL validation could be added here
    if not body.url.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported stream URL format."
        )

    session_id = f"url_stream_{uuid.uuid4().hex[:8]}"

    request_params = {
        "videoPath": session_id,
        "stabilization": body.stabilization,
        "heavy_rain_removal": body.heavyRainRemoval,
        "video_visibility": body.videoVisibility,
        "distance_estimation": body.distanceEstimation,
    }

    try:
        # Start AI pipeline session
        pipeline.start_session(session_id, request_params)

        # Initialize and start StreamReader
        reader = URLStreamReader(session_id, pipeline, body.url)
        reader.start()
        
        # Store reference
        url_readers[session_id] = reader
        
        return {"session_id": session_id, "status": "started"}

    except Exception as e:
        logger.error(f"Failed to start URL stream: {e}")
        pipeline.stop_session(session_id)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/live/stop_url/{session_id}")
async def stop_url_stream(session_id: str, request: Request):
    """
    Stops a URL stream and cleans up resources.
    """
    pipeline = get_live_pipeline(request)
    
    reader = url_readers.pop(session_id, None)
    if reader:
        reader.stop()
        
    pipeline.stop_session(session_id)
    return {"status": "stopped", "session_id": session_id}


async def generate_mjpeg_stream(session_id: str, pipeline: LivePipelineManager):
    """
    Generator that fetches processed frames from the session's output queue,
    encodes them as JPEG, and yields them as a multipart stream.
    """
    session = pipeline.sessions.get(session_id)
    if not session:
        return

    try:
        last_keepalive_time = 0.0
        while session.is_running:
            try:
                # Use a non-blocking get to yield control back to the event loop
                item = session.output_queue.get_nowait()
                last_keepalive_time = 0.0 # reset on actual frame
            except queue.Empty:
                # If stream is initializing, yield a loading frame every 1 second to prevent browser timeout
                if session.current_state not in [SessionState.STREAMING, SessionState.FAILED, SessionState.STOPPING, SessionState.TERMINATED]:
                    now = asyncio.get_event_loop().time()
                    if now - last_keepalive_time > 1.0:
                        last_keepalive_time = now
                        
                        # Create a black frame
                        frame = np.zeros((480, 854, 3), dtype=np.uint8)
                        text = f"Status: {session.current_state.name}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, text, (50, 240), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                        
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                await asyncio.sleep(0.01)
                continue

            if item is None:
                break
                
            if isinstance(item, tuple) and len(item) == 2:
                frame_id, frame = item
            else:
                frame = item

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            # Yield multipart boundary and JPEG bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Briefly yield to the event loop so FastAPI remains highly responsive
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        logger.info(f"MJPEG stream cancelled by client for session {session_id}")
    finally:
        # Ensure session and reader are cleaned up when the client disconnects
        logger.info(f"Closing MJPEG stream for session {session_id}")
        reader = url_readers.pop(session_id, None)
        if reader:
            reader.stop()
        pipeline.stop_session(session_id)


@router.get("/live/stream/{session_id}")
async def get_url_stream(session_id: str, request: Request):
    """
    Returns the real-time processed MJPEG stream for the given URL session.
    """
    pipeline = get_live_pipeline(request)
    
    if session_id not in pipeline.sessions:
        raise HTTPException(status_code=404, detail="Session not found or not initialized.")
        
    return StreamingResponse(
        generate_mjpeg_stream(session_id, pipeline),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
