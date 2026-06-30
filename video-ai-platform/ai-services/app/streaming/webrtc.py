import asyncio
import json
import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from av import VideoFrame

from app.pipeline.live_pipeline import LivePipelineManager
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class WebRTCOfferSchema(BaseModel):
    sdp: str
    type: str
    videoPath: str = Field(default="live_stream", description="ID for this live session")
    stabilization: bool = False
    heavyRainRemoval: bool = False
    videoVisibility: bool = False
    distanceEstimation: bool = False


class WebRTCAnswerSchema(BaseModel):
    sdp: str
    type: str


def get_live_pipeline() -> LivePipelineManager:
    """Dependency to retrieve the LivePipelineManager singleton. We will initialize this in main.py."""
    # We will attach live_pipeline to app.state in main.py
    return None  # Replaced in the actual route


# Global registry to keep RTCPeerConnections alive
pcs: set = set()

class VideoTransformTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that transforms frames from another track using the AI Pipeline.
    """
    kind = "video"

    def __init__(self, track: MediaStreamTrack, pipeline_manager: LivePipelineManager, request: dict):
        super().__init__()
        self.track = track
        self.pipeline_manager = pipeline_manager
        self.request = request
        self.session_id = request.get("videoPath", "live")
        
        # Start the pipeline manager for this session
        self.pipeline_manager.start_session(self.session_id, self.request)

    async def recv(self) -> VideoFrame:
        import time
        start_time = time.perf_counter()
        logger.info(f"[WebRTC] START recv() - Session: {self.session_id}")
        
        try:
            # Get frame from incoming WebRTC stream
            frame = await self.track.recv()

            # Convert to BGR numpy array
            img = frame.to_ndarray(format="bgr24")

            # Process frame via LivePipelineManager
            processed_img = await self.pipeline_manager.process_frame_async(self.session_id, img)

            if processed_img is None:
                # If the pipeline failed or dropped the frame, return original
                processed_img = img

            # Convert back to VideoFrame
            new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base

            elapsed = time.perf_counter() - start_time
            logger.info(f"[WebRTC] END recv() - Session: {self.session_id} | Elapsed: {elapsed:.3f}s")
            return new_frame
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(f"[WebRTC] FAILED recv() - Session: {self.session_id} | Error: {e} | Elapsed: {elapsed:.3f}s")
            raise

    def stop(self):
        super().stop()
        self.pipeline_manager.stop_session(self.session_id)


@router.post("/rtc/offer", response_model=WebRTCAnswerSchema, summary="WebRTC SDP Offer")
async def rtc_offer(body: WebRTCOfferSchema, request: Request):
    """
    Accepts an SDP offer from the frontend, sets up the AI VideoTransformTrack,
    and returns an SDP answer.
    """
    live_pipeline: Optional[LivePipelineManager] = getattr(request.app.state, "live_pipeline", None)
    if live_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Live Pipeline is not initialized."
        )

    try:
        offer = RTCSessionDescription(sdp=body.sdp, type=body.type)
        pc = RTCPeerConnection()
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"WebRTC Connection state is {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                pcs.discard(pc)
                live_pipeline.stop_session(body.videoPath)

        @pc.on("track")
        def on_track(track):
            logger.info(f"WebRTC Track received: {track.kind}")
            if track.kind == "video":
                # Create the transform track and add it to the peer connection
                request_params = {
                    "videoPath": body.videoPath,
                    "stabilization": body.stabilization,
                    "heavy_rain_removal": body.heavyRainRemoval,
                    "video_visibility": body.videoVisibility,
                    "distance_estimation": body.distanceEstimation,
                }
                local_video = VideoTransformTrack(track, live_pipeline, request_params)
                pc.addTrack(local_video)

        # Handle the offer
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return WebRTCAnswerSchema(
            sdp=pc.localDescription.sdp,
            type=pc.localDescription.type
        )
    except Exception as e:
        import traceback
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        filename = tb[-1].filename if tb else "Unknown"
        lineno = tb[-1].lineno if tb else 0
        
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "file": filename,
            "line": lineno,
            "traceback": traceback.format_exc()
        }
        logger.error(f"Failed to process WebRTC offer: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_details
        )
