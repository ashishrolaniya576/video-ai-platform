import asyncio
import numpy as np
import time
from app.pipeline.live_pipeline import LivePipelineManager

async def test_live_pipeline():
    manager = LivePipelineManager()
    
    request = {
        "videoPath": "live",
        "stabilization": True,
        "heavyRainRemoval": False,
        "videoVisibility": False,
        "distanceEstimation": False
    }
    
    print("Starting session...")
    manager.start_session("test_session", request)
    
    session = manager.sessions["test_session"]
    print(f"Session state: {session.current_state.name}")
    print(f"Worker alive: {session.worker_thread.is_alive()}")
    
    # Send some frames
    for i in range(10):
        print(f"Sending frame {i}")
        frame = np.zeros((480, 848, 3), dtype=np.uint8)
        
        processed = await manager.process_frame_async("test_session", frame)
        print(f"Processed returned: {processed is not None}")
        
        print(f"InQ: {session.frame_queue.qsize()} | OutQ: {session.output_queue.qsize()} | Stage: {session.current_stage}")
        await asyncio.sleep(0.1)

    print("Stopping session...")
    manager.stop_session("test_session")
    
if __name__ == "__main__":
    asyncio.run(test_live_pipeline())
