import sys
import logging
from pathlib import Path

# Fix path to load app modules
sys.path.insert(0, str(Path('/home/ashish/video_player/video-ai-platform/ai-services').resolve()))

from app.models.heavy_rain_remove import HeavyRainRemovalModel
from app.models.object_detection import ObjectDetectionModel
from app.models.video_visibility import VideoVisibilityModel
from app.models.stabilize import StabilizationModel
from app.pipeline.pipeline import PipelineManager, ProcessingRequest

logging.basicConfig(level=logging.INFO)

print('Loading models...')
models = {
    'stabilization': StabilizationModel('cpu'),
    'heavy_rain_removal': HeavyRainRemovalModel('cpu'),
    'video_visibility': VideoVisibilityModel('cpu'),
    'object_detection': ObjectDetectionModel('cpu')
}

# Load them
for m in models.values():
    m.load_model()

pm = PipelineManager(models)

import cv2
import numpy as np

temp_video = '/home/ashish/video_player/video-ai-platform/test_vid.mp4'
print(f'Creating dummy video: {temp_video}')
out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
for i in range(30):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, f'Frame {i}', (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)
out.release()

print('Running pipeline (all features)...')
req = ProcessingRequest(
    video_path=temp_video,
    stabilization=True,
    heavy_rain_removal=True,
    video_visibility=True,
    object_detection=True
)
result = pm.run(req)

print('Result Status:', result.status)
print('Error:', result.error)
print('Output Path:', result.output_video)
print('Done!')
