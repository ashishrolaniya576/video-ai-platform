import os
import cv2
import time
from datetime import datetime
from pathlib import Path

RTSP_URL = "rtsp://10.10.15.30:4554/eoss/1072"
OUTDIR = Path("images"); OUTDIR.mkdir(parents=True, exist_ok=True)
SAVE_EVERY_SEC = 5

# Force FFmpeg to use TCP and set timeouts (values are in microseconds for stimeout)
# NOTE: Delimiter between key/value pairs varies by OpenCV build; '|' works for most recent wheels.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;5000000|buffer_size;102400"

# Try to open with FFmpeg backend explicitly
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ OpenCV couldn't open the RTSP stream with TCP. "
          "If ffmpeg is installed, the fallback method below will handle saving frames.")
    # Exit non-zero so a wrapper can detect and run the fallback.
    raise SystemExit(2)

print("✅ Connected via OpenCV (TCP). Saving a frame every 5 seconds. Press 'q' to quit.")
last_saved = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        # Temporary read failure — wait a moment and keep trying
        time.sleep(0.5)
        continue

    now = time.time()
    if now - last_saved >= SAVE_EVERY_SEC:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = OUTDIR / f"frame_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"💾 {path}")
        last_saved = now

    # Optional preview window (comment these two lines out on headless servers)
    cv2.imshow("RTSP (TCP)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
