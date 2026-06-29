import os, cv2
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;10000000|max_delay;500000"
url = "rtsp://10.10.15.30:4554/eoss/1072"
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
print("opened?", cap.isOpened())
ok, frame = cap.read()
print("read?", ok, frame.shape if ok else None)
cap.release()


print("FFMPEG enabled? ->", "FFMPEG" in cv2.getBuildInformation())
print(cv2.getBuildInformation()[:800])  # peek