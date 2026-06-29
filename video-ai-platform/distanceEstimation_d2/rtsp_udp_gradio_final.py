# rtsp_udp_polling_gradio.py
import os, time, subprocess as sp
import numpy as np
import cv2
import gradio as gr

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

STATE = {"running": False}

def snapshot_once(url: str):
    """
    Take one frame from the RTSP UDP stream using the exact command:
    ffmpeg -rtsp_transport udp -i "<url>" -frames:v 1 -f image2pipe -vcodec mjpeg -loglevel error -
    """
    try:
        proc = sp.Popen(
            [FFMPEG_BIN, "-rtsp_transport", "udp", "-i", url,
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg",
             "-loglevel", "error", "-"],
            stdout=sp.PIPE, stderr=sp.PIPE
        )
        data, _ = proc.communicate(timeout=8)
        if proc.returncode != 0 or not data:
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img[:, :, ::-1] if img is not None else None  # BGR→RGB
    except Exception:
        return None

def stream_polling(url: str, width: int, fps: int):
    """
    Continuously poll frames from snapshot_once at ~fps rate, until Stop is clicked.
    """
    STATE["running"] = True
    period = max(1.0 / max(1, int(fps)), 0.1)

    while STATE["running"]:
        img = snapshot_once(url)
        if img is None:
            canvas = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(canvas, "No frame (check URL/stream)", (30, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            yield canvas
        else:
            if width and isinstance(width, (int, float)) and int(width) > 0:
                h, w = img.shape[:2]
                new_w = int(width)
                new_h = int(h * (new_w / w))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            yield img
        time.sleep(period)

def stop_stream():
    STATE["running"] = False
    # Clear the UI with a black frame
    return np.zeros((360, 640, 3), dtype=np.uint8)

with gr.Blocks(title="RTSP UDP Polling Player") as demo:
    gr.Markdown("### RTSP UDP Polling Player\nUses your working one-frame ffmpeg command repeatedly.")

    url = gr.Textbox(label="RTSP URL", value="rtsp://10.10.15.30:4554/eoss/1072")
    width = gr.Number(value=1280, precision=0, label="Resize width (px)")
    fps = gr.Slider(1, 12, value=6, step=1, label="Polling FPS")

    with gr.Row():
        start_btn = gr.Button("▶️ Start", variant="primary")
        stop_btn = gr.Button("⏹️ Stop", variant="secondary")

    out_img = gr.Image(label="Live Stream", streaming=True)

    start_btn.click(stream_polling, inputs=[url, width, fps], outputs=out_img)
    stop_btn.click(stop_stream, outputs=out_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8011)
