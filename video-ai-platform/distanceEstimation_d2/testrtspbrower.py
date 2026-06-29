# rtsp_udp_gradio_v2.py
import os, time, queue, threading, subprocess as sp
import numpy as np
import cv2
import gradio as gr

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

def _ffmpeg_exists():
    try:
        sp.run([FFMPEG_BIN, "-version"], stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True)
        return True
    except Exception:
        return False

class FFmpegUDPReader:
    """
    Continuous RTSP (UDP) reader using only widely-supported flags.
    Converts incoming stream to MJPEG on stdout; parses JPEGs and yields frames.
    """
    def __init__(self, url: str, width: int = 1280, fps: int = 12):
        self.url = url.strip()
        self.width = int(width)
        self.fps = int(fps)
        self.proc: sp.Popen | None = None
        self._stop = threading.Event()
        self._q = queue.Queue(maxsize=1)
        self._t = None

    def _cmd(self):
        vf = f"scale={self.width}:-1:flags=bicubic,fps={self.fps}"
        # Removed -stimeout / -rw_timeout (not available on some builds)
        # Kept low-latency & analysis shrinkers that are broadly supported
        return [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel", "warning",
            "-rtsp_transport", "udp",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-i", self.url,
            "-an",
            "-vf", vf,
            "-f", "mjpeg",
            "-q:v", "7",
            "-"
        ]

    def start(self):
        try:
            self.proc = sp.Popen(self._cmd(), stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**7)
        except Exception as e:
            return False, f"FFmpeg spawn failed: {e}"

        if not self._warmup_first_jpeg(8.0):
            self.stop()
            return False, "No frames during warm-up (UDP). Try Snapshot Test, then Start again."
        self._stop.clear()
        self._t = threading.Thread(target=self._read_loop, daemon=True)
        self._t.start()
        return True, "Started (UDP)."

    def _warmup_first_jpeg(self, timeout):
        if not self.proc or not self.proc.stdout:
            return False
        buf, sos, eos = b"", b"\xff\xd8", b"\xff\xd9"
        t0 = time.time()
        while time.time() - t0 < timeout and not self._stop.is_set():
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                time.sleep(0.02)
                continue
            buf += chunk
            i, j = buf.find(sos), buf.find(eos)
            if i != -1 and j != -1 and j > i:
                return True
        return False

    def _read_loop(self):
        stream = self.proc.stdout
        buf, sos, eos = b"", b"\xff\xd8", b"\xff\xd9"
        while not self._stop.is_set():
            chunk = stream.read(4096)
            if not chunk:
                break
            buf += chunk
            while True:
                i, j = buf.find(sos), buf.find(eos, 2)
                if i != -1 and j != -1 and j > i:
                    jpg = buf[i:j+2]
                    buf = buf[j+2:]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
                    if frame is not None:
                        while not self._q.empty():
                            try: self._q.get_nowait()
                            except queue.Empty: break
                        try: self._q.put_nowait(frame)
                        except queue.Full: pass
                else:
                    break
        self.stop()

    def get_latest(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        try:
            if self.proc and self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass
        self.proc = None

STATE = {"reader": None, "running": False}

def start_stream(url: str, width: int, fps: int):
    if not _ffmpeg_exists():
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, "ffmpeg not found. Set FFMPEG_BIN or add to PATH.",
                    (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        yield img[:, :, ::-1]
        return

    if STATE["reader"]:
        try: STATE["reader"].stop()
        except Exception: pass
        STATE["reader"] = None

    reader = FFmpegUDPReader(url, width=int(width), fps=int(fps))
    ok, msg = reader.start()
    if not ok:
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(img, msg[:90], (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        yield img[:, :, ::-1]
        return

    STATE["reader"] = reader
    STATE["running"] = True

    last = time.time()
    while STATE["running"]:
        frame = reader.get_latest()
        if frame is not None:
            last = time.time()
            yield frame[:, :, ::-1]
        else:
            if time.time() - last > 2.0:
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(img, "Waiting for frames (UDP)...", (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                yield img[:, :, ::-1]
            time.sleep(0.02)

def stop_stream():
    STATE["running"] = False
    if STATE["reader"]:
        try: STATE["reader"].stop()
        except Exception: pass
        STATE["reader"] = None
    return np.zeros((360, 640, 3), dtype=np.uint8)

def snapshot_test(url: str):
    """
    Runs your exact working command pattern:
    ffmpeg -rtsp_transport udp -i "<url>" -frames:v 1 out.jpg -y -loglevel error
    Then returns the image in-memory (no disk writes).
    """
    if not _ffmpeg_exists():
        return None, "ffmpeg not found (set FFMPEG_BIN or PATH)."

    try:
        proc = sp.Popen(
            [FFMPEG_BIN, "-rtsp_transport", "udp", "-i", url, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-loglevel", "error", "-"],
            stdout=sp.PIPE, stderr=sp.PIPE
        )
        data, err = proc.communicate(timeout=10)
        if proc.returncode != 0 or not data:
            return None, f"Snapshot failed. ffmpeg rc={proc.returncode}, err={err.decode(errors='ignore')}"
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Snapshot decode failed."
        return img[:, :, ::-1], "Snapshot OK"
    except sp.TimeoutExpired:
        proc.kill()
        return None, "Snapshot timed out."
    except Exception as e:
        return None, f"Snapshot error: {e}"

with gr.Blocks(title="RTSP UDP Player (No stimeout)") as demo:
    gr.Markdown("### RTSP UDP Player\nFlags `-stimeout` / `-rw_timeout` removed for compatibility.")
    url = gr.Textbox(label="RTSP URL", value="rtsp://10.10.15.30:4554/eoss/1072")
    with gr.Row():
        width = gr.Number(value=1280, precision=0, label="Resize width (px)")
        fps = gr.Slider(1, 30, value=12, step=1, label="Target FPS")

    with gr.Row():
        start_btn = gr.Button("▶️ Start", variant="primary")
        stop_btn = gr.Button("⏹️ Stop")
        snap_btn = gr.Button("📸 Snapshot Test")

    out_img = gr.Image(label="Live Stream", streaming=True)
    snap_img = gr.Image(label="Snapshot Result")
    snap_msg = gr.Textbox(label="Snapshot Status", lines=2)

    start_btn.click(start_stream, inputs=[url, width, fps], outputs=out_img)
    stop_btn.click(stop_stream, outputs=out_img)
    snap_btn.click(snapshot_test, inputs=[url], outputs=[snap_img, snap_msg])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8011)
