#!/usr/bin/env python3
"""
Standalone RTSP snapshot/pipe test using FFmpeg over UDP.
Works with Windows builds of FFmpeg (no -stimeout / -rw_timeout).
"""

import argparse, os, shlex, subprocess as sp, sys, time

def grab_single_frame(url: str, out_path: str, timeout_s: int = 10) -> bool:
    """Take one snapshot from RTSP over UDP and save to out_path."""
    cmd = f'ffmpeg -rtsp_transport udp -i "{url}" -frames:v 1 "{out_path}" -y -loglevel error'
    print("Running:", cmd)
    try:
        proc = sp.run(shlex.split(cmd), check=False, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode == 0 and os.path.isfile(out_path):
            print(f"[OK] Saved snapshot -> {out_path}")
            return True
        else:
            print("[ERR] FFmpeg failed.")
            if proc.stderr:
                print(proc.stderr.strip())
            return False
    except sp.TimeoutExpired:
        print("[ERR] FFmpeg timed out.")
        return False
    except Exception as e:
        print(f"[ERR] {e}")
        return False

def pipe_test(url: str, save_dir: str, frames: int = 5, width: int | None = 1280, fps: int = 12, timeout_s: int = 15) -> bool:
    """Continuously read MJPEG frames from FFmpeg stdout and save them."""
    import numpy as np, cv2
    os.makedirs(save_dir, exist_ok=True)

    vf = []
    if width:
        vf.append(f"scale={int(width)}:-1:flags=bicubic")
    if fps:
        vf.append(f"fps={int(fps)}")
    vf_str = ",".join(vf) if vf else "fps=10"

    cmd = [
        "ffmpeg",
        "-rtsp_transport", "udp",
        "-i", url,
        "-an", "-f", "mjpeg", "-q:v", "7",
        "-vf", vf_str,
        "-"
    ]
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    try:
        proc = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE, bufsize=10**7)
    except Exception as e:
        print(f"[ERR] Could not start FFmpeg: {e}")
        return False

    sos, eos = b"\xff\xd8", b"\xff\xd9"
    buf, got = b"", 0
    t0 = time.time()
    while got < frames and (time.time() - t0) < timeout_s:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            i, j = buf.find(sos), buf.find(eos)
            if i != -1 and j != -1 and j > i:
                jpg = buf[i:j+2]
                buf = buf[j+2:]
                got += 1
                out_file = os.path.join(save_dir, f"frame_{got:03d}.jpg")
                with open(out_file, "wb") as f:
                    f.write(jpg)
                print(f"[OK] Saved {out_file}")
                if got >= frames:
                    break
            else:
                break

    try:
        if proc and proc.poll() is None:
            proc.kill()
    except Exception:
        pass

    return got > 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="RTSP URL, e.g. rtsp://10.10.15.30:4554/eoss/1072")
    ap.add_argument("--out", default="out.jpg", help="Snapshot output file (default: out.jpg)")
    ap.add_argument("--pipe-test", action="store_true", help="Also run MJPEG pipe test")
    ap.add_argument("--frames", type=int, default=5, help="How many frames to save (pipe test)")
    ap.add_argument("--width", type=int, default=1280, help="Resize width (pipe test)")
    ap.add_argument("--fps", type=int, default=12, help="FPS limit (pipe test)")
    ap.add_argument("--save-dir", default="frames_out", help="Directory to save frames")
    args = ap.parse_args()

    ok = grab_single_frame(args.url, args.out)
    if not ok:
        sys.exit(1)

    if args.pipe_test:
        ok2 = pipe_test(args.url, args.save_dir, args.frames, args.width, args.fps)
        if not ok2:
            sys.exit(2)

if __name__ == "__main__":
    main()
