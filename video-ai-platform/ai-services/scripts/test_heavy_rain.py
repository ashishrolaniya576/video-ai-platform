"""
Standalone integration test for the Heavy Rain Removal model.

Run from ai-services/ directory:
    python scripts/test_heavy_rain.py --video path/to/video.mp4

What this tests:
  1. Auto-download logic for repo and checkpoint
  2. Model loads without error (all Python 3 patches applied automatically)
  3. Flow works on a small subset of frames
  4. Video is processed and output correctly
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add ai-services/ to path so app.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def make_synthetic_rain_video(path: Path, num_frames: int = 10, w: int = 256, h: int = 256) -> None:
    """Create a short synthetic video with rain for testing."""
    rng = np.random.default_rng(seed=42)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (w, h),
    )
    # create a simple gradient background
    base = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        base[y, :, :] = [100 + int(y/h*50), 100, 100]
    
    for i in range(num_frames):
        frame = base.copy()
        # Add random rain streaks
        for _ in range(50):
            x1 = int(rng.integers(0, w-10))
            y1 = int(rng.integers(0, h-20))
            x2 = x1 + int(rng.integers(-2, 2))
            y2 = y1 + int(rng.integers(10, 20))
            cv2.line(frame, (x1, y1), (x2, y2), (200, 200, 220), 1)
            
        writer.write(frame)
    writer.release()
    print(f"[synthetic] Created test video: {path} ({num_frames} frames, {w}x{h})")


def run_test(video_path: Path, output_path: Path) -> bool:
    """Run the full heavy rain removal pipeline and return True on success."""
    from app.config.settings import settings
    from app.models.heavy_rain_remove import HeavyRainRemovalModel
    from app.streaming.reader import VideoReader
    from app.streaming.writer import VideoWriter
    from app.utils.logger import setup_logging

    setup_logging("debug")

    print("\n" + "=" * 60)
    print("  Heavy Rain Removal Integration Test")
    print("=" * 60)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    device = settings.resolve_device()
    print(f"\n[1/5] Device: {device.upper()}")

    model = HeavyRainRemovalModel(device=device)
    print("[2/5] Loading Heavy Rain Removal model (auto-downloading if missing)…")
    t0 = time.perf_counter()
    model.load_model()
    print(f"      Loaded in {time.perf_counter()-t0:.2f}s")

    # ── Step 2: Read video ────────────────────────────────────────────────────
    print(f"[3/5] Reading video: {video_path}")
    with VideoReader(str(video_path), buffer_size=32) as reader:
        meta = reader.metadata
        print(f"      {meta}")
        frames = reader.read_all_frames()
    print(f"      Read {len(frames)} frames.")

    if len(frames) < 1:
        print("[FAIL] Need at least 1 frame.")
        return False

    # ── Step 3: Process ────────────────────────────────────────────────────
    print(f"[4/5] Running Heavy Rain Removal on {len(frames)} frames…")
    t1 = time.perf_counter()
    processed = model.process(frames, meta.fps)
    elapsed = time.perf_counter() - t1
    print(f"      Done in {elapsed:.2f}s — {len(processed)} output frames.")

    if not processed:
        print("[FAIL] No output frames produced.")
        return False

    out_h, out_w = processed[0].shape[:2]
    print(f"      Output resolution: {out_w}x{out_h}")
    assert out_w == meta.width,  f"Width changed: {out_w} != {meta.width}"
    assert out_h == meta.height, f"Height changed: {out_h} != {meta.height}"

    # ── Step 4: Write output ─────────────────────────────────────────────────
    print(f"[5/5] Writing output: {output_path}")
    with VideoWriter(output_path, fps=meta.fps, resolution=(out_w, out_h)) as writer:
        writer.write_batch(processed)
    print(f"      Wrote {writer.frames_written} frames.")

    assert output_path.exists(), "Output file was not created!"
    size_kb = output_path.stat().st_size / 1024
    print(f"      ✓ Output file: {output_path} ({size_kb:.1f} KB)")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    model.cleanup()

    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED ✓")
    print("=" * 60 + "\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Heavy Rain Removal integration test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Input video file path")
    group.add_argument("--synthetic", action="store_true",
                       help="Create and use a synthetic rain video for smoke testing")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path (default: output/heavy_rain_test.mp4)")
    args = parser.parse_args()

    ai_services_dir = Path(__file__).resolve().parents[1]

    if args.synthetic:
        video_path = ai_services_dir / "temp" / "synthetic_rain.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_rain_video(video_path, num_frames=10)
    else:
        video_path = args.video.resolve()
        if not video_path.exists():
            print(f"[error] Video not found: {video_path}")
            sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_dir = ai_services_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "heavy_rain_test.mp4"

    success = run_test(video_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
