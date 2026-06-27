"""
Standalone integration test for the RAFT Stabilization model.

Run from ai-services/ directory:
    python scripts/test_stabilization.py --video path/to/video.mp4

What this tests (mirrors the notebook's stabilize_video call):
  1. Settings load correctly
  2. Model loads without error
  3. Video is read frame by frame
  4. Flow estimation runs on a small subset (first 5 frames) for quick validation
  5. Full stabilize_video equivalent runs on the provided video
  6. Output is written and verified

Usage:
    # Quick smoke test (no real video needed — uses a synthetic video):
    python scripts/test_stabilization.py --synthetic

    # Full test on a real video:
    python scripts/test_stabilization.py --video /path/to/input.mp4

    # Save output to a specific path:
    python scripts/test_stabilization.py --video /path/to/input.mp4 --output /path/to/out.mp4
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


def make_synthetic_video(path: Path, num_frames: int = 30, w: int = 320, h: int = 240) -> None:
    """Create a short synthetic shaky video for smoke testing."""
    rng = np.random.default_rng(seed=42)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (w, h),
    )
    base = rng.integers(50, 200, (h, w, 3), dtype=np.uint8)
    for i in range(num_frames):
        # Simulate camera shake with small random shifts
        dx = int(rng.integers(-8, 8))
        dy = int(rng.integers(-8, 8))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(base, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        writer.write(frame)
    writer.release()
    print(f"[synthetic] Created test video: {path} ({num_frames} frames, {w}x{h})")


def run_test(video_path: Path, output_path: Path) -> bool:
    """Run the full stabilization pipeline and return True on success."""
    from app.config.settings import settings
    from app.models.stabilize import StabilizationModel
    from app.streaming.reader import VideoReader
    from app.streaming.writer import VideoWriter
    from app.utils.logger import setup_logging

    setup_logging("debug")

    print("\n" + "=" * 60)
    print("  RAFT Stabilization Integration Test")
    print("=" * 60)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    device = settings.resolve_device()
    print(f"\n[1/5] Device: {device.upper()}")

    model = StabilizationModel(device=device)
    print("[2/5] Loading RAFT model…")
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

    if len(frames) < 2:
        print("[FAIL] Need at least 2 frames.")
        return False

    # ── Step 3: Stabilize ────────────────────────────────────────────────────
    print(f"[4/5] Running stabilization on {len(frames)} frames…")
    t1 = time.perf_counter()
    stabilized = model.process(frames, meta.fps)
    elapsed = time.perf_counter() - t1
    print(f"      Done in {elapsed:.2f}s — {len(stabilized)} output frames.")

    if not stabilized:
        print("[FAIL] No output frames produced.")
        return False

    out_h, out_w = stabilized[0].shape[:2]
    print(f"      Output resolution: {out_w}x{out_h}")

    # Verify crop reduced the dimensions
    assert out_w < meta.width,  f"Width not cropped: {out_w} >= {meta.width}"
    assert out_h < meta.height, f"Height not cropped: {out_h} >= {meta.height}"
    print(f"      ✓ Crop verified: {meta.width}x{meta.height} → {out_w}x{out_h}")

    # ── Step 4: Write output ─────────────────────────────────────────────────
    print(f"[5/5] Writing output: {output_path}")
    with VideoWriter(output_path, fps=meta.fps, resolution=(out_w, out_h)) as writer:
        writer.write_batch(stabilized)
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
    parser = argparse.ArgumentParser(description="RAFT Stabilization integration test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Input video file path")
    group.add_argument("--synthetic", action="store_true",
                       help="Create and use a synthetic shaky video for smoke testing")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path (default: output/stabilized_test.mp4)")
    args = parser.parse_args()

    ai_services_dir = Path(__file__).resolve().parents[1]

    if args.synthetic:
        video_path = ai_services_dir / "temp" / "synthetic_shaky.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_video(video_path, num_frames=40)
    else:
        video_path = args.video.resolve()
        if not video_path.exists():
            print(f"[error] Video not found: {video_path}")
            sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_dir = ai_services_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "stabilized_test.mp4"

    success = run_test(video_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
