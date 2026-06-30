"""
Standalone integration test for the Video Visibility (PromptIR) model.

Run from ai-services/ directory:
    python scripts/test_video_visibility.py --synthetic
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


def make_synthetic_foggy_video(path: Path, num_frames: int = 144, w: int = 848, h: int = 480) -> None:
    """Create a short synthetic video with low contrast/fog for testing."""
    rng = np.random.default_rng(seed=42)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25.0,
        (w, h),
    )
    # create a simple scene
    base = np.zeros((h, w, 3), dtype=np.uint8)
    # Draw some shapes
    cv2.rectangle(base, (50, 50), (100, 100), (0, 0, 255), -1)
    cv2.circle(base, (150, 150), 40, (0, 255, 0), -1)
    
    for _ in range(num_frames):
        frame = base.copy()
        
        # Add "fog" (blend with white/gray and reduce contrast)
        fog = np.full_like(frame, 200)
        frame = cv2.addWeighted(frame, 0.3, fog, 0.7, 0)
        
        # Add some noise
        noise = rng.normal(0, 10, (h, w, 3)).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        writer.write(frame)
    writer.release()
    print(f"[synthetic] Created test video: {path} ({num_frames} frames, {w}x{h})")


def run_test(video_path: Path, output_path: Path) -> bool:
    """Run the full video visibility pipeline and return True on success."""
    from app.config.settings import settings
    from app.models.video_visibility import VideoVisibilityModel
    from app.streaming.reader import VideoReader
    from app.streaming.writer import VideoWriter
    from app.utils.logger import setup_logging

    setup_logging("debug")

    print("\n" + "=" * 60)
    print("  Video Visibility (PromptIR) Integration Test")
    print("=" * 60)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    device = settings.resolve_device()
    print(f"\n[1/5] Device: {device.upper()}")

    model = VideoVisibilityModel(device=device)
    print("[2/5] Loading Video Visibility model (auto-downloading if missing)…")
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
    print(f"[4/5] Running Video Visibility on {len(frames)} frames…")
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
    parser = argparse.ArgumentParser(description="Video Visibility integration test")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=Path, help="Input video file path")
    group.add_argument("--synthetic", action="store_true",
                       help="Create and use a synthetic foggy video for smoke testing")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path (default: output/visibility_test.mp4)")
    args = parser.parse_args()

    ai_services_dir = Path(__file__).resolve().parents[1]

    if args.synthetic:
        video_path = ai_services_dir / "temp" / "synthetic_fog.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_foggy_video(video_path, num_frames=144)
    else:
        video_path = args.video.resolve()
        if not video_path.exists():
            print(f"[error] Video not found: {video_path}")
            sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_dir = ai_services_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "visibility_test.mp4"

    success = run_test(video_path, output_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
