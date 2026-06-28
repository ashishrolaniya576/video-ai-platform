"""
RAFT-based Video Stabilization model.

Converted from: ai-services/notebooks/RAFT_FINAL.ipynb
Source notebook author: Ashish (RAFT_FINAL.ipynb)

Algorithm — faithful to the notebook, refactored for production:

  Pass 1  — Read all frames (required; smoothing is a global operation)
  Pass 2  — Optical flow estimation
              For each consecutive pair (i, i+1):
                a. Resize both frames to FLOW_W × FLOW_H
                b. Run RAFT (iters=20) to get dense flow field
                c. Resize flow back to original resolution & rescale magnitudes
  Pass 3  — Trajectory computation
                d. Convert each flow field to a 2×3 affine transform
                   via RANSAC-fitted estimateAffinePartial2D on a sparse grid
                e. Accumulate transforms to form the global trajectory
  Pass 4  — Trajectory smoothing
                f. Apply uniform_filter1d (size = 2*SMOOTHING_RADIUS+1)
                   independently to all 6 trajectory parameters
  Pass 5  — Correction computation
                g. Compute per-frame correction = smoothed @ inv(cumulative)
  Pass 6  — Warp + crop
                h. warpAffine with INTER_LINEAR + BORDER_REPLICATE to original dims
                i. Crop CROP_RATIO border from all four sides

Tuning parameters (notebook defaults kept, all configurable via .env):
    SMOOTHING_RADIUS = 30
    CROP_RATIO       = 0.07
    FLOW_W, FLOW_H   = 960, 540
    RAFT_ITERS       = 20

Model weights:  raft-sintel.pth
RAFT source:    RAFT/core/  (https://github.com/princeton-vl/RAFT)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from app.config.settings import settings
from app.models.base import BaseModel
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── RAFT Args proxy ───────────────────────────────────────────────────────────
# RAFT's constructor accesses its args object via both attribute access and
# dict-style subscript. This proxy satisfies both interfaces.

class _RAFTArgs:
    """Minimal args object expected by RAFT.__init__ and its sub-modules."""

    small: bool           = False   # Use RAFT-Small (lighter, faster, less accurate)
    mixed_precision: bool = False   # FP16 mixed precision (requires Ampere+ GPU)
    alternate_corr: bool  = False   # Use alternate correlation implementation
    dropout: float        = 0.0     # Dropout (0 = disabled, used only during training)

    def __contains__(self, item: str) -> bool:       # noqa: D105
        return hasattr(self, item)

    def __getitem__(self, item: str) -> object:      # noqa: D105
        return getattr(self, item)


# ── Grid step for sparse-to-affine fitting ────────────────────────────────────
# Notebook uses step=16: sample one point every 16 px in a uniform grid over
# the flow field, then fit estimateAffinePartial2D to those correspondences.
_GRID_STEP: int = 16


class StabilizationModel(BaseModel):
    """
    RAFT optical-flow video stabilizer.

    The model holds the RAFT network and its InputPadder as attributes so they
    are initialised once during load_model() and reused across every call to
    process() without any re-import or re-allocation overhead.
    """

    name = "VideoStabilization"

    def __init__(self, device: str) -> None:
        super().__init__(device)

        # Weights are loaded here — type is the RAFT nn.Module
        self._raft: Optional[torch.nn.Module] = None

        # InputPadder cached after load so _estimate_flow() never re-imports it
        self._InputPadder: Optional[type] = None

        # ── Tuning parameters (all from notebook, all overridable via .env) ──
        self._smoothing_radius: int   = settings.raft_smoothing_radius   # 30
        self._crop_ratio: float       = settings.raft_crop_ratio          # 0.07
        self._flow_w: int             = settings.raft_flow_width          # 960
        self._flow_h: int             = settings.raft_flow_height         # 540
        self._iters: int              = settings.raft_iters               # 20
        self._corrections: List[np.ndarray] = []

    # ═══════════════════════════════════════════════════════════════════════════
    # BaseModel interface
    # ═══════════════════════════════════════════════════════════════════════════

    def load_model(self) -> None:
        """
        Load RAFT weights into GPU/CPU memory.

        Idempotent — calling twice is a no-op.

        Raises:
            FileNotFoundError: RAFT repo or weights are missing.
            ImportError:       RAFT source files cannot be imported.
            RuntimeError:      Weight loading fails (shape mismatch, etc.).
        """
        if self._loaded:
            logger.debug("%s: already loaded — skipping.", self.name)
            return

        raft_core = Path(settings.raft_repo_path) / "core"
        model_path = Path(settings.raft_model_path)

        # ── Validate paths ────────────────────────────────────────────────────
        if not raft_core.exists():
            raise FileNotFoundError(
                f"RAFT core directory not found at '{raft_core}'.\n"
                "Run the setup script:\n"
                "  bash ai-services/scripts/setup_raft.sh\n"
                f"or set RAFT_REPO_PATH in .env (current: '{settings.raft_repo_path}')."
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"RAFT weights not found at '{model_path}'.\n"
                "Run the setup script:\n"
                "  bash ai-services/scripts/setup_raft.sh\n"
                f"or set RAFT_MODEL_PATH in .env (current: '{settings.raft_model_path}')."
            )

        # ── Inject RAFT's core directory into sys.path ────────────────────────
        # The RAFT repo uses bare imports like `from raft import RAFT` and
        # `from utils.utils import InputPadder` — it assumes its core/ directory
        # is on the path, exactly as the notebook does with sys.path.insert(0, 'RAFT/core').
        raft_core_str = str(raft_core.resolve())
        if raft_core_str not in sys.path:
            sys.path.insert(0, raft_core_str)
            logger.debug("Added RAFT core to sys.path: %s", raft_core_str)

        # ── Import RAFT modules ───────────────────────────────────────────────
        try:
            from raft import RAFT  # type: ignore[import]
            from utils.utils import InputPadder  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                f"Failed to import RAFT from '{raft_core}'.\n"
                f"Ensure the repo is fully cloned and core/ contains raft.py. "
                f"Original error: {exc}"
            ) from exc

        # Cache InputPadder so _estimate_flow never re-imports it
        self._InputPadder = InputPadder

        # ── Build and load model ──────────────────────────────────────────────
        logger.info(
            "%s: loading weights — path=%s  device=%s",
            self.name, model_path, self._device,
        )
        t0 = time.perf_counter()

        torch_device = torch.device(self._device)
        model = RAFT(_RAFTArgs())

        # torch.load with weights_only=False is required for RAFT's older
        # checkpoint format which uses pickle (not the newer safetensors format).
        # The notebook does: torch.load(RAFT_MODEL_PATH, map_location=device)
        weights = torch.load(
            str(model_path),
            map_location=torch_device,
            weights_only=False,
        )

        # Strip 'module.' prefix written by nn.DataParallel during training
        state_dict = {k.replace("module.", ""): v for k, v in weights.items()}
        model.load_state_dict(state_dict)
        model = model.to(torch_device).eval()

        self._raft = model
        self._loaded = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "%s: loaded in %.2fs on %s.",
            self.name, elapsed, self._device.upper(),
        )

    def compute_corrections(self, reader) -> None:
        """
        Pass 1: Compute optical flow and global trajectory corrections.
        Reads the entire video stream frame-by-frame, ensuring O(1) RAM usage.
        """
        self._assert_loaded()

        logger.info("%s: Starting Pass 1 (Optical Flow & Trajectory)", self.name)
        t_start = time.perf_counter()

        # Restart reader to frame 0
        reader.seek(0)
        
        transforms: List[np.ndarray] = []
        prev_frame: Optional[np.ndarray] = None
        
        frame_count = 0
        for _, frame in reader.frames():
            if prev_frame is not None:
                flow = self._estimate_flow(prev_frame, frame)
                transforms.append(self._flow_to_transform(flow))
            prev_frame = frame
            frame_count += 1
            
            if frame_count % 50 == 0:
                logger.info("[flow] %d frames analyzed", frame_count)

        if frame_count < 2:
            logger.warning("%s: Need ≥2 frames for stabilization. Disabling.", self.name)
            self._corrections = []
            return

        logger.info("[trajectory] Smoothing %d transforms...", len(transforms))
        cumulative = self._get_cumulative(transforms)
        smoothed = self._smooth_trajectory(cumulative)
        self._corrections = self._get_corrections(cumulative, smoothed)

        elapsed = time.perf_counter() - t_start
        logger.info("%s: Pass 1 complete in %.2fs.", self.name, elapsed)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        **kwargs: object,
    ) -> np.ndarray:
        """
        Pass 2: Apply the pre-computed correction to the current frame.
        """
        self._assert_loaded()

        if not hasattr(self, "_corrections") or not self._corrections:
            return frame

        # Use the last available correction if frame_idx exceeds pre-computed length
        # (e.g. if the video has 10 frames, there are 10 corrections)
        idx = min(frame_idx, len(self._corrections) - 1)
        M = self._corrections[idx]

        h_orig, w_orig = frame.shape[:2]
        warped = self._warp_frame(frame, M, (w_orig, h_orig))
        return self._crop_frame(warped)

    def cleanup(self) -> None:
        """Release GPU memory and reset state."""
        if self._raft is not None:
            del self._raft
            self._raft = None
        self._InputPadder = None
        self._corrections = []
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.debug("%s: CUDA cache cleared.", self.name)
            except Exception:  # noqa: BLE001
                pass
        self._loaded = False
        logger.info("%s: resources released.", self.name)

    # ═══════════════════════════════════════════════════════════════════════════
    # Internal — optical flow
    # ═══════════════════════════════════════════════════════════════════════════

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert a BGR uint8 frame to a float32 RGB tensor [1, 3, H, W] on device.

        Exactly matches the notebook's to_tensor():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0).to(device)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)     # HWC → CHW
            .float()
            .unsqueeze(0)         # → [1, C, H, W]
            .to(self._device)
        )

    def _estimate_flow(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """
        Estimate RAFT optical flow from frame f1 to frame f2.

        Matches notebook's estimate_flow() exactly:
          1. Resize both frames to FLOW_W × FLOW_H (INTER_AREA downscale)
          2. Convert to tensors and pad to RAFT-compatible dimensions
          3. Run RAFT for self._iters iterations in test_mode
          4. Unpad, permute to HWC, move to CPU numpy
          5. Resize flow back to original resolution (INTER_LINEAR)
          6. Rescale x/y components by (w/FLOW_W, h/FLOW_H)

        Args:
            f1: Previous frame (H × W × 3 BGR uint8).
            f2: Current  frame (H × W × 3 BGR uint8).

        Returns:
            Dense flow field as float32 numpy array (H × W × 2) in original
            frame coordinates. flow[y, x, 0] = dx, flow[y, x, 1] = dy.
        """
        h_orig, w_orig = f1.shape[:2]

        # Resize to inference resolution
        s1 = cv2.resize(f1, (self._flow_w, self._flow_h), interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(f2, (self._flow_w, self._flow_h), interpolation=cv2.INTER_AREA)

        with torch.no_grad():
            t1, t2 = self._to_tensor(s1), self._to_tensor(s2)

            # InputPadder ensures dimensions are multiples of 8 (RAFT requirement)
            padder = self._InputPadder(t1.shape)
            t1, t2 = padder.pad(t1, t2)

            # RAFT returns (flow_low, flow_up); we use flow_up (full resolution)
            _, flow_up = self._raft(t1, t2, iters=self._iters, test_mode=True)

            # Unpad, remove batch dim, CHW → HWC, move to CPU
            flow = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()

        # Resize flow from inference resolution back to original frame resolution
        flow = cv2.resize(flow, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # Rescale flow magnitudes to match original pixel coordinates
        flow[..., 0] *= w_orig / self._flow_w   # dx scale
        flow[..., 1] *= h_orig / self._flow_h   # dy scale

        return flow

    # ═══════════════════════════════════════════════════════════════════════════
    # Internal — transform estimation
    # ═══════════════════════════════════════════════════════════════════════════

    def _flow_to_transform(self, flow: np.ndarray) -> np.ndarray:
        """
        Fit a 2×3 partial affine transform to a dense flow field.

        Matches notebook's flow_to_transform() exactly:
          - Sample source points on a regular grid (step=16 px)
          - Compute destination = source + flow at those points
          - Fit estimateAffinePartial2D (RANSAC, threshold=3.0 px)
          - Return identity matrix if RANSAC fails (no inliers)

        A partial affine transform encodes (tx, ty, scale, rotation) — 4 DOF.
        This is more robust than full affine for camera shake estimation.

        Args:
            flow: Dense flow field (H × W × 2) in original frame coordinates.

        Returns:
            2×3 affine matrix as float32 numpy array.
        """
        h, w = flow.shape[:2]

        # Sparse grid of source points
        ys, xs = np.mgrid[
            _GRID_STEP // 2 : h : _GRID_STEP,
            _GRID_STEP // 2 : w : _GRID_STEP,
        ]
        src = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

        # Displacement at sampled locations
        dx = flow[ys.ravel(), xs.ravel(), 0]
        dy = flow[ys.ravel(), xs.ravel(), 1]
        dst = src + np.stack([dx, dy], axis=1).astype(np.float32)

        M, _ = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        # Fallback to identity if RANSAC finds no inliers (e.g. blank frames)
        return M if M is not None else np.eye(2, 3, dtype=np.float32)

    def _estimate_all_transforms(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute N-1 inter-frame transforms for N frames.

        Logs progress every 50 pairs and on the first pair (matching notebook).
        """
        transforms: List[np.ndarray] = []
        total = len(frames) - 1

        for i in range(total):
            flow = self._estimate_flow(frames[i], frames[i + 1])
            transforms.append(self._flow_to_transform(flow))

            # Progress: log on first frame and every 50 thereafter
            if i == 0 or (i + 1) % 50 == 0:
                logger.info("[flow] %d/%d pairs done", i + 1, total)

        return transforms

    # ═══════════════════════════════════════════════════════════════════════════
    # Internal — trajectory smoothing
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_cumulative(self, transforms: List[np.ndarray]) -> List[np.ndarray]:
        """
        Build the cumulative (global) camera trajectory.

        The trajectory starts at the identity transform and is extended by
        composing each incremental inter-frame transform onto the previous
        cumulative position.

        Matches notebook's get_cumulative():
            cum = [eye(2,3)]
            for M in transforms:
                prev3 = vstack([cum[-1], [0,0,1]])
                curr3 = vstack([M, [0,0,1]])
                cum.append( (curr3 @ prev3)[:2, :] )

        Returns:
            List of N 2×3 matrices (float64), one per frame.
            cum[0] = identity (frame 0 reference pose)
            cum[i] = cumulative transform up to frame i
        """
        cum: List[np.ndarray] = [np.eye(2, 3, dtype=np.float64)]
        for M in transforms:
            prev3 = np.vstack([cum[-1],             [0.0, 0.0, 1.0]])
            curr3 = np.vstack([M.astype(np.float64), [0.0, 0.0, 1.0]])
            cum.append((curr3 @ prev3)[:2, :])
        return cum

    def _smooth_trajectory(self, cumulative: List[np.ndarray]) -> np.ndarray:
        """
        Apply a 1-D uniform (box) filter to each of the 6 trajectory parameters.

        Matches notebook's smooth_trajectory():
            arr = array(cum).reshape(-1, 6)
            size = 2*SMOOTHING_RADIUS + 1
            for i in range(6):
                out[:, i] = uniform_filter1d(arr[:, i], size=size)

        The filter window is (2 × SMOOTHING_RADIUS + 1) = 61 frames at the
        default SMOOTHING_RADIUS=30. uniform_filter1d pads with 'reflect' at
        boundaries by default, which handles the start/end of the video.

        Args:
            cumulative: List of N 2×3 trajectory matrices (float64).

        Returns:
            Smoothed trajectory as numpy array (N × 2 × 3) float64.
        """
        arr = np.array(cumulative, dtype=np.float64).reshape(-1, 6)
        filter_size = 2 * self._smoothing_radius + 1
        out = np.zeros_like(arr)
        for col in range(6):
            out[:, col] = uniform_filter1d(arr[:, col], size=filter_size)
        return out.reshape(-1, 2, 3)

    def _get_corrections(
        self,
        cumulative: List[np.ndarray],
        smoothed: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Compute per-frame correction transforms.

        Each correction is the transform that maps the actual (unstabilized)
        camera position to the smoothed (stabilized) position:
            correction = smoothed @ inv(cumulative)

        Matches notebook's get_corrections():
            for o, s in zip(cum, scum):
                o3 = vstack([o, [0,0,1]])
                s3 = vstack([s, [0,0,1]])
                corr.append( (s3 @ inv(o3))[:2, :].astype(float32) )

        Args:
            cumulative: List of N 2×3 matrices (float64).
            smoothed:   Array of N 2×3 smoothed matrices (float64).

        Returns:
            List of N 2×3 correction matrices (float32), one per frame.
        """
        corrections: List[np.ndarray] = []
        for orig, smth in zip(cumulative, smoothed):
            o3 = np.vstack([orig, [0.0, 0.0, 1.0]])
            s3 = np.vstack([smth, [0.0, 0.0, 1.0]])
            correction = (s3 @ np.linalg.inv(o3))[:2, :].astype(np.float32)
            corrections.append(correction)
        return corrections

    # ═══════════════════════════════════════════════════════════════════════════
    # Internal — warp + crop
    # ═══════════════════════════════════════════════════════════════════════════

    def _warp_frame(
        self,
        frame: np.ndarray,
        M: np.ndarray,
        output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Apply a 2×3 affine correction to a frame.

        Matches notebook's apply_transform():
            cv2.warpAffine(frame, M, output_size,
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

        The output_size is (original_width, original_height) — the warp
        preserves the full frame extent before cropping.  BORDER_REPLICATE
        fills any exposed borders with the nearest edge pixel rather than
        black, which looks more natural.

        Args:
            frame:       Source BGR frame.
            M:           2×3 affine correction matrix.
            output_size: (width, height) of the warped output.

        Returns:
            Warped frame (same dtype as input).
        """
        return cv2.warpAffine(
            frame, M, output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Crop CROP_RATIO of each side to remove warp-induced border artefacts.

        Matches notebook's crop_frame():
            h, w = frame.shape[:2]
            dy, dx = int(h * ratio), int(w * ratio)
            return frame[dy:h-dy, dx:w-dx]

        At the default CROP_RATIO=0.07, a 1920×1080 frame becomes
        approximately 1786×1005 (7% cropped from each of 4 sides).

        Args:
            frame: Warped BGR frame.

        Returns:
            Centre-cropped frame.
        """
        h, w = frame.shape[:2]
        dy = int(h * self._crop_ratio)
        dx = int(w * self._crop_ratio)
        return frame[dy : h - dy, dx : w - dx]

    def _apply_corrections(
        self,
        frames: List[np.ndarray],
        corrections: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Warp and crop every frame using its per-frame correction transform.

        The warp output size is always the original frame resolution
        (w_orig, h_orig) so that the correction operates in the full
        pixel space before the border crop.

        Matches notebook's loop:
            for i in range(len(frames)):
                warped = apply_transform(frames[i], corrections[i], (width, height))
                cropped = crop_frame(warped, CROP_RATIO)
                out.write(cropped)
                if (i+1) % 100 == 0:
                    print(f'[warp] {i+1}/{len(frames)} frames written')

        Args:
            frames:      All original BGR frames.
            corrections: List of N 2×3 correction matrices.

        Returns:
            List of stabilized + cropped BGR frames.
        """
        if not frames:
            return []

        h_orig, w_orig = frames[0].shape[:2]
        original_size: Tuple[int, int] = (w_orig, h_orig)  # (width, height) for cv2

        stabilized: List[np.ndarray] = []

        for i, (frame, M) in enumerate(zip(frames, corrections)):
            warped  = self._warp_frame(frame, M, original_size)
            cropped = self._crop_frame(warped)
            stabilized.append(cropped)

            # Progress: every 100 frames (matches notebook)
            if (i + 1) % 100 == 0:
                logger.info("[warp] %d/%d frames written", i + 1, len(frames))

        return stabilized
