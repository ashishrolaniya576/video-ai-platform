"""
Video Visibility Enhancement Model (PromptIR)

Converted from ai-services/notebooks/video_visibility.ipynb.
Automatically downloads repository and checkpoint if missing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import pytorch_lightning as pl

from app.config.settings import settings
from app.models.base import BaseModel
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _download_if_missing() -> None:
    """Download the PromptIR repository and checkpoint if missing."""
    repo_path = Path(settings.promptir_repo_path)
    if not repo_path.exists():
        logger.info(f"Cloning PromptIR repo to {repo_path}...")
        import subprocess
        try:
            subprocess.run(["git", "clone", "https://github.com/va1shn9v/PromptIR.git", str(repo_path)], check=True)
            logger.info("Successfully cloned PromptIR repo.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone PromptIR repo: {e}")

    ckpt_path = Path(settings.promptir_checkpoint)
    if not ckpt_path.exists():
        logger.info(f"Downloading checkpoint to {ckpt_path}...")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        try:
            urllib.request.urlretrieve(settings.promptir_checkpoint_url, str(ckpt_path))
            logger.info("Successfully downloaded checkpoint.")
        except Exception as e:
            raise RuntimeError(f"Failed to download PromptIR checkpoint: {e}")


def pad_to_multiple(tensor: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad tensor dimensions to a multiple of a given number."""
    _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (pad_h, pad_w)


def gaussian_window(size: int) -> torch.Tensor:
    """Generate a 2D Gaussian window for blending tiles."""
    sigma = size / 6.0
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g2d = g.unsqueeze(0) * g.unsqueeze(1)
    return g2d / g2d.max()


class PromptIRWrapper(pl.LightningModule):
    """Wrapper class mimicking the PromptIRModel from the notebook."""
    def __init__(self, net_class):
        super().__init__()
        self.net = net_class(decoder=True)
        
    def forward(self, x):
        return self.net(x)


class VideoVisibilityModel(BaseModel):
    """Video Visibility Enhancement (PromptIR) model."""

    name = "VideoVisibility"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._model: Optional[PromptIRWrapper] = None
        
        self.tile_size = settings.promptir_tile_size
        self.tile_overlap = settings.promptir_tile_overlap
        self.alpha = settings.promptir_contrast_alpha
        self.beta = settings.promptir_contrast_beta
        self.clahe_clip = settings.promptir_clahe_clip

    def load_model(self) -> None:
        if self._loaded:
            logger.debug("%s: already loaded — skipping.", self.name)
            return

        _download_if_missing()

        repo_path = Path(settings.promptir_repo_path).resolve()
        ckpt_path = Path(settings.promptir_checkpoint).resolve()

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            from net.model import PromptIR  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"Failed to import PromptIR from '{repo_path}'. "
                f"Original error: {exc}"
            ) from exc

        logger.info("%s: loading network on %s", self.name, self._device)
        torch_device = torch.device(self._device)

        # Initialize network
        model = PromptIRWrapper(PromptIR)
        
        # Load weights
        logger.info("%s: loading weights from %s", self.name, ckpt_path)
        try:
            checkpoint = torch.load(str(ckpt_path), map_location=torch_device)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint for {self.name}: {e}")

        model.to(torch_device)
        model.eval()

        self._model = model
        self._loaded = True
        logger.info("%s: Checkpoint Loaded. Model loaded successfully.", self.name)

    def process(
        self,
        frames: List[np.ndarray],
        fps: float,
        **kwargs: object,
    ) -> List[np.ndarray]:
        self._assert_loaded()

        if not frames:
            return []

        logger.info("%s: Processing Started on %d frames…", self.name, len(frames))
        t_start = time.perf_counter()

        processed_frames = []
        for idx, frame in enumerate(frames):
            logger.debug("%s: Current Frame %d", self.name, idx)
            processed_frame = self._process_single_frame(frame)
            processed_frames.append(processed_frame)
            logger.debug("%s: Frame Completed %d", self.name, idx)

            if (idx + 1) % 50 == 0:
                logger.info("%s: processed %d/%d frames", self.name, idx + 1, len(frames))

        elapsed = time.perf_counter() - t_start
        h_out, w_out = processed_frames[0].shape[:2]
        logger.info(
            "%s: Execution Time — %d frames → %dx%d in %.2fs.",
            self.name, len(processed_frames), w_out, h_out, elapsed,
        )
        return processed_frames

    def _process_single_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        # OpenCV Frame -> RGB -> Tensor
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = transforms.ToTensor()(frame_rgb)
        
        # PromptIR Tiled Inference
        logger.debug("%s: Tile Processing", self.name)
        torch_device = torch.device(self._device)
        restored_tensor = self._infer_tiled(img_tensor, self.tile_size, self.tile_overlap, torch_device)
        
        # Convert back to numpy array (RGB)
        restored_rgb = (restored_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        
        # RGB -> BGR for postprocessing
        frame = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)
        
        # Contrast Enhancement
        frame = cv2.convertScaleAbs(frame, alpha=self.alpha, beta=self.beta)
        
        # CLAHE
        logger.debug("%s: CLAHE", self.name)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8,8))
        l = clahe.apply(l)
        frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        
        # Sharpening
        logger.debug("%s: Sharpening", self.name)
        kernel = np.array([[ 0, -0.5,  0], [-0.5, 3, -0.5], [ 0, -0.5,  0]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        return frame

    def _infer_tiled(self, img_tensor: torch.Tensor, tile: int, overlap: int, device: torch.device) -> torch.Tensor:
        """Tiled inference identical to the notebook logic."""
        c, h, w = img_tensor.shape
        output = torch.zeros_like(img_tensor)
        weight = torch.zeros(1, h, w, dtype=torch.float32)
        step = tile - overlap
        ys = sorted(set(max(0, y) for y in list(range(0, h - tile + 1, step)) + [h - tile]))
        xs = sorted(set(max(0, x) for x in list(range(0, w - tile + 1, step)) + [w - tile]))
        
        if h <= tile and w <= tile:
            patch, _ = pad_to_multiple(img_tensor, 8)
            with torch.no_grad():
                out = self._model(patch.unsqueeze(0).to(device)).squeeze(0).cpu()  # type: ignore
            return out[:, :h, :w].clamp(0, 1)
            
        blend = gaussian_window(tile)
        for y in ys:
            y2 = min(y + tile, h)
            for x in xs:
                x2 = min(x + tile, w)
                patch = img_tensor[:, y:y2, x:x2]
                patch_padded, _ = pad_to_multiple(patch, 8)
                with torch.no_grad():
                    restored = self._model(patch_padded.unsqueeze(0).to(device)).squeeze(0).cpu() # type: ignore
                restored = restored[:, :y2-y, :x2-x]
                bh, bw = y2-y, x2-x
                w_patch = blend[:bh, :bw]
                output[:, y:y2, x:x2] += restored * w_patch
                weight[:, y:y2, x:x2] += w_patch
                
        return (output / weight.clamp(min=1e-6)).clamp(0, 1)

    def cleanup(self) -> None:
        """Release GPU memory and reset state."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.info("%s: GPU Usage cleared.", self.name)
            except Exception:  # noqa: BLE001
                pass
        self._loaded = False
        logger.info("%s: Cleanup complete. Model resources released.", self.name)
