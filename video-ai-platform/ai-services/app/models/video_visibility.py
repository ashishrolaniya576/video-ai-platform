"""
Video Visibility Enhancement Model (PromptIR)

Converted from ai-services/notebooks/video_visibility.ipynb.
Automatically downloads repository and checkpoint if missing.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
try:
    import lightning as pl  # lightning>=2.0
except ImportError:
    import pytorch_lightning as pl  # type: ignore[no-redef]  # legacy

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
        self._tile_batch_size = max(1, min(8, settings.promptir_tile_batch_size))
        self._enable_amp = settings.promptir_enable_amp
        self._enable_channels_last = settings.promptir_enable_channels_last
        self._enable_compile = settings.promptir_enable_compile
        self._profile = settings.promptir_profile
        
        # GPU Optimizations
        self._gaussian_cache: dict[int, torch.Tensor] = {}
        self._tile_buffers: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._timing_stats: Dict[str, List[float]] = {"prepare": [], "infer": [], "postprocess": []}

    def _apply_model_channels_last(self, model: PromptIRWrapper) -> None:
        """Convert only rank-4 parameters to channels_last; disable flag on failure."""
        if not self._enable_channels_last:
            return
        try:
            for param in model.parameters():
                if param.ndim == 4:
                    param.data = param.data.to(memory_format=torch.channels_last)
        except (TypeError, RuntimeError) as exc:
            logger.debug(
                "%s: channels_last not applied to model weights (%s); continuing in default layout",
                self.name,
                exc,
            )
            self._enable_channels_last = False

    def _to_channels_last_nchw(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply channels_last only to rank-4 NCHW tensors; never touch CHW inputs."""
        if not self._enable_channels_last or tensor.device.type != "cuda":
            return tensor
        if tensor.ndim != 4:
            return tensor
        try:
            return tensor.to(memory_format=torch.channels_last)
        except (TypeError, RuntimeError) as exc:
            logger.debug(
                "%s: channels_last skipped for input tensor (%s); continuing in default layout",
                self.name,
                exc,
            )
            self._enable_channels_last = False
            return tensor

    def _run_model_inference(self, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Run PromptIR on a rank-4 batch tensor with optional AMP and channels_last."""
        batch = self._to_channels_last_nchw(batch)
        if device.type == "cuda" and self._enable_amp:
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16):
                return self._model(batch)
        with torch.inference_mode():
            return self._model(batch)

    def _get_gaussian_window(self, size: int, device: torch.device) -> torch.Tensor:
        """Fetch or create a cached 2D Gaussian window on the GPU."""
        if size not in self._gaussian_cache:
            sigma = size / 6.0
            coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g2d = g.unsqueeze(0) * g.unsqueeze(1)
            self._gaussian_cache[size] = g2d / g2d.max()
        return self._gaussian_cache[size]

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
        
        # Dynamic Tile Scaling based on VRAM (if on CUDA)
        if torch_device.type == "cuda":
            total_vram = torch.cuda.get_device_properties(torch_device).total_memory
            # If VRAM > 12GB, we can afford larger tiles (e.g. 1024), reducing overhead
            if total_vram > 12 * 1024**3:
                self.tile_size = 1024
                self.tile_overlap = 64
                logger.info("%s: Detected >12GB VRAM. Scaling tile_size to %d for throughput.", self.name, self.tile_size)

        # Initialize network
        model = PromptIRWrapper(PromptIR)
        
        # Load weights
        logger.info("%s: loading weights from %s", self.name, ckpt_path)
        try:
            checkpoint = torch.load(str(ckpt_path), map_location=torch_device, weights_only=False)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint for {self.name}: {e}")

        if torch_device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        model.to(torch_device)
        if torch_device.type == "cuda":
            self._apply_model_channels_last(model)
        model.eval()

        if torch_device.type == "cuda" and self._enable_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                logger.info("%s: enabled torch.compile for PromptIR inference.", self.name)
            except Exception as exc:
                logger.warning("%s: torch.compile disabled for PromptIR due to %s", self.name, exc)

        self._model = model
        self._loaded = True
        logger.info("%s: Checkpoint Loaded. Model loaded successfully.", self.name)

    def _prepare_input_tensor(self, frame: np.ndarray, device: torch.device | None = None) -> torch.Tensor:
        """Convert a BGR numpy frame into a compact CHW float tensor on the target device."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous().to(torch.float32)
        tensor.div_(255.0)
        if device is not None:
            tensor = tensor.to(device=device, non_blocking=True)
        return tensor

    def _apply_gpu_postprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply contrast and sharpening on GPU using PyTorch kernels."""
        tensor = tensor.clamp(0.0, 1.0)
        tensor = tensor.mul(self.alpha).add(self.beta / 255.0)
        tensor = tensor.clamp(0.0, 1.0)
        kernel = torch.tensor(
            [[[[0.0, -0.5, 0.0], [-0.5, 3.0, -0.5], [0.0, -0.5, 0.0]]]],
            device=tensor.device,
            dtype=tensor.dtype,
        )
        # Apply the same 3x3 sharpen filter to each channel independently.
        sharpened = F.conv2d(
            tensor.unsqueeze(0),
            kernel.repeat(tensor.size(0), 1, 1, 1),
            groups=tensor.size(0),
            padding=1,
        )
        return sharpened.squeeze(0).clamp(0.0, 1.0)

    def _postprocess_restored(self, restored_tensor: torch.Tensor, device: torch.device | None = None) -> np.ndarray:
        """Convert a restored CHW tensor back into a BGR uint8 numpy frame."""
        tensor = restored_tensor.detach()
        if device is not None and device.type == "cuda" and self._enable_amp:
            tensor = self._apply_gpu_postprocess(tensor)
        else:
            tensor = tensor.clamp(0.0, 1.0)

        rgb = tensor.permute(1, 2, 0).contiguous()
        rgb = rgb.mul(255.0).round().to(torch.uint8)
        if rgb.device.type != "cpu":
            rgb = rgb.cpu()
        frame_rgb = rgb.numpy()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def _record_timing(self, stage: str, elapsed_ms: float) -> None:
        if not self._profile:
            return
        self._timing_stats[stage].append(elapsed_ms)
        if len(self._timing_stats[stage]) > 200:
            self._timing_stats[stage] = self._timing_stats[stage][-200:]

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        **kwargs: object,
    ) -> np.ndarray:
        self._assert_loaded()

        torch_device = torch.device(self._device)
        if frame_idx % 50 == 0:
            logger.info("%s: Processing frame %d", self.name, frame_idx)

        prepare_start = time.perf_counter()
        img_tensor = self._prepare_input_tensor(frame, torch_device)
        self._record_timing("prepare", (time.perf_counter() - prepare_start) * 1000.0)

        infer_start = time.perf_counter()
        restored_tensor = self._infer_tiled(img_tensor, self.tile_size, self.tile_overlap, torch_device)
        self._record_timing("infer", (time.perf_counter() - infer_start) * 1000.0)

        postprocess_start = time.perf_counter()
        frame = self._postprocess_restored(restored_tensor, torch_device)
        self._record_timing("postprocess", (time.perf_counter() - postprocess_start) * 1000.0)

        if self._profile and frame_idx % 100 == 0:
            self._log_timing_summary(frame_idx)

        
        return frame

    def _log_timing_summary(self, frame_idx: int) -> None:
        if not self._profile:
            return
        totals = {name: sum(values) / max(1, len(values)) for name, values in self._timing_stats.items() if values}
        logger.info(
            "%s: frame %d timing stats (ms) prepare=%.2f infer=%.2f postprocess=%.2f",
            self.name,
            frame_idx,
            totals.get("prepare", 0.0),
            totals.get("infer", 0.0),
            totals.get("postprocess", 0.0),
        )

    def _infer_tiled(self, img_tensor: torch.Tensor, tile: int, overlap: int, device: torch.device) -> torch.Tensor:
        """Tiled inference deeply optimized for GPU accumulation and batching."""
        c, h, w = img_tensor.shape
        
        img_tensor_gpu = img_tensor.to(device=device, non_blocking=True)
        shape_key = (c, h, w)
        if shape_key not in self._tile_buffers:
            self._tile_buffers[shape_key] = (
                torch.zeros((c, h, w), dtype=torch.float32, device=device),
                torch.zeros((1, h, w), dtype=torch.float32, device=device)
            )
        output, weight = self._tile_buffers[shape_key]
        output.zero_()
        weight.zero_()

        step = tile - overlap
        ys = sorted(set(max(0, y) for y in list(range(0, h - tile + 1, step)) + [h - tile]))
        xs = sorted(set(max(0, x) for x in list(range(0, w - tile + 1, step)) + [w - tile]))

        if h <= tile and w <= tile:
            patch, _ = pad_to_multiple(img_tensor_gpu, 8)
            out = self._run_model_inference(patch.unsqueeze(0), device).squeeze(0)
            return out[:, :h, :w].clamp(0, 1)

        blend = self._get_gaussian_window(tile, device)

        patches = []
        coords = []
        for y in ys:
            y2 = min(y + tile, h)
            for x in xs:
                x2 = min(x + tile, w)
                patch = img_tensor_gpu[:, y:y2, x:x2]
                patch_padded, _ = pad_to_multiple(patch, 8)
                patches.append(patch_padded)
                coords.append((y, y2, x, x2))

        shape_groups: dict[tuple[int, ...], list[tuple[torch.Tensor, tuple[int, int, int, int]]]] = defaultdict(list)
        for patch, coord in zip(patches, coords):
            shape_groups[patch.shape].append((patch, coord))

        for group_items in shape_groups.values():
            group_patches = [patch for patch, _ in group_items]
            group_coords = [coord for _, coord in group_items]
            for i in range(0, len(group_patches), self._tile_batch_size):
                batch_patches = torch.stack(group_patches[i:i + self._tile_batch_size])
                batch_restored = self._run_model_inference(batch_patches, device)

                for j in range(batch_patches.size(0)):
                    y, y2, x, x2 = group_coords[i + j]
                    restored = batch_restored[j, :, :y2 - y, :x2 - x]
                    bh, bw = y2 - y, x2 - x
                    w_patch = blend[:bh, :bw]
                    output[:, y:y2, x:x2].add_(restored * w_patch)
                    weight[:, y:y2, x:x2].add_(w_patch)

        output.div_(weight.clamp(min=1e-6))
        return output.clamp(0, 1)

    def cleanup(self) -> None:
        """Release GPU memory and reset state."""
        if self._model is not None:
            del self._model
            self._model = None
        self._gaussian_cache.clear()
        self._tile_buffers.clear()
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.info("%s: GPU Usage cleared.", self.name)
            except Exception:
                pass
        self._loaded = False
        logger.info("%s: Cleanup complete. Model resources released.", self.name)
