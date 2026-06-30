"""
Heavy Rain Removal Model

Converted from ai-services/notebooks/HEAVYRAIN.ipynb
Preserves original Python 3 compatibility fixes.
Automatically downloads repository and checkpoint if missing.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from app.config.settings import settings
from app.models.base import BaseModel
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_python3_patches(repo_path: Path) -> None:
    """Apply all Python 3 compatibility fixes from the notebook to the cloned repo files."""
    
    # Fix 1: helper.py
    helper_path = repo_path / "helper.py"
    if helper_path.exists():
        with open(helper_path, "r") as f:
            content = f.read()
        if "imresize" in content or 'print "random' in content:
            content = content.replace(
                'print "random low value leq high value"',
                'print("random low value leq high value")'
            )
            content = content.replace("from scipy.misc import imresize\n", "")
            content = content.replace(
                "img_data = imresize(img_data, [h,w])",
                "img_data = np.array(Image.fromarray(img_data.astype(np.uint8)).resize((w, h)))"
            )
            # Make sure PIL is imported if missing (though the notebook didn't inject import Image, 
            # we should to be safe, but we'll inject it just in case).
            if "from PIL import Image" not in content:
                content = "from PIL import Image\n" + content
            with open(helper_path, "w") as f:
                f.write(content)
            logger.debug("Applied patches to helper.py")

    # Fix 2: test.py (we might not use test.py directly, but applying for completeness)
    test_path = repo_path / "test.py"
    if test_path.exists():
        with open(test_path, "r") as f:
            content = f.read()
        if "h = h / 2" in content:
            content = content.replace("from tensorboard_logger import log_value\n", "")
            content = content.replace("h = h / 2", "h = int(h / 2)")
            content = content.replace("w = w / 2", "w = int(w / 2)")
            content = content.replace("new_h = floor_h * 64", "new_h = int(floor_h * 64)")
            content = content.replace("new_w = floor_w * 64", "new_w = int(floor_w * 64)")
            with open(test_path, "w") as f:
                f.write(content)
            logger.debug("Applied patches to test.py")

    # Fix 3: Base.py - fix torch.load for old checkpoint format
    base_path = repo_path / "Base.py"
    if base_path.exists():
        with open(base_path, "r") as f:
            content = f.read()
        if "weights_only" not in content:
            content = content.replace(
                "ckpt = torch.load(ckpt_path)",
                "ckpt = torch.load(ckpt_path, weights_only=False, encoding='latin1')"
            )
            with open(base_path, "w") as f:
                f.write(content)
            logger.debug("Applied patches to Base.py")

    # Fix 4: model.py - fix integer divisions for channel counts
    model_path = repo_path / "model.py"
    if model_path.exists():
        with open(model_path, "r") as f:
            content = f.read()
        lines = content.split("\n")
        fixed_lines = []
        changed = False
        for line in lines:
            if re.search(r'\w\s*/\s*\d', line) and '//' not in line and '#' not in line.split('/')[0]:
                new_line = re.sub(r'(?<!/)/(?!/)', '//', line)
                fixed_lines.append(new_line)
                changed = True
            else:
                fixed_lines.append(line)
        if changed:
            with open(model_path, "w") as f:
                f.write("\n".join(fixed_lines))
            logger.debug("Applied patches to model.py")


def _download_if_missing() -> None:
    """Download the repository and checkpoint if missing."""
    repo_path = Path(settings.heavy_rain_repo_path)
    if not repo_path.exists():
        logger.info(f"Cloning Heavy Rain Removal repo to {repo_path}...")
        import subprocess
        try:
            subprocess.run(["git", "clone", "https://github.com/liruoteng/HeavyRainRemoval.git", str(repo_path)], check=True)
            logger.info("Successfully cloned HeavyRainRemoval repo.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone HeavyRainRemoval repo: {e}")
    
    # Apply patches
    _apply_python3_patches(repo_path)

    # Checkpoint
    ckpt_path = Path(settings.heavy_rain_checkpoint)
    if not ckpt_path.exists():
        logger.info(f"Downloading checkpoint to {ckpt_path}...")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        import gdown
        try:
            gdown.download(settings.heavy_rain_checkpoint_url, str(ckpt_path), quiet=False)
            logger.info("Successfully downloaded checkpoint.")
        except Exception as e:
            raise RuntimeError(f"Failed to download Heavy Rain Removal checkpoint: {e}")


class HeavyRainRemovalModel(BaseModel):
    """Heavy Rain Removal model."""

    name = "HeavyRainRemoval"

    def __init__(self, device: str) -> None:
        super().__init__(device)
        self._network: Optional[torch.nn.Module] = None
        self._transform: Optional[transforms.Compose] = None
        self._batch_size = max(1, int(settings.heavy_rain_batch_size))
        self._input_cpu_buffer: Optional[torch.Tensor] = None
        self._input_gpu_buffer: Optional[torch.Tensor] = None
        self._output_gpu_buffer: Optional[torch.Tensor] = None
        self._cpu_rgb_buffer: Optional[np.ndarray] = None
        self._cpu_bgr_buffer: Optional[np.ndarray] = None
        self._current_shape: Optional[Tuple[int, int, int]] = None
        self._profile_enabled = bool(settings.heavy_rain_profile)
        self._timing_stats: dict[str, float] = {
            "gpu_upload": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "frames": 0.0,
        }

    def load_model(self) -> None:
        if self._loaded:
            logger.debug("%s: already loaded — skipping.", self.name)
            return

        # Ensure repo and checkpoint exist
        _download_if_missing()

        repo_path = Path(settings.heavy_rain_repo_path).resolve()
        ckpt_path = Path(settings.heavy_rain_checkpoint).resolve()

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        try:
            from model import DecompModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"Failed to import DecompModel from '{repo_path}'. "
                f"Original error: {exc}"
            ) from exc

        logger.info("%s: loading network on %s", self.name, self._device)
        torch_device = torch.device(self._device)

        # Initialize network
        network = DecompModel()
        
        # Load weights
        logger.info("%s: loading weights from %s", self.name, ckpt_path)
        try:
            ckpt = torch.load(str(ckpt_path), map_location=torch_device, weights_only=False, encoding='latin1')
            
            state_dict_raw = ckpt.get('G', ckpt)
            
            # The checkpoint keys have 'module.' prefix because it was trained with DataParallel
            state_dict = {}
            for k, v in state_dict_raw.items():
                # In Base.py they do: name = k[7:] (if 'module' in k)
                if 'module.' in k:
                    name = k.replace("module.", "")
                else:
                    name = k
                state_dict[name] = v
                
            network.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint for {self.name}: {e}")

        if self._device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            network = network.to(torch_device, memory_format=torch.channels_last).eval()
        else:
            network = network.to(torch_device).eval()

        # We use manual numpy conversion in process_frame instead of this transform for speed.
        self._transform = None

        self._network = network
        self._loaded = True
        logger.info("%s: Checkpoint Loaded. Model loaded successfully.", self.name)

    def _ensure_buffers(self, batch_size: int, height: int, width: int) -> None:
        if height % 64 != 0:
            height = (height // 64 + 1) * 64
        if width % 64 != 0:
            width = (width // 64 + 1) * 64

        if self._cpu_rgb_buffer is None or self._cpu_rgb_buffer.shape != (batch_size, height, width, 3):
            self._cpu_rgb_buffer = np.empty((batch_size, height, width, 3), dtype=np.uint8)
        if self._cpu_bgr_buffer is None or self._cpu_bgr_buffer.shape != (batch_size, height, width, 3):
            self._cpu_bgr_buffer = np.empty((batch_size, height, width, 3), dtype=np.uint8)

        if self._input_gpu_buffer is None or self._input_gpu_buffer.shape != (batch_size, 3, height, width):
            self._input_gpu_buffer = torch.empty((batch_size, 3, height, width), dtype=torch.float32, device=self._device)
        if self._output_gpu_buffer is None or self._output_gpu_buffer.shape != (batch_size, 3, height, width):
            self._output_gpu_buffer = torch.empty((batch_size, 3, height, width), dtype=torch.float32, device=self._device)
        if self._input_cpu_buffer is None or self._input_cpu_buffer.shape != (3, height, width):
            pin_memory = self._device == "cuda"
            self._input_cpu_buffer = torch.empty((3, height, width), dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        self._current_shape = (batch_size, height, width)

    def _resolve_batch_size(self, height: int, width: int, requested_batch_size: int) -> int:
        if self._device != "cuda":
            return 1
        try:
            free_mem, _ = torch.cuda.mem_get_info()
            bytes_per_frame = max(1, height * width * 3 * 32 // 8) * 8
            estimated_capacity = max(1, free_mem // max(1_000_000_000, bytes_per_frame * 8))
            return max(1, min(requested_batch_size, estimated_capacity))
        except Exception:
            return max(1, min(requested_batch_size, 2))

    def _prepare_batch_input(self, frames: List[np.ndarray], batch_size: Optional[int] = None) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Reuse preallocated CPU/GPU buffers to build a batch tensor without repeated allocations."""
        if not frames:
            raise ValueError("At least one frame is required")
        effective_batch_size = len(frames) if batch_size is None else int(batch_size)
        effective_batch_size = max(1, min(effective_batch_size, len(frames)))

        target_h = max(frame.shape[0] for frame in frames)
        target_w = max(frame.shape[1] for frame in frames)
        self._ensure_buffers(effective_batch_size, target_h, target_w)

        original_shapes = [(frame.shape[0], frame.shape[1]) for frame in frames]
        for idx, frame in enumerate(frames):
            frame_h, frame_w = frame.shape[:2]
            target_frame_h = self._current_shape[1] if self._current_shape[1] != frame_h else frame_h
            target_frame_w = self._current_shape[2] if self._current_shape[2] != frame_w else frame_w
            if target_frame_h != frame_h or target_frame_w != frame_w:
                resized_frame = cv2.resize(frame, (target_frame_w, target_frame_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized_frame = frame
            cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB, dst=self._cpu_rgb_buffer[idx, :target_frame_h, :target_frame_w, :])
            rgb_view = torch.from_numpy(self._cpu_rgb_buffer[idx, :target_frame_h, :target_frame_w, :])
            self._input_cpu_buffer.copy_(rgb_view.permute(2, 0, 1).contiguous().float())
            self._input_gpu_buffer[idx].copy_(self._input_cpu_buffer, non_blocking=self._device == "cuda")

        input_batch = self._input_gpu_buffer[:len(frames)]
        input_batch = input_batch.mul_(2.0 / 255.0).sub_(1.0)
        return input_batch, original_shapes

    def _postprocess_batch_output(self, output: torch.Tensor, original_shapes: List[Tuple[int, int]]) -> List[np.ndarray]:
        output_view = self._output_gpu_buffer[:len(original_shapes)]
        output_view.copy_(output)
        output_view = output_view.add(0.5).clamp_(0.0, 1.0).mul_(255.0)
        results: List[np.ndarray] = []
        for idx, (orig_h, orig_w) in enumerate(original_shapes):
            frame = output_view[idx].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            if frame.shape[0] != orig_h or frame.shape[1] != orig_w:
                frame = cv2.resize(frame, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            results.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return results

    def process_frames(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int] | None = None,
        **kwargs: object,
    ) -> List[np.ndarray]:
        self._assert_loaded()
        if not frames:
            return []
        if frame_indices is None:
            frame_indices = list(range(len(frames)))

        requested_batch_size = int(kwargs.get("batch_size", self._batch_size))
        results: List[np.ndarray] = []
        for batch_start in range(0, len(frames), requested_batch_size):
            batch_frames = frames[batch_start:batch_start + requested_batch_size]
            if not batch_frames:
                continue

            batch_h = max(frame.shape[0] for frame in batch_frames)
            batch_w = max(frame.shape[1] for frame in batch_frames)
            effective_batch_size = min(len(batch_frames), self._resolve_batch_size(batch_h, batch_w, requested_batch_size))
            effective_batch = batch_frames[:effective_batch_size]

            upload_start = time.perf_counter()
            prepared, original_shapes = self._prepare_batch_input(effective_batch, effective_batch_size)
            self._timing_stats["gpu_upload"] += time.perf_counter() - upload_start

            inference_start = time.perf_counter()
            with torch.inference_mode(), torch.amp.autocast("cuda", enabled=self._device == "cuda"):
                _, _, _, clean_out = self._network(prepared)  # type: ignore[misc]
                clean_out = clean_out.detach()
            self._timing_stats["inference"] += time.perf_counter() - inference_start

            postprocess_start = time.perf_counter()
            batch_results = self._postprocess_batch_output(clean_out, original_shapes)
            self._timing_stats["postprocess"] += time.perf_counter() - postprocess_start

            results.extend(batch_results)
            self._timing_stats["frames"] += len(effective_batch)
            if self._profile_enabled and (len(results) % 8 == 0 or len(results) == len(frames)):
                fps = self._timing_stats["frames"] / max(1e-6, self._timing_stats["inference"])
                logger.info(
                    "%s: batch=%d upload=%.3fms infer=%.3fms post=%.3fms fps=%.2f",
                    self.name,
                    effective_batch_size,
                    self._timing_stats["gpu_upload"] * 1000.0,
                    self._timing_stats["inference"] * 1000.0,
                    self._timing_stats["postprocess"] * 1000.0,
                    fps,
                )
        return results

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        **kwargs: object,
    ) -> np.ndarray:
        """
        Process a single OpenCV BGR frame for heavy rain removal.
        Convert to RGB -> Tensor -> Inference -> Postprocess -> BGR.
        """
        self._assert_loaded()

        if frame_idx % 100 == 0:
            logger.info("%s: Processing frame %d", self.name, frame_idx)

        batch_result = self.process_frames([frame], [frame_idx], **kwargs)
        return batch_result[0]

    def cleanup(self) -> None:
        """Release GPU memory and reset state."""
        if self._network is not None:
            del self._network
            self._network = None
        self._transform = None
        self._input_cpu_buffer = None
        self._input_gpu_buffer = None
        self._output_gpu_buffer = None
        self._cpu_rgb_buffer = None
        self._cpu_bgr_buffer = None
        self._current_shape = None
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.debug("%s: CUDA cache cleared.", self.name)
            except Exception:  # noqa: BLE001
                pass
        self._loaded = False
        logger.info("%s: Model Closed and resources released.", self.name)
