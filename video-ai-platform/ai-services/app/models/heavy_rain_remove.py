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

        network = network.to(torch_device).eval()

        # We use manual numpy conversion in process_frame instead of this transform for speed.
        self._transform = None

        self._network = network
        self._loaded = True
        logger.info("%s: Checkpoint Loaded. Model loaded successfully.", self.name)

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
        
        if frame_idx % 50 == 0:
            logger.info("%s: Processing frame %d", self.name, frame_idx)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Original dims
        orig_h, orig_w = frame_rgb.shape[:2]

        # In notebook test.py, dimensions are resized to be multiples of 64
        floor_h = int(orig_h / 64)
        floor_w = int(orig_w / 64)
        new_h = int(floor_h * 64)
        new_w = int(floor_w * 64)

        if new_h != orig_h or new_w != orig_w:
            img_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = frame_rgb

        # Transform to tensor (manual for speed to avoid transforms.ToTensor overhead)
        img_float = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self._device, non_blocking=True)

        # Inference
        with torch.inference_mode(), torch.autocast(device_type=self._device, enabled=self._device=="cuda"):
            st_out, trans_out, atm_out, clean_out = self._network(input_tensor) # type: ignore
            
            # The notebook's test.py does:
            # clean_out = (image_in_var - st_out - (1 - trans_out) * atm_out) / (trans_out + 0.0001)
            # However, looking at test.py predict() method, it just uses clean_out directly, or recomputes it.
            # Wait, in predict(), it does recompute:
            # clean_out = (input_var - st_out - (1 - trans_out) * atm_out) / (trans_out + 0.0001)
            output_tensor = (input_tensor - st_out - (1 - trans_out) * atm_out) / (trans_out + 0.0001)

        # Postprocessing: Un-normalize and clamp
        output_tensor = output_tensor + 0.5
        output_tensor = output_tensor.clamp(0.0, 1.0)
        
        # Convert to numpy array [H, W, C]
        out_rgb = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_rgb = (out_rgb * 255.0).astype(np.uint8)

        # Resize back to original dimensions if we changed them
        if new_h != orig_h or new_w != orig_w:
            out_rgb = cv2.resize(out_rgb, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Convert back to BGR
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        return out_bgr

    def cleanup(self) -> None:
        """Release GPU memory and reset state."""
        if self._network is not None:
            del self._network
            self._network = None
        self._transform = None
        if self._device == "cuda":
            try:
                torch.cuda.empty_cache()
                logger.debug("%s: CUDA cache cleared.", self.name)
            except Exception:  # noqa: BLE001
                pass
        self._loaded = False
        logger.info("%s: Model Closed and resources released.", self.name)
