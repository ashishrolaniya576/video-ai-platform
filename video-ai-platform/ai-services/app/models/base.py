"""
Abstract base class that every AI model module must implement.

The pipeline manager calls exactly three methods on every model:
    1. load_model()      — load weights into memory (called at startup)
    2. process_frame()   — run inference on a single frame
    3. cleanup()         — release GPU / file handles

Enforcing this interface makes the pipeline completely model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np


class BaseModel(ABC):
    """
    Abstract AI model interface.

    All concrete models (StabilizationModel, HeavyRainRemovalModel,
    VideoVisibilityModel) must subclass this and implement all abstract methods.
    """

    # Human-readable name shown in logs and API responses
    name: str = "BaseModel"

    def __init__(self, device: str) -> None:
        """
        Args:
            device: Torch device string — 'cuda' or 'cpu'.
        """
        self._device = device
        self._loaded: bool = False
        self.available: bool = True
        self.unavailable_reason: str = ""

    # ── Mandatory interface ────────────────────────────────────────────────────

    @abstractmethod
    def load_model(self) -> None:
        """
        Load model weights into memory.

        Called once at application startup. Must be idempotent — calling
        load_model() twice must not load the model twice.
        After a successful call, `self._loaded` must be set to True.
        """

    @abstractmethod
    def process_frame(self, frame: np.ndarray, frame_idx: int, **kwargs: object) -> np.ndarray:
        """
        Run inference on a single BGR frame.

        Args:
            frame:     H×W×3 uint8 numpy array in BGR colour space.
            frame_idx: The 0-based index of the current frame in the video stream.
            **kwargs:  Any model-specific keyword arguments.

        Returns:
            Processed frame as an H×W×3 uint8 numpy array in BGR.

        Raises:
            RuntimeError: if load_model() has not been called yet.
        """

    @abstractmethod
    def cleanup(self) -> None:
        """
        Release GPU memory, close file handles, and reset internal state.

        Called at application shutdown or after a processing error.
        Must be safe to call even if load_model() was never called.
        """

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def is_available(self) -> bool:
        return self.available

    @property
    def device(self) -> str:
        return self._device

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if the model is not loaded."""
        if not self._loaded:
            raise RuntimeError(
                f"{self.name} has not been loaded. "
                "Call load_model() before calling process()."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self._device!r}, loaded={self._loaded})"
