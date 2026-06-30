import numpy as np
import torch

from app.models.heavy_rain_remove import HeavyRainRemovalModel


def test_prepare_batch_input_converts_rgb_frames_to_float_batch():
    model = HeavyRainRemovalModel(device="cpu")

    frames = [
        np.zeros((64, 64, 3), dtype=np.uint8),
        np.full((64, 64, 3), 255, dtype=np.uint8),
    ]

    batch_tensor, _ = model._prepare_batch_input(frames)

    assert batch_tensor.shape == (2, 3, 64, 64)
    assert batch_tensor.dtype == torch.float32
    assert torch.allclose(batch_tensor[0], torch.full((3, 64, 64), -1.0))
    assert torch.allclose(batch_tensor[1], torch.full((3, 64, 64), 1.0))
