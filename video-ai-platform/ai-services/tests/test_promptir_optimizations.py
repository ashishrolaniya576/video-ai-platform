import numpy as np
import pytest
import torch

from app.models.video_visibility import VideoVisibilityModel


def test_promptir_preprocess_and_postprocess_helpers():
    model = VideoVisibilityModel("cpu")
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    frame[2:8, 3:12, 0] = 255
    frame[9:14, 5:20, 2] = 128

    tensor = model._prepare_input_tensor(frame)
    assert tensor.shape == (3, 16, 24)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

    restored = torch.rand(3, 16, 24, dtype=torch.float32)
    output = model._postprocess_restored(restored)
    assert output.shape == (16, 24, 3)
    assert output.dtype == np.uint8


def test_channels_last_skips_rank3_tensors():
    model = VideoVisibilityModel("cpu")
    model._enable_channels_last = True
    chw = torch.randn(3, 16, 24)
    assert model._to_channels_last_nchw(chw) is chw

    nchw = torch.randn(2, 3, 16, 24)
    assert model._to_channels_last_nchw(nchw) is nchw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_channels_last_applies_only_to_rank4_on_cuda():
    model = VideoVisibilityModel("cuda")
    model._enable_channels_last = True
    chw = torch.randn(3, 16, 24, device="cuda")
    assert model._to_channels_last_nchw(chw).ndim == 3

    nchw = torch.randn(1, 3, 16, 24, device="cuda")
    converted = model._to_channels_last_nchw(nchw)
    assert converted.ndim == 4
    assert converted.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_promptir_load_model_with_channels_last_enabled():
    model = VideoVisibilityModel("cuda")
    model.load_model()
    assert model.is_loaded
    frame = np.random.randint(0, 255, (256, 320, 3), dtype=np.uint8)
    output = model.process_frame(frame, 0)
    assert output.shape == frame.shape
    model.cleanup()
