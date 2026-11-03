"""Input format of Video Swin.

The model takes a float tensor ``(B, C, T, H, W) = (B, 3, 32, 224, 224)``:
RGB channels (``to_bgr=False``) normalized per channel with the ImageNet
statistics. The official configs write them on the 0-255 scale
(mean 123.675, 116.28, 103.53; std 58.395, 57.12, 57.375); dividing by 255
gives the values below.

The miniclips are stored as uint8 ``(T, H, W, 3)``. Keep them as uint8 until
they reach the GPU and convert there with ``to_model_input``: it moves 4 times
fewer bytes from CPU to GPU than sending float32.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
CLIP_LEN = 32
CROP_SIZE = 224
CHANNELS = 3


def normalize(clip: np.ndarray, mean: Sequence[float] = IMAGENET_MEAN,
              std: Sequence[float] = IMAGENET_STD) -> np.ndarray:
    """uint8 ``(..., H, W, 3)`` -> float32 ``(x / 255 - mean) / std`` (NumPy, channel last)."""
    x = np.asarray(clip)
    if x.shape[-1] != CHANNELS:
        raise ValueError(f"expected RGB as the last axis, got shape {x.shape}")
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    return (x.astype(np.float32) / np.float32(255.0) - m) / s


def to_cthw(clip: np.ndarray) -> np.ndarray:
    """``(..., T, H, W, C)`` -> ``(..., C, T, H, W)`` (NumPy)."""
    return np.ascontiguousarray(np.moveaxis(np.asarray(clip), -1, -4))


def to_model_input(clips: torch.Tensor | np.ndarray, device: str | torch.device | None = None,
                   mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD,
                   dtype: torch.dtype = torch.float32, non_blocking: bool = True) -> torch.Tensor:
    """uint8 ``(T, H, W, 3)`` or ``(B, T, H, W, 3)`` -> normalized ``(B, 3, T, H, W)`` on ``device``.

    The cast and the normalization run on ``device`` (the GPU), after the transfer.
    A float input is taken as already scaled to [0, 1].
    """
    x = torch.as_tensor(clips)
    if x.ndim == 4:
        x = x.unsqueeze(0)
    if x.ndim != 5 or x.shape[-1] != CHANNELS:
        raise ValueError(f"expected uint8 (B, T, H, W, 3) or (T, H, W, 3), got shape {tuple(x.shape)}")
    is_uint8 = x.dtype == torch.uint8
    if device is not None:
        x = x.to(device, non_blocking=non_blocking)
    x = x.permute(0, 4, 1, 2, 3).to(dtype)
    if is_uint8:
        x = x / 255.0
    m = torch.tensor(mean, dtype=dtype, device=x.device).view(1, 3, 1, 1, 1)
    s = torch.tensor(std, dtype=dtype, device=x.device).view(1, 3, 1, 1, 1)
    return ((x - m) / s).contiguous()


def prepare_clips(clips: np.ndarray | torch.Tensor, device: str | torch.device | None = None,
                  check: bool = True, **kwargs) -> torch.Tensor:
    """Miniclips of one video uint8 ``(K, T, H, W, 3)`` -> model input ``(K, 3, T, H, W)``."""
    x = to_model_input(clips, device=device, **kwargs)
    if check:
        check_input(x)
    return x


def check_input(x: torch.Tensor, clip_len: int = CLIP_LEN, size: int = CROP_SIZE) -> None:
    """Raise if ``x`` does not look like the input the checkpoints were trained with."""
    if x.ndim != 5:
        raise ValueError(f"expected (B, 3, T, H, W), got {tuple(x.shape)}")
    _, c, t, h, w = x.shape
    if c != CHANNELS:
        raise ValueError(f"expected 3 channels on axis 1, got {c} (is the tensor still channel-last?)")
    if (t, h, w) != (clip_len, size, size):
        raise ValueError(f"expected T×H×W = {clip_len}×{size}×{size}, got {t}×{h}×{w}")
    if not torch.is_floating_point(x):
        raise ValueError("the input must be float and normalized (use to_model_input)")


def dummy_input(batch: int = 1, clip_len: int = CLIP_LEN, size: int = CROP_SIZE,
                device: str | torch.device | None = None) -> torch.Tensor:
    """Random normalized input, to check shapes and memory."""
    return torch.randn(batch, CHANNELS, clip_len, size, size, device=device)
