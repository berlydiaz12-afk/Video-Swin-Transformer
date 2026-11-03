"""Prediction and feature extraction for whole videos (K miniclips).

As in mmaction's test (``average_clips='prob'``), the score of a video is the
mean of the softmax probabilities of its miniclips.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import nn

from .colab_setup import amp_dtype
from .miniclip_reading import read_miniclips
from .model import logits_to_probs
from .swin_input import to_model_input


@dataclass
class VideoPrediction:
    probs: np.ndarray        # (num_classes,) mean over miniclips
    clip_probs: np.ndarray   # (K, num_classes)
    label: int               # argmax of probs

    def top_k(self, k: int = 5, class_names: Sequence[str] | None = None) -> list[tuple[str | int, float]]:
        idx = np.argsort(-self.probs)[:k]
        return [((class_names[i] if class_names else int(i)), float(self.probs[i])) for i in idx]


def autocast_context(device: torch.device | str, enabled: bool = True):
    device = torch.device(device)
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=amp_dtype())
    return nullcontext()


def _chunks(x: torch.Tensor, size: int) -> Iterator[torch.Tensor]:
    for i in range(0, len(x), size):
        yield x[i:i + size]


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def clip_logits(model: nn.Module, clips: torch.Tensor | np.ndarray, batch_size: int = 4,
                amp: bool = True) -> torch.Tensor:
    """Logits ``(K, num_outputs)`` (float32, on CPU) for uint8 ``(K, T, H, W, 3)`` miniclips.

    Miniclips go to the GPU ``batch_size`` at a time; lower it if memory runs out.
    """
    device = _device_of(model)
    clips = torch.as_tensor(clips)
    out = []
    for chunk in _chunks(clips, batch_size):
        x = to_model_input(chunk, device=device)
        with autocast_context(device, amp):
            out.append(model(x).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def predict_clips(model: nn.Module, clips: torch.Tensor | np.ndarray, batch_size: int = 4,
                  amp: bool = True) -> VideoPrediction:
    """Prediction for one video given its uint8 miniclips ``(K, T, H, W, 3)``.

    Probabilities come from softmax, or from sigmoid for a binary head.
    """
    was_training = model.training
    model.eval()
    try:
        probs = logits_to_probs(clip_logits(model, clips, batch_size, amp)).numpy()
    finally:
        model.train(was_training)
    mean = probs.mean(0)
    return VideoPrediction(probs=mean, clip_probs=probs, label=int(mean.argmax()))


def predict_video(model: nn.Module, clip_dir: str | Path, batch_size: int = 4,
                  amp: bool = True) -> VideoPrediction:
    """Prediction for one video from its folder ``miniclips/<id>/``."""
    return predict_clips(model, read_miniclips(clip_dir), batch_size, amp)


@torch.no_grad()
def extract_features(model: nn.Module, clips: torch.Tensor | np.ndarray, batch_size: int = 4,
                     amp: bool = True, reduce: str | None = "mean") -> np.ndarray:
    """Backbone features after spatio-temporal average pooling.

    Returns ``(K, D)`` per miniclip if ``reduce`` is None, or ``(D,)`` with
    ``reduce="mean"`` (D = 1024 for Swin-B, 768 for T/S).
    """
    was_training = model.training
    model.eval()
    device = _device_of(model)
    feats = []
    try:
        for chunk in _chunks(torch.as_tensor(clips), batch_size):
            x = to_model_input(chunk, device=device)
            with autocast_context(device, amp):
                feats.append(model.forward_features(x, pooled=True).float().cpu())
    finally:
        model.train(was_training)
    f = torch.cat(feats).numpy()
    if reduce is None:
        return f
    if reduce == "mean":
        return f.mean(0)
    raise ValueError("reduce must be None or 'mean'")


def load_kinetics400_labels() -> list[str]:
    """The 400 Kinetics class names, in the order of the checkpoint head."""
    path = Path(__file__).with_name("label_map_k400.txt")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
