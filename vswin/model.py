"""Video Swin recognizer: SwinTransformer3D backbone + I3D head (same as mmaction's Recognizer3D).

Shapes with the default input (B, 3, 32, 224, 224) and patch (2, 4, 4):

    input        (B, 3,    32, 224, 224)
    patch embed  (B, C,    16,  56,  56)      C = 96 (T/S) or 128 (B)
    stage 1..4   spatial 56 -> 28 -> 14 -> 7, time stays at 16
    backbone     (B, 8C,   16,   7,   7)      8C = 768 (T/S) or 1024 (B)
    head         (B, num_classes)             logits (no softmax)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch import nn

from .checkpoints import LoadReport, load_pretrained
from .swin3d import SwinTransformer3D
from .variants import DEFAULT_VARIANT, SwinVariant, get_variant


class I3DHead(nn.Module):
    """Average pooling over (T, H, W) + dropout + linear layer, with the parameter names of mmaction."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.5, init_std: float = 0.01):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_cls = nn.Linear(in_channels, num_classes)
        nn.init.normal_(self.fc_cls.weight, 0.0, init_std)
        nn.init.zeros_(self.fc_cls.bias)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T, H, W) -> (B, C)."""
        return self.avg_pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_cls(self.dropout(self.pool(x)))


class VideoSwin(nn.Module):
    def __init__(self, variant: str | SwinVariant = DEFAULT_VARIANT, num_classes: int | None = None,
                 drop_path_rate: float | None = None, head_dropout: float = 0.5,
                 use_checkpoint: bool = False, frozen_stages: int = -1, binary: bool = False):
        """``binary=True``: 2 classes with a single output (logit of class 1, sigmoid)
        instead of 2 outputs with softmax. Both give the same probabilities;
        see ``logits_to_probs``."""
        super().__init__()
        v = get_variant(variant)
        if binary and num_classes not in (None, 2):
            raise ValueError("binary=True means 2 classes with one sigmoid output; leave num_classes as None or 2")
        self.variant = v
        self.binary = binary
        self.num_classes = 2 if binary else (num_classes or v.num_classes)
        self.num_outputs = 1 if binary else self.num_classes
        self.backbone = SwinTransformer3D(
            pretrained=None,
            patch_size=v.patch_size,
            embed_dim=v.embed_dim,
            depths=list(v.depths),
            num_heads=list(v.num_heads),
            window_size=v.window_size,
            drop_path_rate=v.drop_path_rate if drop_path_rate is None else drop_path_rate,
            patch_norm=True,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint,
        )
        self.backbone.init_weights()  # random init (trunc_normal); pretrained weights are loaded afterwards
        self.cls_head = I3DHead(v.feature_dim, self.num_outputs, dropout=head_dropout)

    @property
    def feature_dim(self) -> int:
        return self.variant.feature_dim

    def forward_features(self, x: torch.Tensor, pooled: bool = True) -> torch.Tensor:
        """Backbone output: (B, 8C) if ``pooled``, else the map (B, 8C, T/2, H/32, W/32)."""
        feat = self.backbone(x)
        return self.cls_head.pool(feat) if pooled else feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: normalized float tensor (B, 3, T, H, W). Returns logits (B, num_outputs):
        (B, num_classes) for softmax, (B, 1) with ``binary=True``."""
        return self.cls_head(self.backbone(x))

    def freeze_backbone(self, frozen_stages: int = 4) -> None:
        """Freeze the patch embedding and the first ``frozen_stages`` stages (4 = whole backbone except the last norm)."""
        self.backbone.frozen_stages = frozen_stages
        self.backbone._freeze_stages()

    def unfreeze_backbone(self) -> None:
        self.backbone.frozen_stages = -1
        for p in self.backbone.parameters():
            p.requires_grad = True
        self.backbone.train(self.training)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Recompute activations in backward: much less memory, ~30 % slower."""
        for layer in self.backbone.layers:
            layer.use_checkpoint = enabled
            for block in layer.blocks:
                block.use_checkpoint = enabled

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)


def build_model(variant: str | SwinVariant = DEFAULT_VARIANT, num_classes: int | None = None,
                pretrained: bool | str | Path = True, load_head: bool | None = None,
                drop_path_rate: float | None = None, head_dropout: float = 0.5,
                use_checkpoint: bool = False, frozen_stages: int = -1, binary: bool = False,
                device: str | torch.device | None = None, cache: str | Path | None = None,
                verbose: bool = True) -> VideoSwin:
    """Create a ``VideoSwin`` and load its weights.

    * ``pretrained=True``: download and load the official checkpoint of ``variant``.
    * ``pretrained=<path>``: load a local ``.pth`` (official or saved by ``vswin.fine_tuning``).
    * ``pretrained=False``: random weights.

    With ``num_classes`` different from the checkpoint (e.g. 2 for the thesis),
    only the backbone is loaded and the head starts from scratch. ``load_head``
    forces either behavior.

    ``binary=True`` builds a 2-class model with a single sigmoid output
    (1024 -> 1). A checkpoint fine-tuned that way must be loaded with ``binary=True``.
    """
    v = get_variant(variant)
    model = VideoSwin(v, num_classes=num_classes, drop_path_rate=drop_path_rate,
                      head_dropout=head_dropout, use_checkpoint=use_checkpoint,
                      frozen_stages=frozen_stages, binary=binary)
    report: LoadReport | None = None
    if pretrained is True:
        head = (model.num_outputs == v.num_classes) if load_head is None else load_head
        report = load_pretrained(model, v, load_head=head, directory=cache)
    elif pretrained:
        report = load_pretrained(model, path=pretrained, load_head=True if load_head is None else load_head)
    if verbose and report is not None:
        print(report)
    model.load_report = report
    if device is not None:
        model.to(device)
    return model


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Class probabilities (B, num_classes) from the head output.

    * (B, 1), binary head: p = sigmoid(z) and returns [1 - p, p].
    * (B, C), C >= 2: softmax.

    With 2 classes both are the same function: softmax(z)[1] = sigmoid(z1 - z0).
    Returning two columns keeps metrics and predictions identical for both heads.
    """
    logits = logits.float()
    if logits.shape[-1] == 1:
        p = torch.sigmoid(logits)
        return torch.cat([1 - p, p], dim=-1)
    return logits.softmax(-1)


class BinaryBCELoss(nn.Module):
    """``BCEWithLogitsLoss`` for a (B, 1) head and integer labels {0, 1}.

    ``pos_weight`` multiplies the loss of class 1 (use n_negative / n_positive
    to compensate imbalance). ``label_smoothing`` moves the targets to
    ``ls/2`` and ``1 - ls/2``, the same targets that CrossEntropyLoss uses with 2 classes.
    """

    def __init__(self, pos_weight: float | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]) if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target.float().view(-1, 1)
        if self.label_smoothing:
            t = t * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits.float().view(-1, 1), t,
                                                              pos_weight=self.pos_weight)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(total, trainable)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
