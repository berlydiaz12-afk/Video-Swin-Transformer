"""Catalog of the official Video Swin checkpoints.

Each variant reproduces the backbone and head settings of one config in
``Video-Swin-Transformer/configs/recognition/swin``. The ``recommended_*`` fields
copy the optimizer settings of that config (they assume 8 GPUs and gradient
accumulation, so treat them as a starting point, not as fixed values).
"""
from __future__ import annotations

from dataclasses import dataclass, field

_STORAGE = "https://github.com/SwinTransformer/storage/releases/download/v1.0.4"


@dataclass(frozen=True)
class SwinVariant:
    name: str
    url: str
    embed_dim: int
    depths: tuple[int, ...]
    num_heads: tuple[int, ...]
    window_size: tuple[int, int, int]
    drop_path_rate: float
    num_classes: int
    pretrained_on: str
    patch_size: tuple[int, int, int] = (2, 4, 4)
    clip_len: int = 32
    frame_interval: int = 2
    crop_size: int = 224
    recommended_lr: float = 3e-4
    recommended_weight_decay: float = 0.05
    notes: str = field(default="", compare=False)

    @property
    def feature_dim(self) -> int:
        """Channels of the last stage (input size of the classification head)."""
        return self.embed_dim * 2 ** (len(self.depths) - 1)

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


_TINY = dict(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
_SMALL = dict(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
_BASE = dict(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))

VARIANTS: dict[str, SwinVariant] = {
    v.name: v
    for v in (
        SwinVariant("swin_t_k400_1k", f"{_STORAGE}/swin_tiny_patch244_window877_kinetics400_1k.pth",
                    **_TINY, window_size=(8, 7, 7), drop_path_rate=0.1, num_classes=400,
                    pretrained_on="ImageNet-1K -> Kinetics-400",
                    recommended_lr=1e-3, recommended_weight_decay=0.02),
        SwinVariant("swin_s_k400_1k", f"{_STORAGE}/swin_small_patch244_window877_kinetics400_1k.pth",
                    **_SMALL, window_size=(8, 7, 7), drop_path_rate=0.1, num_classes=400,
                    pretrained_on="ImageNet-1K -> Kinetics-400",
                    recommended_lr=1e-3, recommended_weight_decay=0.02),
        SwinVariant("swin_b_k400_1k", f"{_STORAGE}/swin_base_patch244_window877_kinetics400_1k.pth",
                    **_BASE, window_size=(8, 7, 7), drop_path_rate=0.3, num_classes=400,
                    pretrained_on="ImageNet-1K -> Kinetics-400",
                    recommended_lr=1e-3, recommended_weight_decay=0.05),
        SwinVariant("swin_b_k400_22k", f"{_STORAGE}/swin_base_patch244_window877_kinetics400_22k.pth",
                    **_BASE, window_size=(8, 7, 7), drop_path_rate=0.2, num_classes=400,
                    pretrained_on="ImageNet-22K -> Kinetics-400",
                    notes="Default for the thesis (Swin-B, best Kinetics-400 accuracy)."),
        SwinVariant("swin_b_k600_22k", f"{_STORAGE}/swin_base_patch244_window877_kinetics600_22k.pth",
                    **_BASE, window_size=(8, 7, 7), drop_path_rate=0.2, num_classes=600,
                    pretrained_on="ImageNet-22K -> Kinetics-600"),
        SwinVariant("swin_b_k400_22k_w1677", f"{_STORAGE}/swin_base_patch244_window1677_kinetics400_22k.pth",
                    **_BASE, window_size=(16, 7, 7), drop_path_rate=0.4, num_classes=400,
                    pretrained_on="ImageNet-22K -> Kinetics-400 (temporal window 16)",
                    notes="Starting point used by the authors for Something-Something v2."),
        SwinVariant("swin_b_sthv2", f"{_STORAGE}/swin_base_patch244_window1677_sthv2.pth",
                    **_BASE, window_size=(16, 7, 7), drop_path_rate=0.4, num_classes=174,
                    pretrained_on="Kinetics-400 -> Something-Something v2"),
    )
}

DEFAULT_VARIANT = "swin_b_k400_22k"


def get_variant(name: str | SwinVariant) -> SwinVariant:
    if isinstance(name, SwinVariant):
        return name
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"unknown variant {name!r}; available: {', '.join(VARIANTS)}") from None


def list_variants() -> list[str]:
    return list(VARIANTS)
