"""Download and load the official Video Swin checkpoints without mmcv.

The official ``.pth`` files were saved by mmcv: a dict with ``meta`` and
``state_dict``, where the keys start with ``backbone.`` (SwinTransformer3D) and
``cls_head.`` (I3DHead: ``cls_head.fc_cls.weight`` / ``.bias``). ``VideoSwin``
uses the same names, so the weights load without renaming.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .colab_setup import cache_dir
from .variants import SwinVariant, get_variant

HEAD_PREFIX = "cls_head."
BACKBONE_PREFIX = "backbone."


@dataclass
class LoadReport:
    source: str
    loaded: int
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    dropped_head: list[str] = field(default_factory=list)

    @property
    def backbone_complete(self) -> bool:
        return not any(k.startswith(BACKBONE_PREFIX) for k in self.missing)

    def __str__(self) -> str:
        lines = [f"Loaded {self.loaded} tensors from {self.source}"]
        if self.dropped_head:
            lines.append(f"  head not loaded (new number of classes): {', '.join(self.dropped_head)}")
        if self.missing:
            lines.append(f"  missing in checkpoint ({len(self.missing)}): {_short(self.missing)}")
        if self.unexpected:
            lines.append(f"  not used from checkpoint ({len(self.unexpected)}): {_short(self.unexpected)}")
        lines.append(f"  backbone complete: {'yes' if self.backbone_complete else 'NO'}")
        return "\n".join(lines)


def download_checkpoint(variant: str | SwinVariant, directory: str | Path | None = None,
                        progress: bool = True) -> Path:
    """Download the ``.pth`` of ``variant`` once and return its local path."""
    v = get_variant(variant)
    target = cache_dir(directory) / v.filename
    if not target.exists() or target.stat().st_size == 0:
        torch.hub.download_url_to_file(v.url, str(target), progress=progress)
    return target


def read_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Return the tensors of a checkpoint as a flat ``state_dict``.

    Accepts mmcv files (``state_dict``), files saved by ``vswin.fine_tuning``
    (``model``) and plain state dicts. Removes the ``module.`` prefix of
    DataParallel / DistributedDataParallel.
    """
    # The official files keep mmcv metadata in ``meta``, which weights_only=True rejects.
    obj: Any = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], Mapping):
                obj = obj[key]
                break
    if not isinstance(obj, Mapping):
        raise ValueError(f"{path}: no state_dict found")
    return {(k[7:] if k.startswith("module.") else k): v
            for k, v in obj.items() if isinstance(v, torch.Tensor)}


def load_weights(model: nn.Module, state: Mapping[str, torch.Tensor], source: str = "state_dict",
                 load_head: bool = True) -> LoadReport:
    """Copy ``state`` into ``model``.

    The head is skipped when ``load_head`` is false or when its shape differs
    (for example 400 Kinetics classes vs. 2 thesis classes). Buffers that the
    backbone rebuilds (``relative_position_index``) are ignored if missing.
    """
    own = model.state_dict()
    state = dict(state)
    dropped = []
    for k in [k for k in state if k.startswith(HEAD_PREFIX)]:
        if not load_head or k not in own or own[k].shape != state[k].shape:
            dropped.append(k)
            del state[k]
    mismatched = [k for k in state if k in own and own[k].shape != state[k].shape]
    if mismatched:
        raise ValueError("shape mismatch between checkpoint and model (wrong variant?): "
                         + ", ".join(f"{k} {tuple(state[k].shape)} vs {tuple(own[k].shape)}" for k in mismatched))
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if "relative_position_index" not in k]
    return LoadReport(
        source=source,
        loaded=len(state) - len(result.unexpected_keys),
        missing=missing,
        unexpected=list(result.unexpected_keys),
        dropped_head=dropped,
    )


def load_pretrained(model: nn.Module, variant: str | SwinVariant | None = None,
                    path: str | Path | None = None, load_head: bool = True,
                    directory: str | Path | None = None) -> LoadReport:
    """Load an official checkpoint (``variant``, downloaded if needed) or a local ``path``."""
    if path is None:
        if variant is None:
            raise ValueError("pass a variant or a checkpoint path")
        path = download_checkpoint(variant, directory)
    report = load_weights(model, read_state_dict(path), source=str(path), load_head=load_head)
    if not report.backbone_complete:
        raise RuntimeError(f"the backbone was not fully loaded:\n{report}")
    return report


def _short(keys: list[str], n: int = 6) -> str:
    return ", ".join(keys[:n]) + (" …" if len(keys) > n else "")
