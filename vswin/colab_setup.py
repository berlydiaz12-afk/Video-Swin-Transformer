"""Colab helpers: mount Google Drive, inspect the GPU and choose the cache folder."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive"
_ENV_CACHE = "VSWIN_CACHE"


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def mount_drive(force_remount: bool = False) -> Path | None:
    """Mount Google Drive in Colab and return ``/content/drive/MyDrive`` (``None`` outside Colab)."""
    if not in_colab():
        return None
    if not DRIVE_ROOT.exists() or force_remount:
        from google.colab import drive
        drive.mount(str(DRIVE_MOUNT), force_remount=force_remount)
    return DRIVE_ROOT


@dataclass(frozen=True)
class GpuInfo:
    available: bool
    name: str
    total_gb: float
    capability: tuple[int, int]
    bf16: bool

    def __str__(self) -> str:
        if not self.available:
            return "No GPU (Runtime > Change runtime type > GPU)"
        return (f"{self.name}: {self.total_gb:.1f} GB, compute capability "
                f"{self.capability[0]}.{self.capability[1]}, bf16={'yes' if self.bf16 else 'no'}")


def gpu_info() -> GpuInfo:
    if not torch.cuda.is_available():
        return GpuInfo(False, "", 0.0, (0, 0), False)
    props = torch.cuda.get_device_properties(0)
    return GpuInfo(
        available=True,
        name=props.name,
        total_gb=props.total_memory / 1024**3,
        capability=(props.major, props.minor),
        bf16=torch.cuda.is_bf16_supported(),
    )


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype() -> torch.dtype:
    """bfloat16 on Ampere or newer (A100, L4), float16 otherwise (T4)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def cache_dir(path: str | Path | None = None) -> Path:
    """Folder for downloaded checkpoints.

    Order: ``path`` argument, ``$VSWIN_CACHE``, ``MyDrive/vswin_cache`` if Drive is
    mounted (survives runtime resets), and ``~/.cache/vswin`` otherwise.
    """
    if path is None:
        env = os.environ.get(_ENV_CACHE)
        if env:
            path = env
        elif DRIVE_ROOT.exists():
            path = DRIVE_ROOT / "vswin_cache"
        else:
            path = Path.home() / ".cache" / "vswin"
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def describe_environment() -> str:
    lines = [
        f"torch {torch.__version__} (CUDA {torch.version.cuda})",
        str(gpu_info()),
        f"Colab: {'yes' if in_colab() else 'no'}",
        f"Drive mounted: {'yes' if DRIVE_ROOT.exists() else 'no'}",
        f"Checkpoint cache: {cache_dir()}",
    ]
    return "\n".join(lines)
