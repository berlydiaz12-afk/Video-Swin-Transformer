"""Read the miniclips written by the preprocessing package (``preproc``).

Layout of one dataset (``salida/<dataset>/``)::

    miniclips/<video_id>/<video_id>_000.mp4, _001.mp4, ...   K miniclips in temporal order
    meta/<video_id>.json                                     metadata (keys in Spanish)

Each ``.mp4`` is H.264 yuv420p, BT.601 limited range, 32 frames of 224×224.
ffmpeg decodes it to RGB uint8 ``(T, H, W, 3)``, the same way ``preproc`` does.
"""
from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CLIP_NAME = re.compile(r"_(\d+)\.mp4$")


class FfmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class Ffmpeg:
    ffmpeg: str
    ffprobe: str
    major: int

    @property
    def passthrough_args(self) -> list[str]:
        """Keep every decoded frame (no duplication or dropping)."""
        return ["-fps_mode", "passthrough"] if self.major >= 5 else ["-vsync", "passthrough"]


@functools.lru_cache(maxsize=None)
def find_ffmpeg() -> Ffmpeg:
    """Locate ffmpeg and ffprobe (Colab ships both; elsewhere ``apt install ffmpeg``)."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FfmpegError("ffmpeg/ffprobe not found in PATH (in Colab: !apt-get -y install ffmpeg)")
    out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True).stdout
    m = re.search(r"version\s+n?(\d+)", out)
    return Ffmpeg(ffmpeg, ffprobe, int(m.group(1)) if m else 4)


def probe_size(path: str | Path, ff: Ffmpeg | None = None) -> tuple[int, int]:
    """(width, height) of the first video stream."""
    ff = ff or find_ffmpeg()
    res = subprocess.run([ff.ffprobe, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "json", str(path)],
                         stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if res.returncode != 0:
        raise FfmpegError(f"ffprobe could not read {path}: {res.stderr.strip()[-300:]}")
    streams = json.loads(res.stdout or "{}").get("streams") or []
    if not streams:
        raise FfmpegError(f"{path}: no video stream")
    return int(streams[0]["width"]), int(streams[0]["height"])


def read_miniclip(path: str | Path, size: tuple[int, int] | None = None,
                  ff: Ffmpeg | None = None) -> np.ndarray:
    """Decode one miniclip ``.mp4`` as RGB uint8 ``(T, H, W, 3)``.

    ``size = (width, height)`` skips the ffprobe call when it is already known
    (224×224 for the preprocessing output).
    """
    ff = ff or find_ffmpeg()
    width, height = size or probe_size(path, ff)
    cmd = [ff.ffmpeg, "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
           "-an", "-sn", "-dn", *ff.passthrough_args,
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True)
    if res.returncode != 0:
        raise FfmpegError(f"ffmpeg could not decode {path}: {res.stderr.decode('utf-8', 'replace')[-300:]}")
    frame_bytes = width * height * 3
    if not res.stdout or len(res.stdout) % frame_bytes:
        raise FfmpegError(f"{path}: decoded {len(res.stdout)} bytes, not a multiple of a {width}×{height} frame")
    # bytearray makes the array writable (torch.from_numpy warns on read-only buffers)
    return np.frombuffer(bytearray(res.stdout), dtype=np.uint8).reshape(-1, height, width, 3)


def list_miniclips(clip_dir: str | Path) -> list[Path]:
    """``.mp4`` files of ``miniclips/<id>/`` sorted by their index (temporal order)."""
    paths = list(Path(clip_dir).glob("*.mp4"))

    def index(p: Path) -> tuple[int, str]:
        m = _CLIP_NAME.search(p.name)
        return (int(m.group(1)) if m else 1 << 30, p.name)

    return sorted(paths, key=index)


def read_miniclips(clip_dir_or_paths: str | Path | list[str | Path], size: tuple[int, int] | None = None,
                   ff: Ffmpeg | None = None) -> np.ndarray:
    """All miniclips of one video as uint8 ``(K, T, H, W, 3)``."""
    if isinstance(clip_dir_or_paths, (str, Path)):
        paths = list_miniclips(clip_dir_or_paths)
    else:
        paths = [Path(p) for p in clip_dir_or_paths]
    if not paths:
        raise FileNotFoundError(f"no .mp4 miniclips in {clip_dir_or_paths}")
    ff = ff or find_ffmpeg()
    size = size or probe_size(paths[0], ff)
    clips = [read_miniclip(p, size, ff) for p in paths]
    lengths = {len(c) for c in clips}
    if len(lengths) != 1:
        raise FfmpegError(f"miniclips with different frame counts {sorted(lengths)} in {paths[0].parent}")
    return np.stack(clips)


def fit_clip_length(clip: np.ndarray, clip_len: int) -> np.ndarray:
    """Return exactly ``clip_len`` frames: uniform subsampling if longer, repeating frames in order if shorter."""
    n = len(clip)
    if n == clip_len:
        return clip
    idx = np.floor(np.arange(clip_len) * n / clip_len).astype(np.int64)
    return clip[np.clip(idx, 0, n - 1)]
