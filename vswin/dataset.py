"""PyTorch datasets over the preprocessing output (``salida/<dataset>/``).

Two modes:

* ``"clip"`` (training): one sample = one miniclip, uint8 ``(T, H, W, 3)``.
  Videos with many miniclips would dominate an epoch; ``make_clip_sampler``
  gives every video (and optionally every class) the same weight.
* ``"video"`` (evaluation): one sample = all K miniclips of a video,
  uint8 ``(K, T, H, W, 3)``. K changes between videos, so use batch size 1
  (``make_loaders`` does it).

Samples stay uint8; normalize on the GPU with ``vswin.swin_input.to_model_input``.
"""
from __future__ import annotations

import csv
import json
import random
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .miniclip_reading import fit_clip_length, list_miniclips, read_miniclip
from .swin_input import CLIP_LEN

LabelValue = Union[int, str]
LabelSource = Union[Mapping[str, LabelValue], str, Path, Callable[[dict], "LabelValue | None"]]


@dataclass
class VideoRecord:
    video_id: str
    clip_paths: list[Path]
    label: int
    rel_path: str | None = None
    fallback: str | None = None
    size: tuple[int, int] | None = None  # (width, height)
    meta: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_clips(self) -> int:
        return len(self.clip_paths)


# --------------------------------------------------------------------------- labels

def read_label_csv(path: str | Path, id_column: str | None = None, label_column: str = "label") -> dict[str, str]:
    """CSV with a video id column (``video_id`` or ``id``) and a ``label`` column."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        idc = id_column or next((c for c in ("video_id", "id") if c in cols), None)
        if idc is None or label_column not in cols:
            raise ValueError(f"{path}: expected columns 'video_id' (or 'id') and '{label_column}', found {cols}")
        return {row[idc].strip(): row[label_column].strip() for row in reader if row[idc].strip()}


def label_from_path(rules: Mapping[str, int]) -> Callable[[dict], int | None]:
    """Label from the original relative path stored in ``meta["ruta"]``.

    ``rules`` maps a case-insensitive text to a class, e.g.
    ``{"NonPorn": 0, "Porn": 1}``. Longer texts are tested first, so
    ``NonPorn`` wins over ``Porn``.
    """
    ordered = sorted(rules.items(), key=lambda kv: len(kv[0]), reverse=True)

    def label(meta: dict) -> int | None:
        path = str(meta.get("ruta") or "").replace("\\", "/").lower()
        for text, cls in ordered:
            if text.lower() in path:
                return cls
        return None

    return label


def _label_getter(labels: LabelSource) -> Callable[[str, dict], LabelValue | None]:
    if callable(labels):
        return lambda vid, meta: labels(meta)
    table = read_label_csv(labels) if isinstance(labels, (str, Path)) else dict(labels)
    return lambda vid, meta: table.get(vid)


def _to_index(value: LabelValue, class_names: Sequence[str] | None) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if class_names is not None:
        if text not in class_names:
            raise ValueError(f"label {text!r} is not in class_names {list(class_names)}")
        return list(class_names).index(text)
    if text.lstrip("-").isdigit():
        return int(text)
    raise ValueError(f"label {text!r} is text: pass class_names to map it to an index")


# --------------------------------------------------------------------------- index

def index_dataset(root: str | Path, labels: LabelSource, class_names: Sequence[str] | None = None,
                  skip_unlabeled: bool = True) -> list[VideoRecord]:
    """List the videos of one preprocessed dataset with their miniclips and labels.

    ``root`` is ``salida/<dataset>`` (it contains ``miniclips/`` and ``meta/``).
    ``labels``: dict ``{video_id: label}``, a CSV path, or a function
    ``meta -> label`` (see ``label_from_path``). Labels are ints or names in ``class_names``.
    """
    root = Path(root)
    clips_root, meta_root = root / "miniclips", root / "meta"
    if not clips_root.is_dir():
        raise FileNotFoundError(f"{clips_root} does not exist (root must be salida/<dataset>)")
    get_label = _label_getter(labels)
    records, unlabeled, empty = [], [], []
    for clip_dir in sorted(p for p in clips_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        vid = clip_dir.name
        meta_path = meta_root / f"{vid}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        out = meta.get("salida") or {}
        if out.get("archivos"):
            paths = [root / rel for rel in out["archivos"]]
        else:
            paths = list_miniclips(clip_dir)
        paths = [p for p in paths if p.exists()]
        if not paths:
            empty.append(vid)
            continue
        value = get_label(vid, meta)
        if value is None or value == "":
            unlabeled.append(vid)
            continue
        size = (int(out["ancho"]), int(out["alto"])) if out.get("ancho") and out.get("alto") else None
        records.append(VideoRecord(
            video_id=vid,
            clip_paths=paths,
            label=_to_index(value, class_names),
            rel_path=meta.get("ruta"),
            fallback=(meta.get("respaldo") or {}).get("caso"),
            size=size,
            meta=meta,
        ))
    if empty:
        warnings.warn(f"{len(empty)} videos without miniclips were skipped (e.g. {empty[:3]})")
    if unlabeled:
        msg = f"{len(unlabeled)} videos without label (e.g. {unlabeled[:3]})"
        if not skip_unlabeled:
            raise ValueError(msg)
        warnings.warn(msg + " were skipped")
    return records


def describe_records(records: Sequence[VideoRecord], class_names: Sequence[str] | None = None) -> str:
    videos = Counter(r.label for r in records)
    clips = Counter()
    for r in records:
        clips[r.label] += r.num_clips
    ks = [r.num_clips for r in records]
    lines = [f"{len(records)} videos, {sum(ks)} miniclips (K min/median/max = "
             f"{min(ks) if ks else 0}/{int(np.median(ks)) if ks else 0}/{max(ks) if ks else 0})"]
    for c in sorted(videos):
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        lines.append(f"  class {name}: {videos[c]} videos, {clips[c]} miniclips")
    fallbacks = Counter(r.fallback for r in records if r.fallback)
    if fallbacks:
        lines.append(f"  fallback cases: {dict(fallbacks)}")
    return "\n".join(lines)


def split_records(records: Sequence[VideoRecord], val_fraction: float = 0.2, seed: int = 0,
                  stratify: bool = True) -> tuple[list[VideoRecord], list[VideoRecord]]:
    """Split by video (never by miniclip), keeping the class proportions if ``stratify``."""
    rng = random.Random(seed)
    groups: dict[int, list[VideoRecord]] = defaultdict(list)
    for r in records:
        groups[r.label if stratify else 0].append(r)
    train, val = [], []
    for key in sorted(groups):
        items = sorted(groups[key], key=lambda r: r.video_id)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def select_records(records: Sequence[VideoRecord], video_ids: Sequence[str] | str | Path) -> list[VideoRecord]:
    """Keep the given ids (a list, or a text file with one id per line), for predefined folds."""
    if isinstance(video_ids, (str, Path)):
        video_ids = _read_ids(video_ids)
    wanted = set(video_ids)
    return [r for r in records if r.video_id in wanted]


# --------------------------------------------------------------------------- fixed train / val / test split

SPLIT_NAMES = ("train", "val", "test")
SPLIT_INFO = "split_info.json"


def split_train_val_test(records: Sequence[VideoRecord], val_fraction: float = 0.15, test_fraction: float = 0.15,
                         seed: int = 0) -> tuple[list[VideoRecord], list[VideoRecord], list[VideoRecord]]:
    """Stratified split by video into train / val / test.

    Within each class the videos are sorted by id and shuffled with ``seed``;
    the first ``test_fraction`` go to test, the next ``val_fraction`` to val
    and the rest to train. Save the result with ``save_split`` and reuse it
    with ``load_split``: the saved files, not the seed, define the split.
    """
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >= 0 and add up to less than 1")
    _check_unique(records)
    rng = random.Random(seed)
    groups: dict[int, list[VideoRecord]] = defaultdict(list)
    for r in records:
        groups[r.label].append(r)
    train, val, test = [], [], []
    for label in sorted(groups):
        items = sorted(groups[label], key=lambda r: r.video_id)
        rng.shuffle(items)
        n_test = int(round(len(items) * test_fraction))
        n_val = int(round(len(items) * val_fraction))
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    return train, val, test


def save_split(directory: str | Path, train: Sequence[VideoRecord], val: Sequence[VideoRecord],
               test: Sequence[VideoRecord], overwrite: bool = False, **info: Any) -> Path:
    """Write ``train.txt``, ``val.txt``, ``test.txt`` (one video id per line) and ``split_info.json``.

    Refuses to overwrite an existing split unless ``overwrite=True``, so that
    every training run keeps using the same videos. ``info`` (e.g. seed and
    fractions) is stored in ``split_info.json``.
    """
    d = Path(directory)
    if not overwrite and any((d / f"{n}.txt").exists() for n in SPLIT_NAMES):
        raise FileExistsError(f"a split already exists in {d}; load it with load_split or pass overwrite=True")
    parts = dict(zip(SPLIT_NAMES, (train, val, test)))
    _check_disjoint({n: [r.video_id for r in rs] for n, rs in parts.items()})
    d.mkdir(parents=True, exist_ok=True)
    for name, rs in parts.items():
        ids = sorted(r.video_id for r in rs)
        (d / f"{name}.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    summary = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        **info,
        "counts": {n: _class_counts(rs) for n, rs in parts.items()},
    }
    (d / SPLIT_INFO).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return d


def load_split(records: Sequence[VideoRecord], directory: str | Path, strict: bool = True
               ) -> tuple[list[VideoRecord], list[VideoRecord], list[VideoRecord]]:
    """Rebuild train / val / test from the ``.txt`` files of ``save_split`` (or official folds).

    A missing file gives an empty list (e.g. folds with only train and test).
    Ids listed in a file but absent from ``records`` (a video that failed or
    was not copied) raise an error with ``strict=True``, and only a warning
    otherwise. Videos in ``records`` that belong to no file are reported and left out.
    """
    d = Path(directory)
    files = {n: d / f"{n}.txt" for n in SPLIT_NAMES}
    if not any(p.exists() for p in files.values()):
        raise FileNotFoundError(f"no train.txt / val.txt / test.txt in {d}")
    _check_unique(records)
    ids = {n: (_read_ids(p) if p.exists() else []) for n, p in files.items()}
    _check_disjoint(ids)
    by_id = {r.video_id: r for r in records}
    missing = {n: [i for i in lst if i not in by_id] for n, lst in ids.items()}
    missing = {n: lst for n, lst in missing.items() if lst}
    if missing:
        msg = "; ".join(f"{n}: {len(lst)} (e.g. {lst[:3]})" for n, lst in missing.items())
        msg = f"videos of the saved split not found in the dataset -> {msg}"
        if strict:
            raise ValueError(msg + " (strict=False skips them, but the split is no longer the same)")
        warnings.warn(msg)
    listed = set().union(*map(set, ids.values()))
    unused = [r.video_id for r in records if r.video_id not in listed]
    if unused:
        warnings.warn(f"{len(unused)} videos of the dataset are not in the split and are not used (e.g. {unused[:3]})")
    out = tuple([by_id[i] for i in ids[n] if i in by_id] for n in SPLIT_NAMES)
    return out  # type: ignore[return-value]


def get_or_create_split(records: Sequence[VideoRecord], directory: str | Path, val_fraction: float = 0.15,
                        test_fraction: float = 0.15, seed: int = 0, strict: bool = True
                        ) -> tuple[list[VideoRecord], list[VideoRecord], list[VideoRecord]]:
    """Load the split saved in ``directory``; create and save it only the first time."""
    d = Path(directory)
    if any((d / f"{n}.txt").exists() for n in SPLIT_NAMES):
        print(f"Using the saved split in {d}")
        return load_split(records, d, strict=strict)
    train, val, test = split_train_val_test(records, val_fraction, test_fraction, seed)
    save_split(d, train, val, test, seed=seed, val_fraction=val_fraction, test_fraction=test_fraction,
               total_videos=len(records))
    print(f"Created and saved a new split in {d}")
    return train, val, test


def _read_ids(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _check_unique(records: Sequence[VideoRecord]) -> None:
    dup = [i for i, n in Counter(r.video_id for r in records).items() if n > 1]
    if dup:
        raise ValueError(f"repeated video ids in records: {dup[:5]}")


def _check_disjoint(ids: Mapping[str, Sequence[str]]) -> None:
    names = list(ids)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            common = set(ids[names[a]]) & set(ids[names[b]])
            if common:
                raise ValueError(f"{len(common)} videos are in both {names[a]} and {names[b]} "
                                 f"(e.g. {sorted(common)[:3]})")


def _class_counts(records: Sequence[VideoRecord]) -> dict[str, int]:
    c = Counter(r.label for r in records)
    return {"videos": len(records), "miniclips": sum(r.num_clips for r in records),
            **{f"class_{k}": v for k, v in sorted(c.items())}}


# --------------------------------------------------------------------------- dataset

class MiniclipDataset(Dataset):
    def __init__(self, records: Sequence[VideoRecord], mode: str = "clip", clip_len: int = CLIP_LEN,
                 flip_prob: float = 0.0, max_clips: int | None = None,
                 transform: Callable[[np.ndarray], np.ndarray] | None = None):
        """
        ``flip_prob``: horizontal flip probability per miniclip (the official
        training pipeline uses 0.5; keep 0 for evaluation).
        ``max_clips`` (video mode): at most this many miniclips per video,
        evenly spaced in time, to bound GPU memory.
        ``transform``: extra function on the uint8 ``(T, H, W, 3)`` array.
        """
        if mode not in ("clip", "video"):
            raise ValueError("mode must be 'clip' or 'video'")
        self.records = list(records)
        self.mode = mode
        self.clip_len = clip_len
        self.flip_prob = flip_prob
        self.max_clips = max_clips
        self.transform = transform
        self.items = ([(v, c) for v, r in enumerate(self.records) for c in range(r.num_clips)]
                      if mode == "clip" else [(v, -1) for v in range(len(self.records))])

    def __len__(self) -> int:
        return len(self.items)

    @property
    def labels(self) -> list[int]:
        return [self.records[v].label for v, _ in self.items]

    def _load(self, record: VideoRecord, clip_index: int) -> np.ndarray:
        clip = read_miniclip(record.clip_paths[clip_index], size=record.size)
        clip = fit_clip_length(clip, self.clip_len)
        if self.flip_prob > 0 and random.random() < self.flip_prob:
            clip = clip[:, :, ::-1]
        if self.transform is not None:
            clip = self.transform(clip)
        return np.ascontiguousarray(clip)

    def _clip_indices(self, record: VideoRecord) -> list[int]:
        k = record.num_clips
        if self.max_clips is None or k <= self.max_clips:
            return list(range(k))
        return sorted({int(i) for i in np.linspace(0, k - 1, self.max_clips).round()})

    def __getitem__(self, i: int) -> dict[str, Any]:
        v, c = self.items[i]
        r = self.records[v]
        if self.mode == "clip":
            return {"clip": torch.from_numpy(self._load(r, c)), "label": r.label, "video": v, "clip_index": c}
        clips = np.stack([self._load(r, j) for j in self._clip_indices(r)])
        return {"clips": torch.from_numpy(clips), "label": r.label, "video": v, "video_id": r.video_id}


def collate_video(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Video mode works with one video per batch (K differs between videos)."""
    if len(batch) != 1:
        raise ValueError("video mode needs batch_size=1")
    return batch[0]


def clip_sample_weights(dataset: MiniclipDataset, balance_classes: bool = True) -> torch.Tensor:
    """Weight per miniclip: 1/K of its video, times 1/(videos of its class) if ``balance_classes``."""
    if dataset.mode != "clip":
        raise ValueError("weights are defined for clip mode")
    per_class = Counter(r.label for r in dataset.records)
    w = []
    for v, _ in dataset.items:
        r = dataset.records[v]
        weight = 1.0 / r.num_clips
        if balance_classes:
            weight /= per_class[r.label]
        w.append(weight)
    return torch.tensor(w, dtype=torch.double)


def make_clip_sampler(dataset: MiniclipDataset, num_samples: int | None = None, balance_classes: bool = True,
                      seed: int | None = None) -> WeightedRandomSampler:
    """Sampler with replacement; ``num_samples`` per epoch defaults to the number of miniclips."""
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(clip_sample_weights(dataset, balance_classes),
                                 num_samples=num_samples or len(dataset), replacement=True, generator=gen)


def make_loaders(train_records: Sequence[VideoRecord], val_records: Sequence[VideoRecord] | None = None,
                 batch_size: int = 2, num_workers: int = 2, flip_prob: float = 0.5,
                 balanced_sampling: bool = True, samples_per_epoch: int | None = None,
                 val_max_clips: int | None = None, seed: int = 0) -> tuple[DataLoader, DataLoader | None]:
    """Training loader (clip mode) and validation loader (video mode, batch 1)."""
    train_ds = MiniclipDataset(train_records, mode="clip", flip_prob=flip_prob)
    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  persistent_workers=num_workers > 0)
    if balanced_sampling:
        sampler = make_clip_sampler(train_ds, samples_per_epoch, balance_classes=True, seed=seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True, **common)
    else:
        gen = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                  generator=gen, **common)
    val_loader = None
    if val_records:
        val_loader = make_eval_loader(val_records, num_workers=num_workers, max_clips=val_max_clips)
    return train_loader, val_loader


def make_eval_loader(records: Sequence[VideoRecord], num_workers: int = 2,
                     max_clips: int | None = None) -> DataLoader:
    """Video-mode loader (one video per batch, fixed order, no augmentation) for val or test."""
    ds = MiniclipDataset(records, mode="video", max_clips=max_clips)
    return DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_video, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
