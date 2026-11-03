"""Head-only training on cached backbone embeddings.

With the whole backbone frozen, the 1024-d embedding of a miniclip (output of
``backbone.norm`` + spatio-temporal average pooling) never changes. It is
computed once per miniclip and per view (original and horizontally flipped),
saved to disk, and the head is then trained on those vectors: an epoch takes
seconds instead of a full Swin-B forward pass over the dataset.

Cache file (``torch.save``)::

    features     (N, V, D) float16   N miniclips, V views (0 = original, 1 = flip), D = 1024
    clip_video   (N,)      int64     index of the video of each miniclip
    video_ids    list[str]           M videos
    video_labels (M,)      int64
    meta         dict                variant, checkpoint, views, creation time

The trained head is saved inside a full ``best.pth``, so
``build_model(..., pretrained=best.pth)`` and ``predict_video`` work as usual.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import MiniclipDataset, VideoRecord, collate_video
from .fine_tuning import _format_row, classification_metrics, make_criterion, save_checkpoint, set_seed, warmup_cosine
from .inference import autocast_context
from .model import logits_to_probs
from .swin_input import to_model_input

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore[no-redef]
        return it

CACHE_VERSION = 1
VIEW_NAMES = ("original", "hflip")


# --------------------------------------------------------------------------- feature set

@dataclass
class FeatureSet:
    features: torch.Tensor          # (N, V, D) float16
    clip_video: torch.Tensor        # (N,) int64
    video_ids: list[str]
    video_labels: torch.Tensor      # (M,) int64
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_clips(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_videos(self) -> int:
        return len(self.video_ids)

    @property
    def num_views(self) -> int:
        return int(self.features.shape[1])

    @property
    def dim(self) -> int:
        return int(self.features.shape[2])

    @property
    def clip_labels(self) -> torch.Tensor:
        return self.video_labels[self.clip_video]

    def clips_per_video(self) -> torch.Tensor:
        return torch.bincount(self.clip_video, minlength=self.num_videos)

    def to(self, device: str | torch.device) -> "FeatureSet":
        return FeatureSet(self.features.to(device), self.clip_video.to(device), list(self.video_ids),
                          self.video_labels.to(device), dict(self.meta))

    def __repr__(self) -> str:
        mb = self.features.numel() * self.features.element_size() / 1024**2
        return (f"FeatureSet({self.num_videos} videos, {self.num_clips} miniclips, "
                f"{self.num_views} views, dim {self.dim}, {mb:.1f} MB)")


def save_features(fs: FeatureSet, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"version": CACHE_VERSION, "features": fs.features.cpu(), "clip_video": fs.clip_video.cpu(),
            "video_ids": fs.video_ids, "video_labels": fs.video_labels.cpu(), "meta": fs.meta}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)
    return path


def load_features(path: str | Path) -> FeatureSet:
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    if data.get("version") != CACHE_VERSION:
        raise ValueError(f"{path}: cache version {data.get('version')} != {CACHE_VERSION}; rebuild it")
    return FeatureSet(data["features"], data["clip_video"], list(data["video_ids"]), data["video_labels"],
                      data.get("meta", {}))


def model_signature(model: nn.Module) -> dict[str, Any]:
    """What the embeddings depend on: variant and checkpoint file of the backbone."""
    variant = getattr(model, "variant", None)
    report = getattr(model, "load_report", None)
    source = getattr(report, "source", None)
    return {
        "variant": variant.name if variant is not None else None,
        "checkpoint": Path(source).name if source else None,
        "feature_dim": getattr(model, "feature_dim", None),
    }


# --------------------------------------------------------------------------- extraction

@torch.no_grad()
def extract_record_features(model: nn.Module, records: Sequence[VideoRecord], flips: Sequence[bool] = (False, True),
                            batch_size: int = 8, amp: bool = True, num_workers: int = 2,
                            partial_path: str | Path | None = None, save_every: int = 200,
                            progress: bool = True) -> FeatureSet:
    """Embeddings of every miniclip of ``records`` for each view in ``flips``.

    Runs the backbone in eval mode (no drop path). With ``partial_path``, the
    finished videos are saved every ``save_every`` videos, and a later call
    continues from there (useful if Colab disconnects).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    records = list(records)
    flips = [bool(f) for f in flips]
    done: dict[str, torch.Tensor] = {}
    if partial_path is not None and Path(partial_path).exists():
        state = torch.load(str(partial_path), map_location="cpu", weights_only=False)
        if state.get("flips") == flips and state.get("signature") == model_signature(model):
            done = state["per_video"]
            print(f"Resuming extraction: {len(done)} videos already done")
    todo = [r for r in records if r.video_id not in done]
    try:
        if todo:
            loader = DataLoader(MiniclipDataset(todo, mode="video"), batch_size=1, shuffle=False,
                                collate_fn=collate_video, num_workers=num_workers,
                                pin_memory=torch.cuda.is_available())
            bar = tqdm(loader, desc="embeddings", total=len(todo), disable=not progress)
            for n, sample in enumerate(bar, 1):
                clips = sample["clips"]                      # (K, T, H, W, 3) uint8
                views = []
                for flip in flips:
                    v = torch.flip(clips, dims=[3]) if flip else clips   # dim 3 = width
                    feats = []
                    for i in range(0, len(v), batch_size):
                        x = to_model_input(v[i:i + batch_size], device=device)
                        with autocast_context(device, amp):
                            feats.append(model.forward_features(x, pooled=True).float().cpu())
                    views.append(torch.cat(feats))
                done[sample["video_id"]] = torch.stack(views, dim=1).half()   # (K, V, D)
                if partial_path is not None and n % save_every == 0:
                    _save_partial(partial_path, done, flips, model)
    finally:
        model.train(was_training)

    dim = getattr(model, "feature_dim", None) or (next(iter(done.values())).shape[-1] if done else 1024)
    feats = [done[r.video_id] for r in records]
    clip_video = [torch.full((len(f),), i, dtype=torch.long) for i, f in enumerate(feats)]
    return FeatureSet(
        features=torch.cat(feats) if feats else torch.empty(0, len(flips), dim, dtype=torch.float16),
        clip_video=torch.cat(clip_video) if clip_video else torch.empty(0, dtype=torch.long),
        video_ids=[r.video_id for r in records],
        video_labels=torch.tensor([r.label for r in records], dtype=torch.long),
        meta={**model_signature(model), "views": [VIEW_NAMES[int(f)] for f in flips],
              "created": time.strftime("%Y-%m-%d %H:%M:%S")},
    )


def get_or_extract_features(model: nn.Module, records: Sequence[VideoRecord], path: str | Path,
                            flips: Sequence[bool] = (False, True), rebuild: bool = False,
                            **kwargs: Any) -> FeatureSet:
    """Load the cache at ``path`` if it matches ``records`` and ``model``; otherwise extract and save it.

    Labels always come from ``records`` (they can change without re-extracting).
    A cache made for other videos or another checkpoint raises an error;
    pass ``rebuild=True`` to replace it.
    """
    path = Path(path)
    if path.exists() and not rebuild:
        fs = load_features(path)
        _check_cache(fs, model, records, flips, path)
        print(f"Loaded {fs} from {path}")
        return _relabel(fs, records)
    partial = path.with_suffix(".partial.pt")
    start = time.perf_counter()
    fs = extract_record_features(model, records, flips, partial_path=partial, **kwargs)
    fs.meta["extraction_seconds"] = round(time.perf_counter() - start, 1)
    save_features(fs, path)
    partial.unlink(missing_ok=True)
    print(f"Saved {fs} to {path} ({fs.meta['extraction_seconds'] / 60:.1f} min)")
    return fs


def _save_partial(path: str | Path, done: dict[str, torch.Tensor], flips: list[bool], model: nn.Module) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"flips": flips, "signature": model_signature(model), "per_video": done}, tmp)
    tmp.replace(path)


def _check_cache(fs: FeatureSet, model: nn.Module, records: Sequence[VideoRecord], flips: Sequence[bool],
                 path: Path) -> None:
    problems = []
    sig = model_signature(model)
    for key in ("variant", "checkpoint"):
        if fs.meta.get(key) != sig.get(key):
            problems.append(f"{key}: cache {fs.meta.get(key)!r} vs model {sig.get(key)!r}")
    views = [VIEW_NAMES[int(bool(f))] for f in flips]
    if fs.meta.get("views") != views:
        problems.append(f"views: cache {fs.meta.get('views')} vs requested {views}")
    cached, wanted = set(fs.video_ids), {r.video_id for r in records}
    if cached != wanted:
        problems.append(f"videos: {len(wanted - cached)} missing in cache, {len(cached - wanted)} extra")
    if problems:
        raise ValueError(f"{path} does not match: " + "; ".join(problems) + " (rebuild=True re-extracts it)")


def _relabel(fs: FeatureSet, records: Sequence[VideoRecord]) -> FeatureSet:
    labels = {r.video_id: r.label for r in records}
    fs.video_labels = torch.tensor([labels[v] for v in fs.video_ids], dtype=torch.long)
    return fs


# --------------------------------------------------------------------------- head training

@dataclass
class HeadTrainConfig:
    output_dir: str = "/content/drive/MyDrive/vswin_runs/head"
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.05           # on the weight matrix only, not on the bias
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: float = 2.0
    min_lr_ratio: float = 0.0
    batch_size: int = 256                # miniclip embeddings per step
    samples_per_epoch: int | None = None # default: number of training miniclips
    balanced_sampling: bool = True       # same weight per video (any K) and per class
    flip_prob: float = 0.5               # probability of using the flipped view of a miniclip
    label_smoothing: float = 0.0
    class_weights: list[float] | None = None
    pos_weight: float | None = None
    monitor: str = "video_acc"
    eval_every: int = 1
    seed: int = 0
    class_names: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def head_logits(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Head output for pooled embeddings ``(B, D)``: dropout + linear layer."""
    head = model.cls_head
    return head.fc_cls(head.dropout(features))


@torch.no_grad()
def evaluate_features(model: nn.Module, fs: FeatureSet, view: int = 0,
                      batch_size: int = 4096) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Same metrics and rows as ``fine_tuning.evaluate``, on cached embeddings (original view)."""
    if fs.num_clips == 0:
        return {}, []
    was_training = model.training
    model.eval()
    device = model.cls_head.fc_cls.weight.device
    try:
        probs = [logits_to_probs(head_logits(model, fs.features[i:i + batch_size, view].to(device).float())).cpu()
                 for i in range(0, fs.num_clips, batch_size)]
    finally:
        model.train(was_training)
    p = torch.cat(probs)
    cv = fs.clip_video.cpu()
    labels = fs.video_labels.cpu()
    counts = torch.bincount(cv, minlength=fs.num_videos)
    video_p = torch.zeros(fs.num_videos, p.shape[1]).index_add_(0, cv, p) / counts.clamp(min=1).unsqueeze(1)
    metrics = {**classification_metrics(labels[cv].tolist(), p.numpy(), "clip_"),
               **classification_metrics(labels.tolist(), video_p.numpy(), "video_")}
    rows = [{"video_id": vid, "label": int(labels[i]), "pred": int(video_p[i].argmax()),
             "probs": video_p[i].numpy().round(6).tolist(), "num_clips": int(counts[i])}
            for i, vid in enumerate(fs.video_ids)]
    return metrics, rows


class HeadTrainer:
    """Trains only ``model.cls_head`` on cached embeddings.

    Files in ``cfg.output_dir``: ``best.pth`` (full model with the best head,
    loadable with ``build_model(..., pretrained=...)``), ``best_head.pth`` (head
    only), ``history.json``, ``predictions_best.json`` and ``config.json``.
    """

    def __init__(self, model: nn.Module, train: FeatureSet, val: FeatureSet | None, cfg: HeadTrainConfig,
                 device: str | torch.device | None = None):
        if train.num_clips == 0:
            raise ValueError("the training FeatureSet is empty")
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model = model.to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.cls_head.parameters():
            p.requires_grad = True
        self.train_set = train.to(self.device)
        self.val_set = val.to(self.device) if val is not None and val.num_clips else None
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.samples = cfg.samples_per_epoch or train.num_clips
        self.steps_per_epoch = math.ceil(self.samples / cfg.batch_size)
        fc = self.model.cls_head.fc_cls
        self.optimizer = torch.optim.AdamW(
            [{"params": [fc.weight], "weight_decay": cfg.weight_decay},
             {"params": [fc.bias], "weight_decay": 0.0}], lr=cfg.lr, betas=cfg.betas)
        self.scheduler = warmup_cosine(self.optimizer, cfg.epochs * self.steps_per_epoch,
                                       int(round(cfg.warmup_epochs * self.steps_per_epoch)), cfg.min_lr_ratio)
        self.criterion = make_criterion(self.model, cfg).to(self.device)
        self.weights = self._sample_weights()
        self.history: list[dict[str, Any]] = []
        self.best: float | None = None
        self.best_epoch: int | None = None
        self.best_head: dict[str, torch.Tensor] | None = None

    def _sample_weights(self) -> torch.Tensor:
        ts = self.train_set
        if not self.cfg.balanced_sampling:
            return torch.ones(ts.num_clips, device=self.device)
        cv = ts.clip_video
        w = 1.0 / ts.clips_per_video()[cv].float()
        per_class = torch.bincount(ts.video_labels)
        return w / per_class[ts.video_labels[cv]].float()

    def _train_epoch(self, gen: torch.Generator) -> dict[str, float]:
        cfg, ts = self.cfg, self.train_set
        self.model.train()
        start = time.perf_counter()
        idx = torch.multinomial(self.weights, self.samples, replacement=True, generator=gen)
        if ts.num_views > 1 and cfg.flip_prob > 0:
            views = (torch.rand(self.samples, generator=gen, device=self.device) < cfg.flip_prob).long()
        else:
            views = torch.zeros(self.samples, dtype=torch.long, device=self.device)
        total_loss, correct = 0.0, 0
        for i in range(0, self.samples, cfg.batch_size):
            b, v = idx[i:i + cfg.batch_size], views[i:i + cfg.batch_size]
            x = ts.features[b, v].float()
            y = ts.video_labels[ts.clip_video[b]]
            logits = head_logits(self.model, x)
            loss = self.criterion(logits, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += float(loss) * len(b)
            correct += int((logits_to_probs(logits.detach()).argmax(1) == y).sum())
        return {"train_loss": total_loss / self.samples, "train_clip_acc": correct / self.samples,
                "lr_head": self.optimizer.param_groups[0]["lr"], "train_seconds": time.perf_counter() - start}

    def fit(self, verbose: bool = True) -> list[dict[str, Any]]:
        cfg = self.cfg
        set_seed(cfg.seed)
        gen = torch.Generator(device=self.device).manual_seed(cfg.seed)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        best_rows: list[dict[str, Any]] = []
        best_metrics: dict[str, float] = {}
        for epoch in range(cfg.epochs):
            row: dict[str, Any] = {"epoch": epoch + 1, **self._train_epoch(gen)}
            is_best = False
            last = epoch + 1 == cfg.epochs
            if self.val_set is not None and ((epoch + 1) % cfg.eval_every == 0 or last):
                metrics, rows = evaluate_features(self.model, self.val_set)
                row.update(metrics)
                value = metrics.get(cfg.monitor)
                if value is not None and not math.isnan(value) and (self.best is None or value > self.best):
                    is_best = True
                    self.best, best_rows, best_metrics = value, rows, metrics
            elif self.val_set is None and last:
                is_best = True   # without validation, keep the last head
            if is_best:
                self.best_epoch = epoch + 1
                self.best_head = copy.deepcopy(self.model.cls_head.state_dict())
            self.history.append(row)
            if verbose:
                print(_format_row(row, cfg.monitor, is_best))
        self._save(best_rows, best_metrics)
        return self.history

    def _save(self, rows: list[dict[str, Any]], metrics: dict[str, float]) -> None:
        """Put the best head into the model and write the run files (once, at the end)."""
        if self.best_head is not None:
            self.model.cls_head.load_state_dict(self.best_head)
        torch.save({"cls_head": self.model.cls_head.state_dict(), "epoch": self.best_epoch, "best": self.best,
                    "binary": getattr(self.model, "binary", False)}, self.out / "best_head.pth")
        save_checkpoint(self.out / "best.pth", self.model, epoch=(self.best_epoch or 0) - 1, best=self.best,
                        history=self.history, cfg=self.cfg)
        (self.out / "history.json").write_text(json.dumps(self.history, indent=1), encoding="utf-8")
        if rows:
            (self.out / "predictions_best.json").write_text(
                json.dumps({"epoch": self.best_epoch, "metrics": metrics, "videos": rows}, indent=1),
                encoding="utf-8")
        print(f"Best epoch {self.best_epoch} ({self.cfg.monitor} = {self.best}); saved {self.out / 'best.pth'}")
