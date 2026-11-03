"""Fine-tuning of Video Swin on the preprocessed miniclips.

Follows the official recipe (configs/recognition/swin): AdamW, backbone
learning rate × 0.1, no weight decay on norms / position bias tables, linear
warmup of 2.5 epochs and cosine decay to 0, mixed precision and gradient
accumulation. Defaults are those of Swin-B ImageNet-22K -> Kinetics-400.

Difference from mmcv: mmcv applies only the first matching custom key, so in
the official runs backbone norms kept weight decay and position bias tables
got the full learning rate. Here every backbone parameter gets lr × 0.1 and
no 1-D parameter or position table gets weight decay.

Training works per miniclip (clip mode); evaluation works per video (mean of
the miniclip probabilities), and reports both levels.
"""
from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .colab_setup import amp_dtype, default_device
from .inference import autocast_context
from .model import BinaryBCELoss, logits_to_probs
from .swin_input import to_model_input

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **_):  # type: ignore[no-redef]
        return it

_NO_DECAY_KEYS = ("relative_position_bias_table", "absolute_pos_embed", "norm")


# --------------------------------------------------------------------------- configuration

@dataclass
class FineTuneConfig:
    output_dir: str = "/content/drive/MyDrive/vswin_runs/run"
    epochs: int = 30
    lr: float = 3e-4                     # head; the backbone uses lr * backbone_lr_mult
    backbone_lr_mult: float = 0.1
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_epochs: float = 2.5
    min_lr_ratio: float = 0.0            # final lr = lr * min_lr_ratio
    accumulation_steps: int = 8          # effective batch = loader batch × accumulation_steps
    grad_clip: float | None = None       # max gradient norm (None = no clipping, as the official configs)
    amp: bool = True                     # bf16 on A100/L4, fp16 + GradScaler on T4
    label_smoothing: float = 0.0
    class_weights: list[float] | None = None  # softmax head: CrossEntropyLoss(weight=...)
    pos_weight: float | None = None           # binary head: weight of class 1 (None: class_weights[1] / class_weights[0] if given)
    freeze_backbone_epochs: int = 0      # train only the head during the first N epochs
    eval_every: int = 1
    eval_batch_size: int = 4             # miniclips per forward pass during evaluation
    monitor: str = "video_acc"           # metric that selects best.pth (any key of evaluate())
    resume: bool = True                  # continue from output_dir/last.pth if it exists
    seed: int = 0
    class_names: list[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # free notes saved with the run


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- optimizer and schedule

def param_groups(model: nn.Module, lr: float, weight_decay: float,
                 backbone_lr_mult: float = 0.1) -> list[dict[str, Any]]:
    """Four groups: (backbone | head) × (decay | no decay). Frozen parameters are included,
    so unfreezing later does not require a new optimizer."""
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {k: [] for k in
                                                           [(True, True), (True, False), (False, True), (False, False)]}
    for name, p in model.named_parameters():
        is_backbone = name.startswith("backbone.")
        no_decay = p.ndim <= 1 or any(k in name for k in _NO_DECAY_KEYS)
        groups[(is_backbone, not no_decay)].append(p)
    out = []
    for (is_backbone, decay), params in groups.items():
        if params:
            out.append({
                "params": params,
                "lr": lr * (backbone_lr_mult if is_backbone else 1.0),
                "weight_decay": weight_decay if decay else 0.0,
                "name": f"{'backbone' if is_backbone else 'head'}_{'decay' if decay else 'no_decay'}",
            })
    return out


def build_optimizer(model: nn.Module, cfg: FineTuneConfig) -> torch.optim.AdamW:
    return torch.optim.AdamW(param_groups(model, cfg.lr, cfg.weight_decay, cfg.backbone_lr_mult),
                             lr=cfg.lr, betas=cfg.betas)


def warmup_cosine(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int,
                  min_lr_ratio: float = 0.0) -> torch.optim.lr_scheduler.LambdaLR:
    """Per optimizer step: linear warmup from 0, then cosine down to ``min_lr_ratio``."""
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --------------------------------------------------------------------------- metrics

def classification_metrics(labels: Sequence[int], probs: np.ndarray, prefix: str = "") -> dict[str, float]:
    """Accuracy, balanced accuracy and, for 2 classes, precision/recall/F1/AUC of class 1."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if len(y) == 0:
        return {}
    pred = p.argmax(1)
    n_cls = p.shape[1]
    m: dict[str, float] = {f"{prefix}acc": float((pred == y).mean())}
    recalls = [float((pred[y == c] == c).mean()) for c in range(n_cls) if (y == c).any()]
    m[f"{prefix}balanced_acc"] = float(np.mean(recalls))
    if n_cls == 2:
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        m[f"{prefix}precision"] = precision
        m[f"{prefix}recall"] = recall
        m[f"{prefix}f1"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        m[f"{prefix}auc"] = _binary_auc(y, p[:, 1])
    return m


def confusion_matrix(labels: Sequence[int], preds: Sequence[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, q in zip(labels, preds):
        cm[int(t), int(q)] += 1
    return cm


def _binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) formula, with ties averaged."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


# --------------------------------------------------------------------------- loops

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, criterion: nn.Module,
                    scaler: torch.amp.GradScaler, cfg: FineTuneConfig, epoch: int,
                    device: torch.device) -> dict[str, float]:
    """One pass over ``loader`` (clip mode). The scheduler steps once per optimizer step."""
    model.train()
    accum = max(1, cfg.accumulation_steps)
    total_loss, correct, seen, steps = 0.0, 0, 0, 0
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    bar = tqdm(loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False)
    for i, batch in enumerate(bar):
        x = to_model_input(batch["clip"], device=device)
        y = torch.as_tensor(batch["label"], device=device)
        with autocast_context(device, cfg.amp):
            logits = model(x)
        loss = criterion(logits.float(), y)
        scaler.scale(loss / accum).backward()

        if (i + 1) % accum == 0 or i + 1 == n_batches:
            if cfg.grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            steps += 1

        total_loss += float(loss.detach()) * len(y)
        correct += int((logits_to_probs(logits.detach()).argmax(1) == y).sum())
        seen += len(y)
        bar.set_postfix(loss=f"{total_loss / seen:.4f}", acc=f"{correct / seen:.3f}")
    return {
        "train_loss": total_loss / max(1, seen),
        "train_clip_acc": correct / max(1, seen),
        "lr_head": optimizer.param_groups[-1]["lr"],
        "optimizer_steps": steps,
        "train_seconds": time.perf_counter() - start,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device | None = None,
             batch_size: int = 4, amp: bool = True) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate on a video-mode loader.

    Returns metrics (``clip_*`` per miniclip, ``video_*`` per video, with the
    mean probability of its miniclips) and one row per video.
    """
    device = device or next(model.parameters()).device
    model.eval()
    rows, clip_labels, clip_probs, video_labels, video_probs = [], [], [], [], []
    start = time.perf_counter()
    for sample in tqdm(loader, desc="eval", leave=False):
        clips = sample["clips"]
        probs = []
        for i in range(0, len(clips), batch_size):
            x = to_model_input(clips[i:i + batch_size], device=device)
            with autocast_context(device, amp):
                probs.append(logits_to_probs(model(x)).cpu())
        p = torch.cat(probs).numpy()
        label = int(sample["label"])
        mean = p.mean(0)
        clip_labels.extend([label] * len(p))
        clip_probs.append(p)
        video_labels.append(label)
        video_probs.append(mean)
        rows.append({"video_id": sample["video_id"], "label": label, "pred": int(mean.argmax()),
                     "probs": mean.round(6).tolist(), "num_clips": len(p)})
    metrics: dict[str, float] = {}
    if rows:
        metrics.update(classification_metrics(clip_labels, np.concatenate(clip_probs), "clip_"))
        metrics.update(classification_metrics(video_labels, np.stack(video_probs), "video_"))
    metrics["eval_seconds"] = time.perf_counter() - start
    return metrics, rows


# --------------------------------------------------------------------------- checkpoints

def save_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, scaler: torch.amp.GradScaler | None = None, epoch: int = -1,
                    best: float | None = None, history: list[dict] | None = None,
                    cfg: FineTuneConfig | None = None) -> Path:
    """Save to ``path`` atomically (temporary file + rename, safe if Colab disconnects)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    variant = getattr(model, "variant", None)
    data = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best": best,
        "history": history or [],
        "config": asdict(cfg) if cfg is not None else None,
        "variant": variant.name if variant is not None else None,
        "num_classes": getattr(model, "num_classes", None),
        "binary": getattr(model, "binary", False),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, scaler: torch.amp.GradScaler | None = None) -> dict[str, Any]:
    """Restore a checkpoint written by ``save_checkpoint`` and return its other fields."""
    data = torch.load(str(path), map_location="cpu", weights_only=False)
    model.load_state_dict(data["model"])
    if optimizer is not None and data.get("optimizer"):
        optimizer.load_state_dict(data["optimizer"])
    if scheduler is not None and data.get("scheduler"):
        scheduler.load_state_dict(data["scheduler"])
    if scaler is not None and data.get("scaler"):
        scaler.load_state_dict(data["scaler"])
    return {k: data.get(k) for k in ("epoch", "best", "history", "config", "variant", "num_classes", "binary")}


def make_criterion(model: nn.Module, cfg: FineTuneConfig) -> nn.Module:
    """``BinaryBCELoss`` (sigmoid) for a binary head, ``CrossEntropyLoss`` (softmax) otherwise."""
    if getattr(model, "binary", False):
        pos_weight = cfg.pos_weight
        if pos_weight is None and cfg.class_weights:
            if len(cfg.class_weights) != 2:
                raise ValueError("a binary head takes 2 class_weights (or pos_weight)")
            pos_weight = cfg.class_weights[1] / cfg.class_weights[0]
        return BinaryBCELoss(pos_weight=pos_weight, label_smoothing=cfg.label_smoothing)
    if cfg.pos_weight is not None:
        raise ValueError("pos_weight is for binary=True; use class_weights with a softmax head")
    weight = torch.tensor(cfg.class_weights, dtype=torch.float32) if cfg.class_weights else None
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)


# --------------------------------------------------------------------------- orchestration

class FineTuner:
    """Training loop with evaluation, best/last checkpoints and resumption.

    Files in ``cfg.output_dir``: ``last.pth`` (every epoch), ``best.pth``
    (best ``cfg.monitor``), ``history.json`` and ``predictions_best.json``
    (per-video validation predictions of the best epoch).
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader | None,
                 cfg: FineTuneConfig, device: torch.device | str | None = None):
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else default_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.steps_per_epoch = math.ceil(len(train_loader) / max(1, cfg.accumulation_steps))
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = warmup_cosine(self.optimizer, cfg.epochs * self.steps_per_epoch,
                                       int(round(cfg.warmup_epochs * self.steps_per_epoch)), cfg.min_lr_ratio)
        use_scaler = cfg.amp and self.device.type == "cuda" and amp_dtype() == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        self.criterion = make_criterion(model, cfg).to(self.device)

        self.start_epoch = 0
        self.best: float | None = None
        self.history: list[dict[str, Any]] = []
        last = self.out / "last.pth"
        if cfg.resume and last.exists():
            state = load_checkpoint(last, self.model, self.optimizer, self.scheduler, self.scaler)
            self.start_epoch = int(state["epoch"]) + 1
            self.best = state["best"]
            self.history = state["history"] or []
            print(f"Resumed from {last} (next epoch {self.start_epoch + 1}, best {cfg.monitor} = {self.best})")

    def _set_backbone_frozen(self, epoch: int) -> None:
        if not hasattr(self.model, "freeze_backbone"):
            return
        if epoch < self.cfg.freeze_backbone_epochs:
            self.model.freeze_backbone(4)
        elif self.cfg.freeze_backbone_epochs and epoch == self.cfg.freeze_backbone_epochs:
            self.model.unfreeze_backbone()

    def fit(self) -> list[dict[str, Any]]:
        cfg = self.cfg
        set_seed(cfg.seed + self.start_epoch)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False),
                                              encoding="utf-8")
        for epoch in range(self.start_epoch, cfg.epochs):
            self._set_backbone_frozen(epoch)
            row: dict[str, Any] = {"epoch": epoch + 1}
            row.update(train_one_epoch(self.model, self.train_loader, self.optimizer, self.scheduler,
                                       self.criterion, self.scaler, cfg, epoch, self.device))
            is_best = False
            if self.val_loader is not None and ((epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs):
                metrics, rows = evaluate(self.model, self.val_loader, self.device, cfg.eval_batch_size, cfg.amp)
                row.update(metrics)
                value = metrics.get(cfg.monitor)
                if value is not None and not math.isnan(value) and (self.best is None or value > self.best):
                    self.best, is_best = value, True
                    (self.out / "predictions_best.json").write_text(
                        json.dumps({"epoch": epoch + 1, "metrics": metrics, "videos": rows}, indent=1),
                        encoding="utf-8")
            self.history.append(row)
            save_checkpoint(self.out / "last.pth", self.model, self.optimizer, self.scheduler, self.scaler,
                            epoch, self.best, self.history, cfg)
            if is_best:
                save_checkpoint(self.out / "best.pth", self.model, epoch=epoch, best=self.best,
                                history=self.history, cfg=cfg)
            (self.out / "history.json").write_text(json.dumps(self.history, indent=1), encoding="utf-8")
            print(_format_row(row, cfg.monitor, is_best))
        return self.history

    def load_best(self) -> dict[str, Any]:
        """Load ``best.pth`` into the model (weights only)."""
        return load_checkpoint(self.out / "best.pth", self.model)


def _format_row(row: dict[str, Any], monitor: str, is_best: bool) -> str:
    keys = ["train_loss", "train_clip_acc", "clip_acc", "video_acc", "video_balanced_acc", "video_f1", "video_auc"]
    if monitor not in keys:
        keys.append(monitor)
    parts = [f"epoch {row['epoch']}"]
    parts += [f"{k}={row[k]:.4f}" for k in keys if isinstance(row.get(k), float)]
    parts.append(f"lr={row['lr_head']:.2e}")
    parts.append(f"{row['train_seconds'] / 60:.1f} min")
    return " | ".join(parts) + ("  * best" if is_best else "")
