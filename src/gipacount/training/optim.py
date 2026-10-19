"""Optimiser and learning-rate schedule construction.

The paper specifies Adam with lr ``1e-4`` and weight decay ``1e-4`` but names no
schedule (assumption A16), so the default is a constant learning rate.  The
schedule below is a small, self-contained implementation rather than a
``torch.optim.lr_scheduler`` subclass: it is trivially checkpointable and its
``lr(epoch)`` is a pure function, which keeps resumed runs bit-identical.
"""

from __future__ import annotations

import math
from typing import Any

from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from gipacount.config import OptimConfig

__all__ = ["EpochLRScheduler", "build_optimizer", "build_scheduler", "trainable_parameters"]


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only the parameters that require a gradient.

    The paper trains only GFIM and PLAM while the backbone stays frozen (§4.3),
    so the frozen encoder must never enter the optimiser state.
    """
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> Optimizer:
    """Build the optimiser from ``cfg`` (Adam, lr 1e-4, wd 1e-4 by default)."""
    params = trainable_parameters(model)
    if not params:  # pragma: no cover - defensive
        raise RuntimeError("the model has no trainable parameters")

    if cfg.name == "adam":
        return Adam(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.name == "adamw":
        return AdamW(params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)
    if cfg.name == "sgd":
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"unknown optimiser: {cfg.name!r}")


class EpochLRScheduler:
    """Multiplicative LR schedule evaluated once per epoch.

    ``lr(e) = base_lr · warmup(e) · main(e)`` with

    * ``warmup(e) = min(1, (e + 1) / warmup_epochs)`` (no-op when ``warmup_epochs = 0``);
    * ``main(e)`` = 1 (``"none"``), cosine decay to ``min_lr``, or a multiplicative
      drop at ``milestones`` / every ``epochs // 5`` epochs.

    Args:
        optimizer: The optimiser whose ``param_groups`` are updated in place.
        cfg: Optimiser configuration.
        epochs: Total number of epochs (defines the cosine period).
    """

    def __init__(self, optimizer: Optimizer, cfg: OptimConfig, epochs: int) -> None:
        self.optimizer = optimizer
        self.cfg = cfg
        self.epochs = max(1, int(epochs))
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_epoch = -1

    # ------------------------------------------------------------------ #
    def lr(self, epoch: int) -> float:
        """Learning-rate multiplier at ``epoch`` (0-based)."""
        cfg = self.cfg
        warmup = max(0, int(cfg.warmup_epochs))
        factor = 1.0
        if warmup > 0 and epoch < warmup:
            factor *= (epoch + 1) / warmup

        main_epochs = max(1, self.epochs - warmup)
        progress_epoch = max(0, epoch - warmup)
        if cfg.scheduler == "none":
            pass
        elif cfg.scheduler == "cosine":
            progress = min(1.0, progress_epoch / main_epochs)
            floor = cfg.min_lr / cfg.lr if cfg.lr else 0.0
            factor *= floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        elif cfg.scheduler == "multistep":
            drops = sum(1 for m in cfg.milestones if epoch >= m)
            factor *= cfg.gamma**drops
        elif cfg.scheduler == "step":
            size = max(1, self.epochs // 5)
            factor *= cfg.gamma ** (epoch // size)
        else:  # pragma: no cover - guarded by config Literal
            raise ValueError(f"unknown scheduler: {cfg.scheduler!r}")
        return factor

    def step(self, epoch: int | None = None) -> None:
        """Advance one epoch and write the new LR into every param group."""
        self.last_epoch = self.last_epoch + 1 if epoch is None else int(epoch)
        factor = self.lr(self.last_epoch)
        for group, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base * factor

    def get_last_lr(self) -> list[float]:
        """Current learning rate(s)."""
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    # -- checkpointing --------------------------------------------------- #
    def state_dict(self) -> dict[str, Any]:
        """Serialize the schedule position."""
        return {"last_epoch": self.last_epoch, "base_lrs": self.base_lrs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`state_dict`."""
        self.last_epoch = int(state["last_epoch"])
        self.base_lrs = [float(x) for x in state["base_lrs"]]


def build_scheduler(optimizer: Optimizer, cfg: OptimConfig, epochs: int) -> EpochLRScheduler | None:
    """Return a schedule, or ``None`` when a constant LR is requested."""
    if cfg.scheduler == "none" and cfg.warmup_epochs <= 0:
        return None
    return EpochLRScheduler(optimizer, cfg, epochs)
