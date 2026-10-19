"""Checkpoint save/load with full resume support.

A checkpoint carries everything needed to continue a run exactly: model and
optimiser state, schedule position, epoch/step counters, RNG snapshots for
python/numpy/torch, the CCSS collator position, and the config that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gipacount.config import ExperimentConfig
from gipacount.utils.logging import get_logger
from gipacount.utils.seed import RNGState

__all__ = ["CheckpointState", "load_checkpoint", "save_checkpoint"]

_LOGGER = get_logger(__name__)
_FORMAT_VERSION = 1


@dataclass(slots=True)
class CheckpointState:
    """Deserialised checkpoint contents."""

    epoch: int
    global_step: int
    best_metric: float | None
    config: dict[str, Any]
    rng_state: RNGState | None
    collator_state: dict[str, Any] | None
    extra: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    config: ExperimentConfig,
    epoch: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    best_metric: float | None = None,
    rng_state: RNGState | None = None,
    collator_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint to ``path`` (its parent directory is created).

    Returns:
        The path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": _FORMAT_VERSION,
        "model": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "config": config.model_dump(mode="json"),
        "rng_state": None if rng_state is None else rng_state.as_dict(),
        "collator_state": collator_state,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()

    torch.save(payload, path)
    _LOGGER.debug("saved checkpoint %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> CheckpointState:
    """Load a checkpoint and optionally restore model/optimiser/schedule state.

    Args:
        path: Checkpoint file.
        model: If given, ``load_state_dict`` is called with ``strict``.
        optimizer: If given, its state is restored.
        scheduler: If given, its state is restored (anything with
            ``load_state_dict``).
        map_location: Passed to ``torch.load``.
        strict: Strictness of ``model.load_state_dict``.

    Returns:
        A :class:`CheckpointState` with the counters and metadata.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    payload: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    version = int(payload.get("format_version", 0))
    if version != _FORMAT_VERSION:
        _LOGGER.warning("checkpoint format version %s (expected %s)", version, _FORMAT_VERSION)

    if model is not None:
        missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
        if missing or unexpected:  # pragma: no cover - only with strict=False
            _LOGGER.warning("missing keys: %s; unexpected keys: %s", missing, unexpected)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload and payload["scheduler"] is not None:
        scheduler.load_state_dict(payload["scheduler"])

    rng_state = payload.get("rng_state")
    return CheckpointState(
        epoch=int(payload.get("epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
        best_metric=payload.get("best_metric"),
        config=payload.get("config", {}),
        rng_state=None if rng_state is None else RNGState.from_dict(rng_state),
        collator_state=payload.get("collator_state"),
        extra=payload.get("extra", {}) or {},
    )
