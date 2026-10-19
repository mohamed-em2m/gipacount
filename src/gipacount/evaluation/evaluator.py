"""Evaluation loop and metrics computation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import nn

from gipacount.data.types import CountingBatch
from gipacount.metrics import CountMetrics
from gipacount.utils.device import autocast_context
from gipacount.utils.logging import get_logger

__all__ = ["evaluate", "evaluate_ecr"]

_LOGGER = get_logger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterator[CountingBatch] | Any,
    *,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run the model over ``loader`` and compute MAE/RMSE (Eqs. 19-20).

    Args:
        model: The counting network.
        loader: Iterable of :class:`CountingBatch`.
        device: Where to run.
        loss_fn: Optional loss module; when given its mean value is reported too.
        amp: Enable CUDA autocast.
        max_batches: Stop after this many batches (smoke tests).

    Returns:
        ``{"mae", "rmse", "mean_predicted", "mean_target", "num_samples"[, "loss"]}``.
    """
    model.eval()
    metrics = CountMetrics()
    loss_total = 0.0
    loss_batches = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        batch = batch.to(device)
        with autocast_context(device, amp):
            output = model(
                batch.images, batch.boxes, pad_mask=batch.pad_mask, box_mask=batch.box_mask
            )
        metrics.update(output.count.float().cpu(), batch.counts.float().cpu())

        if loss_fn is not None:
            with autocast_context(device, amp):
                loss_out = loss_fn(
                    output.density.float(),
                    batch.points,
                    batch.point_mask,
                    batch.point_mass,
                    density_size=batch.density_size,
                )
            loss_total += float(loss_out.total.detach())
            loss_batches += 1

    if metrics.num_samples == 0:  # pragma: no cover - defensive
        raise RuntimeError("evaluation loader produced no samples")

    result = metrics.compute()
    if loss_batches:
        result["loss"] = loss_total / loss_batches
    return result


@torch.no_grad()
def evaluate_ecr(
    model: nn.Module,
    loader: Iterator[CountingBatch] | Any,
    *,
    device: torch.device,
    max_composites: int | None = None,
    amp: bool = False,
    ecr_eps: float = 1e-8,
) -> dict[str, float]:
    """Compute MAE/RMSE and ECR (Eq. 21) on a composite (2x2 collage) eval set.

    Every sample produced by the loader must carry a ``positive_mask``; this is
    guaranteed when the loader's collator is built with ``force_mix=True``.

    Args:
        model: The counting network.
        loader: Iterable of composite :class:`CountingBatch` (batch size 1).
        device: Where to run.
        max_composites: Cap on the number of composites to evaluate.
        amp: Enable CUDA autocast.
        ecr_eps: ``ε₀`` of Eq. 21.

    Returns:
        ``{"mae", "rmse", "ecr", "mean_predicted", "mean_target", "num_samples"}``.
    """
    model.eval()
    metrics = CountMetrics(ecr_eps=ecr_eps)
    seen = 0

    for batch in loader:
        if max_composites is not None and seen >= max_composites:
            break
        batch = batch.to(device)
        if batch.positive_mask is None:  # pragma: no cover - caller contract
            raise ValueError("ECR evaluation requires batches with a positive_mask")
        with autocast_context(device, amp):
            output = model(
                batch.images, batch.boxes, pad_mask=batch.pad_mask, box_mask=batch.box_mask
            )
        metrics.update(output.count.float().cpu(), batch.counts.float().cpu())
        metrics.update_ecr(output.density.float().cpu(), batch.positive_mask.cpu())
        seen += batch.images.shape[0]

    if metrics.num_samples == 0:  # pragma: no cover - defensive
        raise RuntimeError("ECR loader produced no samples")
    return metrics.compute()
