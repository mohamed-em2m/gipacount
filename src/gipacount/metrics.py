"""Counting metrics: MAE (Eq. 19), RMSE (Eq. 20) and ECR (Eq. 21)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = [
    "CountMetrics",
    "exemplar_consistent_response_ratio",
    "mean_absolute_error",
    "root_mean_squared_error",
]


def mean_absolute_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Eq. (19): mean absolute counting error."""
    return (predicted - target).abs().mean()


def root_mean_squared_error(predicted: Tensor, target: Tensor) -> Tensor:
    """Eq. (20): root mean squared counting error."""
    return torch.sqrt(((predicted - target) ** 2).mean())


def exemplar_consistent_response_ratio(
    density: Tensor,
    positive_mask: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Eq. (21): share of the predicted mass falling in exemplar-consistent regions.

    Args:
        density: ``(B, 1, H, W)`` predicted density map.
        positive_mask: ``(B, 1, H, W)`` boolean, ``True`` inside ``Ω+``.
        eps: ``ε₀``, a small constant for numerical stability.

    Returns:
        ``(B,)`` ratios; a sample with no predicted mass yields 0.
    """
    if density.shape != positive_mask.shape:
        raise ValueError("density and positive_mask must have the same shape")
    total = density.flatten(1).sum(dim=1)
    positive = (density * positive_mask.to(density.dtype)).flatten(1).sum(dim=1)
    return positive / (total + eps)


@dataclass
class CountMetrics:
    """Accumulate counts and targets, then report MAE/RMSE over the whole split.

    Args:
        ecr_eps: ``ε₀`` of Eq. 21.
    """

    ecr_eps: float = 1e-8
    _errors: list[Tensor] = field(default_factory=list, repr=False)
    _predicted: list[Tensor] = field(default_factory=list, repr=False)
    _targets: list[Tensor] = field(default_factory=list, repr=False)
    _ecr: list[Tensor] = field(default_factory=list, repr=False)

    def update(self, predicted: Tensor, target: Tensor) -> None:
        """Record one batch of predicted and ground-truth counts."""
        predicted = predicted.detach().flatten().to(torch.float64)
        target = target.detach().flatten().to(torch.float64)
        if predicted.shape != target.shape:
            raise ValueError("predicted and target must have the same shape")
        self._predicted.append(predicted)
        self._targets.append(target)
        self._errors.append(predicted - target)

    def update_ecr(self, density: Tensor, positive_mask: Tensor) -> None:
        """Record ECR values for a batch of composite samples (Eq. 21)."""
        self._ecr.append(
            exemplar_consistent_response_ratio(density, positive_mask, eps=self.ecr_eps).detach()
        )

    @property
    def num_samples(self) -> int:
        return int(sum(t.numel() for t in self._targets))

    def compute(self) -> dict[str, float]:
        """Return mean MAE/RMSE (and ECR when it was recorded)."""
        if not self._targets:
            raise RuntimeError("no samples were recorded")
        errors = torch.cat(self._errors)
        result = {
            "mae": float(errors.abs().mean()),
            "rmse": float(torch.sqrt((errors**2).mean())),
            "mean_predicted": float(torch.cat(self._predicted).mean()),
            "mean_target": float(torch.cat(self._targets).mean()),
            "num_samples": float(self.num_samples),
        }
        if self._ecr:
            result["ecr"] = float(torch.cat(self._ecr).mean())
        return result

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._errors.clear()
        self._predicted.clear()
        self._targets.clear()
        self._ecr.clear()
