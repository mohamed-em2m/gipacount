"""Metrics: MAE (Eq. 19), RMSE (Eq. 20) and ECR (Eq. 21)."""

from __future__ import annotations

import pytest
import torch

from gipacount.metrics import (
    CountMetrics,
    exemplar_consistent_response_ratio,
    mean_absolute_error,
    root_mean_squared_error,
)


def test_mae_and_rmse_values() -> None:
    predicted = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 2.0, 5.0])
    # errors: -1, 0, -2
    assert mean_absolute_error(predicted, target).item() == pytest.approx(1.0)
    assert root_mean_squared_error(predicted, target).item() == pytest.approx((5 / 3) ** 0.5)


def test_count_metrics_accumulate() -> None:
    metrics = CountMetrics()
    metrics.update(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 2.0]))
    metrics.update(torch.tensor([3.0]), torch.tensor([5.0]))
    result = metrics.compute()
    assert result["num_samples"] == 3.0
    assert result["mae"] == pytest.approx(1.0)
    assert result["mean_predicted"] == pytest.approx(2.0)
    assert result["mean_target"] == pytest.approx(3.0)


def test_count_metrics_reset() -> None:
    metrics = CountMetrics()
    metrics.update(torch.tensor([1.0]), torch.tensor([1.0]))
    metrics.reset()
    with pytest.raises(RuntimeError, match="no samples"):
        metrics.compute()


def test_count_metrics_rejects_shape_mismatch() -> None:
    metrics = CountMetrics()
    with pytest.raises(ValueError, match="same shape"):
        metrics.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


def test_ecr_is_one_when_mass_is_inside_positive_region() -> None:
    density = torch.zeros(1, 1, 4, 4)
    density[0, 0, :2, :2] = 1.0  # all mass top-left
    positive = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    positive[0, 0, :2, :2] = True
    ratio = exemplar_consistent_response_ratio(density, positive)
    assert ratio.item() == pytest.approx(1.0)


def test_ecr_is_zero_when_mass_is_outside_positive_region() -> None:
    density = torch.zeros(1, 1, 4, 4)
    density[0, 0, 3, 3] = 1.0
    positive = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    positive[0, 0, :2, :2] = True
    ratio = exemplar_consistent_response_ratio(density, positive)
    assert ratio.item() == pytest.approx(0.0)


def test_ecr_requires_matching_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        exemplar_consistent_response_ratio(
            torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        )


def test_metrics_report_ecr_when_recorded() -> None:
    metrics = CountMetrics()
    metrics.update(torch.tensor([1.0]), torch.tensor([1.0]))
    density = torch.zeros(1, 1, 4, 4)
    density[0, 0, 0, 0] = 1.0
    positive = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    positive[0, 0, 0, 0] = True
    metrics.update_ecr(density, positive)
    assert metrics.compute()["ecr"] == pytest.approx(1.0)
