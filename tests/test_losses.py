"""Losses: the Generalized Loss (Eq. 12) and its scalar fallbacks."""

from __future__ import annotations

import pytest
import torch

from gipacount.config import LossConfig
from gipacount.losses import GeneralizedLoss, ScalarLoss, build_loss, count_loss


def _density_points(
    size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch of one: ``(density, points, point_mask, point_mass)``."""
    density = torch.rand(1, 1, size, size) + 0.1
    points = torch.tensor([[[8.0, 8.0], [20.0, 24.0]]])
    point_mask = torch.ones(1, 2, dtype=torch.bool)
    point_mass = torch.ones(1, 2)
    return density, points, point_mask, point_mass


def test_generalized_loss_is_finite_and_differentiable() -> None:
    loss_fn = GeneralizedLoss(LossConfig(sinkhorn_iters=5, chunk_size=256))
    density, points, mask, mass = _density_points()
    density.requires_grad_(True)
    out = loss_fn(density, points, mask, mass, density_size=32)
    assert torch.isfinite(out.total)
    assert out.total.item() > 0
    out.total.backward()
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()
    # every term is exposed for logging
    for key in ("transport", "entropy", "pixel", "point"):
        assert torch.isfinite(getattr(out, key))


def test_generalized_loss_zero_without_points() -> None:
    loss_fn = GeneralizedLoss(LossConfig(sinkhorn_iters=5))
    density = torch.rand(1, 1, 32, 32)
    points = torch.zeros(1, 0, 2)
    mask = torch.zeros(1, 0, dtype=torch.bool)
    mass = torch.zeros(1, 0)
    out = loss_fn(density, points, mask, mass, density_size=32)
    assert out.total.item() == pytest.approx(0.0)


def test_generalized_loss_ignores_masked_points() -> None:
    loss_fn = GeneralizedLoss(LossConfig(sinkhorn_iters=5))
    density = torch.rand(1, 1, 32, 32)
    points = torch.tensor([[[8.0, 8.0], [20.0, 24.0]]])
    mass = torch.ones(1, 2)
    full = loss_fn(density, points, torch.ones(1, 2, dtype=torch.bool), mass, density_size=32)
    half = loss_fn(density, points, torch.tensor([[True, False]]), mass, density_size=32)
    assert torch.isfinite(full.total) and torch.isfinite(half.total)
    assert full.total.item() != pytest.approx(half.total.item())


def test_chunking_does_not_change_the_loss() -> None:
    """The chunked cost matrix must match the un-chunked computation."""
    density, points, mask, mass = _density_points()
    chunks = GeneralizedLoss(LossConfig(sinkhorn_iters=5, chunk_size=64))
    whole = GeneralizedLoss(LossConfig(sinkhorn_iters=5, chunk_size=1_000_000))
    a = chunks(density, points, mask, mass, density_size=32).total
    b = whole(density, points, mask, mass, density_size=32).total
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-5)


def test_generalized_loss_decreases_under_optimisation() -> None:
    """Overfit sanity: gradient descent on the density reduces the loss."""
    torch.manual_seed(0)
    loss_fn = GeneralizedLoss(LossConfig(sinkhorn_iters=10, chunk_size=256))
    raw = torch.zeros(1, 1, 32, 32, requires_grad=True)
    points = torch.tensor([[[16.0, 16.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    mass = torch.ones(1, 1)
    opt = torch.optim.Adam([raw], lr=0.1)

    first = None
    for _ in range(30):
        opt.zero_grad()
        density = torch.nn.functional.softplus(raw)
        out = loss_fn(density, points, mask, mass, density_size=32)
        out.total.backward()
        opt.step()
        first = out.total.item() if first is None else first
    last = loss_fn(
        torch.nn.functional.softplus(raw), points, mask, mass, density_size=32
    ).total.item()
    assert last < first


def test_build_loss_selects_implementation() -> None:
    assert isinstance(build_loss(LossConfig(name="generalized")), GeneralizedLoss)
    assert isinstance(build_loss(LossConfig(name="l1")), ScalarLoss)
    assert isinstance(build_loss(LossConfig(name="l2")), ScalarLoss)


def test_scalar_loss_values() -> None:
    loss_fn = ScalarLoss(LossConfig(name="l1"))
    density = torch.full((1, 1, 4, 4), 0.25)  # count = 4
    points = torch.zeros(1, 3, 2)
    mask = torch.ones(1, 3, dtype=torch.bool)
    mass = torch.ones(1, 3)
    out = loss_fn(density, points, mask, mass, density_size=4)
    # |4 - 3| = 1
    assert out.total.item() == pytest.approx(1.0)


def test_count_loss_modes() -> None:
    density = torch.full((1, 1, 2, 2), 1.0)  # count = 4
    target = torch.tensor([2.0])
    assert count_loss(density, target, "l1").item() == pytest.approx(2.0)
    assert count_loss(density, target, "l2").item() == pytest.approx(4.0)


def test_loss_config_validation() -> None:
    with pytest.raises(Exception, match="positive"):
        LossConfig(epsilon=0.0)
    with pytest.raises(Exception, match="positive"):
        LossConfig(tau=-1.0)


def test_point_loss_l2_variant_is_finite() -> None:
    loss_fn = GeneralizedLoss(LossConfig(sinkhorn_iters=5, point_loss="l2", chunk_size=256))
    density, points, mask, mass = _density_points()
    out = loss_fn(density, points, mask, mass, density_size=32)
    assert torch.isfinite(out.total)
