"""Losses.  The paper's default is the Generalized Loss (Eq. 12, §3.5).

The Generalized Loss of Wan et al. (CVPR 2021) is the entropic-regularised
*unbalanced* optimal transport cost between the predicted density map ``a`` and the
ground-truth dot annotations ``(b, y)``::

    L = min_P  <C, P> - ε H(P) + τ D1(P 1_m | a) + τ D2(Pᵀ 1_n | b)

with ``C_ij = ||x_i - y_j||²``, ``D1`` the squared L2 pixel-wise term and ``D2`` the
L1 point-wise term.  ``P = diag(u) K diag(v)`` with ``K = exp(-C/ε)`` and the
generalized Sinkhorn updates::

    u ← (a / Kv)^{τ/(τ+ε)},   v ← (b / Kᵀu)^{τ/(τ+ε)}

As in the reference implementation the optimal ``v*`` is treated as a constant and
``u*`` is kept a differentiable function of ``a``, so gradients flow to the network.

Two practical adaptations, both documented in ``docs/paper_spec.md`` (A10/A11):

* coordinates are normalised to ``[0, 1]`` (with ``ε = 0.01`` the Gibbs kernel
  underflows for raw pixel distances) — switch with ``coord_norm``;
* the ``n × m`` cost matrix is never materialised: every kernel-vector product and
  every loss term is accumulated in chunks along the pixel axis, so the 512²
  density map stays affordable even for the densest FSC-147 images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from gipacount.config import LossConfig
from gipacount.utils.logging import get_logger

__all__ = ["GeneralizedLoss", "LossOutput", "build_loss", "count_loss"]

_LOGGER = get_logger(__name__)


@dataclass(slots=True)
class LossOutput:
    """Decomposed loss.

    Attributes:
        total: The scalar to back-propagate.
        transport: ``<C, P>``.
        entropy: ``-ε H(P)``.
        pixel: ``D1`` term (squared L2 between ``P 1`` and the prediction).
        point: ``D2`` term (L1 between ``Pᵀ 1`` and the annotation masses).
        count: ``(B,)`` predicted counts, for convenience.
    """

    total: Tensor
    transport: Tensor
    entropy: Tensor
    pixel: Tensor
    point: Tensor
    count: Tensor

    def as_dict(self) -> dict[str, float]:
        """Flatten to plain floats for logging."""
        return {
            "loss": float(self.total.detach()),
            "transport": float(self.transport.detach()),
            "entropy": float(self.entropy.detach()),
            "pixel": float(self.pixel.detach()),
            "point": float(self.point.detach()),
        }

    def item(self) -> float:
        """``total`` as a Python float."""
        return float(self.total.detach())


def count_loss(density: Tensor, target_count: Tensor, mode: Literal["l1", "l2"]) -> Tensor:
    """Simple density-integral baselines: ``|ΣD̂ − N|`` (L1) or its square (L2)."""
    predicted = density.flatten(1).sum(dim=1)
    diff = predicted - target_count.to(predicted.dtype)
    return diff.abs().mean() if mode == "l1" else (diff**2).mean()


class GeneralizedLoss(nn.Module):
    """Entropic-regularised unbalanced OT loss (Eq. 12).

    Args:
        cfg: Loss configuration (``ε``, ``τ``, Sinkhorn iterations, chunking).
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
        self._itemsize = 8 if cfg.dtype == "float64" else 4

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pixel_coords(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """``(size*size, 2)`` normalised ``(x, y)`` centre coordinates of the map."""
        axis = (torch.arange(size, device=device, dtype=dtype) + 0.5) / size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    @staticmethod
    def _point_coords(points: Tensor, size: int, dtype: torch.dtype) -> Tensor:
        """Normalise annotation coordinates to ``[0, 1]`` (A10)."""
        return (points.to(dtype) + 0.5) / size

    def _chunks(self, n: int) -> list[slice]:
        step = max(1, int(self.cfg.chunk_size))
        return [slice(start, min(start + step, n)) for start in range(0, n, step)]

    @staticmethod
    def _sqdist(x: Tensor, y: Tensor) -> Tensor:
        """``(N, 2) x (M, 2) -> (N, M)`` squared Euclidean distances.

        Uses the Gram identity ``|x-y|² = |x|² + |y|² - 2 x·yᵀ`` rather than
        ``cdist``: it is the same quantity, but it avoids the square root and
        runs on BLAS.
        """
        dist = x @ y.t()
        dist.mul_(-2.0)
        dist.add_((x * x).sum(dim=1, keepdim=True))
        dist.add_((y * y).sum(dim=1).unsqueeze(0))
        return dist.clamp_min_(0.0)

    def _kernel(self, x: Tensor, y: Tensor, eps: float) -> Tensor | None:
        """Materialise ``K = exp(-C/ε)`` when it fits the memory budget.

        Building ``K`` once per ε step lets both Sinkhorn directions reuse it
        (``Kv`` and ``Kᵀu``) instead of evaluating the exponential twice.  This
        changes nothing numerically — only how often ``exp`` is called.  Returns
        ``None`` when the ``N × M`` matrix would exceed ``kernel_cache_mb``, in
        which case the caller streams it chunk-wise via :meth:`_kernel_vec`.
        """
        n, m = x.shape[0], y.shape[0]
        budget = float(self.cfg.kernel_cache_mb) * 1024.0 * 1024.0
        if m == 0 or n * m * self._itemsize > budget:
            return None
        dist = self._sqdist(x, y)
        dist.div_(-eps)
        return dist.exp_()

    def _kernel_vec(self, x: Tensor, y: Tensor, weights: Tensor, eps: float) -> Tensor:
        """``(K w)_i = Σ_j exp(-C_ij/ε) w_j`` accumulated chunk-wise along ``x``."""
        out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        for chunk in self._chunks(x.shape[0]):
            dist = self._sqdist(x[chunk], y)
            out[chunk] = (torch.exp(-dist / eps) * weights.unsqueeze(0)).sum(dim=1)
        return out

    # ------------------------------------------------------------------ #
    # Sinkhorn
    # ------------------------------------------------------------------ #
    def _sinkhorn_v(
        self,
        a: Tensor,
        b: Tensor,
        x: Tensor,
        y: Tensor,
        eps: float,
    ) -> Tensor:
        """Solve for ``v*`` with the generalized Sinkhorn iterations (Eq. 6 of GL).

        Runs under ``no_grad``: ``v*`` is a constant with respect to the network
        (Wan et al. treat only ``u*`` as a function of the prediction).
        """
        cfg = self.cfg
        floor = cfg.eps_floor
        m = y.shape[0]
        if m == 0:  # pragma: no cover - guarded by the caller
            return torch.zeros(0, device=x.device, dtype=x.dtype)

        v = torch.ones(m, device=x.device, dtype=x.dtype)
        iterations = max(1, int(cfg.sinkhorn_iters))
        start = float(cfg.epsilon_scale_start) if cfg.epsilon_scaling else 1.0
        u_max = float(cfg.u_max)

        with torch.no_grad():
            for step in range(iterations):
                if cfg.epsilon_scaling:
                    progress = step / max(1, iterations - 1)
                    eps_step = eps * float(start ** (1.0 - progress))
                else:
                    eps_step = eps
                p = cfg.tau / (cfg.tau + eps_step)

                # The same kernel drives both directions: K v (row space) and
                # K^T u (column space).  Materialising it once per ε step halves
                # the number of ``exp`` evaluations over the cost matrix.
                kernel = self._kernel(x, y, eps_step)
                kv = self._kernel_vec(x, y, v, eps_step) if kernel is None else kernel @ v
                u = (a / kv.clamp_min(floor)).clamp_max(u_max).pow(p)
                ktu = self._kernel_vec(y, x, u, eps_step) if kernel is None else kernel.t() @ u
                v = (b / ktu.clamp_min(floor)).clamp_max(u_max).pow(p)
        return v

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def _sample_loss(
        self,
        density: Tensor,
        points: Tensor,
        mass: Tensor,
        size: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return ``(<C,P>, -εH(P), D1, D2)`` for a single sample.

        Args:
            density: ``(1, size, size)`` predicted density.
            points: ``(M, 2)`` ``(x, y)`` annotation coordinates in map pixels.
            mass: ``(M,)`` annotation masses ``b``.
            size: Density map side.
        """
        cfg = self.cfg
        eps = float(cfg.epsilon)
        device = density.device
        dtype = self.dtype

        a = density.reshape(-1).to(dtype)
        if points.numel() == 0:
            zero = a.sum() * 0.0
            return zero, zero, zero, zero

        if cfg.coord_norm == "unit":
            x = self._pixel_coords(size, device, dtype)
            y = self._point_coords(points.to(dtype), size, dtype)
        else:
            axis_x = torch.arange(size, device=device, dtype=dtype) + 0.5
            yy, xx = torch.meshgrid(axis_x, axis_x, indexing="ij")
            x = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
            y = points.to(dtype) + 0.5

        b = mass.to(dtype).clamp_min(0.0)
        v = self._sinkhorn_v(a.detach(), b, x, y, eps)

        kv = self._kernel_vec(x, y, v, eps)
        power = cfg.tau / (cfg.tau + eps)
        u = (a / kv.clamp_min(cfg.eps_floor)).clamp_max(float(cfg.u_max)).pow(power)

        transport = torch.zeros((), device=device, dtype=dtype)
        entropy = torch.zeros((), device=device, dtype=dtype)
        pixel = torch.zeros((), device=device, dtype=dtype)
        column = torch.zeros(y.shape[0], device=device, dtype=dtype)

        for chunk in self._chunks(x.shape[0]):
            dist = self._sqdist(x[chunk], y)
            kernel = torch.exp(-dist / eps)
            plan = u[chunk].unsqueeze(1) * kernel * v.unsqueeze(0)  # (c, m)
            transport = transport + (dist * plan).sum()
            # -ε H(P) with H(P) = -Σ P log P  =>  +ε Σ P log P
            entropy = entropy + eps * (plan * (plan + cfg.eps_floor).log()).sum()
            row = plan.sum(dim=1)
            pixel = pixel + ((row - a[chunk]) ** 2).sum()
            column = column + plan.sum(dim=0)

        point = (column - b).abs().sum() if cfg.point_loss == "l1" else ((column - b) ** 2).sum()

        return transport, entropy, pixel, point

    def forward(
        self,
        density: Tensor,
        points: Tensor,
        point_mask: Tensor,
        point_mass: Tensor,
        *,
        density_size: int | None = None,
    ) -> LossOutput:
        """Compute the Generalized Loss for a batch.

        Args:
            density: ``(B, 1, H, W)`` predicted density map (already non-negative
                and padding-masked).
            points: ``(B, M, 2)`` annotation coordinates in density-map pixels.
            point_mask: ``(B, M)`` boolean validity mask.
            point_mass: ``(B, M)`` annotation masses ``b`` (``1`` almost always).
            density_size: ``H``; defaults to ``density.shape[-1]``.

        Returns:
            :class:`LossOutput`.
        """
        size = int(density_size or density.shape[-1])
        if density.shape[-1] != size or density.shape[-2] != size:
            raise ValueError("density map does not match density_size")

        batch = density.shape[0]
        count = density.flatten(1).sum(dim=1).detach()
        totals: list[Tensor] = []
        parts: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for i in range(batch):
            valid = point_mask[i].to(torch.bool)
            pts = points[i][valid]
            mass = point_mass[i][valid]
            losses = self._sample_loss(density[i], pts, mass, size)
            totals.append(losses[0] + losses[1] + losses[2] + losses[3])
            parts.append(losses)

        total = torch.stack(totals).mean()
        transport = torch.stack([p[0] for p in parts]).mean()
        entropy = torch.stack([p[1] for p in parts]).mean()
        pixel = torch.stack([p[2] for p in parts]).mean()
        point = torch.stack([p[3] for p in parts]).mean()
        return LossOutput(
            total=total, transport=transport, entropy=entropy, pixel=pixel, point=point, count=count
        )


class ScalarLoss(nn.Module):
    """Fallback density losses (``l1`` / ``l2`` on the integral) for ablations."""

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        density: Tensor,
        points: Tensor,
        point_mask: Tensor,
        point_mass: Tensor,
        *,
        density_size: int | None = None,
    ) -> LossOutput:
        """See :meth:`GeneralizedLoss.forward`; ``points`` are used only for counts."""
        del density_size
        target = (point_mass * point_mask.to(point_mass.dtype)).sum(dim=1)
        total = count_loss(density, target, self.cfg.name)  # type: ignore[arg-type]
        zero = total * 0.0
        return LossOutput(
            total=total,
            transport=zero,
            entropy=zero,
            pixel=zero,
            point=zero,
            count=density.flatten(1).sum(dim=1).detach(),
        )


def build_loss(cfg: LossConfig) -> nn.Module:
    """Return the loss module selected by ``cfg.name``."""
    if cfg.name == "generalized":
        return GeneralizedLoss(cfg)
    if cfg.name in {"l1", "l2"}:
        _LOGGER.info("using the %s count loss (ablation) instead of the Generalized Loss", cfg.name)
        return ScalarLoss(cfg)
    raise ValueError(f"unknown loss: {cfg.name!r}")
