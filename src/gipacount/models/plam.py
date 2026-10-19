"""Prototype-guided Local Aggregation Module (PLAM), §3.4 of the paper."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gipacount.config import ModelConfig, PLAMConfig
from gipacount.models.layers import Mlp

__all__ = ["PLAM", "PLAMOutput", "prototype_depthwise_conv"]


def prototype_depthwise_conv(
    features: Tensor,
    kernels: Tensor,
    padding: int | None = None,
) -> Tensor:
    """Per-channel convolution of every feature channel by its own prototype kernel.

    Implements Eq. (10)::

        Ŝ[c, i, j] = Σ_m Σ_n  X̂_I[c, i + m - p, j + n - p] · X̂_P[c, m, n]

    i.e. a *depthwise* (grouped) convolution where the weights are predicted
    per sample and per channel.  ``torch.nn.functional.conv2d`` cannot take
    batched weights, so the operation is expressed with ``unfold`` + a
    broadcast multiply, which is numerically identical and vectorised.

    Args:
        features: ``(B, C, H, W)`` = ``X̂_I``.
        kernels: ``(B, C, k, k)`` = ``X̂_P``; the kernel side ``k = √N_P``.
        padding: ``p``; ``None`` selects ``"same"`` padding ``(k - 1) // 2``.

    Returns:
        ``(B, C, H, W)`` responses before the 1x1 channel fusion.
    """
    if features.ndim != 4 or kernels.ndim != 4:
        raise ValueError("features and kernels must both be 4-D")
    b, c, h, w = features.shape
    kb, kc, kh, kw = kernels.shape
    if (b, c) != (kb, kc):
        raise ValueError(f"channel mismatch: features {(b, c)}, kernels {(kb, kc)}")
    if kh != kw:
        raise ValueError("prototype kernels must be square")
    if padding is None:
        padding = (kh - 1) // 2

    patches = F.unfold(features, kernel_size=kh, padding=padding)  # (B, C*k*k, L)
    length = patches.shape[-1]
    patches = patches.reshape(b, c, kh * kw, length)
    weights = kernels.reshape(b, c, kh * kw).unsqueeze(-1)  # (B, C, k*k, 1)
    out = (patches * weights).sum(dim=2)  # (B, C, L)
    return out.reshape(b, c, h, w)


@dataclass(slots=True)
class PLAMOutput:
    """Output of :class:`PLAM`.

    Attributes:
        similarity: ``Ŝ`` of shape ``(B, d, √N_I, √N_I)`` (256x64x64 in the paper).
        kernel: The prototype convolution kernel ``X̂_P`` (``None`` if PLAM is off).
    """

    similarity: Tensor
    kernel: Tensor | None = None


class PLAM(nn.Module):
    """Turn global prototype tokens into a local similarity map.

    Steps (Eqs. 6-10):

    1. Eq. 6 - slice the first ``N_P`` prompt tokens as global prototypes ``X'_P``.
    2. Eq. 7 - MLP + GELU to obtain ``X̃_P``.
    3. Eq. 8 - reshape to a ``√N_P x √N_P`` per-channel convolution kernel ``X̂_P``.
    4. Eq. 9 - reshape the enhanced image tokens back into ``X̂_I``.
    5. Eq. 10 - depthwise convolution + 1x1 channel fusion -> ``Ŝ``.

    When ``cfg.enabled`` is ``False`` (ablations Q1/Q2 of Table 7 and P0 of
    Table 8) the enhanced image features are passed straight through, i.e. "the
    attention-weighted image features are directly decoded into the density map".

    ``kernel_source="exemplar"`` reproduces the P1/P2 ablations, where the kernel
    is built from the updated *exemplar* tokens instead of the prototype tokens;
    one kernel per exemplar is applied and the responses are averaged.

    Args:
        cfg: PLAM configuration.
        model_cfg: Model configuration (token dim, exemplar counts, grid).
    """

    def __init__(self, cfg: PLAMConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = model_cfg.token_dim
        self.n_prototypes = cfg.n_prototypes
        self.n_exemplars = model_cfg.n_exemplars
        self.tokens_per_exemplar = model_cfg.tokens_per_exemplar
        self.n_exemplar_tokens = model_cfg.n_exemplar_tokens
        self.grid = model_cfg.feature_grid

        side = int(round(cfg.n_prototypes**0.5))
        exp_side = int(round(self.tokens_per_exemplar**0.5))
        if side * side != cfg.n_prototypes:
            raise ValueError("n_prototypes must be a perfect square (Eq. 8)")
        if exp_side * exp_side != self.tokens_per_exemplar:
            raise ValueError(
                "tokens_per_exemplar must be a perfect square for kernel_source='exemplar'"
            )

        if cfg.enabled and cfg.kernel_source == "prototype":
            self.kernel_mlp: nn.Module = Mlp(
                self.dim,
                hidden_features=cfg.mlp_hidden or self.dim,
                out_features=self.dim,
                act=cfg.activation,
            )
        else:
            self.kernel_mlp = nn.Identity()

        self.channel_fuse: nn.Module = (
            nn.Conv2d(self.dim, self.dim, kernel_size=1) if cfg.channel_fuse else nn.Identity()
        )

    # ------------------------------------------------------------------ #
    # Kernel construction
    # ------------------------------------------------------------------ #
    def _prototype_kernel(self, prompt: Tensor) -> Tensor:
        """Eqs. 6-8: slice, MLP, reshape to ``(B, d, √N_P, √N_P)``."""
        prototypes = prompt[:, : self.n_prototypes]  # Eq. 6
        if not isinstance(self.kernel_mlp, nn.Identity):
            prototypes = self.kernel_mlp(prototypes)  # Eq. 7
        side = int(round(self.n_prototypes**0.5))
        b = prototypes.shape[0]
        return prototypes.reshape(b, side, side, self.dim).permute(0, 3, 1, 2).contiguous()  # Eq. 8

    def _exemplar_kernels(self, prompt: Tensor) -> Tensor:
        """Ablation P1/P2: one kernel per exemplar, shape ``(B, K, d, s, s)``."""
        exemplar = prompt[:, self.n_prototypes :][:, : self.n_exemplar_tokens]
        side = int(round(self.tokens_per_exemplar**0.5))
        b = exemplar.shape[0]
        return (
            exemplar.reshape(b, self.n_exemplars, side, side, self.dim)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, prompt: Tensor, image_tokens: Tensor) -> PLAMOutput:
        """Aggregate local evidence with prototype-derived kernels.

        Args:
            prompt: ``X'_T`` of shape ``(B, N_T, d)``.
            image_tokens: ``X^L_IT`` of shape ``(B, N_I, d)``.

        Returns:
            :class:`PLAMOutput`.
        """
        b, n_i, d = image_tokens.shape
        if n_i != self.grid * self.grid:
            raise ValueError(f"expected {self.grid * self.grid} image tokens, got {n_i}")

        # Eq. 9: reshape the enhanced image tokens into spatial features.
        features = image_tokens.transpose(1, 2).reshape(b, d, self.grid, self.grid)

        if not self.cfg.enabled:
            # Ablation: decode the attention-weighted features directly.
            return PLAMOutput(similarity=self.channel_fuse(features), kernel=None)

        padding = self.cfg.padding

        if self.cfg.kernel_source == "prototype":
            kernel = self._prototype_kernel(prompt)  # Eq. 8
            similarity = prototype_depthwise_conv(features, kernel, padding)  # Eq. 10
        else:
            kernels = self._exemplar_kernels(prompt)  # (B, K, d, s, s)
            responses = [
                prototype_depthwise_conv(features, kernels[:, k], padding)
                for k in range(self.n_exemplars)
            ]
            similarity = torch.stack(responses, dim=1).mean(dim=1)
            kernel = kernels.mean(dim=1)

        similarity = self.channel_fuse(similarity)
        return PLAMOutput(similarity=similarity, kernel=kernel)
