"""Multi-head scaled dot-product attention with independent projections.

Every attention operation of GFIM (Eqs. 2-5) has its **own** learned projections
``Wᵠ, Wᵏ, Wᵛ`` plus an output projection, so this module owns all four.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = ["MultiheadAttention"]


class MultiheadAttention(nn.Module):
    """Scaled dot-product attention, ``A = softmax(Q Kᵀ / √d)``.

    Shapes (``B`` batch, ``L`` queries, ``S`` keys, ``d`` model dim, ``h`` heads):

    * ``query``   -> ``(B, L, d)``
    * ``key``     -> ``(B, S, d)``
    * ``value``   -> ``(B, S, d)`` (defaults to ``key``)
    * returns     -> ``((B, L, d), (B, h, L, S) | None)``

    Args:
        dim: Model dimension ``d``.
        n_heads: Number of heads ``h`` (8 in the paper).
        attn_dropout: Dropout applied to the attention weights.
        proj_dropout: Dropout applied to the output projection.
        bias: Whether the projections carry a bias term.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        *,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor | None = None,
        *,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Run attention.

        Args:
            query: ``(B, L, d)``.
            key: ``(B, S, d)``.
            value: ``(B, S, d)``; defaults to ``key``.
            key_padding_mask: ``(B, S)`` boolean, ``True`` marks keys to ignore
                (used for the padded image tokens, paper §4.3).
            need_weights: Also return the ``(B, h, L, S)`` attention weights
                (detached) for interpretability / ECR analysis.

        Returns:
            ``(output, weights)`` with ``weights`` ``None`` unless requested.
        """
        value = key if value is None else value
        b, _, _ = query.shape

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, L, S)
        if key_padding_mask is not None:
            # (B, S) -> (B, 1, 1, S); padded keys get -inf before the softmax.
            mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn = attn.masked_fill(mask, torch.finfo(attn.dtype).min)
        attn = F.softmax(attn, dim=-1)
        weights = attn.detach() if need_weights else None
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, -1, self.dim)
        return self.proj_drop(self.out_proj(out)), weights
