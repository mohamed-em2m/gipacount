"""Absolute positional encodings for image and prompt tokens (§4.3, A6/A7)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["TokenPositionalEncoding"]


class TokenPositionalEncoding(nn.Module):
    """Absolute positional encoding over a token sequence of known length.

    The paper says only that "absolute positional encodings are introduced for
    both image and prompt tokens" from the second GFIM layer onward; the type is
    unspecified (assumption A6).  Both a learnable table (default) and the
    classic sinusoidal encoding are provided.

    Args:
        num_tokens: Sequence length the encoding is defined for.
        dim: Token dimension ``d``.
        kind: ``"learnable"``, ``"sinusoidal"`` or ``"none"``.
        grid_size: When given, learnable embeddings are stored as a 2-D grid and
            bilinearly interpolated if the runtime grid differs (the image-token
            stream is inherently 2-D).
    """

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        *,
        kind: str = "learnable",
        grid_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.kind = kind
        self.grid_size = grid_size

        if kind == "learnable":
            self.embedding = nn.Parameter(torch.zeros(1, num_tokens, dim))
            nn.init.trunc_normal_(self.embedding, std=0.02)
        elif kind == "sinusoidal":
            self.register_buffer("embedding", _sinusoidal(num_tokens, dim), persistent=False)
        elif kind == "none":
            self.embedding = None  # type: ignore[assignment]
        else:  # pragma: no cover - guarded by config Literal
            raise ValueError(f"unknown positional encoding kind: {kind}")

    def forward(self, num_tokens: int | None = None) -> Tensor:
        """Return an encoding of shape ``(1, num_tokens, dim)``.

        Args:
            num_tokens: Override the configured length (a 2-D grid of
                ``isqrt(num_tokens)`` side is assumed for learnable encodings).
        """
        if self.kind == "none" or self.embedding is None:
            raise RuntimeError("positional encoding is disabled")

        target = num_tokens or self.num_tokens
        if target == self.num_tokens:
            return self.embedding

        if self.kind != "learnable" or self.grid_size is None:
            # Interpolate along the token axis as a last resort.
            emb = self.embedding.transpose(1, 2)  # (1, d, N)
            emb = torch.nn.functional.interpolate(
                emb, size=target, mode="linear", align_corners=False
            )
            return emb.transpose(1, 2)

        # Reshape to a grid, interpolate, flatten - keeps spatial semantics.
        side_src = self.grid_size
        side_dst = math.isqrt(target)
        if side_dst * side_dst != target:
            raise ValueError(f"expected a square token count, got {target}")
        emb = self.embedding.reshape(1, side_src, side_src, self.dim).permute(0, 3, 1, 2)
        emb = torch.nn.functional.interpolate(
            emb, size=(side_dst, side_dst), mode="bilinear", align_corners=False
        )
        return emb.permute(0, 2, 3, 1).reshape(1, target, self.dim)


def _sinusoidal(num_tokens: int, dim: int) -> Tensor:
    """Classic ``sin``/``cos`` positional encoding (Vaswani et al., 2017)."""
    position = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_tokens, dim)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)
