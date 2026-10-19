"""Reusable building blocks: channel-wise LayerNorm, MLP, normalisation factory."""

from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

__all__ = ["LayerNorm2d", "Mlp", "build_norm"]


class LayerNorm2d(nn.Module):
    """Layer normalisation over the channel dimension of an ``(N, C, H, W)`` map.

    The paper applies "layer normalization" after every decoder layer, whose
    activations are spatial feature maps; normalising per spatial location over
    the channels is the standard reading.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # (N, C, H, W) -> (N, H, W, C) -> normalise -> back
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"num_channels={self.norm.normalized_shape[0]}"


def build_norm(
    kind: Literal["layernorm", "groupnorm", "none"],
    num_channels: int,
) -> nn.Module:
    """Return the requested channel-wise normalisation layer."""
    if kind == "layernorm":
        return LayerNorm2d(num_channels)
    if kind == "groupnorm":
        return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=True)
    return nn.Identity()


def build_activation(kind: Literal["relu", "gelu"]) -> nn.Module:
    """Return the requested activation module."""
    return nn.ReLU(inplace=False) if kind == "relu" else nn.GELU()


class Mlp(nn.Module):
    """Transformer feed-forward block: ``Linear -> act -> dropout -> Linear``.

    Used for the hidden MLPs of GFIM (§4.3: "two linear layers and a GELU
    activation"), for the non-linear fusion ``M(·)`` of Eq. 3 and for the
    prototype MLP of Eq. 7.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        *,
        act: Literal["relu", "gelu"] = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation(act)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Args: ``x`` of shape ``(..., in_features)``."""
        return self.fc2(self.drop(self.act(self.fc1(x))))
