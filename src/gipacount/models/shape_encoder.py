"""Exemplar shape encoding ``X_S`` - the 3-layer FFN of §3.3."""

from __future__ import annotations

from torch import Tensor, nn

__all__ = ["ShapeEncoder"]


class ShapeEncoder(nn.Module):
    """Encode each exemplar box into a high-dimensional shape feature.

    Paper (§3.3): "for the exemplar bounding box set ``E_k``, a three-layer
    feed-forward network following the LOCA design encodes each box into a
    high-dimensional shape feature ``X_S ∈ R^{N_E x d}``, projecting geometric
    cues into the visual feature space."

    Boxes are normalised to ``[0, 1]`` by the image size (assumption A4) and
    encoded as ``(x1, y1, x2, y2)``.

    Args:
        token_dim: ``d``; also the width of the hidden layers.
        hidden_dim: Hidden width; defaults to ``token_dim``.
        num_layers: Number of linear layers (3 in the paper).
    """

    def __init__(self, token_dim: int, hidden_dim: int | None = None, num_layers: int = 3) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("the paper's shape encoder has three layers")
        hidden_dim = hidden_dim or token_dim
        dims = [4] + [hidden_dim] * (num_layers - 1) + [token_dim]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*layers)

    def forward(self, boxes_normalised: Tensor) -> Tensor:
        """Args: ``(B, K, 4)`` in ``[0, 1]``. Returns: ``(B, K, d)``."""
        return self.net(boxes_normalised)
