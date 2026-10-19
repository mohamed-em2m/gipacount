"""Density decoder ``Dec(·)`` - Eq. (11) and §4.3."""

from __future__ import annotations

from torch import Tensor, nn

from gipacount.config import DecoderConfig
from gipacount.models.layers import build_activation, build_norm

__all__ = ["DensityDecoder"]


class DensityDecoder(nn.Module):
    """Three transposed convolutions plus a point-wise convolution.

    Paper §4.3: "``Dec(·)`` consists of three transposed convolution layers
    followed by a point-wise convolution layer.  Each layer is followed by layer
    normalization and ReLU activation.  The decoder progressively upsamples the
    feature maps from ``64 x 64`` to ``512 x 512``."

    The channel widths are not stated (assumption A8); ``(128, 64, 32)`` is used,
    which gives the required ``x8`` upsampling with stride 2.  The final 1x1
    layer maps to the single density channel and is *not* normalised (a
    LayerNorm over a single channel would normalise the whole map); the output
    activation is applied by :class:`~gipacount.models.gipacount.GIPACount`.

    Args:
        in_channels: Channels of the incoming similarity map ``Ŝ`` (``d`` = 256).
        cfg: Decoder configuration.
        in_size: Spatial side of the input (64 in the paper).
    """

    def __init__(self, in_channels: int, cfg: DecoderConfig, in_size: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_size = in_size

        layers: list[nn.Module] = []
        channels = [in_channels, *cfg.channels]
        for i in range(len(cfg.channels)):
            layers.append(
                nn.ConvTranspose2d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride,
                    padding=cfg.padding,
                    bias=cfg.norm == "none",
                )
            )
            layers.append(build_norm(cfg.norm, channels[i + 1]))
            layers.append(build_activation(cfg.activation))
        self.upsample = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels[-1], cfg.out_channels, kernel_size=1)

        self.upsample_factor = cfg.stride ** len(cfg.channels)
        self.out_size = in_size * self.upsample_factor

    def forward(self, similarity: Tensor) -> Tensor:
        """Map ``Ŝ`` ``(B, d, s, s)`` to a raw density map ``(B, 1, s*8, s*8)``."""
        if similarity.ndim != 4:
            raise ValueError("similarity must be a 4-D tensor")
        if similarity.shape[-1] != self.in_size or similarity.shape[-2] != self.in_size:
            raise ValueError(
                f"expected a {self.in_size}x{self.in_size} similarity map, "
                f"got {tuple(similarity.shape[-2:])}"
            )
        return self.head(self.upsample(similarity))

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_size={self.in_size}, out_size={self.out_size}, channels={tuple(self.cfg.channels)}"
        )
