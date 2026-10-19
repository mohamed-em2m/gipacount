"""Frozen pretrained image encoder producing the image tokens ``X_I`` (§3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gipacount.config import BackboneConfig

__all__ = ["BackboneOutput", "FrozenBackbone"]


@dataclass(slots=True)
class BackboneOutput:
    """Output of the frozen encoder.

    Attributes:
        tokens: ``(B, N_I, d)`` flattened image tokens ``X_I``.
        feature_map: ``(B, d, √N_I, √N_I)`` spatial features, used for ROI pooling.
        grid: Side ``√N_I`` of the feature map.
    """

    tokens: Tensor
    feature_map: Tensor

    @property
    def grid(self) -> int:
        return self.feature_map.shape[-1]


class FrozenBackbone(nn.Module):
    """Pretrained ``timm`` encoder with a linear 1x1 projection to ``token_dim``.

    The encoder is frozen by default and always kept in ``eval`` mode so that its
    normalisation statistics never drift (paper §4.3: "the backbone encoder
    remains frozen, and only the parameters of GFIM and PLAM are optimized").

    The checkpoint is not named by the paper (assumption A1): ``kind="vit"`` uses
    a patch-16 ViT, which yields exactly ``64 x 64`` tokens at 1024 px as
    reported; ``kind="cnn"`` uses a ResNet trunk at ``out_indices`` for the
    GIPACount-R variant.

    Args:
        cfg: Backbone configuration.
        image_size: Square input resolution (1024 in the paper).
        token_dim: ``d``, the projected token dimension (256 in the paper).
    """

    def __init__(self, cfg: BackboneConfig, image_size: int, token_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.stride = cfg.stride
        self.token_dim = token_dim

        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError("timm is required for the backbone; `uv sync`") from exc

        if cfg.kind == "vit":
            self.encoder: Any = timm.create_model(
                cfg.name,
                pretrained=cfg.pretrained,
                num_classes=0,
                global_pool="",
                img_size=image_size,
                dynamic_img_size=True,
            )
            embed_dim = int(getattr(self.encoder, "embed_dim", token_dim))
            self.project: nn.Module = nn.Linear(embed_dim, token_dim)
        elif cfg.kind == "cnn":
            self.encoder = timm.create_model(
                cfg.name,
                pretrained=cfg.pretrained,
                features_only=True,
                out_indices=cfg.out_indices,
            )
            channels = self.encoder.feature_info.channels()
            embed_dim = int(channels[-1])
            self.project = nn.Conv2d(embed_dim, token_dim, kernel_size=1)
        else:  # pragma: no cover - guarded by config Literal
            raise ValueError(f"unknown backbone kind: {cfg.kind}")

        self._freeze()

    # ------------------------------------------------------------------ #
    # Freezing
    # ------------------------------------------------------------------ #
    def _freeze(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        if self.cfg.unfreeze_last_n > 0:
            self._unfreeze_last_blocks(self.cfg.unfreeze_last_n)

        for param in self.project.parameters():
            param.requires_grad = self.cfg.train_projection

    def _unfreeze_last_blocks(self, n: int) -> None:
        """Unfreeze the last ``n`` encoder blocks (ablation Q7 of Table 7).

        Works for the ``timm`` ViT (``encoder.blocks``) and for ResNet-style
        trunks (``encoder.layer4``/``layer3``/...).
        """
        blocks: list[nn.Module] = []
        if hasattr(self.encoder, "blocks"):
            blocks = list(self.encoder.blocks)
        else:
            for name in ("layer4", "layer3", "layer2", "layer1"):
                layer = getattr(self.encoder, name, None)
                if layer is not None:
                    blocks.append(layer)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def train(self, mode: bool = True) -> FrozenBackbone:
        """Keep the frozen encoder in ``eval`` mode regardless of the parent mode."""
        super().train(mode)
        if self.cfg.frozen and self.cfg.unfreeze_last_n == 0:
            self.encoder.eval()
        return self

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, images: Tensor) -> BackboneOutput:
        """Encode ``images`` of shape ``(B, 3, H, W)``.

        Returns:
            :class:`BackboneOutput` with ``tokens`` ``(B, N_I, d)`` and
            ``feature_map`` ``(B, d, √N_I, √N_I)``.
        """
        if self.cfg.kind == "vit":
            features = self._forward_vit(images)
        else:
            features = self._forward_cnn(images)

        b, d, h, w = features.shape
        tokens = features.flatten(2).transpose(1, 2).contiguous()  # (B, N_I, d)
        assert tokens.shape == (b, h * w, d)
        return BackboneOutput(tokens=tokens, feature_map=features)

    def _forward_vit(self, images: Tensor) -> Tensor:
        out: Any = self.encoder.forward_features(images)
        if isinstance(out, (list, tuple)):  # pragma: no cover - some timm variants
            out = out[0]
        if out.ndim != 3:  # pragma: no cover - defensive
            raise RuntimeError(f"unexpected ViT feature shape {tuple(out.shape)}")
        num_prefix = int(getattr(self.encoder, "num_prefix_tokens", 0))
        if num_prefix:
            out = out[:, num_prefix:]
        grid = int(round(out.shape[1] ** 0.5))
        features = self.project(out)
        return features.transpose(1, 2).reshape(out.shape[0], self.token_dim, grid, grid)

    def _forward_cnn(self, images: Tensor) -> Tensor:
        out: Any = self.encoder(images)
        feats: Tensor = out[-1] if isinstance(out, (list, tuple)) else out
        return self.project(feats)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def token_mask_from_padding(self, pad_mask: Tensor, grid: int) -> Tensor:
        """Down-sample an image padding mask to a per-token validity mask.

        Args:
            pad_mask: ``(B, 1, H, W)`` with ``True`` on real pixels.
            grid: Target token grid side.

        Returns:
            ``(B, N_I)`` boolean, ``True`` where the token is *valid*.
        """
        pooled = F.adaptive_avg_pool2d(pad_mask.to(torch.float32), grid)
        return (pooled.flatten(1) > 0.5).to(torch.bool)

    @property
    def num_parameters(self) -> int:
        """Total number of parameters, including frozen ones."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Number of parameters that receive gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"name={self.cfg.name}, kind={self.cfg.kind}, frozen={self.cfg.frozen}"
