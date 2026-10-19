"""The full GIPACount network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gipacount.config import ExperimentConfig, ModelConfig
from gipacount.models.backbone import FrozenBackbone
from gipacount.models.decoder import DensityDecoder
from gipacount.models.exemplar import RoIExemplarEncoder
from gipacount.models.gfim import GFIM
from gipacount.models.plam import PLAM
from gipacount.models.shape_encoder import ShapeEncoder

__all__ = ["GIPACount", "GIPACountOutput", "count_from_density"]


def count_from_density(density: Tensor) -> Tensor:
    """Integrate a density map: ``(B, 1, H, W) -> (B,)`` (the paper's count)."""
    if density.ndim != 4:
        raise ValueError("density must be (B, 1, H, W)")
    return density.flatten(1).sum(dim=1)


@dataclass(slots=True)
class GIPACountOutput:
    """Everything a caller (loss, metrics, visualiser) may need.

    Attributes:
        density: ``D̂`` of shape ``(B, 1, 512, 512)`` in the paper's setting.
        count: ``(B,)`` integrated counts.
        similarity: ``Ŝ`` before decoding, ``(B, d, 64, 64)``.
        prototype_kernel: ``X̂_P``, ``(B, d, 3, 3)``.
        prompt_tokens: ``X'_T``, ``(B, N_T, d)``.
        image_tokens: ``X^L_IT``, ``(B, N_I, d)``.
        attn: Optional detached attention weights.
        extras: Free-form slot for debugging / analysis.
    """

    density: Tensor
    count: Tensor
    similarity: Tensor | None = None
    prototype_kernel: Tensor | None = None
    prompt_tokens: Tensor | None = None
    image_tokens: Tensor | None = None
    attn: dict[str, Tensor] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class GIPACount(nn.Module):
    """Global Interaction and Prototype-guided Aggregation counting network.

    Pipeline (Fig. 2 of the paper)::

        image I ──► frozen backbone ──► X_I (N_I x d)
        boxes E ──► ROI pool (3x3) ──► X_E (K*9 x d) ─┐
        boxes E ──► 3-layer FFN    ──► X_S            ├─► X_T = [X_P; X_E + X_S]   (Eq. 1)
        learnable X_P (N_P x d) ──────────────────────┘
        GFIM (L x [attn_TT, attn_T->I, attn_I->T] + fusion) (Eqs. 2-5)
            ├─► X'_T  ──► PLAM: slice N_P prototypes (Eq. 6), MLP (Eq. 7),
            │            reshape to kernel (Eq. 8), depthwise conv (Eq. 10)
            └─► X^L_IT ──► reshape to X̂_I (Eq. 9) ──┘
        Ŝ ──► Dec(·) ──► D̂  (Eq. 11) ──► count = Σ D̂

    Args:
        cfg: Either a :class:`ModelConfig` or a full :class:`ExperimentConfig`.
    """

    def __init__(self, cfg: ModelConfig | ExperimentConfig) -> None:
        super().__init__()
        model_cfg = cfg.model if isinstance(cfg, ExperimentConfig) else cfg
        self.cfg = model_cfg

        d = model_cfg.token_dim
        grid = model_cfg.feature_grid

        self.backbone = FrozenBackbone(
            model_cfg.backbone, image_size=model_cfg.image_size, token_dim=d
        )
        self.exemplar_encoder = RoIExemplarEncoder(
            tokens_per_exemplar=model_cfg.tokens_per_exemplar,
            stride=model_cfg.backbone.stride,
        )
        self.shape_encoder = ShapeEncoder(token_dim=d)

        #: Learnable prototype-query tokens ``X_P`` (Eq. 1).
        self.prototype_queries = nn.Parameter(torch.randn(1, model_cfg.plam.n_prototypes, d) * 0.02)

        self.gfim = GFIM(
            model_cfg.gfim,
            dim=d,
            n_prompt_tokens=model_cfg.n_prompt_tokens,
            n_image_tokens=model_cfg.n_image_tokens,
            grid=grid,
        )
        self.plam = PLAM(model_cfg.plam, model_cfg)
        self.decoder = DensityDecoder(in_channels=d, cfg=model_cfg.decoder, in_size=grid)

        if self.decoder.out_size != model_cfg.density_size:
            raise ValueError(
                f"decoder produces {self.decoder.out_size}px but density_size is "
                f"{model_cfg.density_size}; density_size must be image_size/2"
            )

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #
    def _prepare_image_tokens(
        self, feature_map: Tensor, pad_mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """Flatten the feature map and derive the per-token validity mask."""
        _, _, h, _ = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2).contiguous()
        if pad_mask is None:
            return tokens, None
        image_mask = self.backbone.token_mask_from_padding(pad_mask, h)
        return tokens, image_mask

    def _build_prompt_tokens(
        self,
        feature_map: Tensor,
        boxes: Tensor,
        box_mask: Tensor | None,
        image_hw: tuple[int, int],
    ) -> Tensor:
        """Eq. 1: ``X_T = [X_P; X_E + X_S]``."""
        b = feature_map.shape[0]
        x_e = self.exemplar_encoder(feature_map, boxes, box_mask)

        h, w = image_hw
        scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
        x_s = self.shape_encoder(boxes / scale)  # (B, K, d)
        x_s = x_s.repeat_interleave(self.cfg.tokens_per_exemplar, dim=1)  # A3

        x_p = self.prototype_queries.expand(b, -1, -1)
        return torch.cat([x_p, x_e + x_s], dim=1)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: Tensor,
        boxes: Tensor,
        *,
        pad_mask: Tensor | None = None,
        box_mask: Tensor | None = None,
        need_attn: bool = False,
        return_intermediates: bool = False,
    ) -> GIPACountOutput:
        """Predict a density map for a batch of query images and exemplars.

        Args:
            images: ``(B, 3, H, W)`` normalised, square-padded query images.
            boxes: ``(B, K, 4)`` exemplar boxes in the **same pixel frame** as
                ``images``, ordered ``(x1, y1, x2, y2)``.
            pad_mask: ``(B, 1, H, W)`` boolean, ``True`` on real (non-padded)
                pixels.  ``None`` means "no padding".
            box_mask: ``(B, K)`` boolean, ``True`` for valid exemplars.
            need_attn: Collect detached attention weights (interpretability).
            return_intermediates: Keep ``Ŝ``, ``X̂_P``, ``X'_T`` and ``X^L_IT`` in
                the output (off by default to save memory).

        Returns:
            :class:`GIPACountOutput`.
        """
        cfg = self.cfg
        if images.ndim != 4:
            raise ValueError("images must be (B, 3, H, W)")
        _, _, h, w = images.shape
        if h != w:
            raise ValueError("images must be square-padded")
        grid = h // cfg.backbone.stride
        if grid != cfg.feature_grid:
            raise ValueError(
                f"input {h}px gives a {grid}x{grid} grid but the model was built for "
                f"{cfg.feature_grid}x{cfg.feature_grid} ({cfg.image_size}px)"
            )

        backbone_out = self.backbone(images)
        x_i, image_mask = self._prepare_image_tokens(backbone_out.feature_map, pad_mask)
        x_t = self._build_prompt_tokens(backbone_out.feature_map, boxes, box_mask, image_hw=(h, w))

        gfim_out = self.gfim(
            x_i,
            x_t,
            image_mask=image_mask if cfg.mask_padding_in_attention else None,
            prompt_mask=None,
            need_weights=need_attn,
        )
        plam_out = self.plam(gfim_out.prompt, gfim_out.image)
        raw = self.decoder(plam_out.similarity)
        density = self._activate(raw)

        if cfg.mask_padding_in_output and pad_mask is not None:
            density = density * self._output_mask(pad_mask, density.shape[-1])

        out = GIPACountOutput(
            density=density,
            count=count_from_density(density),
            attn=gfim_out.attn,
        )
        if return_intermediates:
            out.similarity = plam_out.similarity
            out.prototype_kernel = plam_out.kernel
            out.prompt_tokens = gfim_out.prompt
            out.image_tokens = gfim_out.image
        return out

    def _activate(self, raw: Tensor) -> Tensor:
        """Apply the output activation (assumption A9)."""
        mode = self.cfg.output_activation
        if mode == "softplus":
            return F.softplus(raw)
        if mode == "relu":
            return F.relu(raw)
        return raw

    @staticmethod
    def _output_mask(pad_mask: Tensor, size: int) -> Tensor:
        """Down-sample ``(B, 1, H, W)`` padding to the density resolution."""
        return (F.adaptive_avg_pool2d(pad_mask.to(torch.float32), size) > 0.5).to(pad_mask.dtype)

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    @property
    def num_parameters(self) -> int:
        """Total parameter count (frozen backbone included)."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Parameters that receive gradients."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Trainable / total parameters per top-level module (for Table 6 checks)."""
        breakdown: dict[str, int] = {}
        for name, module in self.named_children():
            breakdown[name] = sum(p.numel() for p in module.parameters())
            breakdown[f"{name}.trainable"] = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
        breakdown["total"] = self.num_parameters
        breakdown["trainable"] = self.num_trainable_parameters
        return breakdown

    def freeze_backbone(self, frozen: bool = True) -> None:
        """Toggle gradients for the pretrained encoder (normally always frozen)."""
        for param in self.backbone.encoder.parameters():
            param.requires_grad = not frozen
        self.backbone.encoder.train(not frozen)
