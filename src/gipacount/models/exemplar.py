"""Exemplar region encoding: ROI pooling of boxes into ``X_E`` (§3.2)."""

from __future__ import annotations

from torch import Tensor, nn
from torchvision.ops import roi_align

__all__ = ["RoIExemplarEncoder"]


class RoIExemplarEncoder(nn.Module):
    """Turn ``K`` exemplar boxes into ``K * tokens_per_exemplar`` tokens.

    The paper (§3.2, §4.3): "the exemplar bounding boxes ``E_k`` are downscaled by
    a factor of 16 to align with the feature map resolution, and ROI pooling is
    applied to obtain exemplar tokens ``X_E``, where ``N_E = K x 9`` under the
    K-shot setting".

    We pool on the *projected* feature map (assumption A2) with a ``3 x 3``
    output grid, which yields exactly 9 tokens per exemplar.

    Args:
        tokens_per_exemplar: Tokens per box; must be a perfect square (9 -> 3x3).
        stride: Backbone stride mapping image pixels to feature-map cells (16).
    """

    def __init__(self, tokens_per_exemplar: int = 9, stride: int = 16) -> None:
        super().__init__()
        side = int(round(tokens_per_exemplar**0.5))
        if side * side != tokens_per_exemplar:
            raise ValueError("tokens_per_exemplar must be a perfect square")
        self.side = side
        self.tokens_per_exemplar = tokens_per_exemplar
        self.stride = stride

    def forward(
        self,
        feature_map: Tensor,
        boxes: Tensor,
        box_mask: Tensor | None = None,
    ) -> Tensor:
        """Pool exemplar boxes out of ``feature_map``.

        Args:
            feature_map: ``(B, d, Hf, Wf)`` projected backbone features.
            boxes: ``(B, K, 4)`` boxes in **input-image pixel** coordinates,
                ordered ``(x1, y1, x2, y2)``.
            box_mask: Optional ``(B, K)`` boolean, ``True`` = valid box.  Invalid
                boxes are pooled from a degenerate region and zeroed afterwards.

        Returns:
            ``(B, K * tokens_per_exemplar, d)`` exemplar tokens ``X_E``.
        """
        b, d, hf, wf = feature_map.shape
        k = boxes.shape[1]
        if boxes.shape != (b, k, 4):
            raise ValueError(f"boxes must be (B, K, 4), got {tuple(boxes.shape)}")

        safe_boxes = boxes
        if box_mask is not None:
            # Degenerate (empty) boxes keep roi_align happy; outputs are zeroed.
            valid = box_mask.to(device=boxes.device, dtype=boxes.dtype).unsqueeze(-1)
            safe_boxes = boxes * valid

        rois = [safe_boxes[i] for i in range(b)]
        pooled = roi_align(
            feature_map,
            rois,
            output_size=(self.side, self.side),
            spatial_scale=1.0 / self.stride,
            sampling_ratio=2,
            aligned=True,
        )  # (B*K, d, side, side): torchvision concatenates the per-image boxes
        pooled = pooled.view(b, k, d, self.side, self.side)
        tokens = pooled.flatten(3, 4).permute(0, 1, 3, 2)  # (B, K, side*side, d)

        if box_mask is not None:
            tokens = tokens * box_mask.to(tokens.dtype).view(b, k, 1, 1)

        if hf * wf == 0:  # pragma: no cover - defensive
            raise RuntimeError("empty feature map")
        return tokens.reshape(b, k * self.tokens_per_exemplar, d)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"tokens_per_exemplar={self.tokens_per_exemplar}, stride={self.stride}"
