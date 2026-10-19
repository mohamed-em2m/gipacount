"""Global Feature Interaction Module (GFIM), §3.3 of the paper."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor, nn

from gipacount.config import GFIMConfig
from gipacount.models.attention import MultiheadAttention
from gipacount.models.layers import Mlp
from gipacount.models.pos_embed import TokenPositionalEncoding

__all__ = ["GFIM", "BidirectionalBlock", "GFIMOutput"]


@dataclass(slots=True)
class GFIMOutput:
    """Output of :class:`GFIM`.

    Attributes:
        prompt: ``X'_T`` of shape ``(B, N_T, d)``; its first ``N_P`` tokens are the
            global prototype tokens used by PLAM.
        image: ``X^L_IT`` of shape ``(B, N_I, d)``, the exemplar-enhanced image
            tokens.
        attn: Optional dict of detached attention weights per operation.
    """

    prompt: Tensor
    image: Tensor
    attn: dict[str, Tensor] = field(default_factory=dict)


class BidirectionalBlock(nn.Module):
    """One GFIM layer: prompt self-attention, p->i, i->p cross-attention.

    Implements Eqs. (2)-(4):

    * Eq. 2 - ``A^{l}_{T->T} = softmax(Q Kᵀ/√d)``, ``X^l_TT = A V + X^{l-1}_TI``
    * Eq. 3 - ``A^{l}_{T->I} = softmax(Q Kᵀ/√d)``,
      ``X^l_TI = M(A V + X^l_TT)`` (see ``cross_attn_fusion``)
    * Eq. 4 - ``A^{l}_{I->T} = softmax(Q Kᵀ/√d)``,
      ``X^l_IT = A V + X^{l-1}_IT``

    All three use 8-head attention over ``d = 256`` in the paper, with a
    pre-LayerNorm and a residual connection around every sub-block (§4.3).

    Args:
        cfg: GFIM configuration (which of the three ops are active, fusion mode).
        dim: Token dimension ``d``.
    """

    def __init__(self, cfg: GFIMConfig, dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = dim

        # --- Eq. 2: prompt self-attention -------------------------------------
        if cfg.use_prompt_self_attention:
            self.ln_tt = nn.LayerNorm(dim)
            self.self_attn = MultiheadAttention(
                dim, cfg.n_heads, attn_dropout=cfg.attn_dropout, proj_dropout=cfg.dropout
            )

        # --- Eq. 3: prompt -> image cross-attention ---------------------------
        if cfg.use_prompt_to_image:
            self.ln_ti_q = nn.LayerNorm(dim)
            self.ln_ti_kv = nn.LayerNorm(dim)
            self.ln_ti_out = nn.LayerNorm(dim)
            self.t2i_attn = MultiheadAttention(
                dim, cfg.n_heads, attn_dropout=cfg.attn_dropout, proj_dropout=cfg.dropout
            )
            self.fusion_mlp = Mlp(
                dim,
                hidden_features=int(dim * cfg.fusion_mlp_ratio),
                act="gelu",
                dropout=cfg.dropout,
            )

        # --- Eq. 4: image -> prompt cross-attention ---------------------------
        if cfg.use_image_to_prompt:
            self.ln_it_q = nn.LayerNorm(dim)
            self.ln_it_kv = nn.LayerNorm(dim)
            self.ln_it_out = nn.LayerNorm(dim)
            self.i2t_attn = MultiheadAttention(
                dim, cfg.n_heads, attn_dropout=cfg.attn_dropout, proj_dropout=cfg.dropout
            )

        # --- Transformer MLP (§4.3: two linear layers, GELU, hidden 2048) -----
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(
            dim, hidden_features=int(dim * cfg.mlp_ratio), act="gelu", dropout=cfg.dropout
        )

        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x_t: Tensor,
        x_i: Tensor,
        *,
        image_mask: Tensor | None = None,
        prompt_mask: Tensor | None = None,
        need_weights: bool = False,
        tag: str = "",
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Run one interaction layer.

        Args:
            x_t: ``(B, N_T, d)`` prompt tokens, i.e. ``X^{l-1}_TI``.
            x_i: ``(B, N_I, d)`` image tokens, i.e. ``X^{l-1}_IT``.
            image_mask: ``(B, N_I)`` boolean validity mask for image tokens.
            prompt_mask: ``(B, N_T)`` boolean validity mask for prompt tokens.
            need_weights: Also return detached attention weights.
            tag: Prefix for the returned attention keys.

        Returns:
            ``(X^l_TI, X^l_IT, aux)``.
        """
        aux: dict[str, Tensor] = {}
        img_pad = None if image_mask is None else ~image_mask
        prompt_pad = None if prompt_mask is None else ~prompt_mask

        # ---- Eq. 2: prompt self-attention ------------------------------------
        if self.cfg.use_prompt_self_attention:
            normed = self.ln_tt(x_t)
            attn_out, w = self.self_attn(
                normed,
                normed,
                normed,
                key_padding_mask=prompt_pad,
                need_weights=need_weights,
            )
            x_tt = x_t + self.drop(attn_out)
            if w is not None:
                aux[f"{tag}t2t"] = w
        else:
            x_tt = x_t

        # ---- Eq. 3: prompt -> image cross-attention --------------------------
        if self.cfg.use_prompt_to_image:
            q = self.ln_ti_q(x_tt)
            kv = self.ln_ti_kv(x_i)
            attn_out, w = self.t2i_attn(
                q, kv, kv, key_padding_mask=img_pad, need_weights=need_weights
            )
            if self.cfg.cross_attn_fusion == "residual_mlp":
                # LN(X_TT + M(A V))                    (assumption A5, default)
                x_ti = self.ln_ti_out(x_tt + self.drop(self.fusion_mlp(attn_out)))
            else:
                # LN(M(A V + X_TT))                    (assumption A5, literal)
                x_ti = self.ln_ti_out(self.drop(self.fusion_mlp(attn_out + x_tt)))
            if w is not None:
                aux[f"{tag}t2i"] = w
        else:
            x_ti = x_tt

        # ---- Eq. 4: image -> prompt cross-attention --------------------------
        if self.cfg.use_image_to_prompt:
            q = self.ln_it_q(x_i)
            kv = self.ln_it_kv(x_ti)
            attn_out, w = self.i2t_attn(
                q, kv, kv, key_padding_mask=prompt_pad, need_weights=need_weights
            )
            x_it = self.ln_it_out(x_i + self.drop(attn_out))
            if w is not None:
                aux[f"{tag}i2t"] = w
        else:
            x_it = x_i

        # ---- Transformer MLP with residual (per layer) -----------------------
        x_it = x_it + self.drop(self.mlp(self.ln_mlp(x_it)))

        return x_ti, x_it, aux

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(dim={self.dim}, heads={self.cfg.n_heads}, "
            f"self_attn={self.cfg.use_prompt_self_attention}, "
            f"t2i={self.cfg.use_prompt_to_image}, i2t={self.cfg.use_image_to_prompt})"
        )


class _FinalFusion(nn.Module):
    """The extra prompt->image attention of Eq. (5).

    ``A'_{T->I} = softmax(Q Kᵀ/√d)``, ``X'_T = A' V + X^L_TI`` with its own
    learned projections ``W'^q, W'^k, W'^v`` for the output layer.
    """

    def __init__(self, dim: int, n_heads: int, *, attn_dropout: float, proj_dropout: float) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ln_out = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(
            dim, n_heads, attn_dropout=attn_dropout, proj_dropout=proj_dropout
        )
        self.drop = nn.Dropout(proj_dropout)

    def forward(
        self,
        x_t: Tensor,
        x_i: Tensor,
        *,
        image_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Returns ``(X'_T, attention weights | None)``."""
        img_pad = None if image_mask is None else ~image_mask
        attn_out, w = self.attn(
            self.ln_q(x_t),
            self.ln_kv(x_i),
            self.ln_kv(x_i),
            key_padding_mask=img_pad,
            need_weights=need_weights,
        )
        return self.ln_out(x_t + self.drop(attn_out)), w


class GFIM(nn.Module):
    """Stack of ``L`` bidirectional interaction layers plus the final fusion.

    Paper §4.3: "In the first layer, self-attention is applied only to the prompt
    tokens without positional encoding. From the second layer onward, absolute
    positional encodings are introduced for both image and prompt tokens."  The
    image tokens are *never* self-attended (that is what keeps the complexity at
    ``O(N_I · N_T)`` rather than ``O(N_I²)``, §4.4.3).

    Args:
        cfg: GFIM configuration.
        dim: Token dimension ``d`` (256 in the paper).
        n_prompt_tokens: ``N_T = N_P + N_E``.
        n_image_tokens: ``N_I`` (4096 in the paper).
        grid: Side ``√N_I`` (64 in the paper).
    """

    def __init__(
        self,
        cfg: GFIMConfig,
        dim: int,
        n_prompt_tokens: int,
        n_image_tokens: int,
        grid: int,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.n_prompt_tokens = n_prompt_tokens
        self.n_image_tokens = n_image_tokens
        self.grid = grid

        self.blocks = nn.ModuleList(BidirectionalBlock(cfg, dim) for _ in range(cfg.layers))

        # Per-layer absolute positional encodings, active from ``pos_from_layer``.
        self.prompt_pos = nn.ModuleList()
        self.image_pos = nn.ModuleList()
        for layer_idx in range(cfg.layers):
            active = cfg.pos_encoding != "none" and layer_idx >= cfg.pos_from_layer
            self.prompt_pos.append(
                TokenPositionalEncoding(n_prompt_tokens, dim, kind=cfg.pos_encoding)
                if active and cfg.pos_on_prompt
                else nn.Identity()
            )
            self.image_pos.append(
                TokenPositionalEncoding(n_image_tokens, dim, kind=cfg.pos_encoding, grid_size=grid)
                if active and cfg.pos_on_image
                else nn.Identity()
            )

        self.final_fusion = (
            _FinalFusion(dim, cfg.n_heads, attn_dropout=cfg.attn_dropout, proj_dropout=cfg.dropout)
            if cfg.use_final_fusion
            else None
        )

    def _add_pos(self, module: nn.Module, tokens: Tensor) -> Tensor:
        if isinstance(module, nn.Identity):
            return tokens
        return tokens + module(tokens.shape[1]).to(dtype=tokens.dtype)

    def forward(
        self,
        image_tokens: Tensor,
        prompt_tokens: Tensor,
        *,
        image_mask: Tensor | None = None,
        prompt_mask: Tensor | None = None,
        need_weights: bool = False,
    ) -> GFIMOutput:
        """Run the full GFIM stack.

        Args:
            image_tokens: ``X_I``, ``(B, N_I, d)``.
            prompt_tokens: ``X_T``, ``(B, N_T, d)``.
            image_mask: Optional ``(B, N_I)`` boolean validity mask.
            prompt_mask: Optional ``(B, N_T)`` boolean validity mask.
            need_weights: Collect detached attention weights for analysis.

        Returns:
            :class:`GFIMOutput` with ``X'_T`` and ``X^L_IT``.
        """
        if image_tokens.shape[1] != self.n_image_tokens:
            raise ValueError(
                f"expected {self.n_image_tokens} image tokens, got {image_tokens.shape[1]}"
            )
        if prompt_tokens.shape[1] != self.n_prompt_tokens:
            raise ValueError(
                f"expected {self.n_prompt_tokens} prompt tokens, got {prompt_tokens.shape[1]}"
            )

        x_i = image_tokens
        x_t = prompt_tokens
        aux: dict[str, Tensor] = {}

        for layer_idx, block in enumerate(self.blocks):
            prep_t = self._add_pos(self.prompt_pos[layer_idx], x_t)
            prep_i = self._add_pos(self.image_pos[layer_idx], x_i)
            x_t, x_i, layer_aux = block(
                prep_t,
                prep_i,
                image_mask=image_mask,
                prompt_mask=prompt_mask,
                need_weights=need_weights,
                tag=f"layer{layer_idx}.",
            )
            aux.update(layer_aux)

        x_t_prime = x_t
        if self.final_fusion is not None:
            x_t_prime, w = self.final_fusion(
                x_t, x_i, image_mask=image_mask, need_weights=need_weights
            )
            if w is not None:
                aux["final.t2i"] = w

        return GFIMOutput(prompt=x_t_prime, image=x_i, attn=aux)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
