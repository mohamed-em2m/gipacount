"""Typed configuration objects for GIPACount.

No hyper-parameter is hard-coded in the model or training code: every value the
paper specifies (and every value it does not, see ``docs/paper_spec.md`` §4) lives
here and is loaded from YAML through :func:`load_config`.

Layout
------
``configs/*.yaml`` mirrors the nested structure of :class:`ExperimentConfig`, e.g.::

    model:
      gfim:
        layers: 6
        n_heads: 8

Source tags in the docstrings refer to ``docs/paper_spec.md``: **[P]** paper,
**[D]** our documented default for an unspecified detail.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BackboneConfig",
    "CCSSConfig",
    "DataConfig",
    "DecoderConfig",
    "EvalConfig",
    "ExperimentConfig",
    "GFIMConfig",
    "LossConfig",
    "ModelConfig",
    "OptimConfig",
    "PLAMConfig",
    "TrainerConfig",
    "load_config",
    "save_config",
]


class _Base(BaseModel):
    """Common base: forbid unknown keys so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=False)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class BackboneConfig(_Base):
    """Frozen pretrained image encoder (§4.3 of the paper).

    The paper only says "a frozen pretrained ViT or ResNet backbone" with
    ``N_I = 64**2`` tokens and ``d = 256``; the exact checkpoint is never named
    (assumption A1).  ``kind`` selects the tokeniser:

    * ``"vit"`` -> ``timm`` Vision Transformer, tokens = patch grid.
    * ``"cnn"`` -> ``timm`` CNN feature extractor at ``out_indices``.
    """

    kind: Literal["vit", "cnn"] = "vit"
    name: str = "vit_small_patch16_224"
    pretrained: bool = True
    #: For ``kind="cnn"``: which feature-pyramid levels to take.  ``(2,)`` is
    #: ResNet layer3 (stride 16) -> 64x64 at 1024 input, as in GIPACount-R.
    out_indices: tuple[int, ...] = (2,)
    #: Spatial stride of the extracted feature map (16 in the paper).
    stride: int = 16
    #: Token/feature dimension ``d`` after projection (256 in the paper).
    token_dim: int = 256
    #: Keep the encoder frozen and train only the head (paper's main setting).
    frozen: bool = True
    #: Number of *final* backbone blocks to unfreeze (ablation Q7 uses the last 3).
    unfreeze_last_n: int = 0
    #: Whether the ``d``-projection that follows the backbone is trainable (A17).
    train_projection: bool = True

    @model_validator(mode="after")
    def _check_unfreeze(self) -> BackboneConfig:
        if self.unfreeze_last_n < 0:
            raise ValueError("unfreeze_last_n must be >= 0")
        if not self.frozen and self.unfreeze_last_n:
            raise ValueError("set frozen=True with unfreeze_last_n>0, or frozen=False")
        return self


class GFIMConfig(_Base):
    """Global Feature Interaction Module (§3.3)."""

    #: ``L``, number of stacked bidirectional interaction layers. [P] = 6
    layers: int = 6
    #: Attention heads. [P] = 8
    n_heads: int = 8
    #: Ratio of the MLP hidden width to ``d``: hidden = mlp_ratio * token_dim.
    #: [P] hidden = 2048 with d = 256  =>  mlp_ratio = 8.
    mlp_ratio: float = 8.0
    dropout: float = 0.0
    attn_dropout: float = 0.0
    #: Pre-LN (paper: "layer normalization is applied before each attention and
    #: MLP block").
    pre_norm: bool = True
    #: Ablations from Table 7/8.  Disabling one of these degrades the module.
    use_prompt_self_attention: bool = True
    use_prompt_to_image: bool = True
    use_image_to_prompt: bool = True
    #: The extra prompt->image attention after the stack, Eq. 5 ("final fusion").
    use_final_fusion: bool = True
    #: How Eq. 3's ``M(A V + X_TT)`` is realised (A5).
    #: ``"residual_mlp"``: ``LN(X_TT + M(A V))``; ``"literal"``: ``LN(M(A V + X_TT))``.
    cross_attn_fusion: Literal["residual_mlp", "literal"] = "residual_mlp"
    #: Absolute positional encoding type (A6).
    pos_encoding: Literal["learnable", "sinusoidal", "none"] = "learnable"
    #: 0-based layer index from which positional encodings are added.
    #: [P] "from the second layer onward" => 1.
    pos_from_layer: int = 1
    #: Add positional encodings to image tokens as well as prompt tokens.
    pos_on_image: bool = True
    pos_on_prompt: bool = True
    #: Non-linear fusion MLP width multiplier for Eq. 3's ``M`` (relative to d).
    fusion_mlp_ratio: float = 8.0

    @field_validator("n_heads")
    @classmethod
    def _check_heads(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_heads must be positive")
        return v


class PLAMConfig(_Base):
    """Prototype-guided Local Aggregation Module (§3.4)."""

    enabled: bool = True
    #: ``N_P``, number of learnable prototype-query tokens. [P] = 9 (3x3 kernel).
    n_prototypes: int = 9
    #: Where the depthwise kernel comes from (Table 8 Panel B).
    #: ``"prototype"`` = full PLAM; ``"exemplar"`` = ablation P1/P2.
    kernel_source: Literal["prototype", "exemplar"] = "prototype"
    #: Hidden width of the prototype MLP, Eq. 7.  ``None`` -> keep ``d`` wide.
    mlp_hidden: int | None = None
    activation: Literal["gelu", "relu"] = "gelu"
    #: 1x1 convolution that fuses channel-wise responses (Eq. 10 -> Ŝ).
    channel_fuse: bool = True
    #: Padding ``p`` of the depthwise convolution, Eq. 10.  ``None`` -> "same".
    padding: int | None = None

    @model_validator(mode="after")
    def _check_square(self) -> PLAMConfig:
        root = math.isqrt(self.n_prototypes)
        if root * root != self.n_prototypes:
            raise ValueError(
                f"n_prototypes must be a perfect square (Eq. 8), got {self.n_prototypes}"
            )
        return self


class DecoderConfig(_Base):
    """Density decoder ``Dec(·)``, Eq. 11 and §4.3."""

    #: Output channels of each of the three transposed convolutions.
    channels: tuple[int, ...] = (128, 64, 32)
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    norm: Literal["layernorm", "groupnorm", "none"] = "layernorm"
    activation: Literal["relu", "gelu"] = "relu"
    #: Final point-wise (1x1) convolution to a single density channel.
    out_channels: int = 1

    @model_validator(mode="after")
    def _check_len(self) -> DecoderConfig:
        if len(self.channels) != 3:
            raise ValueError("the paper uses exactly 3 transposed convolution layers")
        return self


class ModelConfig(_Base):
    """Top-level architecture configuration."""

    #: Square input resolution used for training *and* inference. [P] = 1024
    image_size: int = 1024
    #: Square output density resolution. [P] = 512
    density_size: int = 512
    #: Feature/token dimension ``d``. [P] = 256
    token_dim: int = 256
    #: Number of exemplars per sample, ``K``. [P] = 1 or 3
    n_exemplars: int = 3
    #: Tokens produced per exemplar box (3x3 ROI pool -> 9). [P] = K * 9
    tokens_per_exemplar: int = 9
    #: Activation applied to the raw decoder output (A9).
    output_activation: Literal["softplus", "relu", "none"] = "softplus"
    #: Zero the predicted density in padded regions (A9).
    mask_padding_in_output: bool = True
    #: Mask padded image tokens as attention keys (paper §4.3).
    mask_padding_in_attention: bool = True
    backbone: BackboneConfig = Field(default_factory=BackboneConfig)
    gfim: GFIMConfig = Field(default_factory=GFIMConfig)
    plam: PLAMConfig = Field(default_factory=PLAMConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> ModelConfig:
        if self.image_size % self.backbone.stride:
            raise ValueError("image_size must be divisible by the backbone stride")
        grid = self.image_size // self.backbone.stride
        side = math.isqrt(self.plam.n_prototypes)
        if side * side != self.plam.n_prototypes:  # pragma: no cover - guarded above
            raise ValueError("n_prototypes must be a perfect square")
        if self.density_size != self.image_size // 2:
            raise ValueError(
                "the paper predicts the density map at half the input resolution "
                "(1024 -> 512): density_size must equal image_size // 2"
            )
        if grid * (self.decoder.stride ** len(self.decoder.channels)) != self.density_size:
            raise ValueError(
                f"decoder with {len(self.decoder.channels)} stride-{self.decoder.stride} "
                f"transposed convolutions maps a {grid}px grid to "
                f"{grid * self.decoder.stride ** len(self.decoder.channels)}px, "
                f"but density_size={self.density_size}"
            )
        return self

    @property
    def feature_grid(self) -> int:
        """``√N_I``: side of the backbone feature map in tokens."""
        return self.image_size // self.backbone.stride

    @property
    def n_image_tokens(self) -> int:
        """``N_I`` (4096 in the paper)."""
        return self.feature_grid**2

    @property
    def n_exemplar_tokens(self) -> int:
        """``N_E = K × tokens_per_exemplar`` (K × 9 in the paper)."""
        return self.n_exemplars * self.tokens_per_exemplar

    @property
    def n_prompt_tokens(self) -> int:
        """``N_T = N_P + N_E``."""
        return self.plam.n_prototypes + self.n_exemplar_tokens


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class CCSSConfig(_Base):
    """Category-Contrastive Supervision Strategy (§3.5, Table 9)."""

    enabled: bool = True
    #: ``P_mix``, probability of using the collage instead of single images. [P] 0.5
    p_mix: float = 0.5
    #: Grid of the collage; Table 9 finds 2x2 better than 3x3. [P] 2
    grid: int = 2
    #: Long side each sub-image is resized to before padding. [P] 512 for 2x2
    sub_image_size: int = 512
    #: Zero point supervision in sub-regions whose category != anchor category
    #: (Eq. 15).  Disable to reproduce the "same category" ablation of Table 9.
    category_filter: bool = True
    #: "random" = Table 9 default; "same"/"different" = ablations.
    composition: Literal["random", "same", "different"] = "random"
    #: Deterministic search radius for duplicate-point reassignment (§3.5). [P] 1
    duplicate_radius: int = 1
    seed: int | None = None

    @field_validator("p_mix")
    @classmethod
    def _check_p(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("p_mix must be in [0, 1]")
        return v

    @field_validator("grid")
    @classmethod
    def _check_grid(cls, v: int) -> int:
        if v < 2:
            raise ValueError("grid must be >= 2")
        return v


class DataConfig(_Base):
    """Dataset / preprocessing configuration (§4.1, §4.3)."""

    name: Literal["fsc147", "synthetic"] = "fsc147"
    #: Root that holds the raw datasets (never committed, see `.gitignore`).
    root: Path = Path("data")
    #: Long side each single image is resized to before square padding.
    #: In CCSS mode this is ``ccss.sub_image_size`` instead. [P] 1024 / 512
    image_size: int = 1024
    #: Number of workers for the training loader (0 => in-process, deterministic).
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False
    #: Dataset split file names (FSC-147 standard release).
    split_file: str = "Train_Test_Val_FSC_147.json"
    annotation_file: str = "annotation_FSC147_384.json"
    image_dir: str = "images_384_VarV2"
    #: Also read per-image category names (needed by CCSS, Eq. 15).
    class_file: str = "ImageClasses_FSC147.txt"
    #: Fall back to the 1024-px release if the 384-px one is absent.
    image_dir_fallback: str = "images_384_VarV2"
    #: Deterministic exemplar sampling: which K of the 3 provided boxes to use.
    exemplar_sampling: Literal["first", "random"] = "random"
    #: ImageNet statistics (A18), used for backbone normalisation.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    #: Augmentation (paper §4.3: hflip + colour jitter, LOCA-style).
    random_hflip: bool = True
    color_jitter: bool = True
    color_jitter_strength: float = 0.4
    #: Synthetic dataset only.
    synthetic_train_size: int = 32
    synthetic_val_size: int = 8
    synthetic_image_size: int = 256
    ccss: CCSSConfig = Field(default_factory=CCSSConfig)

    @model_validator(mode="after")
    def _check_dirs(self) -> DataConfig:
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean/std must have 3 entries")
        if self.ccss.enabled:
            grid = self.ccss.grid
            if self.image_size % grid:
                raise ValueError(
                    f"image_size={self.image_size} must be divisible by the CCSS grid={grid}"
                )
            expected = self.image_size // grid
            if self.ccss.sub_image_size != expected:
                raise ValueError(
                    f"CCSS sub_image_size must be image_size/grid = {expected}, "
                    f"got {self.ccss.sub_image_size}"
                )
        return self


# --------------------------------------------------------------------------- #
# Loss / optimisation
# --------------------------------------------------------------------------- #
class LossConfig(_Base):
    """Generalized Loss, Eq. 12 (Wan et al., CVPR 2021)."""

    name: Literal["generalized", "l1", "l2"] = "generalized"
    #: Entropic-regularisation strength ``ε``. [P] 0.01
    epsilon: float = 0.01
    #: Transport term weight ``τ``. [P] 0.5
    tau: float = 0.5
    #: Number of (generalized) Sinkhorn iterations.
    sinkhorn_iters: int = 50
    #: ε-scaling heuristic from Wan et al. for faster convergence (A11).
    epsilon_scaling: bool = True
    #: Initial ε as a multiple of ``epsilon`` when scaling is on.
    epsilon_scale_start: float = 16.0
    #: Pixel-axis chunk size for the ``N_pixels × N_points`` cost matrix (A10).
    chunk_size: int = 4096
    #: Coordinate frame of the cost matrix: unit-normalised (A10) or raw pixels.
    coord_norm: Literal["unit", "pixel"] = "unit"
    #: Precision of the optimal-transport solve (float64 avoids underflow in
    #: ``exp(-C/ε)`` at the paper's tiny ``ε``).
    dtype: Literal["float32", "float64"] = "float64"
    #: Numerical floor for divisions (``a`` and ``b``).
    eps_floor: float = 1e-6
    #: Clamp on the Sinkhorn scale factors, so that ``u * K`` cannot produce NaN.
    u_max: float = 1e12
    #: Add ``tau * ||R V - D||_1`` style point-wise term; matches Eq. 12's D2.
    point_loss: Literal["l1", "l2"] = "l1"

    @field_validator("epsilon", "tau")
    @classmethod
    def _positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("epsilon and tau must be positive")
        return v


class OptimConfig(_Base):
    """Adam optimiser settings (§4.3)."""

    name: Literal["adam", "adamw", "sgd"] = "adam"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    #: No schedule is specified by the paper (A16).
    scheduler: Literal["none", "cosine", "multistep", "step"] = "none"
    warmup_epochs: int = 0
    min_lr: float = 0.0
    milestones: tuple[int, ...] = (100, 150)
    gamma: float = 0.1
    grad_clip_norm: float | None = None


class TrainerConfig(_Base):
    """Training loop settings (Algorithm 1)."""

    epochs: int = 200
    batch_size: int = 4
    #: ``None`` -> one epoch = ``len(train_loader)`` iterations (A15).
    steps_per_epoch: int | None = None
    #: Validate every N epochs.
    val_every: int = 1
    seed: int = 0
    deterministic: bool = True
    amp: bool = False
    device: str = "auto"
    #: Where checkpoints and logs are written.
    output_dir: Path = Path("runs")
    run_name: str | None = None
    #: Keep the best-validation-MAE checkpoint and the last one.
    save_best: bool = True
    save_last: bool = True
    save_every: int = 0
    resume: Path | None = None
    #: Experiment tracking (off by default).
    tensorboard: bool = False
    wandb: bool = False
    wandb_project: str = "gipacount"
    #: Log every N optimisation steps.
    log_every: int = 50
    num_workers: int | None = None
    #: Optional cap on the number of training iterations (smoke tests).
    max_steps: int | None = None

    @field_validator("batch_size")
    @classmethod
    def _check_bs(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v


class EvalConfig(_Base):
    """Evaluation settings (§4.2)."""

    split: str = "val"
    batch_size: int = 1
    num_workers: int = 0
    #: Build the composite 2x2 evaluation set for ECR (Eq. 21).
    compute_ecr: bool = False
    ecr_num_composites: int | None = None
    #: ``ε₀`` in Eq. 21.
    ecr_eps: float = 1e-8
    save_density_maps: bool = False
    checkpoint: Path | None = None


class ExperimentConfig(_Base):
    """Root configuration object."""

    name: str = "gipacount"
    #: Reproduce one of the paper's ablations by toggling modules (Tables 7/8).
    ablation: Literal[
        "none",
        "q1",  # one-way p->i cross-attention only, no prototypes, no PLAM
        "q2",  # + bidirectional cross-attention
        "q3",  # one-way + PLAM
        "q4",  # GFIM off, CCSS only
        "q5",  # GFIM + PLAM, no CCSS
        "q6",  # full
        "q7",  # full + last 3 backbone layers unfrozen
    ] = "none"
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optim: OptimConfig = Field(default_factory=OptimConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)

    @model_validator(mode="after")
    def _apply_ablation(self) -> ExperimentConfig:
        """Translate an ablation name into concrete module switches."""
        if self.data.image_size != self.model.image_size:
            raise ValueError(
                f"data.image_size ({self.data.image_size}) must equal "
                f"model.image_size ({self.model.image_size})"
            )
        if self.data.image_size // 2 != self.model.density_size:
            raise ValueError(
                "data.image_size/2 must equal model.density_size "
                f"({self.data.image_size // 2} != {self.model.density_size})"
            )
        a = self.ablation
        g, p = self.model.gfim, self.model.plam
        d = self.data.ccss

        if a in {"q1", "q3"}:
            # One-way exemplar->image cross-attention; no global prototypes.
            g.use_prompt_self_attention = False
            g.use_image_to_prompt = False
            g.use_prompt_to_image = True
            g.use_final_fusion = a == "q3"
            p.n_prototypes = 1
        if a == "q2":
            g.use_prompt_self_attention = False
            g.use_image_to_prompt = True
            g.use_prompt_to_image = True
        if a in {"q1", "q2"}:
            p.enabled = False
            d.enabled = False
        if a == "q3":
            p.enabled = True
            d.enabled = False
        if a == "q4":
            # GFIM disabled, PLAM disabled, CCSS only.
            g.use_prompt_self_attention = False
            g.use_image_to_prompt = False
            g.use_final_fusion = False
            p.enabled = False
        if a == "q5":
            d.enabled = False
        if a == "q7":
            self.model.backbone.frozen = True
            self.model.backbone.unfreeze_last_n = 3
        return self


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml_with_bases(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Read ``path``, recursively resolving its optional ``_base_`` entries.

    ``_base_`` may be a path or a list of paths (relative to the including file);
    later bases and the file's own keys take precedence.
    """
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"circular `_base_` chain: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    bases = raw.pop("_base_", None)
    if not bases:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        if not base_path.is_file():
            raise FileNotFoundError(f"`_base_` not found: {base_path} (from {path})")
        merged = _deep_merge(merged, _read_yaml_with_bases(base_path, (*seen, resolved)))
    return _deep_merge(merged, raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Unknown keys raise, so a typo in a config never silently changes a run.  A
    top-level ``_base_: ../default.yaml`` key (or a list of them) pulls in another
    config as a base, which keeps experiment and ablation files short.

    Args:
        path: YAML file to read.
        overrides: Optional ``["trainer.epochs=5", "model.gfim.layers=2"]``
            dotted-path overrides applied before validation.

    Returns:
        The validated configuration.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    raw = _read_yaml_with_bases(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return ExperimentConfig.model_validate(raw)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.path=value`` overrides to a nested config dict (in place)."""
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not of the form key.path=value")
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ValueError(f"override {item!r} has an empty key")
        node: Any = raw
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            if not isinstance(child, dict):
                raise ValueError(f"cannot descend into non-mapping key {part!r}")
            node = child
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def save_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    """Serialise ``cfg`` to YAML (used to snapshot every run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cfg.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=False)
    return path
