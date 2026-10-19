"""Model: shapes, parameter budget, gradient flow, determinism, ablations."""

from __future__ import annotations

import pytest
import torch

from gipacount.config import ExperimentConfig, ModelConfig
from gipacount.models.decoder import DensityDecoder
from gipacount.models.exemplar import RoIExemplarEncoder
from gipacount.models.gfim import GFIM
from gipacount.models.gipacount import GIPACount, count_from_density
from gipacount.models.plam import PLAM, prototype_depthwise_conv
from tests.conftest import make_config


def _inputs(cfg: ExperimentConfig, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    size = cfg.model.image_size
    images = torch.randn(batch, 3, size, size)
    boxes = torch.tensor([[[4.0, 4.0, 24.0, 24.0]] * cfg.model.n_exemplars] * batch)
    return images, boxes


def test_forward_shapes(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config)
    model.eval()
    images, boxes = _inputs(tiny_config, batch=2)
    with torch.no_grad():
        out = model(images, boxes)
    assert out.density.shape == (2, 1, 32, 32)
    assert out.count.shape == (2,)
    assert torch.isfinite(out.density).all()
    assert (out.density >= 0).all()  # softplus activation (A9)


def test_intermediates_shapes(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config)
    model.eval()
    images, boxes = _inputs(tiny_config, batch=2)
    with torch.no_grad():
        out = model(images, boxes, return_intermediates=True)
    d = tiny_config.model.token_dim
    grid = tiny_config.model.feature_grid
    assert out.similarity is not None and out.similarity.shape == (2, d, grid, grid)
    assert out.prototype_kernel is not None and out.prototype_kernel.shape == (2, d, 3, 3)
    assert out.image_tokens is not None and out.image_tokens.shape == (2, grid * grid, d)
    assert out.prompt_tokens is not None
    assert out.prompt_tokens.shape[1] == tiny_config.model.n_prompt_tokens


def test_count_from_density_integrates() -> None:
    density = torch.ones(3, 1, 4, 4)
    counts = count_from_density(density)
    assert counts.shape == (3,)
    assert torch.allclose(counts, torch.full((3,), 16.0))


def test_parameter_breakdown_is_consistent(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config)
    breakdown = model.parameter_breakdown()
    assert breakdown["total"] == model.num_parameters
    assert breakdown["trainable"] == model.num_trainable_parameters
    assert breakdown["trainable"] < breakdown["total"]  # frozen backbone
    # of the backbone, only the trainable 1x1 projection receives gradients (A17)
    encoder_params = sum(p.numel() for p in model.backbone.encoder.parameters())
    assert breakdown["backbone"] - encoder_params == breakdown["backbone.trainable"]


def test_backbone_is_frozen(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config)
    assert all(not p.requires_grad for p in model.backbone.encoder.parameters())
    assert all(p.requires_grad for p in model.gfim.parameters())


def test_gradient_flow_to_head_only(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config)
    model.train()
    images, boxes = _inputs(tiny_config, batch=2)
    out = model(images, boxes)
    loss = out.density.sum()
    loss.backward()

    head_grads = [p.grad for p in model.gfim.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in head_grads)
    assert all(p.grad is None for p in model.backbone.encoder.parameters())


def test_eval_is_deterministic(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config)
    model.eval()
    images, boxes = _inputs(tiny_config, batch=1)
    with torch.no_grad():
        a = model(images, boxes).density
        b = model(images, boxes).density
    assert torch.equal(a, b)


def test_input_validation(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config)
    wrong = torch.randn(1, 3, 128, 128)  # wrong resolution -> wrong token grid
    boxes = torch.rand(1, 3, 4)
    with pytest.raises(ValueError, match="grid"):
        model(wrong, boxes)


def test_padding_mask_zeroes_output(tiny_config: ExperimentConfig) -> None:
    """Padded pixels must contribute zero density (A9)."""
    model = GIPACount(tiny_config)
    model.eval()
    size = tiny_config.model.image_size
    images, boxes = _inputs(tiny_config, batch=1)
    pad_mask = torch.ones(1, 1, size, size, dtype=torch.bool)
    pad_mask[:, :, size // 2 :, :] = False  # bottom half is padding
    with torch.no_grad():
        out = model(images, boxes, pad_mask=pad_mask)
    assert out.density[0, :, 16:, :].abs().sum() == 0


def test_attention_mask_avoids_nans(tiny_config: ExperimentConfig) -> None:
    """Fully masked image tokens must not produce NaNs (softmax of -inf)."""
    model = GIPACount(tiny_config)
    model.eval()
    size = tiny_config.model.image_size
    images, boxes = _inputs(tiny_config, batch=1)
    pad_mask = torch.ones(1, 1, size, size, dtype=torch.bool)
    with torch.no_grad():
        out = model(images, boxes, pad_mask=pad_mask)
    assert torch.isfinite(out.density).all()


# --------------------------------------------------------------------------- #
# Sub-modules
# --------------------------------------------------------------------------- #
def test_roi_encoder_token_count() -> None:
    encoder = RoIExemplarEncoder(tokens_per_exemplar=9, stride=16)
    feature_map = torch.randn(2, 8, 4, 4)
    boxes = torch.tensor([[[0.0, 0.0, 32.0, 32.0]] * 3] * 2)
    tokens = encoder(feature_map, boxes)
    assert tokens.shape == (2, 27, 8)  # K=3 x 9


def test_roi_encoder_zeroes_invalid_boxes() -> None:
    encoder = RoIExemplarEncoder(tokens_per_exemplar=9, stride=16)
    feature_map = torch.randn(1, 8, 4, 4)
    boxes = torch.tensor([[[0.0, 0.0, 32.0, 32.0], [0.0, 0.0, 32.0, 32.0]]])
    box_mask = torch.tensor([[True, False]])
    tokens = encoder(feature_map, boxes, box_mask)
    assert tokens.shape == (1, 18, 8)
    assert tokens[0, 9:].abs().sum() == 0  # second exemplar masked out


def test_gfim_shapes(tiny_config: ExperimentConfig) -> None:
    cfg = tiny_config.model
    gfim = GFIM(
        cfg.gfim,
        dim=cfg.token_dim,
        n_prompt_tokens=cfg.n_prompt_tokens,
        n_image_tokens=cfg.n_image_tokens,
        grid=cfg.feature_grid,
    )
    x_i = torch.randn(2, cfg.n_image_tokens, cfg.token_dim)
    x_t = torch.randn(2, cfg.n_prompt_tokens, cfg.token_dim)
    out = gfim(x_i, x_t)
    assert out.image.shape == x_i.shape
    assert out.prompt.shape == x_t.shape


def test_gfim_validates_token_counts(tiny_config: ExperimentConfig) -> None:
    cfg = tiny_config.model
    gfim = GFIM(
        cfg.gfim,
        dim=cfg.token_dim,
        n_prompt_tokens=cfg.n_prompt_tokens,
        n_image_tokens=cfg.n_image_tokens,
        grid=cfg.feature_grid,
    )
    with pytest.raises(ValueError, match="image tokens"):
        gfim(torch.randn(1, 5, cfg.token_dim), torch.randn(1, cfg.n_prompt_tokens, cfg.token_dim))


def test_prototype_depthwise_conv_matches_naive() -> None:
    """Eq. 10 via unfold must equal an explicit loop."""
    features = torch.randn(1, 2, 5, 5)
    kernels = torch.randn(1, 2, 3, 3)
    out = prototype_depthwise_conv(features, kernels)
    padded = torch.nn.functional.pad(features, (1, 1, 1, 1))
    naive = torch.zeros(1, 2, 5, 5)
    for c in range(2):
        for i in range(5):
            for j in range(5):
                patch = padded[0, c, i : i + 3, j : j + 3]
                naive[0, c, i, j] = (patch * kernels[0, c]).sum()
    assert torch.allclose(out, naive, atol=1e-5)


def test_plam_disabled_passes_features_through(tiny_config: ExperimentConfig) -> None:
    cfg = tiny_config.model
    cfg.plam.enabled = False
    plam = PLAM(cfg.plam, cfg)
    grid = cfg.feature_grid
    image_tokens = torch.randn(2, grid * grid, cfg.token_dim)
    prompt = torch.randn(2, cfg.n_prompt_tokens, cfg.token_dim)
    out = plam(prompt, image_tokens)
    assert out.kernel is None
    assert out.similarity.shape == (2, cfg.token_dim, grid, grid)


def test_plam_exemplar_kernel_source(tiny_config: ExperimentConfig) -> None:
    cfg = tiny_config.model
    cfg.plam.kernel_source = "exemplar"
    plam = PLAM(cfg.plam, cfg)
    grid = cfg.feature_grid
    image_tokens = torch.randn(2, grid * grid, cfg.token_dim)
    prompt = torch.randn(2, cfg.n_prompt_tokens, cfg.token_dim)
    out = plam(prompt, image_tokens)
    assert out.similarity.shape == (2, cfg.token_dim, grid, grid)


def test_decoder_output_geometry() -> None:
    from gipacount.config import DecoderConfig

    decoder = DensityDecoder(in_channels=16, cfg=DecoderConfig(channels=(16, 8, 4)), in_size=4)
    assert decoder.out_size == 32
    out = decoder(torch.randn(2, 16, 4, 4))
    assert out.shape == (2, 1, 32, 32)


# --------------------------------------------------------------------------- #
# Geometry invariance across resolutions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("image_size", [64, 128, 256])
def test_geometry_invariant_across_resolutions(image_size: int) -> None:
    """The x8 decoder keeps ``density_size == image_size / 2`` at any resolution."""
    cfg = make_config(
        **{
            "model.image_size": image_size,
            "model.density_size": image_size // 2,
            "data.image_size": image_size,
            "data.synthetic_image_size": image_size,
            "data.ccss.sub_image_size": image_size // 2,
        }
    )
    grid = image_size // 16
    assert grid * 8 == image_size // 2
    model = GIPACount(cfg)
    model.eval()
    images = torch.randn(1, 3, image_size, image_size)
    boxes = torch.tensor([[[4.0, 4.0, 20.0, 20.0]] * cfg.model.n_exemplars])
    with torch.no_grad():
        out = model(images, boxes)
    assert out.density.shape == (1, 1, image_size // 2, image_size // 2)


def test_model_config_rejects_bad_grid() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(image_size=100, density_size=50, backbone={"stride": 16})  # type: ignore[arg-type]


def test_model_accepts_model_config_only(tiny_config: ExperimentConfig) -> None:
    model = GIPACount(tiny_config.model)
    assert isinstance(model, GIPACount)
