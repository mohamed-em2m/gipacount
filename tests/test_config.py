"""Configuration: validation, defaults, ``_base_`` inheritance and overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gipacount.config import ExperimentConfig, apply_overrides, load_config, save_config
from tests.conftest import make_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"


def test_defaults_match_paper() -> None:
    """The bare defaults reproduce the paper's headline geometry."""
    cfg = ExperimentConfig()
    assert cfg.model.image_size == 1024
    assert cfg.model.density_size == 512
    assert cfg.model.feature_grid == 64
    assert cfg.model.n_image_tokens == 4096
    assert cfg.model.n_exemplar_tokens == 27  # K=3 x 9
    assert cfg.model.n_prompt_tokens == 36  # 9 prototypes + 27 exemplar tokens
    assert cfg.model.gfim.layers == 6
    assert cfg.loss.epsilon == pytest.approx(0.01)
    assert cfg.loss.tau == pytest.approx(0.5)


def test_unknown_key_is_rejected() -> None:
    """Typos in a config fail loudly rather than silently doing nothing."""
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        ExperimentConfig.model_validate({"model": {"image_size": 64, "typo": 1}})


def test_density_size_must_be_half_the_image() -> None:
    """§4.3: the density map is predicted at half resolution."""
    payload = {"model": {"image_size": 64, "density_size": 64, "backbone": {"stride": 16}}}
    with pytest.raises(Exception, match="density_size"):
        ExperimentConfig.model_validate(payload)


def test_decoder_upsampling_is_enforced() -> None:
    """A decoder that does not reach ``density_size`` is rejected."""
    payload = {
        "model": {
            "image_size": 64,
            "density_size": 32,
            "backbone": {"stride": 16},
            "decoder": {"channels": [8, 4, 2], "stride": 4},  # x64 -> 256 != 32
        }
    }
    with pytest.raises(Exception, match="transposed convolutions"):
        ExperimentConfig.model_validate(payload)


def test_data_and_model_image_sizes_must_agree() -> None:
    payload = {
        "model": {"image_size": 64, "density_size": 32},
        "data": {"image_size": 128, "ccss": {"enabled": False}},
    }
    with pytest.raises(Exception, match="data.image_size"):
        ExperimentConfig.model_validate(payload)


def test_ccss_sub_image_size_is_validated() -> None:
    """A 2x2 collage of 1024-px images needs 512-px quadrants."""
    payload = {
        "model": {"image_size": 64, "density_size": 32},
        "data": {
            "image_size": 64,
            "ccss": {"enabled": True, "grid": 2, "sub_image_size": 100},
        },
    }
    with pytest.raises(Exception, match="sub_image_size"):
        ExperimentConfig.model_validate(payload)


def test_n_prototypes_must_be_square() -> None:
    payload = {"model": {"image_size": 64, "density_size": 32, "plam": {"n_prototypes": 7}}}
    with pytest.raises(Exception, match="perfect square"):
        ExperimentConfig.model_validate(payload)


def test_load_config_smoke() -> None:
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert cfg.name == "smoke"
    assert cfg.model.image_size == 128
    assert cfg.model.density_size == 64
    assert cfg.trainer.device == "cpu"


def test_base_inheritance_merges_and_overrides() -> None:
    """`vit_fsc147_1shot.yaml` inherits `default.yaml` and changes only K."""
    one_shot = load_config(CONFIGS / "vit_fsc147_1shot.yaml")
    default = load_config(CONFIGS / "default.yaml")
    assert one_shot.model.n_exemplars == 1
    # everything else is inherited
    assert one_shot.model.gfim.layers == default.model.gfim.layers == 6
    assert one_shot.trainer.epochs == default.trainer.epochs == 200
    assert one_shot.model.backbone.kind == default.model.backbone.kind == "vit"


def test_base_inheritance_cnn_variant() -> None:
    cnn = load_config(CONFIGS / "cnn_fsc147_3shot.yaml")
    assert cnn.model.backbone.kind == "cnn"
    assert cnn.model.backbone.out_indices == (2,)
    assert cnn.model.image_size == 1024


def test_all_shipped_configs_load() -> None:
    """Every config in ``configs/`` must validate (catches typos in ablations)."""
    paths = sorted(CONFIGS.rglob("*.yaml"))
    assert paths, "no configs found"
    for path in paths:
        load_config(path)


def test_ablation_switches() -> None:
    q1 = load_config(CONFIGS / "ablations" / "q1.yaml")
    assert q1.model.plam.enabled is False
    assert q1.data.ccss.enabled is False
    assert q1.model.gfim.use_image_to_prompt is False
    assert q1.model.gfim.use_prompt_self_attention is False

    q5 = load_config(CONFIGS / "ablations" / "q5.yaml")
    assert q5.data.ccss.enabled is False
    assert q5.model.plam.enabled is True

    q7 = load_config(CONFIGS / "ablations" / "q7.yaml")
    assert q7.model.backbone.frozen is True
    assert q7.model.backbone.unfreeze_last_n == 3


def test_circular_base_detection(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text("_base_: ./b.yaml\nname: a\n", encoding="utf-8")
    b.write_text("_base_: ./a.yaml\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(a)


def test_missing_base_detection(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    a.write_text("_base_: ./does-not-exist.yaml\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="_base_"):
        load_config(a)


def test_overrides_parse_yaml_values() -> None:
    raw = {"trainer": {"epochs": 1}, "loss": {"epsilon": 0.01}}
    apply_overrides(raw, ["trainer.epochs=5", "model.gfim.layers=2", "loss.epsilon=0.001"])
    assert raw["trainer"]["epochs"] == 5
    assert raw["model"]["gfim"]["layers"] == 2
    assert raw["loss"]["epsilon"] == pytest.approx(0.001)


def test_overrides_require_equals() -> None:
    with pytest.raises(ValueError, match="key.path=value"):
        apply_overrides({}, ["trainer.epochs"])


def test_save_config_round_trip(tmp_path: Path) -> None:
    cfg = make_config()
    out = save_config(cfg, tmp_path / "nested" / "config.yaml")
    assert out.is_file()
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert payload["model"]["image_size"] == cfg.model.image_size
    reloaded = ExperimentConfig.model_validate(payload)
    assert reloaded.model_dump() == cfg.model_dump()
