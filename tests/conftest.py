"""Shared pytest fixtures and helpers.

The whole suite runs on CPU with *no pretrained weights* and *no dataset
download*, so it is fast and offline-friendly.  Every fixture builds a small but
structurally faithful :class:`~gipacount.config.ExperimentConfig` whose geometry
preserves the paper's relationships (token grid ``image_size / 16``, density map
``image_size / 2``, CCSS collage ``grid x 1`` sub-images).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from gipacount.config import ExperimentConfig

#: A 64-px image gives a 4x4 token grid and a 32-px density map: small enough to
#: run in milliseconds, yet it exercises every stride/padding interaction.
TINY_IMAGE_SIZE = 64
TINY_TOKEN_DIM = 32


def _tiny_payload(
    image_size: int = TINY_IMAGE_SIZE, token_dim: int = TINY_TOKEN_DIM
) -> dict[str, Any]:
    """Return a minimal valid config dict for the requested geometry."""
    return {
        "name": "test",
        "model": {
            "image_size": image_size,
            "density_size": image_size // 2,
            "token_dim": token_dim,
            "n_exemplars": 3,
            "backbone": {
                "kind": "vit",
                "name": "vit_small_patch16_224",
                "pretrained": False,
                "stride": 16,
                "token_dim": token_dim,
                "frozen": True,
            },
            "gfim": {
                "layers": 2,
                "n_heads": 2,
                "mlp_ratio": 2.0,
                "fusion_mlp_ratio": 2.0,
                "pos_encoding": "learnable",
                "pos_from_layer": 0,
            },
            "plam": {"enabled": True, "n_prototypes": 9, "kernel_source": "prototype"},
            "decoder": {"channels": [16, 8, 4], "norm": "layernorm", "activation": "relu"},
        },
        "data": {
            "name": "synthetic",
            "image_size": image_size,
            "synthetic_image_size": image_size,
            "synthetic_train_size": 16,
            "synthetic_val_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "random_hflip": True,
            "color_jitter": True,
            "ccss": {
                "enabled": True,
                "p_mix": 0.5,
                "grid": 2,
                "sub_image_size": image_size // 2,
                "category_filter": True,
                "composition": "random",
            },
        },
        "loss": {"name": "generalized", "sinkhorn_iters": 5, "chunk_size": 512},
        "optim": {"name": "adam", "lr": 1e-3, "scheduler": "none"},
        "trainer": {
            "epochs": 2,
            "batch_size": 4,
            "device": "cpu",
            "num_workers": 0,
            "log_every": 100,
            "run_name": "test",
            "amp": False,
            "deterministic": True,
        },
    }


def make_config(**overrides: Any) -> ExperimentConfig:
    """Build a tiny :class:`ExperimentConfig`, applying dotted overrides.

    ``overrides`` use the same ``key.path=value`` syntax as the CLI, except the
    value is a Python object already, e.g. ``make_config(**{"trainer.epochs": 1})``.
    """
    payload = _tiny_payload()
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node: Any = payload
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return ExperimentConfig.model_validate(payload)


@pytest.fixture
def config_factory() -> Callable[..., ExperimentConfig]:
    """Return the :func:`make_config` helper as a fixture."""
    return make_config


@pytest.fixture
def tiny_config() -> ExperimentConfig:
    """A default tiny config (64 px, 32-d tokens, synthetic data)."""
    return make_config()


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Make every test start from the same RNG state."""
    torch.manual_seed(0)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A throwaway output directory for trainers and checkpoints."""
    path = tmp_path / "run"
    path.mkdir(parents=True, exist_ok=True)
    return path
