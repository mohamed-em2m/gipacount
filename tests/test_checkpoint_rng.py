"""Regression tests: RNG state must survive a checkpoint round trip.

A resumed run is only reproducible if the saved RNG snapshot is restored
*exactly*.  The subtle failure mode this file guards against is the interaction
with ``torch.load(map_location=...)``: ``resume`` loads a checkpoint onto the
training device, which also moves the (CPU) RNG ByteTensors to CUDA.  A
CUDA-resident state is rejected by ``torch.random.set_rng_state`` with
``TypeError: RNG state must be a torch.ByteTensor`` — which made GPU resume
impossible while the CPU-only suite stayed green.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gipacount.training.checkpoint import load_checkpoint, save_checkpoint
from gipacount.utils.seed import RNGState, capture_rng_state, restore_rng_state
from tests.conftest import make_config


def _draw() -> tuple[float, float, torch.Tensor]:
    """Sample all three RNGs so a mis-restore is detectable."""
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(3),
    )


def _save(path: Path) -> Path:
    from gipacount.models.gipacount import GIPACount

    cfg = make_config()
    model = GIPACount(cfg.model)
    return save_checkpoint(
        path,
        model=model,
        config=cfg,
        epoch=3,
        global_step=7,
        rng_state=capture_rng_state(),
    )


def test_round_trip_restores_every_rng(tmp_path: Path) -> None:
    """A CPU round trip reproduces the identical random stream."""
    path = _save(tmp_path / "ckpt.pt")
    expected = capture_rng_state()

    # Consume randomness so the state under test is definitely not the saved one.
    for _ in range(5):
        _draw()

    state = load_checkpoint(path, map_location="cpu")
    assert state.rng_state is not None
    restore_rng_state(state.rng_state)

    assert torch.equal(torch.random.get_rng_state(), expected.torch)
    assert np.array_equal(np.random.get_state()[1], expected.numpy[1])
    assert random.getstate() == expected.python


def test_restore_accepts_non_cpu_state_tensors() -> None:
    """Device/dtype of a loaded RNG snapshot must not matter."""
    expected = capture_rng_state()
    mangled = RNGState(
        python=expected.python,
        numpy=expected.numpy,
        torch=expected.torch.clone().to(dtype=torch.int64),
        cuda=[],
    )
    with pytest.raises(TypeError):
        torch.random.set_rng_state(mangled.torch)  # the raw API really does reject it
    restore_rng_state(mangled)
    assert torch.equal(torch.random.get_rng_state(), expected.torch)
    assert _draw()[2].shape == (3,)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_resume_from_a_checkpoint_loaded_onto_cuda(tmp_path: Path) -> None:
    """The exact failure: map_location='cuda' moves the CPU RNG state to GPU."""
    path = _save(tmp_path / "ckpt.pt")
    expected = capture_rng_state()

    state = load_checkpoint(path, map_location="cuda")
    assert state.rng_state is not None
    restore_rng_state(state.rng_state)

    assert torch.equal(torch.random.get_rng_state().cpu(), expected.torch.cpu())
    assert _draw()[2].shape == (3,)
