"""Reproducibility helpers: one entry point covering python / numpy / torch / CUDA."""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

__all__ = ["RNGState", "capture_rng_state", "restore_rng_state", "set_seed"]


@dataclass(slots=True)
class RNGState:
    """Snapshot of every RNG that influences a training run.

    Stored inside checkpoints so that a resumed run continues the exact same
    data/init stream (prompt requirement: "save ... RNG state").
    """

    python: Any
    numpy: Any
    torch: Any
    cuda: list[Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "numpy": self.numpy,
            "torch": self.torch,
            "cuda": self.cuda,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RNGState:
        return cls(
            python=payload["python"],
            numpy=payload["numpy"],
            torch=payload["torch"],
            cuda=list(payload.get("cuda", [])),
        )


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed python, numpy and torch, and make kernels deterministic when asked.

    Args:
        seed: The global seed.
        deterministic: When ``True`` also sets ``torch.use_deterministic_algorithms``
            and the cuDNN flags.  ``CUBLAS_WORKSPACE_CONFIG`` is set before the
            first CUDA call because cuBLAS otherwise raises on deterministic
            reductions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def capture_rng_state() -> RNGState:
    """Return a checkpoint-safe snapshot of all RNG states."""
    return RNGState(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.random.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng_state(state: RNGState) -> None:
    """Restore a snapshot produced by :func:`capture_rng_state`.

    The tensors are forced back to the CPU first: ``torch.load`` is normally
    called with ``map_location=<device>``, which would otherwise move the
    checkpoint's CPU RNG ByteTensors to CUDA and make ``set_rng_state`` raise
    ``TypeError: RNG state must be a torch.ByteTensor`` (a CUDA-resident state
    only breaks :func:`torch.random.set_rng_state`, not the CUDA one).
    """
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(_cpu_byte_tensor(state.torch))
    if state.cuda and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_cpu_byte_tensor(s) for s in state.cuda])


def _cpu_byte_tensor(state: Any) -> Any:
    """Return ``state`` as a CPU ``uint8`` tensor (no-op when already a CPU state)."""
    if torch.is_tensor(state):
        return state.detach().to(device="cpu", dtype=torch.uint8)
    return state
