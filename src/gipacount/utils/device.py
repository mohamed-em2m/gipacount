"""Device resolution and mixed-precision helpers."""

from __future__ import annotations

import contextlib
from typing import Any

import torch

__all__ = ["autocast_context", "resolve_device"]


def autocast_context(device: torch.device, enabled: bool) -> Any:
    """Return an autocast context manager (a no-op off CUDA or when disabled)."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def resolve_device(spec: str = "auto") -> torch.device:
    """Map a config string to a concrete :class:`torch.device`.

    Args:
        spec: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``.

    Returns:
        The resolved device.  ``"auto"`` prefers CUDA, then MPS, then CPU.
    """
    spec = spec.lower().strip()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {spec!r} but CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():  # pragma: no cover
        raise RuntimeError(f"requested {spec!r} but MPS is not available")
    return device
