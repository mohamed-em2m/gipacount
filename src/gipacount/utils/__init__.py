"""Cross-cutting utilities: seeding, logging, IO and device selection."""

from __future__ import annotations

from gipacount.utils.device import resolve_device
from gipacount.utils.io import git_hash, load_json, save_json, utc_timestamp
from gipacount.utils.logging import configure_logging, get_logger
from gipacount.utils.seed import RNGState, capture_rng_state, restore_rng_state, set_seed

__all__ = [
    "RNGState",
    "capture_rng_state",
    "configure_logging",
    "get_logger",
    "git_hash",
    "load_json",
    "resolve_device",
    "restore_rng_state",
    "save_json",
    "set_seed",
    "utc_timestamp",
]
