#!/usr/bin/env python
"""Train GIPACount from a YAML config.

Thin wrapper around :class:`gipacount.training.trainer.Trainer` so that the same
run can be launched without the CLI (e.g. from a scheduler that prefers plain
``python``).  All real logic lives in the package.

Examples::

    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/smoke.yaml --set trainer.epochs=1
    python scripts/train.py --config configs/default.yaml --resume runs/run/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gipacount.config import load_config
from gipacount.training.trainer import Trainer
from gipacount.utils.logging import configure_logging, get_logger

_LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(description="Train GIPACount.")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--set",
        "-s",
        dest="overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Config override, e.g. -s trainer.epochs=5 (repeatable).",
    )
    parser.add_argument("--device", default=None, help="Override the configured device.")
    parser.add_argument("--run-name", default=None, help="Name of the run directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a checkpoint.")
    parser.add_argument("--epochs", type=int, default=None, help="Override trainer.epochs.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments and run training."""
    args = build_parser().parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else "INFO", force=args.verbose)

    try:
        cfg = load_config(args.config, args.overrides)
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.error("config error: %s", exc)
        return 2

    if args.run_name:
        cfg.trainer.run_name = args.run_name
    if args.resume:
        cfg.trainer.resume = args.resume
    if args.epochs is not None:
        cfg.trainer.epochs = args.epochs

    trainer = Trainer(cfg, device=args.device)
    _LOGGER.info("run directory: %s", trainer.run_dir)
    metrics = trainer.fit()
    _LOGGER.info("final validation metrics: %s", metrics)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
