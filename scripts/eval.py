#!/usr/bin/env python
"""Evaluate a trained checkpoint (MAE/RMSE, optionally ECR).

Examples::

    python scripts/eval.py --config configs/default.yaml --checkpoint runs/run/best.pt
    python scripts/eval.py -c configs/smoke.yaml --checkpoint runs/smoke/best.pt --split val
    python scripts/eval.py -c configs/default.yaml --checkpoint best.pt --ecr --output metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gipacount.config import load_config
from gipacount.data.datamodule import build_dataloader, build_dataset
from gipacount.evaluation.ecr import build_ecr_loader
from gipacount.evaluation.evaluator import evaluate, evaluate_ecr
from gipacount.training.trainer import load_model_for_eval
from gipacount.utils.io import save_json
from gipacount.utils.logging import configure_logging, get_logger

_LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(description="Evaluate a GIPACount checkpoint.")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint (.pt).")
    parser.add_argument(
        "--set",
        "-s",
        dest="overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Config override (repeatable).",
    )
    parser.add_argument("--split", default=None, help="val | test.")
    parser.add_argument("--device", default=None, help="Override the configured device.")
    parser.add_argument("--ecr", action="store_true", help="Also compute ECR on the composite set.")
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Limit the number of batches."
    )
    parser.add_argument("--output", type=Path, default=None, help="Write metrics JSON here.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments and run evaluation."""
    args = build_parser().parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else "INFO", force=args.verbose)

    try:
        cfg = load_config(args.config, args.overrides)
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.error("config error: %s", exc)
        return 2

    if args.split:
        cfg.eval.split = args.split
    if cfg.eval.split not in {"val", "test"}:
        _LOGGER.error("unknown split %r", cfg.eval.split)
        return 2

    model, device = load_model_for_eval(cfg, args.checkpoint, device=args.device)
    dataset = build_dataset(cfg.data, cfg.eval.split, n_exemplars=cfg.model.n_exemplars)
    loader = build_dataloader(
        cfg.data, dataset, batch_size=cfg.eval.batch_size, train=False, num_workers=0
    )
    metrics = evaluate(model, loader, device=device, max_batches=args.max_batches)
    _LOGGER.info("%s split: %s", cfg.eval.split, metrics)

    if args.ecr:
        ecr_loader = build_ecr_loader(
            cfg.data,
            dataset,
            n_composites=cfg.eval.ecr_num_composites,
            seed=cfg.trainer.seed,
        )
        metrics.update(evaluate_ecr(model, ecr_loader, device=device, ecr_eps=cfg.eval.ecr_eps))
        _LOGGER.info("composite ECR: %s", metrics["ecr"])

    if args.output:
        save_json(metrics, args.output)
        _LOGGER.info("metrics written to %s", args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
