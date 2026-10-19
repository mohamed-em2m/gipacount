#!/usr/bin/env python
"""Benchmark GIPACount: parameters, latency, throughput and peak memory.

Runs a synthetic forward (and optional backward) pass through the full network at
a chosen resolution, so it works without the FSC-147 download.  Useful to check
that a config is trainable on the target hardware before committing to a full run.

Examples::

    python scripts/benchmark.py -c configs/smoke.yaml
    python scripts/benchmark.py -c configs/default.yaml --batch-size 1 --iters 5
    python scripts/benchmark.py -c configs/default.yaml -s model.backbone.pretrained=false
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

from gipacount.config import load_config
from gipacount.models.gipacount import GIPACount
from gipacount.utils.device import resolve_device
from gipacount.utils.logging import configure_logging, get_logger
from gipacount.utils.seed import set_seed

_LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for this script."""
    parser = argparse.ArgumentParser(description="Benchmark GIPACount.")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument(
        "--set",
        "-s",
        dest="overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Config override (repeatable).",
    )
    parser.add_argument("--device", default=None, help="Override the configured device.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=3, help="Warm-up iterations.")
    parser.add_argument("--backward", action="store_true", help="Also time the backward pass.")
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse arguments and run the benchmark."""
    args = build_parser().parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else "INFO", force=args.verbose)

    try:
        cfg = load_config(args.config, args.overrides)
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.error("config error: %s", exc)
        return 2

    set_seed(cfg.trainer.seed, deterministic=False)
    device = (
        resolve_device(cfg.trainer.device) if args.device is None else torch.device(args.device)
    )
    model = GIPACount(cfg).to(device)
    if args.backward:
        model.train()
    else:
        model.eval()

    size = cfg.model.image_size
    k = cfg.model.n_exemplars
    images = torch.randn(args.batch_size, 3, size, size, device=device)
    boxes = torch.rand(args.batch_size, k, 4, device=device)
    boxes = torch.stack(
        [boxes[..., :2] * size * 0.5, boxes[..., 2:] * size * 0.5 + size * 0.4], dim=-1
    )

    _LOGGER.info(
        "device=%s batch=%d image=%d params=%.2fM trainable=%.2fM",
        device,
        args.batch_size,
        size,
        model.num_parameters / 1e6,
        model.num_trainable_parameters / 1e6,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def run_once() -> None:
        output = model(images, boxes)
        if args.backward:
            output.count.sum().backward()
            model.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        run_once()
    _synchronise(device)

    timings: list[float] = []
    for _ in range(args.iters):
        start = time.perf_counter()
        run_once()
        _synchronise(device)
        timings.append(time.perf_counter() - start)

    mean = statistics.fmean(timings)
    p50 = statistics.median(timings)
    _LOGGER.info(
        "%.4f s/iter (mean) | %.4f s (median) | %.2f iter/s | %.2f img/s",
        mean,
        p50,
        1.0 / mean,
        args.batch_size / mean,
    )
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        reserved = torch.cuda.max_memory_reserved(device) / 1e9
        _LOGGER.info("peak GPU memory: %.2f GB allocated | %.2f GB reserved", peak, reserved)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
