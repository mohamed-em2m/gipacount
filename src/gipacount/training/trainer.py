"""The training loop (Algorithm 1 of the paper)."""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from gipacount.config import ExperimentConfig, save_config
from gipacount.data.collate import CountingCollator
from gipacount.data.datamodule import build_dataloader, build_dataset
from gipacount.evaluation.evaluator import evaluate
from gipacount.losses import LossOutput, build_loss
from gipacount.models.gipacount import GIPACount
from gipacount.training.checkpoint import load_checkpoint, save_checkpoint
from gipacount.training.optim import EpochLRScheduler, build_optimizer, build_scheduler
from gipacount.utils.device import autocast_context, resolve_device
from gipacount.utils.io import git_hash, save_json, utc_timestamp
from gipacount.utils.logging import get_logger
from gipacount.utils.seed import capture_rng_state, restore_rng_state, set_seed

__all__ = ["Trainer", "TrainerState"]

_LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TrainerState:
    """Mutable training progress."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None


class Trainer:
    """End-to-end training of :class:`~gipacount.models.gipacount.GIPACount`.

    Responsibilities: model/loss/optimiser construction, the CCSS-augmented epoch
    loop, periodic validation, checkpointing with full resume support, optional
    experiment tracking, and provenance capture (config + git hash + environment).

    Args:
        cfg: The experiment configuration.
        device: Override the configured device.
        run_dir: Override the output directory for this run.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        *,
        device: str | torch.device | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        set_seed(cfg.trainer.seed, deterministic=cfg.trainer.deterministic)

        self.device = resolve_device(cfg.trainer.device) if device is None else torch.device(device)
        self.amp_enabled = bool(cfg.trainer.amp) and self.device.type == "cuda"

        self.model = GIPACount(cfg).to(self.device)
        self.loss_fn = build_loss(cfg.loss).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.optim)
        self.scheduler: EpochLRScheduler | None = build_scheduler(
            self.optimizer, cfg.optim, cfg.trainer.epochs
        )
        self.scaler = self._build_scaler()
        self.state = TrainerState()

        self.run_dir = self._resolve_run_dir(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_provenance()

        self.train_loader: DataLoader[Any]
        self.val_loader: DataLoader[Any]
        self.train_collator: CountingCollator
        self._build_data()

        self.tracker = _Tracker(cfg, self.run_dir)
        self._log_model_summary()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _build_scaler(self) -> Any:
        try:
            return torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        except (TypeError, AttributeError):  # pragma: no cover - older torch
            return torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _resolve_run_dir(self, run_dir: str | Path | None) -> Path:
        if run_dir is not None:
            return Path(run_dir)
        name = self.cfg.trainer.run_name or f"{self.cfg.name}-{utc_timestamp()}"
        return Path(self.cfg.trainer.output_dir) / name

    def _write_provenance(self) -> None:
        save_config(self.cfg, self.run_dir / "config.yaml")
        info = {
            "git_hash": git_hash(Path.cwd()),
            "timestamp": utc_timestamp(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(self.device),
            "params_total": self.model.num_parameters,
            "params_trainable": self.model.num_trainable_parameters,
        }
        save_json(info, self.run_dir / "run_info.json")

    def _build_data(self) -> None:
        cfg = self.cfg
        seed = cfg.trainer.seed
        n_exemplars = cfg.model.n_exemplars

        train_dataset = build_dataset(cfg.data, "train", n_exemplars=n_exemplars, seed=seed)
        val_split = cfg.eval.split if cfg.eval.split in {"val", "test"} else "val"
        val_dataset = build_dataset(cfg.data, val_split, n_exemplars=n_exemplars, seed=seed)

        workers = cfg.trainer.num_workers
        self.train_collator = CountingCollator(cfg.data, train=True, seed=seed)
        self.train_loader = build_dataloader(
            cfg.data,
            train_dataset,
            batch_size=cfg.trainer.batch_size,
            train=True,
            seed=seed,
            num_workers=workers,
            collator=self.train_collator,
        )
        self.val_loader = build_dataloader(
            cfg.data,
            val_dataset,
            batch_size=cfg.eval.batch_size,
            train=False,
            seed=seed,
            num_workers=workers if workers is not None else cfg.eval.num_workers,
        )

    def _log_model_summary(self) -> None:
        breakdown = self.model.parameter_breakdown()
        _LOGGER.info(
            "model: %d parameters (%.2fM total, %.2fM trainable) on %s",
            breakdown["total"],
            breakdown["total"] / 1e6,
            breakdown["trainable"] / 1e6,
            self.device,
        )
        for key, value in breakdown.items():
            if key.endswith(".trainable"):
                _LOGGER.debug("  %s: %.3fM", key, value / 1e6)
        if self.cfg.model.backbone.frozen and self.cfg.model.backbone.unfreeze_last_n == 0:
            _LOGGER.info("backbone frozen; only the interaction/aggregation head is trained (§4.3)")

    # ------------------------------------------------------------------ #
    # Epoch / step
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one epoch, alternating single-image and CCSS batches (Algorithm 1)."""
        self.model.train()
        running = 0.0
        running_parts: dict[str, float] = {}
        steps = 0
        started = time.perf_counter()
        limit = self.cfg.trainer.steps_per_epoch

        for batch_index, batch in enumerate(self.train_loader):
            if limit is not None and batch_index >= limit:
                break
            if self.cfg.trainer.max_steps is not None and self.state.global_step >= (
                self.cfg.trainer.max_steps
            ):
                break

            loss_out = self._train_step(batch)
            self.state.global_step += 1
            steps += 1
            running += loss_out.item()
            for key, value in loss_out.as_dict().items():
                if key == "loss":
                    continue
                running_parts[key] = running_parts.get(key, 0.0) + value

            if steps % max(1, self.cfg.trainer.log_every) == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                _LOGGER.info(
                    "epoch %d | step %d | loss %.4f | lr %.3e | mixed=%s",
                    epoch,
                    self.state.global_step,
                    running / steps,
                    lr,
                    bool(batch.is_composite[0]) if batch.is_composite is not None else False,
                )

        if steps == 0:  # pragma: no cover - defensive
            raise RuntimeError("the training loader produced no batches")

        metrics = {
            "train/loss": running / steps,
            **{f"train/{k}": v / steps for k, v in running_parts.items()},
            "train/epoch_seconds": time.perf_counter() - started,
        }
        self.tracker.log(metrics, step=epoch)
        return metrics

    def _train_step(self, batch: Any) -> LossOutput:
        """One optimisation step: forward, Generalized Loss, backward, update."""
        batch = batch.to(self.device)
        self.optimizer.zero_grad(set_to_none=True)

        with autocast_context(self.device, self.amp_enabled):
            output = self.model(
                batch.images, batch.boxes, pad_mask=batch.pad_mask, box_mask=batch.box_mask
            )
        density = output.density.float()
        loss_out = self.loss_fn(
            density,
            batch.points,
            batch.point_mask,
            batch.point_mass,
            density_size=batch.density_size,
        )
        loss = loss_out.total

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss ({loss.item()}) at step {self.state.global_step}"
            )

        if self.amp_enabled:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self._clip_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()

        return loss_out

    def _clip_gradients(self) -> None:
        if self.cfg.optim.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.optim.grad_clip_norm,
            )

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate on the validation split."""
        return evaluate(
            self.model,
            self.val_loader,
            device=self.device,
            loss_fn=self.loss_fn,
            amp=self.amp_enabled,
            max_batches=None,
        )

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self) -> dict[str, Any]:
        """Train for ``cfg.trainer.epochs`` epochs and return the best metrics."""
        cfg = self.cfg
        if cfg.trainer.resume is not None:
            self.resume(cfg.trainer.resume)

        start_epoch = self.state.epoch
        if start_epoch >= cfg.trainer.epochs:
            _LOGGER.info("nothing to do: epoch %d >= %d", start_epoch, cfg.trainer.epochs)

        for epoch in range(start_epoch, cfg.trainer.epochs):
            train_metrics = self.train_epoch(epoch)
            if self.scheduler is not None:
                self.scheduler.step(epoch)

            is_last = epoch == cfg.trainer.epochs - 1
            should_validate = (epoch + 1) % max(1, cfg.trainer.val_every) == 0 or is_last
            val_metrics: dict[str, float] = {}
            if should_validate:
                val_metrics = self.validate()
                _LOGGER.info(
                    "epoch %d | val MAE %.4f | val RMSE %.4f | train loss %.4f",
                    epoch,
                    val_metrics["mae"],
                    val_metrics["rmse"],
                    train_metrics["train/loss"],
                )
                self.tracker.log({f"val/{k}": v for k, v in val_metrics.items()}, step=epoch)

            self.state.epoch = epoch + 1
            self._maybe_save(val_metrics, epoch, is_last)

        if cfg.trainer.max_steps is None and self.state.epoch < cfg.trainer.epochs:
            self.state.epoch = cfg.trainer.epochs

        finals = self.validate()
        _LOGGER.info("finished | val MAE %.4f | val RMSE %.4f", finals["mae"], finals["rmse"])
        save_json(finals, self.run_dir / "final_metrics.json")
        self.tracker.close()
        return finals

    def _maybe_save(self, val_metrics: dict[str, float], epoch: int, is_last: bool) -> None:
        cfg = self.cfg
        save_kwargs: dict[str, Any] = {
            "model": self.model,
            "config": cfg,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "best_metric": self.state.best_metric,
            "rng_state": capture_rng_state(),
            "collator_state": self.train_collator.get_state(),
        }

        improved = bool(val_metrics) and (
            self.state.best_metric is None or val_metrics["mae"] < self.state.best_metric
        )
        if improved:
            self.state.best_metric = val_metrics["mae"]
        if improved and cfg.trainer.save_best:
            save_checkpoint(self.run_dir / "best.pt", **save_kwargs)
            _LOGGER.info(
                "new best val MAE %.4f -> %s", self.state.best_metric, self.run_dir / "best.pt"
            )
        if cfg.trainer.save_last:
            save_checkpoint(self.run_dir / "last.pt", **save_kwargs)
        if cfg.trainer.save_every and (epoch + 1) % cfg.trainer.save_every == 0:
            save_checkpoint(self.run_dir / f"epoch{epoch + 1:04d}.pt", **save_kwargs)
        if is_last and not cfg.trainer.save_last and not improved:
            save_checkpoint(self.run_dir / "last.pt", **save_kwargs)

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #
    def resume(self, path: str | Path) -> None:
        """Restore model/optimiser/schedule/RNG state and counters from ``path``."""
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        self.state.epoch = state.epoch
        self.state.global_step = state.global_step
        self.state.best_metric = state.best_metric
        if state.rng_state is not None:
            restore_rng_state(state.rng_state)
        if state.collator_state is not None:
            self.train_collator.set_state(state.collator_state)
        _LOGGER.info(
            "resumed %s at epoch %d (step %d, best MAE %s)",
            path,
            self.state.epoch,
            self.state.global_step,
            self.state.best_metric,
        )


class _Tracker:
    """Minimal experiment-tracking shim (TensorBoard and/or Weights & Biases).

    Both backends are optional and disabled by default; failures degrade to
    logging rather than aborting a run.
    """

    def __init__(self, cfg: ExperimentConfig, run_dir: Path) -> None:
        self._tb: Any = None
        self._wandb: Any = None

        if cfg.trainer.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            except ImportError:  # pragma: no cover - optional dependency
                _LOGGER.warning(
                    "tensorboard requested but not installed (`uv sync --extra tracking`)"
                )
        if cfg.trainer.wandb:
            try:
                import wandb

                self._wandb = wandb.init(
                    project=cfg.trainer.wandb_project,
                    name=cfg.trainer.run_name,
                    dir=str(run_dir),
                    config=cfg.model_dump(mode="json"),
                )
            except ImportError:  # pragma: no cover - optional dependency
                _LOGGER.warning("wandb requested but not installed (`uv sync --extra tracking`)")

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Record ``metrics`` at ``step`` with every enabled backend."""
        if self._tb is not None:
            for key, value in metrics.items():
                self._tb.add_scalar(key, value, step)
        if self._wandb is not None:  # pragma: no cover - requires wandb
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        """Flush and close the backends."""
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        if self._wandb is not None:  # pragma: no cover - requires wandb
            self._wandb.finish()


def load_model_for_eval(
    cfg: ExperimentConfig,
    checkpoint: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[nn.Module, torch.device]:
    """Rebuild the model from ``cfg`` and load ``checkpoint`` for evaluation.

    Args:
        cfg: Experiment configuration (must match the checkpoint's architecture).
        checkpoint: Path to the ``.pt`` file.
        device: Override the configured device.

    Returns:
        ``(model, device)`` with the model in ``eval`` mode.
    """
    resolved = resolve_device(cfg.trainer.device) if device is None else torch.device(device)
    model = GIPACount(cfg)
    meta = load_checkpoint(checkpoint, model=model, map_location="cpu")
    model.to(resolved)
    model.eval()
    save_json(
        {"checkpoint": str(checkpoint), "epoch": meta.epoch, "global_step": meta.global_step},
        Path(checkpoint).parent / "loaded_meta.json",
    )
    return model, resolved
