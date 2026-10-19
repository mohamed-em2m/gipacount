"""End-to-end: training, determinism, checkpointing/resume and the smoke budget."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

from gipacount.config import load_config
from gipacount.models.gipacount import GIPACount
from gipacount.training.checkpoint import load_checkpoint, save_checkpoint
from gipacount.training.trainer import Trainer, load_model_for_eval
from gipacount.utils.seed import capture_rng_state, set_seed
from tests.conftest import make_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_train_epoch_reduces_loss(run_dir: Path) -> None:
    """Overfit sanity: the loss decreases across epochs on a fixed toy set."""
    cfg = make_config()
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    losses = [trainer.train_epoch(epoch)["train/loss"] for epoch in range(4)]
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(torch.tensor(x)) for x in losses)


def test_validation_reports_mae_rmse(run_dir: Path) -> None:
    cfg = make_config()
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    metrics = trainer.validate()
    assert {"mae", "rmse", "num_samples"} <= set(metrics)
    assert metrics["num_samples"] > 0


def test_training_is_deterministic(tmp_path: Path) -> None:
    cfg = make_config()
    a = Trainer(cfg, device="cpu", run_dir=tmp_path / "a")
    b = Trainer(cfg, device="cpu", run_dir=tmp_path / "b")
    loss_a = a.train_epoch(0)["train/loss"]
    loss_b = b.train_epoch(0)["train/loss"]
    assert loss_a == pytest.approx(loss_b, rel=1e-6)

    # and the initial weights are identical for the same seed
    cfg2 = make_config()
    set_seed(0)
    m1 = GIPACount(cfg).gfim
    set_seed(0)
    m2 = GIPACount(cfg2).gfim
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_fit_writes_provenance_and_metrics(run_dir: Path) -> None:
    cfg = make_config()
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    finals = trainer.fit()
    assert finals["mae"] >= 0
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "run_info.json").is_file()
    assert (run_dir / "final_metrics.json").is_file()
    assert (run_dir / "last.pt").is_file()


def test_checkpoint_round_trip(run_dir: Path) -> None:
    cfg = make_config()
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    trainer.train_epoch(0)
    path = run_dir / "ckpt.pt"
    save_checkpoint(
        path,
        model=trainer.model,
        config=cfg,
        epoch=3,
        global_step=12,
        optimizer=trainer.optimizer,
        best_metric=1.25,
        rng_state=capture_rng_state(),
        collator_state=trainer.train_collator.get_state(),
    )

    fresh = GIPACount(cfg)
    meta = load_checkpoint(path, model=fresh)
    assert meta.epoch == 3
    assert meta.global_step == 12
    assert meta.best_metric == pytest.approx(1.25)
    assert meta.rng_state is not None
    assert meta.collator_state is not None
    for p1, p2 in zip(trainer.model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_resume_restores_progress(tmp_path: Path) -> None:
    cfg = make_config()
    trainer = Trainer(cfg, device="cpu", run_dir=tmp_path / "a")
    trainer.train_epoch(0)
    path = save_checkpoint(
        tmp_path / "a" / "resume.pt",
        model=trainer.model,
        config=cfg,
        epoch=1,
        global_step=4,
        optimizer=trainer.optimizer,
        best_metric=2.0,
        rng_state=capture_rng_state(),
        collator_state=trainer.train_collator.get_state(),
    )

    resumed = Trainer(cfg, device="cpu", run_dir=tmp_path / "b")
    resumed.resume(path)
    assert resumed.state.epoch == 1
    assert resumed.state.global_step == 4
    assert resumed.state.best_metric == pytest.approx(2.0)


def test_load_model_for_eval(run_dir: Path) -> None:
    cfg = make_config()
    trainer = Trainer(cfg, device="cpu", run_dir=run_dir)
    path = save_checkpoint(
        run_dir / "for_eval.pt",
        model=trainer.model,
        config=cfg,
        epoch=0,
        global_step=0,
    )
    model, device = load_model_for_eval(cfg, path, device="cpu")
    assert device.type == "cpu"
    assert not model.training


@pytest.mark.slow
def test_smoke_config_trains_within_budget(tmp_path: Path) -> None:
    """`make smoke`: the shipped smoke config trains + validates in under 60s on CPU."""
    cfg = load_config(CONFIGS / "smoke.yaml")
    started = time.perf_counter()
    trainer = Trainer(cfg, device="cpu", run_dir=tmp_path / "smoke")
    finals = trainer.fit()
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"smoke run took {elapsed:.1f}s"
    assert finals["mae"] >= 0
    assert (tmp_path / "smoke" / "best.pt").is_file()
