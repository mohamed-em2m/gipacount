"""CLI smoke tests (thin wrappers over the library)."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gipacount.cli import app

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "gipacount" in result.stdout


def test_info_offline_parameter_count() -> None:
    """`info` builds the model; `pretrained=false` keeps it offline."""
    result = runner.invoke(
        app,
        [
            "info",
            "--config",
            str(CONFIGS / "smoke.yaml"),
            "-s",
            "model.backbone.pretrained=false",
        ],
    )
    assert result.exit_code == 0
    assert "parameter budget" in result.stdout


def test_info_rejects_bad_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  typo: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["info", "--config", str(bad)])
    assert result.exit_code == 2


def test_data_reports_synthetic() -> None:
    result = runner.invoke(app, ["data", "--dataset", "synthetic"])
    assert result.exit_code == 0
    assert "synthetic" in result.stdout


def test_train_help() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
