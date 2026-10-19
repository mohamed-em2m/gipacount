"""Command-line interface: ``gipacount train | eval | data | predict | info``.

Thin wrappers only - all logic lives in the package so it stays importable and
testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from rich.console import Console
from rich.table import Table

from gipacount import __version__
from gipacount.config import ExperimentConfig, load_config
from gipacount.data.datamodule import build_dataloader, build_dataset
from gipacount.data.transforms import resize_and_pad
from gipacount.evaluation.ecr import build_ecr_loader
from gipacount.evaluation.evaluator import evaluate, evaluate_ecr
from gipacount.models.gipacount import GIPACount, count_from_density
from gipacount.training.trainer import Trainer, load_model_for_eval
from gipacount.utils.io import save_json
from gipacount.utils.logging import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="GIPACount: exemplar-conditioned few-shot object counting (Neural Networks 2027).",
)
console = Console()
_LOGGER = get_logger(__name__)

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Path to a YAML config.")]
SetOption = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override config keys, e.g. -s trainer.epochs=5."),
]


def _load(config: Path, overrides: list[str] | None, verbose: bool = False) -> ExperimentConfig:
    configure_logging(level="DEBUG" if verbose else "INFO", force=verbose)
    try:
        return load_config(config, overrides)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _print_metrics(title: str, metrics: dict[str, Any]) -> None:
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
@app.command()
def train(
    config: ConfigOption,
    set_: SetOption = None,
    device: Annotated[str | None, typer.Option(help="Override the configured device.")] = None,
    run_name: Annotated[str | None, typer.Option(help="Name of the run directory.")] = None,
    resume: Annotated[Path | None, typer.Option(help="Resume from a checkpoint.")] = None,
    epochs: Annotated[int | None, typer.Option(help="Override trainer.epochs.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Train GIPACount (Algorithm 1)."""
    cfg = _load(config, set_, verbose)
    if run_name:
        cfg.trainer.run_name = run_name
    if resume:
        cfg.trainer.resume = resume
    if epochs is not None:
        cfg.trainer.epochs = epochs

    trainer = Trainer(cfg, device=device)
    console.print(f"[bold]run dir:[/bold] {trainer.run_dir}")
    metrics = trainer.fit()
    _print_metrics("final validation", metrics)


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@app.command()
def eval(
    config: ConfigOption,
    checkpoint: Annotated[Path, typer.Option("--checkpoint", help="Model checkpoint (.pt).")],
    set_: SetOption = None,
    split: Annotated[str | None, typer.Option(help="val | test")] = None,
    ecr: Annotated[bool, typer.Option(help="Also compute ECR on the composite eval set.")] = False,
    max_batches: Annotated[int | None, typer.Option(help="Limit the number of batches.")] = None,
    output: Annotated[Path | None, typer.Option(help="Write metrics JSON here.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Evaluate a checkpoint on a split (MAE/RMSE, optionally ECR)."""
    cfg = _load(config, set_, verbose)
    if split:
        cfg.eval.split = split
    if cfg.eval.split not in {"val", "test"}:
        console.print(f"[red]error:[/red] unknown split {cfg.eval.split!r}")
        raise typer.Exit(code=2)

    model, device = load_model_for_eval(cfg, checkpoint)
    dataset = build_dataset(cfg.data, cfg.eval.split, n_exemplars=cfg.model.n_exemplars)
    loader = build_dataloader(
        cfg.data, dataset, batch_size=cfg.eval.batch_size, train=False, num_workers=0
    )
    metrics = evaluate(model, loader, device=device, max_batches=max_batches)
    _print_metrics(f"{cfg.eval.split} split", metrics)

    if ecr:
        ecr_loader = build_ecr_loader(
            cfg.data,
            dataset,
            n_composites=cfg.eval.ecr_num_composites,
            seed=cfg.trainer.seed,
        )
        metrics.update(evaluate_ecr(model, ecr_loader, device=device, ecr_eps=cfg.eval.ecr_eps))
        _print_metrics(
            "composite ECR set",
            {"ecr": metrics["ecr"], "mae": metrics["mae"], "rmse": metrics["rmse"]},
        )

    if output:
        save_json(metrics, output)
        console.print(f"metrics written to {output}")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@app.command()
def data(
    root: Annotated[Path, typer.Option(help="Dataset root directory.")] = Path("data"),
    dataset: Annotated[str, typer.Option(help="fsc147 | synthetic")] = "fsc147",
    check: Annotated[bool, typer.Option(help="Verify that the expected files exist.")] = True,
) -> None:
    """Show dataset status and preparation instructions."""
    if dataset == "synthetic":
        console.print("[green]synthetic[/green]: generated on the fly; nothing to download.")
        return

    expected = [
        "Train_Test_Val_FSC_147.json",
        "annotation_FSC147_384.json",
        "ImageClasses_FSC147.txt",
        "images_384_VarV2",
    ]
    table = Table(title=f"FSC-147 under {root}", show_header=True, header_style="bold")
    table.add_column("expected path")
    table.add_column("present")
    all_present = True
    for name in expected:
        path = root / name
        present = path.exists()
        all_present &= present
        table.add_row(name, "[green]yes[/green]" if present else "[red]no[/red]")
    console.print(table)

    if not all_present and check:
        console.print(
            "\n[bold]Prepare the data with:[/bold]\n"
            f"  python scripts/download_data.py --dataset fsc147 --root {root}\n\n"
            "Sources (paper §4.1):\n"
            "  FSC-147  https://github.com/cvlab-stonybrook/LearningToCountEverything\n"
            "  CARPK/PUCPR+  https://lafi.github.io/LPN/\n"
            "  ShanghaiTech  https://github.com/svishwa/crowdcount-mcnn"
        )


# --------------------------------------------------------------------------- #
# info
# --------------------------------------------------------------------------- #
@app.command()
def info(
    config: ConfigOption,
    set_: SetOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Print the resolved config and the model's parameter budget."""
    cfg = _load(config, set_, verbose)
    console.print_json(json.dumps(cfg.model_dump(mode="json"), default=str))

    if cfg.model.backbone.pretrained:
        console.print(
            "[yellow]note[/yellow] building the model to count parameters downloads pretrained "
            "weights; pass -s model.backbone.pretrained=false for an offline count."
        )
    model = GIPACount(cfg)
    table = Table(title="parameter budget", show_header=True, header_style="bold")
    table.add_column("module")
    table.add_column("params", justify="right")
    breakdown = model.parameter_breakdown()
    for key, value in breakdown.items():
        if key.endswith(".trainable"):
            table.add_row(f"  {key}", f"{value / 1e6:.3f} M")
        elif key in {"total", "trainable"}:
            table.add_row(f"[bold]{key}[/bold]", f"[bold]{value / 1e6:.3f} M[/bold]")
        else:
            table.add_row(key, f"{value / 1e6:.3f} M")
    console.print(table)
    console.print(
        "paper (Table 6): 11.31 M trainable; see docs/paper_spec.md A14 for the "
        "paper's own inconsistency here."
    )


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
@app.command()
def predict(
    config: ConfigOption,
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    image: Annotated[Path, typer.Option("--image", help="Query image.")],
    boxes: Annotated[
        str,
        typer.Option("--boxes", help="Exemplar boxes as 'x1,y1,x2,y2;x1,y1,x2,y2'."),
    ],
    set_: SetOption = None,
    save_density: Annotated[Path | None, typer.Option(help="Write the density map as .pt.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Count objects in a single image given exemplar boxes."""
    import numpy as np
    from PIL import Image

    cfg = _load(config, set_, verbose)
    model, device = load_model_for_eval(cfg, checkpoint)

    pil = Image.open(image).convert("RGB")
    array = np.asarray(pil, dtype=np.float32).copy()
    image_t = torch.from_numpy(array).permute(2, 0, 1) / 255.0

    parsed = _parse_boxes(boxes)
    box_t = torch.tensor(parsed, dtype=torch.float32)

    geom = resize_and_pad(image_t, box_t, torch.zeros(0, 2), cfg.model.image_size)
    mean = torch.tensor(cfg.data.mean).view(3, 1, 1)
    std = torch.tensor(cfg.data.std).view(3, 1, 1)
    normalised = ((geom.image - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            normalised,
            geom.boxes.unsqueeze(0).to(device),
            pad_mask=geom.pad_mask.unsqueeze(0).to(device),
        )

    count = float(count_from_density(output.density).item())
    console.print(f"[bold]predicted count:[/bold] {count:.2f}")
    if save_density:
        torch.save(output.density.cpu(), save_density)
        console.print(f"density map saved to {save_density}")


def _parse_boxes(spec: str) -> list[list[float]]:
    """Parse ``"x1,y1,x2,y2;x1,y1,x2,y2"`` into a list of boxes."""
    boxes: list[list[float]] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        values = [float(v) for v in chunk.split(",")]
        if len(values) != 4:
            raise typer.BadParameter(f"each box needs 4 numbers, got {chunk!r}")
        boxes.append(values)
    if not boxes:
        raise typer.BadParameter("at least one exemplar box is required")
    return boxes


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"gipacount {__version__}")


def main() -> None:
    """Entry point used by the console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
