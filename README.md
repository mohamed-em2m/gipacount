# GIPACount

A clean, reproducible re-implementation of

> **GIPACount: Global interaction and prototype-guided aggregation network for
> exemplar-conditioned few-shot object counting** — *Neural Networks* (2027).

GIPACount counts instances of an arbitrary, *previously unseen* category in an
image, given only a handful of exemplar boxes that delineate that category.  It
combines three ideas:

1. **Global Feature Interaction Module (GFIM)** — stacked bidirectional
   attention between learnable prototype queries, exemplar tokens and image
   tokens, so that category knowledge flows *both* ways (exemplar → image and
   image → exemplar).
2. **Prototype-guided Local Aggregation Module (PLAM)** — the prototype queries
   are reshaped into a depthwise convolution kernel that is applied to the image
   tokens, producing a similarity map that highlights instances of the exemplar
   category.
3. **Category-Contrastive Supervision Strategy (CCSS)** — a training-time
   collage of images from different categories teaches the network to respond
   only to the exemplar's category.

The predicted similarity map is decoded into a density map whose integral is the
count.

This repository is an independent re-implementation from the paper's text (no
official code was available).  It is *not* a fork of any existing project.

---

## Quick start

```bash
# 1. install (uv handles the interpreter, deps and lockfile)
uv sync

# 2. run the CPU smoke test: data -> CCSS -> GFIM -> PLAM -> loss -> metrics
make smoke                      # or: uv run gipacount train --config configs/smoke.yaml

# 3. run the test suite
make test
```

Everything is driven by typed YAML configs; see `configs/`.

## Installation

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # runtime + dev dependencies
uv sync --extra tracking         # + TensorBoard / Weights & Biases
uv sync --extra viz              # + matplotlib (density-map plots)
uv sync --extra onnx             # + ONNX export
```

`uv sync` installs CPU PyTorch by default.  For CUDA, edit the index URLs in
`pyproject.toml` (`[tool.uv.index]` / `[tool.uv.sources]`) — see the comment
there — or pre-install the matching `torch`/`torchvision` wheels.

## Data

```bash
# FSC-147 (the paper's main benchmark): metadata is fetched automatically,
# the image archive must be supplied manually (see the script's --help)
python scripts/download_data.py --dataset fsc147 --root data/fsc147 \
    --images-zip /path/to/FSC147_384_V2.zip

# verify
uv run gipacount data --root data/fsc147
```

The **synthetic** dataset (`data.name=synthetic`) is generated on the fly and
needs no download; it backs the smoke test and CI.

## Training & evaluation

```bash
# full paper setting (frozen ViT-S/16, 1024px, 200 epochs, 3-shot)
uv run gipacount train --config configs/default.yaml

# 1-shot variant (Table 2)
uv run gipacount train --config configs/vit_fsc147_1shot.yaml

# ResNet variant ("GIPACount-R")
uv run gipacount train --config configs/cnn_fsc147_3shot.yaml

# evaluate a checkpoint (MAE/RMSE, optionally ECR)
uv run gipacount eval --config configs/default.yaml \
    --checkpoint runs/<run>/best.pt --split test --ecr --output metrics.json

# ablations (Table 7) and module panels (Tables 8-9)
uv run gipacount train --config configs/ablations/q5.yaml
```

Alternatively use the plain scripts: `python scripts/train.py`,
`python scripts/eval.py`, `python scripts/benchmark.py`.

Any config value can be overridden on the command line:

```bash
uv run gipacount train -c configs/default.yaml \
    -s trainer.epochs=5 -s model.gfim.layers=2 -s trainer.batch_size=2
```

## Configuration

Configs are Pydantic models (`src/gipacount/config.py`): unknown keys are
rejected, and every value the paper specifies — plus every value it leaves
unspecified — has a documented default.  A config may inherit from another with
a top-level `_base_` key:

```yaml
_base_: ./default.yaml
name: my-run
model:
  n_exemplars: 1
```

The directory `configs/ablations/` contains one file per ablation row/panel.

## Reproducibility

* One seeding entry point (`gipacount.utils.seed.set_seed`) covers Python,
  NumPy, PyTorch and CUDA.
* Every run writes its resolved config, a git hash, environment info and
  parameter counts next to its checkpoints (`runs/<run>/run_info.json`).
* Checkpoints store model, optimiser, scheduler, epoch/step counters **and the
  full RNG state**, so `--resume` continues the exact data/init stream.
* The device is chosen from config (`auto`/`cpu`/`cuda`/`cuda:N`/`mps`).
* Mixed precision and TensorBoard/W&B are opt-in flags.

## Project layout

```
src/gipacount/
  config.py            typed configuration objects
  models/              backbone, GFIM, PLAM, decoder, full network
  data/                FSC-147 / synthetic datasets, CCSS, collation
  losses.py            Generalized Loss (Eq. 12)
  metrics.py           MAE, RMSE, ECR
  training/            optimiser, checkpointing, training loop
  evaluation/          evaluation loop, ECR composite set
  utils/               seeding, logging, IO, device helpers
  cli.py               `gipacount` command-line interface
configs/               experiment, variant and ablation configs
scripts/               train / eval / benchmark / download_data
tests/                 shape, gradient, determinism, smoke-train tests
docs/                  paper spec + reproduction report
```

## Documentation

* [`docs/paper_spec.md`](docs/paper_spec.md) — the paper distilled into an
  equation→module map, a hyper-parameter table with source tags, an
  *Ambiguities & Assumptions* table (A1–A20 and A12b) and the expected results
  with tolerances.
* [`docs/reproduction_report.md`](docs/reproduction_report.md) — what was
  reproduced, how it was verified, and where this implementation deviates.

## Citation

```bibtex
@article{gipacount2027,
  title   = {GIPACount: Global interaction and prototype-guided aggregation network
             for exemplar-conditioned few-shot object counting},
  journal = {Neural Networks},
  year    = {2027}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
