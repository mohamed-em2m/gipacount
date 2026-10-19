# Reproduction Report — GIPACount

> Paper: J. He, B. Liu, F. Cao, N. Liu, Y. Xiao, *"GIPACount: Global interaction and
> prototype-guided aggregation network for exemplar-conditioned few-shot object
> counting"*, Neural Networks 206 (2027) 109719.
> This report describes what was rebuilt, how it was verified, and where it deviates.

---

## 1. Scope

This repository is a **clean-room re-implementation** of the paper's method from its
text and figures. No official code exists (verified — the only similarly named local
directory, `float-fast/`, is an unrelated audio project), so nothing was copied.

Delivered:

* a typed, `uv`-managed Python package (`src/` layout) implementing the full
  **inference and training** pipeline: frozen backbone → exemplar ROI tokens → GFIM
  (Eqs. 1–5) → PLAM (Eqs. 6–10) → density decoder (Eq. 11), the Generalized Loss
  (Eq. 12), the CCSS collage supervision (Eqs. 13–18), and MAE/RMSE/ECR (Eqs. 19–21);
* configs for the paper's main setting, the 1-shot variant, the ResNet variant and
  every ablation panel (Tables 7–9);
* scripts to download data, train, evaluate and benchmark;
* a test suite covering shapes, gradient flow, parameter budget, determinism,
  loss behaviour, data/geometry conventions and an end-to-end smoke run;
* `docs/paper_spec.md` (the Phase-0 specification) with an equation→module map and a
  table of 20 documented assumptions (A1–A20).

**Not delivered / out of scope** (see §6): the full 200-epoch FSC-147 training run, and
the cross-dataset transfer experiments (CARPK, PUCPR+, ShanghaiTech).

---

## 2. Environment & how to run

* Python 3.11, PyTorch (CPU wheels by default; CUDA-ready configs), `timm`, `pydantic`,
  `typer`; dev tooling via PEP-735 dependency groups.
* Everything is driven by `uv`:

```bash
uv sync                 # runtime + dev deps (CPU torch; see pyproject for CUDA)
uv lock                 # writes uv.lock (committed)
make check              # ruff + mypy + fast tests
make test               # full suite incl. the slow smoke-budget test
make smoke              # uv run gipacount train --config configs/smoke.yaml
```

* `scripts/download_data.py` fetches the small FSC-147 metadata from GitHub and can
  ingest a manually downloaded image archive; the **synthetic** dataset needs nothing.

---

## 3. Verification results (Phase 3)

Run on the development machine (CPU, 8-core container), Python 3.11.16.

| Check | Command | Result |
|---|---|---|
| Lint | `ruff check .` | **clean** |
| Format | `ruff format --check .` | **clean** (53 files) |
| Types | `mypy src` (strict) | **clean** (38 files) |
| Tests | `pytest` | **92 passed**, ~65 s |
| Smoke train | `gipacount train -c configs/smoke.yaml` | **~8 s**, loss **1301.3 → 1108.7** over 2 epochs |

The suite asserts, among other things:

* **Shapes** at every stage — `X_I (B,64²,256)`-style grids at the tiny resolution,
  prototype kernel `(B,d,3,3)`, similarity `(B,d,√N_I,√N_I)`, density `(B,1,H/2,H/2)`.
* **Gradient flow** — GFIM/PLAM receive gradients; frozen backbone parameters do not.
* **Parameter budget** — `parameter_breakdown()` totals are internally consistent and
  only the head (+ the trainable 1×1 projection, A17) is trainable.
* **Determinism** — identical seeds give identical weights and identical epoch losses.
* **Geometry invariance** — the ×8 decoder keeps `density == image/2` at 64/128/256 px.
* **Eq. 10** — the vectorised depthwise convolution matches an explicit reference loop.
* **Padding** — padded regions contribute exactly zero density.
* **GL** — finite, differentiable, chunked == un-chunked, and gradient descent on the
  density reduces it (overfit sanity).
* **Data** — resize/pad, hflip consistency, duplicate-point reassignment with mass
  preservation, CCSS quadrant offsets and category filtering (Eq. 15).
* **End-to-end** — training reduces loss, checkpoints round-trip and resume restores
  epoch/step/best-metric, and the shipped smoke config trains+validates in **< 60 s**.

### Bugs caught by the tests (and fixed)

1. `roi_align` returns `(B·K, C, h, w)` when given a per-image list — the exemplar
   encoder was reshaping it wrongly. Fixed in `models/exemplar.py`.
2. The config's decoder-geometry check used a hard-coded stride 2 instead of the
   configured stride, so it could not actually fire. Fixed in `config.py`.
3. `build_point_map` used fancy-index assignment, which drops all but one of a set of
   duplicate cells; switched to `index_add_` so mass accumulates. Fixed in
   `data/point_map.py`.

---

## 4. Parameter budget (Table 6 check)

Measured with `gipacount info` / `GIPACount.parameter_breakdown()` (backbone built with
`pretrained=false`; the architecture is identical):

| Config | Total | Trainable |
|---|---|---|
| `default.yaml` (ViT, 3-shot, full) | 47.21 M | **24.05 M** |
| `vit_fsc147_1shot.yaml` | 47.19 M | 24.02 M |
| `cnn_fsc147_3shot.yaml` (ResNet-50, layer-3) | 25.52 M | 24.08 M |
| `ablations/q4.yaml` (no interaction) | 43.64 M | 20.48 M |
| `ablations/q7.yaml` (+ last 3 blocks) | 47.21 M | 29.37 M |

The paper reports **11.31 M trainable** (and 80.42 M / 37.73 M total). Our trainable
count is ≈24 M. Decomposition of the ≈24 M head:

| Component | Params |
|---|---|
| 6 × GFIM layer (3 attention ops + MLP + fusion MLP) | 17.37 M |
| per-layer learnable positional encodings (A7) | 5.29 M |
| decoder | 0.69 M |
| shape encoder + PLAM + prototypes | ≈0.33 M |
| backbone 1×1/linear projection | 0.10 M |

This is the **A14** inconsistency: `L=6`, `d=256`, `hidden=2048` and three attention
operations per layer cannot sum to 11.31 M. The dominant unaccounted term is the
per-layer learnable positional encodings that the paper never mentions — sharing a
single PE across layers would bring the count to ≈18.7 M. We implement the architecture
as stated and report the true number rather than shrinking the model to fit the table.
Likewise, total parameters differ from the paper's 80.42 M because the exact backbone
checkpoint is unnamed (A1) — we use `vit_small_patch16_224` (21.7 M of weights) plus
`resnet50` for the CNN variant.

---

## 5. Reproducibility features

* **One seeding entry point** (`utils/seed.set_seed`) covering Python/NumPy/Torch/CUDA,
  plus `torch.use_deterministic_algorithms` (warn-only) and cuDNN flags.
* **Provenance**: every run writes its resolved `config.yaml`, `run_info.json`
  (git hash, timestamp, versions, device, parameter counts) and `final_metrics.json`.
* **Checkpointing with full resume**: model, optimiser, LR schedule position,
  epoch/global-step/best-metric, the **full RNG state**, and the CCSS collator's
  generator state — so a resumed run continues the exact data/init stream.
* **Device-agnostic**: `trainer.device` ∈ {`auto`,`cpu`,`cuda`,`cuda:N`,`mps`}; AMP and
  TensorBoard/W&B are opt-in flags.
* **No hard-coded hyper-parameters**: every value lives in the typed config
  (`config.py`), which forbids unknown keys and documents each [P]/[D] source.

---

## 6. Deviations, limitations and not-reproduced items

1. **Full training run.** We do not run the paper's 200-epoch, 1024-px FSC-147 training
   here (data access + GPU time). All code paths, configs and the evaluation protocol
   are in place; `configs/default.yaml` is the runnable recipe.
2. **Cross-dataset transfer.** CARPK/PUCPR+/ShanghaiTech configs are intentionally
   **not shipped** (the chosen data scope was FSC-147 + synthetic only). The paper does
   not specify how exemplars are obtained on these single-category datasets; assumption
   A19 (synthetic exemplar sampling from GT boxes) would be the protocol. The expected
   numbers remain in `docs/paper_spec.md` §5 for completeness.
3. **1-shot CNN variant** is a one-line config (drop `n_exemplars` to 1 in
   `cnn_fsc147_3shot.yaml`); not shipped to avoid clutter.
4. **Backbone checkpoint unnamed (A1).** We use `vit_small_patch16_224` (patch-16 ⇒ the
   paper's 64×64 grid at 1024 px) and `resnet50` layer-3. Absolute metrics will differ
   from the paper with a different pretrained checkpoint.
5. **Positional encoding (A6/A7).** Per-layer *learnable* absolute PE from layer 2
   onward, for image and prompt streams. This is a literal reading and the main driver
   of the parameter gap (A14); `pos_encoding: sinusoidal` is available and a shared-PE
   variant is a config change.
6. **Output activation (A9).** `softplus` (non-negative mass for the Generalized Loss),
   multiplied by a down-sampled padding mask.
7. **Generalized Loss (A10/A11).** Chunked Sinkhorn with unit-normalised coordinates
   (the cost matrix would be 262144×M otherwise, and `exp(-C/ε)` underflows with pixel
   units at `ε=0.01`); `ε=0.01`, `τ=0.5` per the paper, ε-scaling on, 50 iterations.
8. **CCSS batching (A12).** With `P_mix=0.5`, one collage is built per iteration from
   the drawn batch (forward batch = 1); otherwise the batch is trained as single images.
9. **Duplicate-point rule (A20).** Deterministic radius-1 (then radius-2) reassignment
   of colliding annotations onto neighbouring empty cells, preserving mass.

### Paper self-inconsistencies (flagged, not "fixed")

* **A13** — Table 7 Q6 (full method) reports MAE 9.15, while Table 8 Panel B reports
  10.46 for what is described as the same full-PLAM configuration. We implement the
  method, not the table bookkeeping.
* **A14** — the 11.31 M trainable-parameter claim (Table 6) is incompatible with the
  stated architecture (see §4).

---

## 7. Mapping to the assignment's requirements

| Requirement | Where |
|---|---|
| Phase 0 spec, Eq.→module map, assumptions | `docs/paper_spec.md` |
| Typed configs, no hard-coded hyper-parameters | `src/gipacount/config.py` |
| Modern Python, full type hints, Google docstrings | throughout `src/`; `mypy --strict` clean |
| `uv`, `src/` layout, PEP-735 groups, lockfile | `pyproject.toml`, `uv.lock` |
| Ruff / mypy / pytest / coverage / pre-commit / CI | `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` |
| Every equation cited with tensor shapes | docstrings across `models/`, `losses.py`, `metrics.py` |
| Shape / gradient / determinism / loss / overfit / smoke tests | `tests/` |
| Training < 60 s on CPU for the smoke config | `tests/test_smoke_train.py::test_smoke_config_trains_within_budget` |
| Reproduction report | this file |

---

## 8. Quick command reference

```bash
uv run gipacount train  -c configs/default.yaml                 # paper setting
uv run gipacount train  -c configs/smoke.yaml                   # CPU smoke
uv run gipacount train  -c configs/ablations/q5.yaml            # an ablation
uv run gipacount eval   -c configs/default.yaml \
                        --checkpoint runs/<run>/best.pt --split test --ecr
uv run gipacount info   -c configs/default.yaml \
                        -s model.backbone.pretrained=false      # offline param count
uv run gipacount predict -c configs/default.yaml \
                        --checkpoint runs/<run>/best.pt \
                        --image path/to/img.jpg --boxes "x1,y1,x2,y2"
python scripts/benchmark.py -c configs/default.yaml --batch-size 1
```
