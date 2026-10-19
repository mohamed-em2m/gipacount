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
  table of documented assumptions (A1–A20 plus A12b).

**Not delivered / out of scope** (see §6): the cross-dataset transfer experiments
(CARPK, PUCPR+, ShanghaiTech). The 200-epoch FSC-147 run **is** launched and
documented in §5b (it takes about a day on a single RTX A6000).

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
  The FSC-147 image archive (`images_384_VarV2.zip`, 6146 files) was obtained from a
  public Hugging Face mirror and extracted under `data/fsc147/`.

### 2.1 GPU environment used for the full run

The shipped dependency group pins **CPU** torch wheels so `uv sync` works anywhere.
For the full training run the venv was switched to CUDA in place, without touching the
project metadata:

```bash
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu130 --reinstall torch torchvision
# -> torch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA available: True
```

Because `uv run` re-syncs the environment back to the locked CPU wheels, the long run is
launched with the venv interpreter directly:

```bash
.venv/bin/python scripts/train.py --config configs/default.yaml \
    --run-name fsc147_vit_3shot_200ep -s trainer.num_workers=16
```

---

## 3. Verification results (Phase 3)

Run on the development machine (CPU, 8-core container), Python 3.11.16.

| Check | Command | Result |
|---|---|---|
| Lint | `ruff check .` | **clean** |
| Format | `ruff format --check .` | **clean** (44 files) |
| Types | `mypy src` (strict) | **clean** (38 files) |
| Tests | `pytest` | **92 passed**, ~70 s |
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
4. `scripts/download_data.py::_strip_common_prefix` treated the archive's own
   `images_384_VarV2/` directory as a wrapper and stripped it, flattening 6146 images
   into the dataset root (so the dataset failed its own verification check). Found by
   actually running the downloader on the real archive. Fixed in `scripts/download_data.py`.

### Profiling the real 1024 px step (and why the loss was ~80 % of it)

The first real-data step on the A6000 took **4.5 s**. Timing each component in isolation
revealed the network was *not* the problem:

| Component | Time (batch 4, 1024 px) |
|---|---|
| frozen ViT-S/16 backbone | 31 ms |
| prompt tokens (Eq. 1) | 0.6 ms |
| GFIM, 6 layers (Eqs. 2–5) | 13 ms |
| PLAM (Eqs. 6–10) | 2 ms |
| decoder (Eq. 11) | 6 ms |
| **Generalized Loss (Eq. 12)** | **1417 ms** |

The forward pass totals ~53 ms; the unbalanced-OT solve was ~27× more expensive. Three
fixes, none of which change the objective:

1. **Kernel reuse.** The Sinkhorn loop evaluated `exp(−C/ε)` *twice* per iteration (once
   for `Kv`, once for `Kᵀu`) even though both use the same `K`. It is now materialised
   once per ε step (streamed chunk-wise above `loss.kernel_cache_mb`). Verified
   equivalent: max relative difference in `v*` ≈ 5e-7 (fp32 round-off).
2. **Gram distances.** `torch.cdist(...,p=2).pow(2)` computed a square root only to
   square it again; replaced by `|x|² + |y|² − 2x·yᵀ` (BLAS, no `sqrt`).
3. **Iteration budget.** `sinkhorn_iters` 50 → **20**, per Wan et al.'s own statement
   that ε-scaling "needs less than 20 iterations until converge"; and the loss runs in
   `float32` (measured difference from `float64`: ~5e-7 relative).

Effect on the median step, batch 4 at 1024 px:

| Configuration | Median step | 200-epoch projection |
|---|---|---|
| start (`float64`, 50 iters, `cdist`, double exp) | 4.53 s | ~9.4 days |
| + 16 loader workers | 1.87 s | ~4.6 days |
| + kernel reuse & Gram distances | 0.34 s | ~20 h |
| + 20 iterations, `float32` | **0.25 s** | **~15 h** |

Peak GPU memory is 5.9 GB, so there is ample headroom on a 48 GB card.

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

## 5b. The full FSC-147 run (in progress)

The paper's headline setting (`configs/default.yaml`: 1024 px, ViT-S/16, 3-shot,
batch 4, 200 epochs, CCSS p=0.5, Generalized Loss) was launched on the real FSC-147
dataset on a single RTX A6000:

```bash
.venv/bin/python scripts/train.py --config configs/default.yaml \
    --run-name fsc147_vit_3shot_200ep -s trainer.num_workers=16
```

* train 3659 images / 915 steps per epoch, val 1286 images;
* ~0.14 s per training step ⇒ ~2–3 min per epoch, plus ~1.6 min validation ⇒
  **≈15–20 h for all 200 epochs**;
* peak GPU memory 5.9 GB; checkpoints are written every epoch (`last.pt` + `best.pt`
  by val MAE), so the run is resumable with `--resume runs/fsc147_vit_3shot_200ep/last.pt`;
* artefacts land in `runs/fsc147_vit_3shot_200ep/` (`config.yaml`, `run_info.json`,
  `last.pt`, `best.pt`, `final_metrics.json`).

Observed at launch: epoch 0 loss **66031 → 65168** over the first 150 steps, i.e. the
model is already descending from its initial (softplus-over-the-whole-map) scale.

> **Status.** Training was still running when this report was written; the table below
> is filled in as the run progresses (per-epoch val MAE/RMSE come from the logger and
> `final_metrics.json`).

| Checkpoint | val MAE | val RMSE | Paper (val) |
|---|---|---|---|
| epoch 1 | *pending* | *pending* | 8.92 / 31.78 |
| epoch 50 | *pending* | *pending* | |
| epoch 100 | *pending* | *pending* | |
| epoch 200 (final) | *pending* | *pending* | |

---

## 6. Deviations, limitations and not-reproduced items

1. **Full training run.** The 200-epoch, 1024-px FSC-147 run is launched and documented
   in §5b; it takes about a day of wall-clock on one A6000, so its final numbers are
   reported when it completes. All code paths, configs and the evaluation protocol are
   in place; `configs/default.yaml` is the runnable recipe.
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
7. **Generalized Loss (A10/A11).** Unit-normalised coordinates (the cost matrix would be
   262144×M otherwise, and `exp(-C/ε)` underflows with pixel units at `ε=0.01`);
   `ε=0.01`, `τ=0.5` per the paper, ε-scaling on, **20** iterations (Wan et al.: ε-scaling
   "needs less than 20 iterations until converge"), `float32`. The Gibbs kernel is
   materialised once per ε step and reused for both Sinkhorn directions; results are
   identical to the two-`exp` formulation to ~5e-7 relative.
8. **CCSS batching (A12).** With `P_mix=0.5`, one collage is built per iteration from
   the drawn batch (forward batch = 1); otherwise the batch is trained as single images.
9. **Duplicate-point rule (A20).** Deterministic radius-1 (then radius-2) reassignment
   of colliding annotations onto neighbouring empty cells, preserving mass.
10. **Determinism on CUDA (A12b).** `trainer.deterministic` defaults to `false`: with
    deterministic algorithms enabled, torchvision routes CUDA RoI-align through a
    lazily `torch.compile`d pure-Python kernel (≈0.5 s/step and it crashes on torch
    builds whose Inductor is inconsistent, as the CUDA 13.0 wheel tested here). Seeding
    is unaffected; `deterministic: true` remains the right choice on CPU/CI.

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
