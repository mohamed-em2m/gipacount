# Paper Specification — GIPACount

> Source: J. He, B. Liu, F. Cao, N. Liu, Y. Xiao, *"GIPACount: Global interaction and
> prototype-guided aggregation network for exemplar-conditioned few-shot object
> counting"*, **Neural Networks 206 (2027) 109719**.
> DOI: `10.1016/j.neunet.2026.109719`. Local file: `1-s2.0-S0893608026011743-main.pdf`.
> Official repository: **none found** (verified: the only directory in the workspace
> named `float-fast/` is an unrelated audio-driven talking-face project).

---

## 1. Contribution summary (one paragraph)

GIPACount reformulates exemplar-conditioned few-shot counting (FSC) as an explicit
**global-to-local** process. A pretrained image encoder is kept **frozen** and produces
image tokens; exemplar boxes are pooled into a short prompt sequence built from
exemplar tokens concatenated with **learnable prototype-query tokens** `X_P`. A
**Global Feature Interaction Module (GFIM)** stacks `L` layers of three operations —
prompt self-attention, prompt→image cross-attention, image→prompt cross-attention —
so that exemplar priors and query context are refined *bidirectionally* rather than
one-way. After the stack, an extra prompt→image attention consolidates a final prompt
sequence `X'_T` whose first `N_P` tokens are **global prototype tokens**. A
**Prototype-guided Local Aggregation Module (PLAM)** turns those prototypes into a
per-channel `√N_P × √N_P` depthwise convolution kernel that is applied to the
exemplar-enhanced image feature map, producing a target-specific similarity map which a
transposed-convolution decoder upsamples into the density map `D̂` (its integral is the
count). A **Category-Contrastive Supervision Strategy (CCSS)** mixes four training
images into a `2×2` collage at probability `P_mix` and *zeroes* the point supervision of
sub-regions whose exemplar category differs from the anchor's, which removes the
dataset-induced shortcut of "count the salient/dense stuff". Training uses the
**Generalized Loss** (entropic-regularized unbalanced optimal transport, Wan et al.
CVPR 2021). Reported gains: FSC-147 3-shot test 9.15/45.90 MAE/RMSE with a frozen ViT
(9.64/48.31 with a frozen CNN), and large cross-dataset gains on CARPK, PUCPR+ and
ShanghaiTech, with only **11.31 M trainable parameters**.

---

## 2. Equations / algorithms / architecture → planned modules

Notation (Table 1): `I` query image; `{E_k}_{k=1..K}` exemplar boxes; `X_I ∈ R^{N_I×d}`
image tokens; `X_E ∈ R^{N_E×d}` exemplar tokens; `X_S ∈ R^{N_E×d}` box shape features;
`X_P ∈ R^{N_P×d}` prototype-query tokens; `X_T = [X_P; X_E+X_S] ∈ R^{N_T×d}` prompt
tokens with `N_T = N_P + N_E`; `V` binary point map; `D̂` predicted density map.

| Paper item | Content | Implementation |
|---|---|---|
| §3.3, Eq. 1 | Shape features `X_S` (3-layer FFN, LOCA-style) added to `X_E`; prompt tokens `X_T = [X_P; X_E + X_S]` | `models/shape_encoder.py::ShapeEncoder`; concat in `models/gipacount.py::GIPACount._build_prompt_tokens` |
| §3.3, Eq. 2 | Prompt self-attention: `A_T→T = softmax(Q Kᵀ/√d)`, `X_TT = A·V + X_TI^{l-1}` | `models/gfim.py::BidirectionalBlock.forward` (self-attention branch) |
| §3.3, Eq. 3 | Prompt→image cross-attention with 2-layer MLP fusion `M(·)`: `A_T→I`, `X_TI = M(A·V + X_TT)` | `models/gfim.py::BidirectionalBlock.forward` (prompt→image branch, `cross_attn_fusion`) |
| §3.3, Eq. 4 | Image→prompt cross-attention: `A_I→T`, `X_IT = A·V + X_IT^{l-1}` | `models/gfim.py::BidirectionalBlock.forward` (image→prompt branch) |
| §3.3, Eq. 5 | Final extra prompt→image attention producing `X'_T` | `models/gfim.py::_FinalFusion` (used by `GFIM.forward`) |
| §3.3 (text) | 3 operations stacked `L=6` times; residual + LayerNorm around every block; pre-LN | `models/gfim.py::GFIM` |
| §3.3 (text) | First layer: self-attention on prompts **without** positional encoding; from layer 2 on, absolute positional encodings for image **and** prompt tokens | `models/gfim.py::GFIM.forward` (`pos_encoding`, `pos_from_layer=1`), `models/pos_embed.py::TokenPositionalEncoding` |
| §3.3 (text) | 8-head multi-head attention, `d=256`, MLP hidden 2048 (GELU) | `GFIMConfig{n_heads=8, mlp_ratio=8}`, `models/attention.py::MultiheadAttention` |
| §3.1/3.2 (text) | Frozen pretrained backbone → flattened image tokens `X_I`, `N_I=64²`, `d=256`; boxes downscaled ×16 then ROI-pooled → `X_E`, `N_E=K×9` | `models/backbone.py::FrozenBackbone`, `models/exemplar.py::RoIExemplarEncoder` |
| §3.2 (text) | Exemplar boxes `E_k=[x1,y1,x2,y2]`, `K ∈ {1,3}` | `config.py::ModelConfig{n_exemplars}` |
| Eq. 6 | `X'_P = Slice(X'_T; 1:N_P)` (first `N_P` tokens) | `models/plam.py::PLAM._prototype_kernel` |
| Eq. 7 | `X̃_P = GELU(X'_P W_p1 + b_p1) W_p2 + b_p2` | `models/plam.py::PLAM.kernel_mlp` (`models/layers.py::Mlp`) |
| Eq. 8 | Reshape `X̃_P` → prototype kernel `X̂_P ∈ R^{d×√N_P×√N_P}` (`N_P=9` ⇒ `3×3`) | `models/plam.py::PLAM._prototype_kernel` |
| Eq. 9 | Reshape `X^L_IT` → `X̂_I ∈ R^{d×√N_I×√N_I}` (`256×64×64`) | `models/plam.py::PLAM.forward` (transpose + reshape) |
| Eq. 10 | Depthwise (grouped) convolution of `X̂_P` over each channel of `X̂_I` with padding `p`; then 1×1 conv to fuse channels → `Ŝ ∈ R^{d×64×64}` | `models/plam.py::prototype_depthwise_conv`, `PLAM.channel_fuse` |
| Eq. 11 | `D̂ = Dec(Ŝ)`: 3 transposed convs + 1 point-wise conv, each + LayerNorm + ReLU, 64×64 → 512×512 | `models/decoder.py::DensityDecoder` |
| Eq. 12 | Generalized Loss (entropic-regularized unbalanced OT), `ε=0.01`, `τ=0.5` | `losses.py::GeneralizedLoss` |
| §3.5, Eqs. 13–17 | CCSS: resize each of 4 samples to 512 long side, pad to 512², stitch 2×2 ⇒ `Ĩ ∈ R^{3×1024×1024}`; points `/2` ⇒ 256² per quadrant with quadrant offsets; category filter Eq. 15; anchor-only exemplars with offset, Eq. 17 | `data/ccss.py::build_composite`, `data/collate.py::CountingCollator` |
| §3.5, Eq. 18 | `L_train = L(Ṽ, D̂)` computed once on the whole stitched sample | `training/trainer.py::Trainer._train_step` |
| §4.3, Alg. 1 | Alternation between CCSS (`P_mix=0.5`) and standard single-image training over 200 epochs | `data/collate.py::CountingCollator._should_mix`; `training/trainer.py::Trainer.train_epoch` |
| §4.3 (text) | Single-image mode: resize+pad to 1024², point map at 512² | `data/transforms.py::SamplePreprocessor`, `resize_and_pad`; `data/point_map.py::build_point_map` |
| §4.2, Eq. 19 | `MAE` | `metrics.py::mean_absolute_error`, `CountMetrics` |
| §4.2, Eq. 20 | `RMSE` | `metrics.py::root_mean_squared_error`, `CountMetrics` |
| §4.2, Eq. 21 | `ECR` (exemplar-consistent response ratio) on the 2×2 composite eval set | `metrics.py::exemplar_consistent_response_ratio`, `evaluation/ecr.py::build_ecr_loader`, `evaluation/evaluator.py::evaluate_ecr` |
| §4.3 (text) | Image padding mask used to suppress padded regions in image attention | `models/backbone.py::FrozenBackbone.token_mask_from_padding`; mask propagated through `GFIM` (`key_padding_mask`) |
| §4.4.3 (text) | No global self-attention among image tokens; complexity `O(N_I·N_T)` | structural property of `GFIM` (image tokens only ever appear as keys/values of cross-attention) |

---

## 3. Hyperparameters, data, optimisation, metrics (with source tags)

Source tags: **[P]** paper body/table, **[A]** appendix/figure, **[D]** derived/default
chosen by us (see §4).

### 3.1 Model
| Setting | Value | Source |
|---|---|---|
| Input resolution (train + inference) | `3×1024×1024` | [P] §4.3 |
| Density map resolution | `512×512` (stride 2) | [P] §4.3 |
| Backbone | frozen pretrained ViT or ResNet; **exact checkpoint not named** | [P] §4.3 → [D] `timm` `vit_small_patch16_224` (64² tokens @1024) / `resnet50` layer-3 for GIPACount-R |
| Token dimension `d` | 256 | [P] §4.3 |
| `N_I` | 64² = 4096 | [P] §4.3 |
| Exemplar tokens `N_E` | `K × 9` (3×3 ROI pool, `K∈{1,3}`) | [P] §4.3 |
| Prototype-query tokens `N_P` | 9 (⇒ 3×3 kernel) | [P] §4.3, Table 8 |
| GFIM depth `L` | 6 | [P] §4.3 |
| Heads | 8 | [P] §4.3 |
| MLP hidden | 2048 | [P] §4.3 |
| Shape encoder | 3-layer FFN (LOCA design) | [P] §3.3 |
| Decoder | 3 transposed conv + 1 pointwise conv, LayerNorm + ReLU | [P] §4.3 |
| Trainable parameters (reported) | 11.31 M | [P] Table 6 |
| Total parameters (reported) | 80.42 M (ViT) / 37.73 M (CNN) | [P] Table 6 |
| FLOPs @1024 (reported) | 275.43 G | [P] Table 6 |

### 3.2 Training (Algorithm 1, §4.3)
| Setting | Value | Source |
|---|---|---|
| Optimizer | Adam | [P] §4.3 |
| LR | `1e-4` | [P] §4.3 |
| Weight decay | `1e-4` | [P] §4.3 |
| Scheduler | **not specified** | [P] → [D] constant LR |
| Epochs | 200 | [P] §4.3 / Alg. 1 |
| Batch size | 4 | [P] §4.3 |
| Frozen backbone | yes (all but last 3 layers in Q7 ablation) | [P] §4.3, Table 7 |
| CCSS probability `P_mix` | 0.5 | [P] §4.3, Alg. 1 |
| CCSS grid | 2×2 (3×3 ablated, worse) | [P] Table 9 |
| CCSS category filter | zero supervision where `C(E_i) ≠ C(E_1)` | [P] Eq. 15 |
| Augmentation | random horizontal flip + colour jitter (LOCA config) | [P] §4.3 |
| Loss | Generalized Loss, `ε=0.01`, `τ=0.5`, 20 Sinkhorn iterations | [P] Eq. 12, §4.3; iteration budget per Wan et al. (see A11) |
| Mixed precision | not specified | [P] → [D] off by default, flag available |

### 3.3 Data
| Dataset | Role | Source |
|---|---|---|
| FSC-147 (6135 images, 147 classes, 89/29/29 split, 3 exemplar boxes/image) | train/val/test (base→novel) | [P] §4.1 |
| CARPK (1448 aerial parking images) | cross-dataset test | [P] §4.1 |
| PUCPR+ (100 images) | cross-dataset test | [P] §4.1 |
| ShanghaiTech A/B (1198 imgs, 330165 heads) | cross-dataset test | [P] §4.1 |
| Mean objects/image (FSC-147) | ~56 (range ~7–3700) | [P] §4.1 |
| Preprocessing | resize long side + pad to square; boxes scaled to feature res (`/16`) | [P] §4.3 |

### 3.4 Metrics
| Metric | Definition | Source |
|---|---|---|
| MAE | mean \|N_target − Σ D̂\| | [P] Eq. 19 |
| RMSE | √(mean (N_target − Σ D̂)²) | [P] Eq. 20 |
| ECR | Σ_{Ω+} D̂ / (Σ_Ω D̂ + ε₀) on 2×2 composites | [P] Eq. 21 |

---

## 4. Ambiguities & Assumptions

Nothing below is silently invented: each row states the gap, our default, and why.

| # | Gap in the paper | Chosen default | Reasoning / impact |
|---|---|---|---|
| A1 | Backbone checkpoint never named; only "pretrained ViT or ResNet", `N_I=64²`, `d=256`, stride 16 | `timm` **`vit_small_patch16_224`** (frozen, 64² tokens at 1024²) for GIPACount; **`resnet50` layer-3** for GIPACount-R. Both followed by a `Linear(·, 256)` projection. | Patch-16 at 1024 ⇒ exactly the paper's 64×64 grid. DINOv2/`vit_base` alternatives are a config change. Affects absolute numbers, not the architecture. |
| A2 | Whether the exemplar ROI pooling acts before or after the 256-d projection | Pool **after** projection (roi_align `output_size=3`, `spatial_scale=1/16`) | `X_E ∈ R^{K·9×256}` must already be `d`-dimensional for the concatenation in Eq. 1. |
| A3 | `X_S` is one vector per box but `X_E` has 9 tokens per box | Shape feature is **broadcast** to the box's 9 tokens | Consistent with LOCA (box-level geometry) and with `N_E=K×9`. |
| A4 | Shape encoder input normalisation | boxes normalised to `[0,1]` by image `W,H`, encoded as `(x1,y1,x2,y2)` | Stable input scale; LOCA uses normalised box coordinates. |
| A5 | Eq. 3 writes `M(A·V + X_TT)` and also states "residual connections are used throughout" | Default `cross_attn_fusion="residual_mlp"`: `X_TI = LN(X_TT + M(A·V))`, `M` = `Linear(d,4d)-GELU-Linear(4d,d)`. `"literal"` mode implements `LN(M(A·V + X_TT))`. | Both readings are offered; the difference is where the residual is applied. Small numerical effect. |
| A6 | "Absolute positional encodings" — type unspecified | **Learnable** absolute PE (per-layer, image + prompt), introduced from layer index 1 (0-based), interpolated if the grid differs | Cosine alternatives behind `pos_encoding: sinusoidal`. Only affects convergence speed. |
| A7 | Which prompt tokens receive positional encodings and whether the same PE is shared across layers | Per-layer learnable PE added to both streams at every layer `l ≥ 1` | Literal reading of "from the second layer onward, absolute PEs are introduced for both image and prompt tokens". |
| A8 | Decoder channel widths not given | `256→128→64→32` transposed convs (k=4, s=2, p=1) + `1×1` conv → 1 channel | Delivers exactly the 64→512 upsample of §4.3. See A13 on the parameter count. |
| A9 | Output activation for a density map not specified; GL requires non-negative "mass" `a` | `softplus` (configurable `relu`/`none`), multiplied by the down-sampled padding mask | Counting needs `D̂ ≥ 0`; Sinkhorn divides by `a`, so negatives would break it. Masking prevents mass in padded regions. |
| A10 | GL operates on the 512² map, but its cost matrix would be `262144 × M` | Coordinates normalised to `[0,1]`; the Gibbs kernel `exp(−C/ε)` is **materialised once per ε step** (reused for both Sinkhorn directions) and streamed chunk-wise above `loss.kernel_cache_mb`; squared distances use the Gram identity instead of `cdist` | Normalised coordinates are required: with pixel units `exp(−C/ε)` underflows to 0 for `ε=0.01`. 262144×3700 would be ~4 GB. The materialisation is a pure implementation detail — same objective, ~2× less work. |
| A11 | GL hyper-parameters: paper says "defaults of Wan et al. with `ε=0.01`, `τ=0.5`"; Wan et al. use `ε=0.005` | `ε=0.01`, `τ=0.5` (paper's explicit values), ε-scaling on, **20** Sinkhorn iterations, log-domain off, `float32` | The paper's explicit numbers win over the citation. 20 iterations because Wan et al. state that with ε-scaling "less than 20 iterations until converge" — the earlier 50 was both unfaithful and ~2.5× slower. `float32` was measured against `float64` on real FSC-147 batches: max relative difference ≈ 5e-7, i.e. below the noise the objective cares about. |
| A12 | Batch size 4 in CCSS mode: 4 samples → 1 collage (batch 1?) or 4 collages? | Per iteration draw `batch_size` samples; with `P_mix` build **one** collage from them (forward batch = 1); otherwise forward all 4 | Literal reading of "four samples are randomly selected from the current training batch". Both modes emit a 1024² input / 512² target, so the loss is unchanged in form. |
| A12b | `trainer.deterministic`: the paper never demands bit-exact reruns, but `torch.use_deterministic_algorithms(True)` on CUDA makes torchvision route RoI-align through a lazily `torch.compile`d pure-Python kernel (seconds/step; broken Inductor on some torch builds) | Default `false` (seeded, `cudnn.benchmark=True`); `true` remains available for CPU/CI bit-exact runs | Determinism was costing ~4× wall-clock and crashing on this torch build. Seeding still makes runs reproducible to the usual standard. |
| A13 | Ablation Q4/Q6 (Table 7) list MAE 18.83/9.15 for "CCSS only"/"full", while Table 8 Panel B lists full PLAM as 10.46 — **the same configuration has two different numbers in the paper** | We reproduce the paper's *method*, not its table bookkeeping; the report flags the inconsistency | We cannot reconcile the paper's own numbers; documented in `docs/reproduction_report.md`. |
| A14 | Table 6 reports 11.31 M trainable params, but a faithful implementation of the stated architecture computes **≈24.0 M**: 17.4 M for the 6 GFIM layers (`d=256`, `hidden=2048` ⇒ a `256→2048→256` MLP *and* a `256→2048→256` fusion MLP *and* 3× `4·d²` attention per layer), 5.3 M for the per-layer learnable positional encodings (A7), 0.7 M for the decoder, ≈0.3 M for the shape encoder + PLAM, 0.1 M for the backbone projection | Implement the paper's stated architecture (`hidden=2048`); report the true count | The paper's own numbers are mutually inconsistent (also Table 7 vs Table 8, see A13). The gap is dominated by the per-layer learnable PEs the paper never mentions: a single PE shared across layers would give ≈18.7 M. We do not shrink the model to fit a table. |
| A15 | Steps per epoch not specified | `steps_per_epoch: null` ⇒ `len(train_loader)` | Standard FSC-147 protocol. |
| A16 | Validation frequency, EMA, early stopping, LR schedule — none specified | validate each epoch, keep best-val-MAE checkpoint, no EMA, constant LR | Minimal, faithful default. |
| A17 | Whether the 256-d projection is trainable | Trainable (`projection.trainable: true`) but excluded by default from the "GFIM+PLAM only" freeze rule for the *backbone* | The paper freezes only the backbone; the projection is part of the head. Toggle available. |
| A18 | Normalisation statistics for input images | ImageNet mean/std | Universal default for `timm` backbones. |
| A19 | CarPK/PUCPR+/ShanghaiTech preprocessing for cross-dataset eval (exemplar boxes are not provided) | Synthetic exemplar sampling from GT boxes (3 random instances per image) | The paper does not describe how exemplars are obtained on single-category datasets; sampling from the GT boxes is the only sensible class-agnostic protocol. |
| A20 | "radius 1" reassignment rule for duplicate points (§3.5) | For each collision, search the 8 neighbours in a deterministic order (then radius 2 as fallback) | Deterministic, reproducible, and matches the stated intent. |

---

## 5. Expected results (targets to reproduce, with tolerances)

Numbers below are the paper's own (Table 2–9). Absolute reproduction is **not** expected
without the exact backbone checkpoint (A1) and the paper's own inconsistencies (A13/A14);
the tolerances are the acceptance band for a faithful re-run.

| Experiment | Config | Metric | Paper | Tolerance |
|---|---|---|---|---|
| FSC-147 3-shot, ViT | `configs/default.yaml` | test MAE / RMSE | 9.15 / 45.90 | ±1.0 / ±5.0 |
| FSC-147 3-shot, ViT | `configs/default.yaml` | val MAE / RMSE | 8.92 / 31.78 | ±1.0 / ±4.0 |
| FSC-147 3-shot, CNN | `configs/cnn_fsc147_3shot.yaml` | test MAE / RMSE | 9.64 / 48.31 | ±1.0 / ±5.0 |
| FSC-147 1-shot, ViT | `configs/vit_fsc147_1shot.yaml` | test MAE / RMSE | 10.97 / 56.85 | ±1.5 / ±6.0 |
| FSC-147 1-shot, ViT | `configs/vit_fsc147_1shot.yaml` | val MAE / RMSE | 10.83 / 36.82 | ±1.5 / ±4.0 |
| FSC-147 1-shot, CNN | *(not shipped — see report)* | test MAE / RMSE | 11.81 / 60.46 | ±1.5 / ±6.0 |
| CARPK (FSC-147 → CARPK) | *(not shipped — see report)* | MAE / RMSE | 6.40 / 7.87 | ±1.0 / ±1.5 |
| PUCPR+ | *(not shipped — see report)* | MAE / RMSE | 36.52 / 59.34 | ±4.0 / ±7.0 |
| ShanghaiTech A | *(not shipped — see report)* | MAE / RMSE | 128.6 / 224.1 | ±10 / ±20 |
| ShanghaiTech B | *(not shipped — see report)* | MAE / RMSE | 35.3 / 50.2 | ±4 / ±6 |
| Ablation Q1→Q6 (Table 7) | `configs/ablations/q1..q6.yaml` | ordering of MAE/RMSE | Q1 worst … Q6 best | **ordering** must hold, absolute ±15 % |
| GFIM panel (Table 8A) | `configs/ablations/gfim_*.yaml` | ECR ordering: full 0.78 > w/o final 0.74 > p→i 0.73 > w/o self-attn 0.70 > i→p 0.56 | ordering | full GFIM best ECR |
| PLAM panel (Table 8B) | `configs/ablations/plam_*.yaml` | `N_P=9` best balance; `N_P=16` no gain | ordering | `N_P=9` ≥ `N_P=1`,`16` |
| CCSS panel (Table 9) | `configs/ablations/ccss_*.yaml` | random-mix 2×2 best MAE/RMSE | ordering | random-mix ≥ same/different-category |
| Efficiency (Table 6) | `scripts/benchmark.py` | trainable params | 11.31 M | see A14 — we report the actual ≈24.0 M |
| Smoke test | `configs/smoke.yaml` | loss decreases on the synthetic set | — | loss ↓ across epochs (verified: 1301 → 1109) |

### Verification checklist (Phase 3)
1. `tests/test_models.py` — tensor shapes at every stage (`X_I`, `X_T`, `X'_T`, kernel, `Ŝ`, `D̂`), gradient flow to GFIM/PLAM, determinism under a fixed seed, `N_E = K·9`, `D̂` ∈ `[0, ∞)` and zero on padded regions, depthwise conv (Eq. 10) matched against an explicit reference loop.
2. `tests/test_losses.py` — `GL(V, V) ≈ 0` (the theoretical optimum), GL ≥ 0, gradient flow, chunked == unchunked.
3. `tests/test_metrics.py` — MAE/RMSE/ECR against hand-computed values.
4. `tests/test_data.py` — point-map construction and duplicate reassignment, resize/pad box transformation, CCSS quadrant offsets (Eq. 14/17), category filtering (Eq. 15).
5. `tests/test_smoke_train.py` — full train+eval on the synthetic dataset, < 60 s on CPU; single-batch overfit drives the count error down.
