# SumoTrack Results

Short empirical notes from local runs. Treat these as terrain markers, not claims of optimizer quality.

## 2026-06-13: LFM/SYNTH matrix-path smoke

Setup:

- Model: `LiquidAI/LFM2.5-1.2B-Base`, local cache, `HF_HUB_OFFLINE=1`.
- Data: local SYNTH parquet shards from `/home/djg/.cache/nanochat/base_data_synth`.
- Scope: `--param-scope matrices-no-embeddings`.
- Trainable: 92 non-embedding 2D tensors, 1,035,993,088 params.
- Excluded: 134,219,776 embedding/lm-head params and 61,440 tiny 3D conv params.
- Sequence/batch: `--seq-len 192 --batch-size 1`.
- Timing: `--warmup-steps 1 --measure-steps 3`; reported step time excludes warmup and first basis initialization.
- Optimizer path: `SumoTrack`, Grassmann refresh, no torch compile.

Random-init rank sweep:

| rank | state bytes | measured step seconds | peak CUDA bytes | initial val loss | final val loss |
| ---: | ----------: | --------------------: | --------------: | ---------------: | -------------: |
| 64 | 88,866,816 | 0.181071 | 4,864,393,216 | 1.943316 | 1.939324 |
| 128 | 177,733,632 | 0.321723 | 4,961,828,864 | 1.943316 | 1.942642 |
| 256 | 355,467,264 | 0.721046 | 5,152,342,528 | 1.943316 | 1.944172 |

Rank-64 SVD-init comparison:

| init | rank | state bytes | measured step seconds | peak CUDA bytes | initial val loss | final val loss |
| --- | ---: | ----------: | --------------------: | --------------: | ---------------: | -------------: |
| random | 64 | 88,866,816 | 0.181071 | 4,864,393,216 | 1.943316 | 1.939324 |
| svd | 64 | 88,866,816 | 0.165038 | 4,864,393,216 | 1.943316 | 1.944222 |

Notes:

- State size scales linearly with rank, as expected.
- All tested ranks fit comfortably on the local CUDA device in this short sequence-length setting.
- The three measured steps are too few to interpret loss movement. The random rank-64 final-val improvement is a smoke signal only.
- The SVD-init run used warmup to absorb cold basis initialization. Its measured step time is therefore not SVD cold-start timing.
- Rank 256 exposes the eager projected-space SVD tax: step time grows faster than the state-memory story alone would suggest.
- Next meaningful comparison needs more measured steps, AdamW/Muon baselines, and eventually HeavyBall-style orthogonalization instead of eager SVD inside projected space.

## 2026-06-13: SUMO orthogonalization ablation

Question: does SUMO/Muon-style orthogonalization of the projected first moment help, or is SubTrack-style projected momentum enough?

Setup:

- Model/data/scope: same LFM/SYNTH `matrices-no-embeddings` setup as above.
- Trainable: 92 non-embedding 2D tensors, 1,035,993,088 params.
- Rank: 64.
- Init: SVD basis initialization.
- Steps: 20 optimizer steps.
- Tokens/update: 768 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=4`).
- Validation texts: 16.
- No torch compile.

Results:

| optimizer | lr | state bytes | peak CUDA bytes | initial val | final val | mean train | step seconds |
| --- | ---: | ----------: | --------------: | ----------: | --------: | ---------: | -----------: |
| SumoTrack, SVD ortho | 0.001 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.800248 | 2.192415 | 1.981366 |
| SumoTrack, SVD ortho | 0.0025 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.701340 | 2.046949 | 1.976006 |
| SumoTrack, SVD ortho | 0.01 | 88,866,816 | 5,035,031,040 | 2.345853 | 2.625968 | 2.397669 | 1.980586 |
| Projected momentum, no ortho | 0.0025 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.975293 | 2.309294 | 1.865864 |
| Projected momentum, no ortho | 0.01 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.815476 | 2.182461 | 1.866572 |
| torch AdamW quality anchor | 0.00002 | 4,143,972,720 | 9,097,523,712 | 2.345853 | 1.655455 | 1.962064 | 0.200987 |

Notes:

- The SUMO-style orthogonalized projected moment beat the no-orthogonalization projected-momentum ablation at the tested LRs.
- No-ortho improved when moved from `0.0025` to `0.01`, but still did not catch orthogonalized `0.0025`.
- Orthogonalized `0.01` overshot badly, so LR tuning matters; HeavyBall's `0.0025` default landed near the useful region for this setup.
- AdamW remains a quality anchor, not the target budget. Torch fused AdamW fit this matrix-only scope but used about 4.14 GB of optimizer state and 9.1 GB peak CUDA before embeddings/full fallback params. HeavyBall AdamW OOMed on the same scope during a one-step plumbing check.
- At the time of this ablation, local compiled HeavyBall paths were blocked by missing `Python.h`, so HeavyBall polar/NS was treated as a separate integration task rather than judged from uncompiled one-step smoke timing. Later notes below record the machine fix and compiled Newton-Schulz results.

## 2026-06-13: Norm-logged longer SUMO vs projected momentum

Question: after adding grad/param/update norm telemetry, does no-orthogonalization projected momentum catch SUMO when given a wider LR search and more tokens per optimizer update?

Setup:

- Same LFM/SYNTH `matrices-no-embeddings` scope as above: 92 2D tensors, 1,035,993,088 trainable params.
- Rank: 64.
- Init: SVD basis initialization.
- Steps: 1 warmup + 40 measured optimizer steps.
- Tokens/update: 1536 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=8`).
- Validation texts: 16.
- Norm logging: enabled. `update_norm` is the actual `SumoTrack` update vector norm after LR scaling, excluding decoupled weight decay; weight decay was `0` for these runs. AdamW-style generic update norms are intentionally not computed because cloning/streaming 1B params would distort the memory/perf story.

Representative command shape at the time:

```bash
HF_HUB_OFFLINE=1 uv run python experiments/llm_synth_smoke.py \
  --optimizers subspace --param-scope matrices-no-embeddings \
  --warmup-steps 1 --measure-steps 40 \
  --seq-len 192 --batch-size 1 --grad-accum-steps 8 --val-texts 16 \
  --rank 64 --basis-init svd --log-norms
```

Results:

| orthogonalization | lr | final val | mean train | mean grad norm | mean update norm | update/param | step seconds |
| --- | ---: | --------: | ---------: | -------------: | ---------------: | -----------: | -----------: |
| SVD SUMO | 0.0025 | 1.593341 | 1.861647 | 10.945538 | 0.191900 | 0.000491 | 0.559974 |
| SVD SUMO | 0.003 | 1.616218 | 1.863523 | 10.740152 | 0.230280 | 0.000590 | 0.555029 |
| SVD SUMO | 0.004 | 1.663988 | 1.893716 | 10.412884 | 0.307040 | 0.000786 | 0.553619 |
| no ortho | 0.01 | 1.699018 | 2.031259 | 11.752147 | 0.023460 | 0.000060 | 0.449519 |
| no ortho | 0.02 | 1.631776 | 1.948476 | 11.283048 | 0.036870 | 0.000094 | 0.437304 |
| no ortho | 0.04 | 1.612252 | 1.911619 | 10.968435 | 0.060896 | 0.000156 | 0.437575 |
| no ortho | 0.08 | 1.639280 | 1.943709 | 10.999331 | 0.108202 | 0.000277 | 0.437675 |

Notes:

- No-ortho was under-tuned in the earlier ablation; it improves substantially when moved above `0.01`, with the best tested point at `0.04`.
- SUMO still has the best tested validation loss (`1.593341` at LR `0.0025`) at the same 88.9 MB optimizer-state budget.
- The norm telemetry argues this is not just “take a bigger raw projected-momentum step.” No-ortho degrades by LR `0.08` while its mean update norm (`0.108`) remains well below SUMO LR `0.0025` (`0.192`). SUMO is changing the update geometry and scale together.
- Eager projected-space SVD costs about 25-30% step time in this norm-logged run (`~0.56s` vs `~0.44s`). This strengthens the case for integrating HeavyBall polar/Newton-Schulz as the next performance cut, but not at the cost of losing the observed SUMO geometry.

## 2026-06-13: Compiled HeavyBall Newton-Schulz orthogonalization

Question: can HeavyBall's compiled Newton-Schulz orthogonalization replace exact projected-space SVD without drifting from the SVD correctness baseline?

Machine fix:

- Torch/Inductor initially failed because `/usr/include/python3.14/Python.h` was missing.
- Fedora package fix: install `python3-devel` matching Python `3.14.5`.
- Downstream signal after install: HeavyBall compiled `inplace_orthogonal_` ran on CUDA for `newtonschulz`, `thinky_polar_express`, and `svd` modes.

Scale semantics:

- HeavyBall Muon's `orthogonalize_update` defaults to `scale_mode="scale"`, which flattens the full parameter to `[rows, cols]` and multiplies by `sqrt(max(1, rows / cols))`.
- Applying that same `"scale"` mode directly to a SUMO projected tensor can multiply by `sqrt(full_dim / rank)`, which is a different and often much larger scale.
- Added `orthogonalization_scale_mode="muon"`: orthogonalize in projected space, but scale according to the original full matrix shape. This preserves HeavyBall Muon aspect scaling without letting rank become an accidental LR multiplier.
- HeavyBall's `thinky_polar_express` path is only selected for square matrices; HeavyBall falls back to Newton-Schulz for rectangular tensors. SUMO projected updates are generally rectangular, so the practical HeavyBall hot path here is Newton-Schulz.

Short scale smoke:

| mode | scale mode | final val | mean update norm | note |
| --- | --- | ---: | ---: | --- |
| HeavyBall NS | projected `scale` | 2.898771 | 1.513056 | unstable at LR `0.0025`; projected-rank scaling too large |
| HeavyBall NS | `none` | 2.058131 | 0.095534 | descends, but drops HeavyBall Muon aspect scaling |
| HeavyBall NS | `muon` | 1.984581 | 0.143622 | descends and preserves full-matrix Muon scaling |

Rank-64 LFM/SYNTH comparison:

- Same matrix-only LFM/SYNTH setup as above.
- Rank: 64.
- Init: SVD basis initialization.
- Steps: 1 warmup + 40 measured optimizer steps.
- Tokens/update: 1536.
- LR: `0.0025`.
- Scale mode: `muon`.

| orthogonalization | mode | final val | mean train | mean update norm | update/param | step seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exact SVD | baseline | 1.632313 | 1.860567 | 0.288551 | 0.000739 | 0.560159 |
| HeavyBall | newtonschulz | 1.629435 | 1.871363 | 0.287822 | 0.000737 | 0.479736 |

Notes:

- HeavyBall Newton-Schulz closely matches exact SVD quality and update scale in this short run, with lower measured step time.
- Exact SVD remains valuable as a correctness/debug baseline, but it should no longer be the main performance path.
- The previous best unscaled SVD run (`1.593341`) is not directly comparable to the Muon-scaled runs; scale and LR are coupled. Muon-scale mode likely needs its own LR sweep.

## 2026-06-14: Broad no-embedding LFM/SYNTH quality smoke

Question: does rank-64 broad-scope `SumoTrack` preserve the state-memory story and keep useful movement when real model topology adds fallback tensors, or was the positive matrix-only signal too narrow?

Setup:

- Model/data: `LiquidAI/LFM2.5-1.2B-Base` from local cache, local SYNTH shards, `HF_HUB_OFFLINE=1`.
- Scope: `--param-scope broad-no-embeddings`; embeddings/lm-head frozen.
- Trainable: 1,036,120,832 params across 146 tensors: 92 matrix tensors / 1,035,993,088 params and 54 fallback tensors / 127,744 params.
- Rank/init: rank 64, random basis initialization.
- Steps: 1 warmup + 20 measured optimizer steps.
- Tokens/update: 768 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=4`).
- Validation texts: 8.
- Norm logging: enabled for `SumoTrack` runs.

Results:

| optimizer | ortho | lr | state bytes | matrix/fallback state bytes | peak CUDA bytes | final val | mean train | mean update/param | step seconds |
| --- | --- | ---: | ----------: | ---: | --------------: | --------: | ---------: | ----------------: | -----------: |
| SumoTrack | HeavyBall NS + Muon scale | 0.0025 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 1.734250 | 2.072461 | 0.001181 | 0.323579 |
| SumoTrack | HeavyBall NS + Muon scale | 0.005 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 1.793099 | 2.168231 | 0.002193 | 0.326059 |
| SumoTrack | none | 0.04 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 7.085179 | 7.481242 | 0.010982 | 0.285293 |
| SumoTrack | none | 0.005 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 1.811141 | 2.219735 | 0.001681 | 0.284458 |
| torch AdamW | n/a | 0.00002 | 4,144,483,912 | n/a / 4,144,483,912 | 9,129,377,280 | 1.512150 | 1.927465 | n/a | 0.203278 |

Notes:

- Broad topology preserved the SumoTrack state-memory story: fallback state was ~1.0 MB and did not dominate the ~89.9 MB total.
- Orthogonalized SumoTrack stayed numerically stable at LR `0.0025` and `0.005`, but `0.005` looked too hot on this short run.
- The no-ortho matrix-only LR prior did not transfer. LR `0.04` was an obvious broad-topology failure, not a noisy loss comparison. LR `0.005` was stable but still worse than orthogonalized LR `0.0025`.
- AdamW remains materially better in this short quality anchor, but it used about 46x SumoTrack optimizer state and much higher peak CUDA. That is the intended product tension, not a contradiction.
- Next useful gate is a longer broad run centered on orthogonalized LR `0.0025`, with a lower no-ortho bracket and possibly SVD init as a quality-initialization comparison. Continuing should be falsified if the AdamW quality gap fails to narrow with more steps/tokens or if no-ortho catches up after a fair broad LR tune.

## 2026-06-14: Aurora and two-sided projected geometry check

Question: should SumoTrack's projected orthogonalization remain one-sided rectangular HeavyBall NS, use Aurora-style leverage-uniform rectangular polar, or move to a two-sided square-core projection?

Setup:

- Model/data: `LiquidAI/LFM2.5-1.2B-Base`, local SYNTH, `HF_HUB_OFFLINE=1`.
- Scope: `--param-scope broad-no-embeddings`.
- Rank/init: rank 64, fixed bases unless noted.
- Steps: 1 warmup + 10 measured optimizer steps.
- Tokens/update: 768 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=4`).
- Validation texts: 8.
- Norm/leverage logging: enabled.

Results:

| projection | init | ortho | state bytes | final val | mean train | update/param | leverage CV | leverage min/max vs mean | step seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one-sided | random | HeavyBall NS | 89,888,768 | 1.735645 | 2.123392 | 0.001307 | 1.277685 | 0.109 / 36.609 | 0.347413 |
| one-sided | random | Aurora | 89,888,768 | 1.751235 | 2.145004 | 0.001313 | 0.025380 | 0.829 / 1.066 | 0.418227 |
| two-sided | random | HeavyBall NS | 90,642,432 | 1.801806 | 2.180583 | 0.001318 | 0.023224 | 0.917 / 1.043 | 0.341462 |
| two-sided | SVD | HeavyBall NS | 90,642,432 | 1.813795 | 2.162647 | 0.001303 | 0.019022 | 0.946 / 1.042 | 0.335513 |

Notes:

- One-sided HeavyBall NS has extreme large-axis leverage concentration in this diagnostic: average projected row-energy max was ~36x mean.
- Aurora fixes that mechanically, reducing leverage CV to ~0.025 and max/mean to ~1.07, with similar update/param. It slightly lost immediate target movement and cost ~20% step time over this short run.
- Two-sided square-core projection is stable and nearly state-neutral, but underperformed one-sided rectangular HeavyBall on immediate target movement. SVD init did not rescue it in this short fixed-basis comparison.
- Interpretation: one-sided rectangular updates remain the mainline for target adaptation. Aurora is still worth a retention-aware run because its value may be smoother/less forgetting-prone movement, not faster 10-step target loss. Two-sided should stay experimental until it shows retention or rank-budget advantage.

## 2026-06-17: Packed SDPA throughput sanity check

Question: can the LFM broad-no-embeddings harness run a realistic no-grad-accum throughput shape without accidentally disabling fast attention?

Findings:

- Padded `padding=max_length` throughput batches were the wrong test shape. The resulting non-null `attention_mask` made PyTorch SDPA reject flash attention (`Flash Attention does not support non-null attn_mask`) and made HF expand GQA key/value heads instead of using native SDPA GQA.
- Added packed full-sequence batches (`--pack-sequences`) that omit `attention_mask`; this preserves causal SDPA flash eligibility for throughput tests.
- `causal-conv1d` was temporarily built and verified during measurement, making LFM report its short-conv fast path available. It is not kept as a repo dependency because proper installation needs a coherent CUDA extension toolchain, not ad-hoc local build scaffolding. `flash-attn` source build was abandoned after memory pressure; do not source-build it casually in this environment.
- With `LiquidAI/LFM2.5-1.2B-Base`, broad-no-embeddings, rank 64, residual-facing side, CCE loss, packed `seq_len=1024`, and no grad accumulation, the local 12GB card fits `batch_size=4` but not `5`, `6`, `8`, or `16`.

Measured fitting point:

| batch × seq | tokens/step | checkpointing | compile | Aurora PP / NS | step seconds | tokens/sec | peak CUDA | state bytes |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 × 1024 | 4096 | off | off | 2 / 5 | 0.623791 | 6566 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | on | off | 2 / 5 | 0.625118 | 6552 | 10.75 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | on | 2 / 5 | 0.622031 | 6585 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | off | 1 / 5 | 0.607037 | 6748 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | off | 2 / 3 | 0.612473 | 6688 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | off | 1 / 3 | 0.601221 | 6813 | 10.70 GB | 89.9 MB |

Notes:

- Checkpointing and `torch.compile` were noise at this shape.
- Reducing Aurora/polar cycles gave a modest optimizer-side speedup; `1` Aurora preconditioning pass and `3` NS steps was ~3.6% faster than the default timing point. This is throughput-only evidence, not quality evidence.
- The remaining memory wall is LFM layer temporary/activation memory under dense broad training, not optimizer state and not SDPA masking.

## 2026-06-18: Packed SDPA, CCE path, and Aurora-cycle smoke

Question: do packed batches keep SDPA on the flash path, does the CCE harness path run, and do Aurora cycle knobs buy real throughput on the local 12GB card?

Setup:

- Model/data: `LiquidAI/LFM2.5-1.2B-Base`, local SYNTH, `HF_HUB_OFFLINE=1`.
- Scope: `--param-scope broad-no-embeddings`, rank 64, residual-facing side.
- Shape: packed `seq_len=256`, `batch_size=1`, `--skip-validation`.
- Init: `basis_init=random` for perf-path smoke; exact SVD init OOMed at this shape on the local card.
- Loss paths: `hf` and `cce` (`--chunked-lm-loss-tokens 128`).

Results:

| loss / Aurora cycles | tokens/step | step seconds | tokens/sec | peak CUDA |
| --- | ---: | ---: | ---: | ---: |
| HF / default `2/5` | 256 | 0.220295 | 1162.079 | 4.53 GB |
| HF / reduced `1/3` | 256 | 0.086944 | 2944.421 | 4.53 GB |
| CCE / default `2/5` | 256 | 0.187094 | 1368.299 | 4.53 GB |

Notes:

- Packed batches omit `attention_mask`, so the harness keeps SDPA flash eligibility instead of falling back to padded-mask behavior.
- The CCE path works in this harness shape with random basis init.
- On this tiny-token smoke, lowering Aurora preconditioning / Newton-Schulz cycles was a big throughput win with no visible memory change. That is optimizer-side overhead amortization, not a quality claim.
- SVD basis initialization is still the cold-start tax; on this 12GB card it OOMed at the same packed shape, so throughput-path runs should use random init unless the init cost is the point.
- Follow-up packed smoke at `seq_len=512`, `batch_size=2` (1024 tokens/step, random init, HF loss) sharpened the cycle story: default `2/5` cycles OOMed on this card, while reduced `1/3` cycles fit and ran at `5701 tok/s` with `5.27 GB` peak CUDA. Cycle count is therefore not just a micro-optimization at this shape; it can be the difference between fit and no fit.
- These throughput numbers were taken without `causal-conv1d`; the environment fell back to LFM2's Torch `Conv1d` short-conv path because no Triton/`causal-conv1d` package was installed.

## 2026-06-18: Qwen3.5-2B-Base + FLA pull

Question: does pulling Qwen3.5-2B-Base plus `flash-linear-attention` lift the local 12GB throughput ceiling?

Setup:

- Model: `Qwen/Qwen3.5-2B-Base`, now fully cached locally.
- Extra dependency: `flash-linear-attention` installed successfully; `causal-conv1d` build was attempted but failed because the environment lacks `nvcc` and the package does not cleanly build under the current Python/CUDA toolchain.
- Fast-path status: transformers still reports Qwen3.5 fast path unavailable because one required library is missing, so the model falls back to Torch for the short-conv path.

Results:

| shape | outcome | tokens/sec | peak CUDA |
| --- | --- | ---: | ---: |
| `batch=1, seq=128` | fit | 661 | 6.66 GB |
| `batch=2, seq=1024` | OOM in MLP | n/a | ~10.30 GB |
| `batch=4, seq=1024` | OOM | n/a | ~10.30 GB |

Notes:

- Qwen3.5 does load and train offline, but without `causal-conv1d` it does not unlock the intended fast path.
- The model switch alone does not buy the bs4×1k target on this 12GB card; it still blows up around the MLP/activation side before the optimizer matters.
- Net: `flash-linear-attention` is useful, but it is not the missing piece by itself.

## 2026-06-18: causal-conv1d build blocker

Question: can `causal-conv1d` be made uv-native on this Fedora box without hand-built hacks?

Findings:

- The project floor was dropped to Python `3.13`, and `uv sync --python 3.13` works.
- `uv add causal-conv1d --no-build-isolation-package causal-conv1d --no-binary-package causal-conv1d` still fails when the build uses the local CUDA 13.3 toolkit.
- The actual blocker is the host compiler: Fedora ships GCC `16.1`, while CUDA 13.3 rejects compilers newer than `15`.

Required next step:

- Install GCC 15 (`gcc15`, `gcc15-c++`) and point `CC`/`CXX` at it for the build shell, then retry the `uv add` command under `--python 3.13`.

## 2026-06-18: causal-conv1d built and Qwen fast path enabled

Question: did the Fedora box now satisfy the real build contract for `causal-conv1d`?

Findings:

- `.python-version` now pins the repo to Python `3.13`, so `uv run` / `uv sync` stop drifting back to system 3.14.
- Building `causal-conv1d` succeeded when the shell used `/usr/bin/gcc-15` and `/usr/bin/g++-15` explicitly together with CUDA 13.3 and Python 3.13.
- Verification signal: `transformers.utils.import_utils.is_causal_conv1d_available()` returns `True`, and `transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn` / `causal_conv1d_update` are both bound.

Meaning:

- Qwen3.5’s short-conv fast path is now live in this repo environment; the remaining work is throughput measurement, not kernel archaeology.

## 2026-06-18: Qwen3.5 throughput try

Question: does Qwen3.5-2B-Base with `causal-conv1d` and FLA actually clear the earlier bs4×1k ceiling?

Result:

- No. `batch=4, seq=1024` still OOMed during the Qwen3.5 linear-attention / MLP path.
- `batch=1, seq=1024` also OOMed, and activation checkpointing did not rescue it.

Meaning:

- Enabling the fast short-conv path is necessary plumbing, but it does not by itself make Qwen3.5 the 12GB bs4×1k vehicle. The remaining wall is still model activations / temporary tensors, not optimizer state.

## 2026-06-18: Qwen3.5 memory math check

Follow-up smoke runs with `basis_init=random` separated the loss path from the basis-init path:

- `batch=1, seq=1024, loss_impl=cce` fit at `7.28 GB` peak CUDA.
- `batch=1, seq=1024, loss_impl=hf` fit at `10.04 GB` peak CUDA.
- `batch=4, seq=1024, loss_impl=cce` OOMed in `causal_conv1d_fn` while allocating a `48.00 MiB` buffer (`empty_like(x)`), with only `31.88 MiB` free on the GPU.

Math:

- Qwen3.5-2B-Base has `2,274,069,824` parameters total.
- The text-side language model is `1,881,825,088` params; the non-embedding text-side trainable set is `1,373,265,728` params.
- One packed `bf16` activation at `batch=4, seq=1024, hidden=6144` is `4*1024*6144*2 = 50,331,648` bytes, i.e. the exact `48 MiB` buffer that failed.

Takeaway:

- The OOM is not a mystery leak. `batch=4, seq=1024` is already at the edge of the 11.59 GiB card once weights + grads + optimizer state + per-layer temporaries are resident, and the causal-conv output buffer is the shove that finally tips it over.

## 2026-06-21: Clean LFM 350M packed CCE throughput ceiling

Question: after cleaning the harness to one faithful route — EOS-packed no-mask batches plus CCE loss, no model fallback, no HF/chunked loss switch — what token step fits on the local 11.59 GiB 4070 SUPER with the default `LiquidAI/LFM2.5-350M-Base`?

Setup:

- Model/data: `LiquidAI/LFM2.5-350M-Base`, local SYNTH shards.
- Scope: `--param-scope broad-no-embeddings`, rank 64, residual-facing side.
- Batching/loss: EOS-packed fixed-length blocks, no `attention_mask`, CCE loss.
- Init: `--basis-init random`, because this was a throughput/fit check rather than an SVD cold-start quality claim.
- Shape: `seq_len=1024`, no grad accumulation, `--skip-validation`, one warmup + three measured steps.

Results:

| batch × seq | tokens/step | checkpointing | outcome | tokens/sec | step seconds | peak CUDA | state bytes |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4 × 1024 | 4096 | off | fit | 18,873 | 0.217 | 5.24 GB | 48.5 MB |
| 8 × 1024 | 8192 | off | fit | 18,859 | 0.434 | 9.70 GB | 48.5 MB |
| 9 × 1024 | 9216 | off | OOM in forward MLP | n/a | n/a | ~10.2 GB allocated at failure | n/a |
| 10 × 1024 | 10240 | off | OOM in forward MLP | n/a | n/a | ~10.2 GB allocated at failure | n/a |
| 16 × 1024 | 16384 | off/on | OOM in forward MLP | n/a | n/a | ~10.2 GB allocated at failure | n/a |

Notes:

- The clean no-mask CCE route works and reports `attn_implementation=sdpa`.
- The practical no-grad-accum ceiling at `seq_len=1024` is `batch_size=8` / `8192` tokens per optimizer step on the current card state.
- OOMs happen during the model forward MLP path before optimizer state matters. Activation checkpointing did not rescue `batch_size=16`, consistent with a forward temporary/activation wall rather than a backward-only saved-activation wall.
- SumoTrack state is ~48.5 MB total at this 350M broad-no-embeddings topology: ~48.0 MB matrix state and ~0.5 MB fallback state.
