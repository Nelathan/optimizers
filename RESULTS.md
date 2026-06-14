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
- Init: `subspace_init=svd`.
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
- Init: `subspace_init=svd`.
- Steps: 1 warmup + 40 measured optimizer steps.
- Tokens/update: 1536 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=8`).
- Validation texts: 16.
- Norm logging: enabled. `update_norm` is the actual `SumoTrack` update vector norm after LR scaling, excluding decoupled weight decay; weight decay was `0` for these runs. AdamW-style generic update norms are intentionally not computed because cloning/streaming 1B params would distort the memory/perf story.

Representative command shape:

```bash
HF_HUB_OFFLINE=1 uv run python experiments/llm_synth_smoke.py \
  --optimizers subspace --param-scope matrices-no-embeddings \
  --warmup-steps 1 --measure-steps 40 \
  --seq-len 192 --batch-size 1 --grad-accum-steps 8 --val-texts 16 \
  --rank 64 --subspace-init svd --orthogonalization <svd|none> \
  --subspace-lr <lr> --log-norms
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
- Init: `subspace_init=svd`.
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
- Rank/init: rank 64, `--subspace-init random`.
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
