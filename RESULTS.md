# SUMOTrack Results

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
- Optimizer path: `SubspaceMuon`, Grassmann refresh, no torch compile.

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
| SubspaceMuon, SVD ortho | 0.001 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.800248 | 2.192415 | 1.981366 |
| SubspaceMuon, SVD ortho | 0.0025 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.701340 | 2.046949 | 1.976006 |
| SubspaceMuon, SVD ortho | 0.01 | 88,866,816 | 5,035,031,040 | 2.345853 | 2.625968 | 2.397669 | 1.980586 |
| Projected momentum, no ortho | 0.0025 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.975293 | 2.309294 | 1.865864 |
| Projected momentum, no ortho | 0.01 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.815476 | 2.182461 | 1.866572 |
| torch AdamW quality anchor | 0.00002 | 4,143,972,720 | 9,097,523,712 | 2.345853 | 1.655455 | 1.962064 | 0.200987 |

Notes:

- The SUMO-style orthogonalized projected moment beat the no-orthogonalization projected-momentum ablation at the tested LRs.
- No-ortho improved when moved from `0.0025` to `0.01`, but still did not catch orthogonalized `0.0025`.
- Orthogonalized `0.01` overshot badly, so LR tuning matters; HeavyBall's `0.0025` default landed near the useful region for this setup.
- AdamW remains a quality anchor, not the target budget. Torch fused AdamW fit this matrix-only scope but used about 4.14 GB of optimizer state and 9.1 GB peak CUDA before embeddings/full fallback params. HeavyBall AdamW OOMed on the same scope during a one-step plumbing check.
- Direct eager HeavyBall orthogonalization is wired, but local compiled HeavyBall paths are blocked by missing `Python.h`. With compile disabled, the first-step SVD initialization dominates tiny one-step checks. Fast HeavyBall polar/NS should be treated as an integration task, not judged from uncompiled one-step smoke timing.
