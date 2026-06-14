# SumoTrack Insights

Brief durable findings for continuing SumoTrack without re-deriving the terrain.

## Current implementation

- `SumoTrack` exists as an eager PyTorch optimizer with projected first moments for 2D matrix params and HeavyBall-backed fused AdamW fallback for non-2D params.
- Matrix params do not store full-size first or second moments in the main path.
- Basis initialization supports exact SVD by default and random QR as an opt-out for ablation/performance-path measurement.
- Basis refresh supports exact SVD and a device-safe Grassmann/Stiefel tangent update with QR retraction.
- Projected moments are transported across basis changes by lifting through the old basis and projecting into the new basis. This uses a temporary full tensor but does not store a full moment.
- Projected update modes now include `orthogonalization="svd"`, `"none"`, compiled HeavyBall-backed Newton-Schulz, and `"aurora"`. Aurora is implemented only as a projected direction map: SumoTrack still owns projected EMA, basis tracking, LR, full-matrix Muon scaling, fallback semantics, and accounting. The local Fedora machine needed `python3-devel` so Torch/Inductor could find `/usr/include/python3.14/Python.h`; after installing it, HeavyBall compiled orthogonalization runs on CUDA.
- `orthogonalization_scale_mode="muon"` means SUMO orthogonalizes in projected space but scales as HeavyBall Muon would have scaled the original full matrix: `sqrt(max(1, rows / cols))`. This is distinct from `"scale"`, which applies HeavyBall's rule to the projected tensor and can accidentally introduce a `sqrt(full_dim / rank)` multiplier.
- `projection_mode="two_sided"` exists as an experimental branch with left/right bases and square `[rank, rank]` projected moments. It supports random/SVD initialization and fixed bases or explicit SVD refresh. It deliberately rejects Grassmann tracking for now because a coupled left/right Grassmann update has not been designed; silently substituting SVD refresh would corrupt the smoothing question.
- HeavyBall ECC/param-ECC is not integrated into `SumoTrack`; unsupported options fail loudly. The fallback path uses HeavyBall's `fused_adam_` math and fp32 moments for bf16 params, but ECC still requires HeavyBall `ChainOpt` state hooks. HeavyBall itself can run bf16 ECC in this environment when its compile wrappers are disabled in-test.

## Evaluation stance

- Toy fresh-init runs are not the main signal. Subspace methods can be weak in the first few thousand steps from a fresh initialization.
- The target regime is continued/post-training on already-useful models, where constrained state and reduced forgetting may matter more than newborn convergence speed.
- Use real pretrained LLM smoke tests on local SYNTH shards before drawing optimizer conclusions.
- For convergence-quality comparisons, use SVD initialization explicitly; initialization quality is part of the algorithm and may be worth the cold cost.
- Random initialization is for ablation and performance-path measurement, not the default quality story.
- Do not silently optimize experiments around excluding warmup/cold-start if the user says to ignore that cost. If the framing seems wrong, discuss it explicitly.
- Orthogonalized/Muon-style updates should be tuned around HeavyBall-scale LRs first, not AdamW's tiny LLM LR priors.

## Local terrain

- Local SYNTH data lives at `~/.cache/nanochat/base_data_synth` with shards `synth_001.parquet`, `synth_002.parquet`, `synth_003.parquet`, and `synth_500.parquet`.
- SYNTH rows contain `query`, `synthetic_reasoning`, and `synthetic_answer`, which can be concatenated directly for causal LM training text.
- `LiquidAI/LFM2.5-1.2B-Base` has been pulled and is the practical default pretrained model for short local tests.
- `Qwen/Qwen3-4B` is cached locally and complete, but larger.
- `Qwen/Qwen3.5-2B-Base` can be inspected through Hugging Face safetensor metadata but was not pulled during this session.

## Shape notes

- LFM 1.2B has small 3D convolution kernels: `model.layers.*.conv.conv.weight` with shape `[2048, 1, 3]`, 6,144 params each.
- Qwen3.5-2B has 18 small 3D linear-attention convolution kernels: `model.language_model.layers.*.linear_attn.conv1d.weight` with shape `[6144, 1, 4]`, 24,576 params each.
- These 3D kernels are tiny relative to 2D attention and MLP matrices. Ignore or separately account for them for now; the 2D matrices carry the state-memory story.
- Rank-64 one-sided projected moments are often extreme rectangles: LFM produces `8192×64` / `64×8192` and Qwen3-4B MLP reaches `9728×64` / `64×9728`. This makes large-axis leverage imbalance a real SumoTrack geometry question, not an abstract Aurora curiosity.

## Empirical signals so far

- LFM/SYNTH one-attention-layer smoke with rank 128 ran successfully offline on CUDA.
- On that narrow slice, SumoTrack state was about 3.4 MB versus AdamW about 41.9 MB for the same 10.5M trainable params.
- That one-layer state-size result is only a plumbing signal, not meaningful evidence for whole-model post-training.
- Grassmann refresh was faster than repeated exact SVD refresh in tiny local comparisons and is the preferred baseline path.
- LFM matrix-scope rank sweep with random init ran at ranks 64/128/256 over 92 non-embedding 2D tensors, about 1.036B trainable params. Optimizer state scaled linearly: about 88.9 MB, 177.7 MB, and 355.5 MB.
- SUMO orthogonalization has first positive evidence: rank-64 SVD-init orthogonalized `SumoTrack` at LR `0.0025` beat the no-orthogonalization projected-momentum ablation at tested LRs over 20 LFM/SYNTH optimizer steps.
- Longer norm-logged LFM/SYNTH runs keep the SUMO signal alive but sharpen the caveat: no-ortho projected momentum was under-tuned at low LR and improves up to about LR `0.04`; best tested SUMO still won (`1.593341` final val vs no-ortho `1.612252`) over 40 measured steps at 1536 tokens/update.
- Norm telemetry suggests SUMO is not merely winning by taking a bigger raw projected-momentum step. No-ortho at LR `0.08` degraded while its mean update norm stayed below SUMO LR `0.0025`, so the geometry change remains the interesting signal.
- HeavyBall Newton-Schulz with full-Muon scaling now tracks exact SVD orthogonalization closely on the LFM/SYNTH rank-64 setup while being faster in the measured loop (`~0.48s` vs `~0.56s` per step). This makes HeavyBall NS the practical hot path; exact SVD remains the correctness rail.
- Torch AdamW on the same matrix-only scope reached slightly better short-run validation loss but used about 4.14 GB optimizer state and 9.1 GB peak CUDA. Treat AdamW as a quality anchor, not a feasible target budget.
- HeavyBall AdamW OOMed on the same matrix-only scope during a one-step plumbing check. HeavyBall integration remains important for fallback/ECC machinery, but full AdamW state is outside the target regime.
- Broad no-embedding LFM/SYNTH plumbing smoke now runs with HeavyBall Newton-Schulz, rank-64 random init, and `orthogonalization_scale_mode="muon"`. One warmup + one measured 64-token step selected 1.036B trainable params across 146 tensors: 92 matrix tensors and 54 fallback tensors, with embeddings/lm-head frozen. State accounting reported ~88.9 MB matrix state, ~1.0 MB fallback state, ~89.9 MB total optimizer state, ~4.76 GB peak CUDA, and measured update/param ratio ~0.00179. This is topology/accounting evidence, not convergence evidence.
- Broad no-embedding LFM/SYNTH quality smoke preserved the memory story and kept the SUMO signal alive. With rank-64 random init, 1 warmup + 20 measured steps, 768 tokens/update, HeavyBall Newton-Schulz + `orthogonalization_scale_mode="muon"` at LR `0.0025` reached final val `1.734250` with ~89.9 MB optimizer state, ~5.07 GB peak CUDA, mean update/param `0.001181`, and ~0.324s measured steps. LR `0.005` stayed numerically stable but was worse (`1.793099`, mean update/param `0.002193`). No-ortho projected momentum did not inherit the matrix-only LR `0.04` prior in broad topology: it blew out to final val `7.085179` and mean update/param `0.010982`; a lower LR `0.005` was stable but still worse than orthogonalized LR `0.0025` (`1.811141`). Torch AdamW remains the quality anchor (`1.512150`) but used ~4.14 GB optimizer state and ~9.13 GB peak CUDA on the same broad-no-embeddings scope.
- Aurora projected orthogonalization does exactly what it is supposed to do mechanically: on synthetic extreme rectangular updates it collapses large-axis leverage CV from HeavyBall NS levels around `0.17` to about `0.003` with only a small direction-alignment cost. On a short fixed-basis broad LFM/SYNTH comparison, one-sided Aurora reduced measured projected leverage CV from `~1.28` to `~0.025` and max row-energy ratio from `~36x` to `~1.07x`, but was slightly worse on target loss over 10 measured steps (`1.751235` vs one-sided HeavyBall `1.735645`) and slower (`0.418s` vs `0.347s`). Treat this as “leverage uniformity is real but not automatically better immediate descent,” not as a final Aurora rejection.
- Two-sided square-core SumoTrack is stable and has nearly the same state budget as one-sided rank-64 (~90.6 MB total broad-no-embedding state vs ~89.9 MB). In the same 10-step fixed-basis broad LFM/SYNTH comparison, two-sided random init reached final val `1.801806`; two-sided SVD init reached `1.813795`. Both had smooth square-core leverage (`CV ~0.02`) and similar update/param, but underperformed one-sided rectangular HeavyBall. Immediate read: two-sided is a stronger bottleneck and should be kept as a stability/forgetting or rank-budget branch, not promoted over the hidden-state one-sided mainline from this evidence.
