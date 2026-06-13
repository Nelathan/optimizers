# SUMOTrack Insights

Brief durable findings for continuing SUMOTrack / `SubspaceMuon` without re-deriving the terrain.

## Current implementation

- `SubspaceMuon` exists as an eager PyTorch optimizer with projected first moments for 2D matrix params and local AdamW fallback for non-2D params.
- Matrix params do not store full-size first or second moments in the main path.
- Basis initialization supports exact SVD by default and random QR as an opt-out for ablation/performance-path measurement.
- Basis refresh supports exact SVD and a device-safe Grassmann/Stiefel tangent update with QR retraction.
- Projected moments are transported across basis changes by lifting through the old basis and projecting into the new basis. This uses a temporary full tensor but does not store a full moment.
- Projected update modes now include `orthogonalization="svd"`, `"none"`, and an eager HeavyBall-backed path. The HeavyBall path is wired for experiments but disables HeavyBall compile locally because this Python 3.14 environment lacks `Python.h` for Triton support.
- HeavyBall ECC/param-ECC is not integrated into `SubspaceMuon`; unsupported options fail loudly. HeavyBall itself can run bf16 ECC in this environment when its compile wrappers are disabled in-test.

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

## Empirical signals so far

- LFM/SYNTH one-attention-layer smoke with rank 128 ran successfully offline on CUDA.
- On that narrow slice, SubspaceMuon state was about 3.4 MB versus AdamW about 41.9 MB for the same 10.5M trainable params.
- That one-layer state-size result is only a plumbing signal, not meaningful evidence for whole-model post-training.
- Grassmann refresh was faster than repeated exact SVD refresh in tiny local comparisons and is the preferred baseline path.
- LFM matrix-scope rank sweep with random init ran at ranks 64/128/256 over 92 non-embedding 2D tensors, about 1.036B trainable params. Optimizer state scaled linearly: about 88.9 MB, 177.7 MB, and 355.5 MB.
- SUMO orthogonalization has first positive evidence: rank-64 SVD-init orthogonalized `SubspaceMuon` at LR `0.0025` beat the no-orthogonalization projected-momentum ablation at tested LRs over 20 LFM/SYNTH optimizer steps.
- Torch AdamW on the same matrix-only scope reached slightly better short-run validation loss but used about 4.14 GB optimizer state and 9.1 GB peak CUDA. Treat AdamW as a quality anchor, not a feasible target budget.
- HeavyBall AdamW OOMed on the same matrix-only scope during a one-step plumbing check. HeavyBall integration remains important for fallback/ECC machinery, but full AdamW state is outside the target regime.

## Next cuts

1. Run a longer SUMO-vs-projected-momentum comparison with more tokens/update and a small LR sweep around each method's stable region.
2. Integrate HeavyBall Newton-Schulz / PolarExpress orthogonalization without globally disabling compile, or provide a clean local eager fallback only for tests.
3. Add update-norm / param-norm logging so LR and orthogonalization quality are not compared blind.
4. Implement HeavyBall-backed fallback AdamW semantics for non-2D/full-parameter mode, with ECC/param-ECC compatibility and no state quantization unless explicitly chosen.
5. Add an explicit full-parameter post-training mode with embeddings and non-2D fallback accounted for.
6. Compare against additional feasible low-state baselines, not only full AdamW.
