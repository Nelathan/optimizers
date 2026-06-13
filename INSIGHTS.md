# SUMOTrack Insights

Brief durable findings for continuing SUMOTrack / `SubspaceMuon` without re-deriving the terrain.

## Current implementation

- `SubspaceMuon` exists as an eager PyTorch optimizer with projected first moments for 2D matrix params and local AdamW fallback for non-2D params.
- Matrix params do not store full-size first or second moments in the main path.
- Basis refresh supports exact SVD and a device-safe Grassmann/Stiefel tangent update with QR retraction.
- Projected moments are transported across basis changes by lifting through the old basis and projecting into the new basis. This uses a temporary full tensor but does not store a full moment.
- HeavyBall ECC/param-ECC is not integrated into `SubspaceMuon`; unsupported options fail loudly. HeavyBall itself can run bf16 ECC in this environment when its compile wrappers are disabled in-test.

## Evaluation stance

- Toy fresh-init runs are not the main signal. Subspace methods can be weak in the first few thousand steps from a fresh initialization.
- The target regime is continued/post-training on already-useful models, where constrained state and reduced forgetting may matter more than newborn convergence speed.
- Use real pretrained LLM smoke tests on local SYNTH shards before drawing optimizer conclusions.
- For performance timing, use random orthogonal subspace initialization and measure only post-warmup steps. Do not let first-step SVD dominate the reported number.
- For convergence-quality comparisons, use SVD initialization explicitly; initialization quality is part of the algorithm.

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

## Next cuts

1. Add random orthogonal projector initialization.
2. Wire `subspace_init="svd" | "random"` into `SubspaceMuon`.
3. Update `experiments/llm_synth_smoke.py` to train all non-embedding 2D matrices by default.
4. Add an explicit full-parameter mode for actual post-training behavior, with non-2D fallback accounted for.
5. Run short no-compile LFM/SYNTH rank sweeps at `64`, `128`, and `256`, measuring only post-warmup steps.
6. Separately run SVD-init convergence checks once the performance path is clean.
