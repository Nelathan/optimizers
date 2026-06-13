# SUMOTrack Insights

Brief durable findings for continuing SUMOTrack / `SubspaceMuon` without re-deriving the terrain.

## Current implementation

- `SubspaceMuon` exists as an eager PyTorch optimizer with projected first moments for 2D matrix params and local AdamW fallback for non-2D params.
- Matrix params do not store full-size first or second moments in the main path.
- Basis initialization supports exact SVD by default and random QR as an opt-out for ablation/performance-path measurement.
- Basis refresh supports exact SVD and a device-safe Grassmann/Stiefel tangent update with QR retraction.
- Projected moments are transported across basis changes by lifting through the old basis and projecting into the new basis. This uses a temporary full tensor but does not store a full moment.
- Projected update modes now include `orthogonalization="svd"`, `"none"`, and a compiled HeavyBall-backed Newton-Schulz path. The local Fedora machine needed `python3-devel` so Torch/Inductor could find `/usr/include/python3.14/Python.h`; after installing it, HeavyBall compiled orthogonalization runs on CUDA.
- `orthogonalization_scale_mode="muon"` means SUMO orthogonalizes in projected space but scales as HeavyBall Muon would have scaled the original full matrix: `sqrt(max(1, rows / cols))`. This is distinct from `"scale"`, which applies HeavyBall's rule to the projected tensor and can accidentally introduce a `sqrt(full_dim / rank)` multiplier.
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
- Longer norm-logged LFM/SYNTH runs keep the SUMO signal alive but sharpen the caveat: no-ortho projected momentum was under-tuned at low LR and improves up to about LR `0.04`; best tested SUMO still won (`1.593341` final val vs no-ortho `1.612252`) over 40 measured steps at 1536 tokens/update.
- Norm telemetry suggests SUMO is not merely winning by taking a bigger raw projected-momentum step. No-ortho at LR `0.08` degraded while its mean update norm stayed below SUMO LR `0.0025`, so the geometry change remains the interesting signal.
- HeavyBall Newton-Schulz with full-Muon scaling now tracks exact SVD orthogonalization closely on the LFM/SYNTH rank-64 setup while being faster in the measured loop (`~0.48s` vs `~0.56s` per step). This makes HeavyBall NS the practical hot path; exact SVD remains the correctness rail.
- Torch AdamW on the same matrix-only scope reached slightly better short-run validation loss but used about 4.14 GB optimizer state and 9.1 GB peak CUDA. Treat AdamW as a quality anchor, not a feasible target budget.
- HeavyBall AdamW OOMed on the same matrix-only scope during a one-step plumbing check. HeavyBall integration remains important for fallback/ECC machinery, but full AdamW state is outside the target regime.

## Next cuts

1. Move from matrix-only experiments to a full/broad fine-tuning path. The next decisive question is whether SUMOTrack survives real model parameter topology while keeping the matrix-state memory promise.
2. Implement HeavyBall-backed fallback semantics for non-2D/full-parameter mode, with ECC/param-ECC compatibility and no state quantization unless explicitly chosen.
3. Add explicit LLM harness parameter scopes that account separately for matrix params, fallback params, embeddings/lm-head, tiny 3D kernels, and frozen params.
4. Add optimizer state byte accounting by category: matrix projected state, fallback state, total state.
5. Add a mixed-path save/load/resume smoke: matrix projected moments and bases reload with the same shapes, fallback state reloads, and the next step changes parameters.
6. Run a full/broad LFM/SYNTH smoke with HeavyBall Newton-Schulz + `orthogonalization_scale_mode="muon"`, reporting loss, state bytes by category, peak CUDA, step time after compile warmup, and matrix update norms.
7. Only after the full/broad path is real, run longer LR-aware quality comparisons. Longer matrix-only runs are useful but no longer the highest-leverage proof.
