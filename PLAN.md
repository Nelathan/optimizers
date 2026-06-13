# SUMOTrack Plan

SUMOTrack is an experimental optimizer line for memory-constrained full fine-tuning and continued pretraining on consumer GPUs. The public optimizer class should probably be named `SubspaceMuon`: clear, searchable, and honest. The project name `SUMOTrack` names the core synthesis: SUMO-style moment orthogonalization inside a SubTrack-style tracked gradient subspace, implemented on top of HeavyBall.

The target hardware is mid-range consumer NVIDIA cards, especially cards in the RTX 4070 Super / 12 GB class and adjacent 16-24 GB cards. The target use cases are:

- continued pretraining where full AdamW state is too expensive,
- replacing or injecting a freshly initialized attention mechanism while freezing the rest,
- full fine-tuning in situations where LoRA would normally be chosen only for memory reasons,
- small-batch, high-noise training where stable spectral geometry matters.

## Integration stance

Do not fork HeavyBall first. Do not vendor HeavyBall internals. Do not add HeavyBall as a git submodule unless a concrete packaging need appears.

Start this repo as the experimental companion package and depend on `../HeavyBall` as an editable path dependency. HeavyBall already provides the right runtime substrate: chainable transforms, compiled update paths, ECC, parameter ECC, Muon orthogonalization, clipping, MARS, cautioning, FSDP shape handling, and optimizer API compatibility.

The first implementation should live here, probably under:

```text
sumotrack/
  __init__.py
  optimizer.py              # SubspaceMuon public optimizer
  projector.py              # SubTrack/SUMO projector state
  rotation.py               # round-robin subspace refresh scheduler
  param_groups.py           # matrix/fallback grouping helpers
  diagnostics.py            # optional lightweight metrics
experiments/
  smoke_sumotrack.py
  compare_tiny_transformer.py
```

If the design stabilizes and its transform fits cleanly into HeavyBall's chainable model, upstreaming into HeavyBall can be considered later. Until then, local iteration speed matters more than library purity.

## What HeavyBall already gives us

HeavyBall already integrates the most important FlashOptim idea for this work: 24-bit-like error correction through `ecc="bf16+8"` and `param_ecc="bf16+8"`. Use that path first. Do not mix FlashOptim kernels into v1.

HeavyBall also already provides Muon-style orthogonalization through `heavyball.chainable.orthogonalize_update` and polar/Newton-Schulz utilities in `heavyball.utils`. SUMOTrack's novelty is not full-matrix Muon. The novelty is orthogonalization inside a tracked low-rank gradient subspace, with memory-shaped state.

HeavyBall's `PSGDLRA` is relevant but not equivalent. It is a low-rank PSGD preconditioner. SUMOTrack should track per-matrix gradient subspaces and maintain projected first moments, closer to SubTrack + SUMO.

## Core algorithm v1

Default rank: `32`.

For each eligible 2D parameter matrix `W` with gradient `G`:

1. Maintain an orthonormal subspace basis `Q`.
   - If `m >= n`, use a right basis `Q` with shape `[r, n]` and project as `G_hat = G @ Q.T`.
   - If `m < n`, use a left basis `Q` with shape `[m, r]` and project as `G_hat = Q.T @ G`.

2. Maintain the first moment in projected space:
   - `M_hat = beta * M_hat + G_hat`, or a Nesterov variant.
   - Do not maintain a full-size first moment for the matrix path.

3. Orthogonalize inside the projected space:
   - v1 can use exact SVD for correctness and debugging.
   - The performance path should use HeavyBall's optimized Newton-Schulz / polar matmul implementation because it avoids expensive decomposition kernels.

4. Project back:
   - right basis: `O = O_hat @ Q`
   - left basis: `O = Q @ O_hat`

5. Optionally recover some discarded gradient component:
   - `G_perp = G - project_back(G_hat)`
   - update direction can include `O + recovery_scale * G_perp`.
   - Start with `recovery_scale=0.0` or `0.25` as an ablation knob, not dogma.

6. Apply HeavyBall-compatible parameter update behavior:
   - learning rate and warmup,
   - decoupled weight decay where appropriate,
   - clipping/caution/MARS where compatible,
   - `ecc="bf16+8"` and `param_ecc="bf16+8"`.

Non-2D parameters should fall back to a boring HeavyBall optimizer path such as AdamW/LaProp. Use `SplitOpt` or helper-generated parameter groups rather than contorting the matrix path to handle embeddings, norms, and biases.

## Subspace rotation

Only subspace refresh/rotation should be round-robin. Ordinary projection, moment update, orthogonalization, and parameter update happen every step for every active matrix.

Use a stable parameter order and a refresh budget:

```text
eligible = all trainable matrix parameters above the minimum size
cursor = persistent integer
budget = subspace_refresh_budget
refresh eligible[cursor : cursor + budget] each optimizer step, wrapping around
```

Expose both direct and derived controls:

- `subspace_refresh_budget=1` as the concrete scheduling primitive,
- `target_refresh_interval=None` as sugar that derives a budget from the number of eligible matrices,
- `rank=32`,
- `subspace_update_method="grassmann" | "svd_refresh" | "random"`.

If the target interval is shorter than the number of eligible modules, overlapping refreshes are allowed by increasing the budget. Avoid ambiguous modulo semantics.

## Strong belief: Grassmann update

The Grassmannian update is expected to be central, not decorative. Full SVD refresh is acceptable as a baseline and for initialization, but a good SUMOTrack implementation should prefer SubTrack-style geometry-aware tracking once the shape/state machinery is correct.

The SubTrack update must be adapted carefully:

- no hard-coded CUDA device strings,
- correct dtype promotion and restoration,
- stable behavior for tiny or skinny matrices,
- optional accumulated-gradient tracking,
- projected moment transport when the basis changes.

## Performance risks

The main performance enemy is not just FLOPs; it is kernel-launch spray. Past experiments showed that bucketing similarly shaped modules and calling batched eigendecompositions improved speed. SUMOTrack should keep this in mind, but not prematurely optimize v1.

Round-robin refresh reduces decomposition pressure because only rotations are sparse in time. Projection and orthogonalization still run every step, so they should be matmul-shaped and batchable where practical.

Likely performance ladder:

1. Correct eager implementation.
2. Use HeavyBall chainable transforms enough to inherit ECC/update behavior.
3. Replace exact SVD orthogonalization with Newton-Schulz/polar for the hot path.
4. Bucket same-shape projected moments for batched NS or batched eig/SVD where useful.
5. Only then consider invasive autograd tricks.

## Future memory frontier: projected activations

A later optimization is to avoid materializing the full-rank gradient only to project it. Mathematically, for linear modules, projected gradients can be computed from projected activations / backprops directly. This could reduce peak memory pressure substantially.

This is not v1. Prior attempts required heavy PyTorch hacking: returning `None` as the normal gradient and storing a projected gradient side channel. That path is powerful but sharp. It should be isolated behind explicit hooks and tested against vanilla gradients for equivalence.

The safe sequence is:

1. Prove optimizer behavior using normal gradients.
2. Add optional projected-gradient capture for selected `nn.Linear` modules.
3. Verify exact/near-exact equivalence against full-gradient projection on tiny models.
4. Measure actual peak-memory reduction, not just theoretical allocation savings.

## First success criteria

SUMOTrack v1 is worth continuing if it can show:

- stable loss descent on tiny regression and tiny transformer tests,
- lower optimizer-state memory than AdamW for matrix-heavy models,
- credible convergence versus HeavyBall AdamW, Muon, and PSGDLRA,
- no accidental full-size matrix moments on the matrix path,
- sane behavior with bf16 params plus HeavyBall ECC,
- refresh spikes that are visible but tolerable.

Do not declare victory from a green unit test. The observable downstream signals are loss curves, peak memory, step time, state size, and restart correctness.
