# SUMOTrack TODO

This is the next-iteration queue for SUMOTrack / `SubspaceMuon`. Prefer small falsifiable cuts over heroic optimizer theatre.

## Phase 0: repo/package setup

- [ ] Add an editable HeavyBall path dependency in `pyproject.toml`.
- [ ] Create the `sumotrack/` package.
- [ ] Export `SubspaceMuon` from `sumotrack/__init__.py`.
- [ ] Add a minimal smoke script under `experiments/`.
- [ ] Add a tiny test harness that can run without downloading a large model.

## Phase 1: projector correctness

- [ ] Implement `SubspaceProjector` for 2D tensors.
- [ ] Support right-basis projection for `m >= n`.
- [ ] Support left-basis projection for `m < n`.
- [ ] Clamp rank to valid matrix dimensions.
- [ ] Initialize basis with exact SVD.
- [ ] Add random orthogonal initialization as an option for ablation.
- [ ] Implement `project()` and `project_back()` shape tests.
- [ ] Test basis orthonormality after initialization.
- [ ] Test dtype/device preservation.
- [ ] Remove all hard-coded CUDA device assumptions from borrowed SubTrack logic.

## Phase 2: round-robin refresh scheduler

- [ ] Implement stable eligible-parameter ordering.
- [ ] Implement `subspace_refresh_budget` scheduling.
- [ ] Implement derived `target_refresh_interval` scheduling.
- [ ] Make wrapping behavior explicit and tested.
- [ ] Record which parameter IDs refresh on each step for diagnostics.
- [ ] Verify only rotations refresh round-robin; ordinary updates still run for all parameters.

## Phase 3: minimal optimizer

- [ ] Implement first eager `SubspaceMuon` optimizer using normal PyTorch gradients.
- [ ] Matrix path: projected first moment only, no full-size first moment.
- [ ] Fallback path: delegate non-2D params to HeavyBall AdamW or LaProp.
- [ ] Add config for `rank=32`, `beta`, `recovery_scale`, `orthogonalization`, and refresh scheduling.
- [ ] Implement exact-SVD orthogonalization inside projected space for correctness mode.
- [ ] Implement HeavyBall Newton-Schulz/polar orthogonalization for speed mode.
- [ ] Add optional Nesterov projected momentum.
- [ ] Add optional perpendicular gradient recovery.
- [ ] Verify state dict save/load round trip.

## Phase 4: HeavyBall-native integration

- [ ] Convert the matrix path into a HeavyBall-compatible chainable transform where practical.
- [ ] Preserve compatibility with HeavyBall `ecc="bf16+8"`.
- [ ] Preserve compatibility with HeavyBall `param_ecc="bf16+8"`.
- [ ] Verify warmup/clipping/caution/MARS behavior or explicitly disable unsupported combinations.
- [ ] Decide whether to use `SplitOpt` or local grouped delegation for fallback params.
- [ ] Add tests for bf16 parameters with ECC enabled.

## Phase 5: Grassmann tracking

- [ ] Port SubTrack's Grassmannian update into `sumotrack/projector.py`.
- [ ] Replace hard SVD refresh with `subspace_update_method="grassmann"` after initialization.
- [ ] Implement projected moment transport across basis updates.
- [ ] Test moment transport against explicit old-basis/new-basis projection formulas.
- [ ] Add accumulated-gradient tracking option.
- [ ] Compare SVD refresh vs Grassmann update on tiny transformer loss and step time.

## Phase 6: performance work

- [ ] Count kernel launches for a representative optimizer step.
- [ ] Profile exact SVD orthogonalization vs Newton-Schulz/polar.
- [ ] Bucket same-shape projected moments for batched NS.
- [ ] Investigate bucketing refreshed modules by shape for batched `eigh`/SVD if refresh spikes matter.
- [ ] Measure whether round-robin refresh makes batched decompositions unnecessary.
- [ ] Add diagnostics for refresh time, projection time, orthogonalization time, and update time.
- [ ] Avoid optimizing decomposition refresh before the hot per-step projection/ortho path is understood.

## Phase 7: experiments

- [ ] Tiny linear regression sanity check.
- [ ] Tiny MLP classification sanity check.
- [ ] Tiny transformer language-modeling smoke test.
- [ ] Compare against HeavyBall AdamW, Muon, PSGDLRA, and possibly SOAP if memory allows.
- [ ] Measure peak VRAM, optimizer state bytes, tokens/sec, and loss curves.
- [ ] Test rank `16`, `32`, `64`.
- [ ] Test refresh budget `1`, `2`, `4`.
- [ ] Test recovery scale `0`, `0.1`, `0.25`, `0.5`, `1.0`.
- [ ] Test orthogonalization `svd` vs `newtonschulz`.
- [ ] Test freshly initialized attention module with rest frozen.

## Phase 8: projected-gradient hooks, later and dangerous

- [ ] Design an opt-in hook API for selected `nn.Linear` modules.
- [ ] Compute projected gradients from saved projected activations/backprops.
- [ ] Avoid materializing full-rank gradients where possible.
- [ ] Verify projected-hook gradients against `project(full_gradient)` on tiny models.
- [ ] Ensure returned `None` gradients do not break distributed, AMP, checkpointing, or trainer assumptions.
- [ ] Measure actual peak-memory reduction.
- [ ] Keep this path isolated from the baseline optimizer.

## Open questions

- [ ] Should the first public class be `SubspaceMuon`, `SUMOTrack`, or both aliasing the same implementation?
- [ ] Is perpendicular recovery beneficial in the target fine-tuning regime, or does it reintroduce too much noise?
- [ ] Does rank 32 need faster rotation than higher-rank SubTrack defaults?
- [ ] Does Grassmann tracking beat periodic SVD refresh at the same wall-clock budget?
- [ ] Is Aurora-style rectangular balancing useful inside projected spaces, or only for full-matrix Muon?
- [ ] Which fallback optimizer is best for embeddings/norms/biases: AdamW, LaProp, or HeavyBall's existing split MuonAdamW pattern?

## Red flags

- [ ] Any full-size first moment on matrix params.
- [ ] Any full-size second moment on matrix params unless deliberately added for an ablation.
- [ ] Silent dtype degradation when bf16 params are used.
- [ ] Hard-coded `.cuda()` calls.
- [ ] Kernel-launch count exploding with module count.
- [ ] Loss improves in a toy test while memory is worse than AdamW.
- [ ] Passing tests without state dict resume checks.
