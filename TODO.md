# SumoTrack TODO

This is the next-iteration queue for SumoTrack. Prefer small falsifiable cuts over heroic optimizer theatre.

## Gates before advancing

Progression is gated by evidence, not by vibes or a locally attractive diff. Each gate should be satisfied by committed code, tests, and when relevant a small script that emits the observable signal.

- [x] **Projector gate:** tall and wide projections have correct shapes, lift back to the original shape, clamp rank correctly, preserve dtype/device expectations, and maintain orthonormal bases.
- [x] **Scheduler gate:** refresh order is deterministic, budgeted, wraps explicitly, supports derived target intervals, and is proven not to skip ordinary per-step updates.
- [x] **Optimizer state gate:** `SumoTrack.step()` updates matrix and fallback params, matrix params store projected moments only, and optimizer state dict save/load round-trips without shape drift.
- [x] **Descent gate:** a no-download smoke script shows loss descent and reports optimizer state bytes, including a comparison that would catch accidental full-size matrix moments.
- [x] **HeavyBall/ECC gate:** bf16 params plus HeavyBall ECC/param-ECC either work in a smoke test or unsupported combinations fail loudly.
- [x] **Grassmann gate:** Grassmann basis updates preserve orthonormality, transport projected moments correctly, and are compared against SVD refresh on tiny loss and step-time signals.
- [x] **Performance gate:** optimization work is justified by measured step time, refresh spike size, state bytes, and a profiler/kernel-launch signal.

Projected-gradient hooks stay locked until the ordinary-gradient baseline clears the optimizer state, descent, and HeavyBall/ECC gates.

## Phase 0: repo/package setup

- [x] Add an editable HeavyBall path dependency in `pyproject.toml`.
- [x] Create the `sumotrack/` package.
- [x] Export `SumoTrack` from `sumotrack/__init__.py`.
- [x] Add a minimal smoke script under `experiments/`.
- [x] Add a tiny test harness that can run without downloading a large model.

## Phase 1: projector correctness

- [x] Implement `SubspaceProjector` for 2D tensors.
- [x] Support right-basis projection for `m >= n`.
- [x] Support left-basis projection for `m < n`.
- [x] Clamp rank to valid matrix dimensions.
- [x] Initialize basis with exact SVD.
- [x] Add random orthogonal initialization for performance/algo-path measurement.
- [x] Implement `project()` and `project_back()` shape tests.
- [x] Test basis orthonormality after initialization.
- [x] Test dtype/device preservation.
- [x] Remove all hard-coded CUDA device assumptions from borrowed SubTrack logic.

## Phase 2: round-robin refresh scheduler

- [x] Implement stable eligible-parameter ordering.
- [x] Implement `subspace_refresh_budget` scheduling.
- [x] Implement derived `target_refresh_interval` scheduling.
- [x] Make wrapping behavior explicit and tested.
- [x] Record which parameter IDs refresh on each step for diagnostics.
- [x] Verify only rotations refresh round-robin; ordinary updates still run for all parameters.

## Phase 3: minimal optimizer

- [x] Implement first eager `SumoTrack` optimizer using normal PyTorch gradients.
- [x] Matrix path: projected first moment only, no full-size first moment.
- [x] Fallback path: update non-2D params with local AdamW baseline.
- [x] Add config for `rank=32`, `beta`, `recovery_scale`, `orthogonalization`, and refresh scheduling.
- [x] Implement exact-SVD orthogonalization inside projected space for correctness mode.
- [x] Wire HeavyBall Newton-Schulz/polar orthogonalization for eager experiments.
- [x] Integrate HeavyBall Newton-Schulz orthogonalization without disabling compile.
- [x] Add full-matrix Muon scale mode for projected SUMO updates.
- [ ] Add optional Nesterov projected momentum.
- [x] Add optional perpendicular gradient recovery.
- [x] Verify state dict save/load round trip.

## Phase 4: HeavyBall-native integration

- [ ] Convert the matrix path into a HeavyBall-compatible chainable transform where practical.
- [ ] Preserve compatibility with HeavyBall `ecc="bf16+8"`.
- [ ] Preserve compatibility with HeavyBall `param_ecc="bf16+8"`.
- [x] Verify warmup/clipping/caution/MARS behavior or explicitly disable unsupported combinations.
- [ ] Replace local toy AdamW fallback with HeavyBall-backed AdamW/LaProp-style fallback where practical.
- [ ] Decide whether to use `SplitOpt` or local grouped delegation for fallback params.
- [ ] Report matrix projected state bytes, fallback state bytes, and total optimizer state bytes separately.
- [ ] Add mixed matrix+fallback state-dict save/load/resume smoke.
- [x] Add tests for bf16 parameters with ECC enabled.

## Phase 5: Grassmann tracking

- [x] Implement device-safe Grassmann/Stiefel tangent update in `sumotrack/projector.py`.
- [x] Replace hard SVD refresh with `subspace_update_method="grassmann"` after initialization.
- [x] Implement projected moment transport across basis updates.
- [x] Test moment transport against explicit old-basis/new-basis projection formulas.
- [ ] Add accumulated-gradient tracking option.
- [x] Compare SVD refresh vs Grassmann update on tiny regression loss and step time.

## Phase 6: performance work

- [x] Count profiler CUDA events for a representative optimizer step.
- [x] Profile exact SVD orthogonalization vs compiled HeavyBall Newton-Schulz on LFM/SYNTH smoke.
- [ ] Bucket same-shape projected moments for batched NS.
- [ ] Investigate bucketing refreshed modules by shape for batched `eigh`/SVD if refresh spikes matter.
- [ ] Measure whether round-robin refresh makes batched decompositions unnecessary.
- [x] Add diagnostics for step time, refresh spike ratio, state bytes, peak CUDA memory, and profiler CUDA events.
- [ ] Avoid optimizing decomposition refresh before the hot per-step projection/ortho path is understood.

## Phase 7: experiments

- [x] Tiny linear regression sanity check.
- [x] Cached pretrained-LLM SYNTH smoke test.
- [x] Update LLM SYNTH smoke to train all non-embedding 2D matrices by default.
- [ ] Add full/broad post-training mode that trains matrix params plus explicit fallback params.
- [ ] Add LLM harness accounting for embeddings/lm-head, non-2D fallback tensors, tiny 3D kernels, and frozen tensors.
- [x] Exclude or separately account for tiny 3D conv/linear-attention kernels.
- [x] Add warmup/measurement split for no-compile smoke runs.
- [x] Add opt-in grad/param/update norm logging to LLM SYNTH smoke.
- [ ] Tiny MLP classification sanity check.
- [ ] Tiny transformer language-modeling smoke test.
- [x] Compare against torch AdamW as an infeasible quality anchor on LFM matrix scope.
- [ ] Compare against HeavyBall AdamW, Muon, PSGDLRA, and possibly SOAP if memory allows.
- [ ] Measure peak VRAM, optimizer state bytes, tokens/sec, and loss curves.
- [ ] Test rank `16`, `32`, `64`.
- [x] Test rank `64`, `128`, `256` on cached pretrained-LLM SYNTH/post-training smoke.
- [ ] Test refresh budget `1`, `2`, `4`.
- [ ] Test recovery scale `0`, `0.1`, `0.25`, `0.5`, `1.0`.
- [x] Test orthogonalization `svd` vs `none` for SUMO-vs-SubTrack ablation.
- [x] Test compiled HeavyBall `newtonschulz` orthogonalization.
- [ ] Test whether any square projected spaces make `thinky_polar_express` relevant; rectangular projected updates route through Newton-Schulz.
- [ ] Run full/broad LFM/SYNTH smoke with HeavyBall NS, `orthogonalization_scale_mode="muon"`, state bytes by category, peak CUDA, post-compile step time, and matrix update norms.
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

- [x] Public optimizer class is `SumoTrack`; no old optimizer-name alias is exported.
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
