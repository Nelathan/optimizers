# AGENTS.md

## Repo purpose

This repo is a lab for home-scale machine-learning optimizers. The current main thread is **SUMOTrack**, with public optimizer name **SubspaceMuon** unless a better name earns its keep.

SUMOTrack targets high effectiveness on mid-range consumer NVIDIA GPUs, especially memory-constrained cards such as the RTX 4070 Super. The aim is to make full fine-tuning or continued pretraining practical where users would otherwise retreat to LoRA for memory reasons.

## Current design direction

Build SUMOTrack directly on top of `../HeavyBall`. HeavyBall is a sibling repo and should be treated as the primary optimizer substrate. Prefer an editable path dependency over vendoring, forking, or submodules.

The intended synthesis:

- HeavyBall for chainable optimizer machinery, compiled update paths, ECC, param ECC, Muon/NS orthogonalization, clipping, MARS, caution, and API compatibility.
- FlashOptim mainly as conceptual background; HeavyBall already implements the important 24-bit ECC path through `ecc="bf16+8"` and `param_ecc="bf16+8"`.
- SubTrack for Grassmannian gradient subspace tracking and memory-efficient projected optimization.
- SUMO for moment orthogonalization inside a low-dimensional subspace.
- Aurora as a later possible improvement for rectangular/non-square orthogonalization, not a v1 dependency.

## Collaboration and reporting

Prefer signal over ceremony. A useful progress report should say what changed in the system's meaning: which invariant now holds, which bottleneck became visible, which assumption broke, and what the next high-leverage cut is. Avoid process-shaped summaries that merely list files, commits, and validations unless those details carry decision value.

Do not silently work around the user's stated framing. If the user says to ignore warmup/cold-start cost, do not optimize the report or experiment around excluding it as though that were the primary concern. If that framing seems wrong, stop and discuss the disagreement explicitly.

For SUMOTrack specifically, do not become overfocused on first-step SVD cost. Exact SVD initialization is allowed and often desirable because it may start stronger; random initialization exists to opt out when measuring the algorithmic/performance path. Torch compile latency can dominate first-step SVD cost in practice, so treat cold-start complaints proportionally.

## Ground rules

- Do not fork HeavyBall unless explicitly asked.
- Do not copy large chunks of HeavyBall into this repo.
- Do not add a git submodule casually.
- Do not add speculative abstractions before the optimizer works.
- Do not optimize kernel launches before there is a correct measurable baseline.
- Preserve user work and unrelated files.
- Prefer small, reversible changes with clear tests.
- Keep algorithmic assumptions explicit.

## Implementation priorities

The first useful implementation is not the fanciest one. Build in this order:

1. Correct projector math with exact SVD initialization.
2. Minimal eager optimizer using normal full gradients.
3. Round-robin subspace refresh for rotations only.
4. Projected first moment and projected-space orthogonalization.
5. HeavyBall ECC/param-ECC compatibility.
6. Grassmannian SubTrack update.
7. Newton-Schulz/polar fast path.
8. Bucketing/batching for performance.
9. Projected-gradient autograd hooks, only after the baseline is proven.

## Algorithm invariants

- Default subspace rank is `32`.
- Matrix parameters should not store full-size first moments.
- Matrix parameters should not store full-size second moments in the main path.
- Non-2D params should use a boring fallback optimizer path.
- Fallback optimizer behavior should be classic AdamW/FlashOptim-style, implemented through HeavyBall where possible, with no moment quantization unless explicitly chosen. ECC/state ECC and param ECC are important compatibility targets.
- Round-robin scheduling applies to subspace refresh/rotation, not to ordinary per-step optimization.
- Orthogonalization happens inside the projected subspace.
- Grassmannian tracking is expected to become the main update method after v1 correctness is established.
- Exact SVD initialization is the default because it may improve early adaptation. Random initialization is an opt-out for ablation and performance-path measurement.
- Fast orthogonalization should move up quickly: prefer HeavyBall's Newton-Schulz/polar machinery, and check whether HeavyBall is using a PolarExpress-style algorithm before inventing a local substitute.

## Performance notes

Past optimizer experiments hit a kernel-launch wall. Too many tiny decompositions or per-module kernels can dominate runtime. Batching by module shape and using batched `eigh`/SVD helped in previous implementations.

For SUMOTrack, round-robin refresh may reduce decomposition pressure because only rotations are staggered. However, projection and orthogonalization still happen every step, so watch kernel-launch count and module-count scaling.

Prefer matmul-shaped hot paths. HeavyBall's optimized Newton-Schulz/polar implementation is likely faster than decomposition-based orthogonalization because it is mostly matmuls.

Orthogonalization quality and optimizer learning rate are coupled. When swapping exact SVD orthogonalization for a fast approximate path, expect LR tuning to be part of the experiment rather than treating equal LR as a fair comparison.

## Future projected-gradient path

A later optimization may avoid materializing the full-rank gradient before projection. For linear layers, projected gradients can be computed from projected activations/backprops and stored directly. This can reduce memory pressure, but prior attempts required heavy PyTorch/autograd hacking, including returning `None` for normal gradients and storing projected gradients out-of-band.

Do not make this the baseline. If implemented, isolate it behind explicit opt-in hooks and prove equivalence against projecting the full gradient on tiny models.

## Validation expectations

Do not report optimizer work as done because the code imports. Optimizers fail silently and convincingly.

Use the repo's `uv` environment for Python commands. System Python may not have torch installed. Preferred forms:

```bash
uv run python -m unittest discover -s tests
uv run python experiments/<script>.py
```

If adding dependencies, update `pyproject.toml` and let `uv` update `uv.lock`; do not work around the environment by using global Python packages.

Useful evidence includes:

- projector shape and orthonormality tests,
- state dict save/load tests,
- bf16 + ECC smoke tests,
- loss descent on tiny regression / tiny transformer tasks,
- peak VRAM measurement,
- optimizer state byte accounting,
- step-time and refresh-spike measurement,
- comparison against HeavyBall AdamW, Muon, and PSGDLRA.

If validation cannot be run, say exactly what remains unverified.

## Naming

Use **SUMOTrack** for the project/optimizer family and **SubspaceMuon** for the first public optimizer class unless the user chooses otherwise. Avoid churny renaming once code exists.

## Style

Favor direct, readable PyTorch first. Once the math is right, move hot paths toward HeavyBall-style transforms and compiled utilities. Keep comments for non-obvious math, state-shape invariants, and performance traps. Do not narrate obvious tensor operations.
