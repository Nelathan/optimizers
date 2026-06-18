# AGENTS.md

This file is the operating contract for agents working in this repo. It is not the plan, lab notebook, or task ledger.

- Current direction and future leads live in `PLAN.md`.
- Durable empirical facts live in `INSIGHTS.md`.
- Brief experiment records live in `RESULTS.md`: why the run existed, what was tested, what moved, and what was observed.

## Repo purpose

This repo is the SumoTrack lab: a place to design and test a memory-efficient optimizer for high-capacity continued pretraining on consumer GPUs.

The product target is usable distribution adaptation under memory pressure: move a pretrained model across a real data shift without full AdamW state, slow gradient accumulation, or adapter-only capacity limits. The near-term hardware target is a single RTX 5090-class machine training Gemma 4 12B-class models with high tokens per step.

## Current optimizer invariants

- Public optimizer name: **SumoTrack**.
- Matrix params do not store full-size first moments.
- Matrix params do not store full-size second moments in the main path.
- Non-2D params use boring fallback semantics or are frozen by task policy.
- Fallback state is accounted separately.
- Orthogonalization happens in projected space.
- Aurora with Muon scale semantics is the forward projected direction.
- HeavyBall Newton-Schulz is the internal polar primitive inside Aurora.
- Grassmann tracking is the forward basis-update path after initialization.
- Burst refresh is the basis schedule; round-robin was removed as complexity rent.
- Two-sided square-core projection was removed from the active path.
- Unsupported ECC/param-ECC fails loudly.
- Exact SVD remains a correctness rail and possible initialization choice, not the steady-state performance path.

## Harness defaults and coordinate convention

The LLM harness defaults to broad no-embedding training, uniform rank 64, and residual-facing projection side. Override only the axis being tested; do not cargo-cult long CLI invocations that restate defaults.

Rank allocation is uniform. Do not reintroduce rank-allocation knobs without a fresh product-shaped reason and a direct comparison against uniform rank 64.

Optimizer side names are PyTorch weight-storage coordinates: `Linear.weight == [out_features, in_features]`.

- MLP up/gate and attention q/k/v map the residual/activation input axis to storage `right`.
- MLP down and attention output map the residual output/top axis to storage `left`.

The harness policy is named `residual-facing` so callers do not need to think in storage left/right.

## Product discipline

Prefer algorithmic signal over harness ceremony. Keep asking:

- Does this let us train more tokens per step?
- Does this preserve enough source behavior while moving target loss?
- Does this spend state on tensors that matter?
- Does this improve optimizer geometry, or merely make tables tidier?
- Would this help Gemma 12B on a 5090, or only make LFM smoke tables prettier?

Avoid these traps:

- repeating broad topology smokes after accounting already works,
- polishing LR brackets before geometry is right,
- treating AdamW as the opponent rather than a quality anchor,
- adding HeavyBall-native/ECC plumbing before the algorithm earns it,
- implementing projected-gradient hooks before ordinary-gradient SumoTrack is proven,
- expanding harness ceremony without a sharper algorithm question.

## HeavyBall and adjacent repos

Build on `../HeavyBall` directly. Use it as the primary optimizer substrate for compiled orthogonalization, future ECC/param-ECC, chainable machinery, clipping, MARS, caution, and API compatibility.

Do not fork HeavyBall unless explicitly asked. Do not vendor HeavyBall internals. Do not add a submodule casually.

Aurora is a projected direction map inside SumoTrack, not a wholesale optimizer replacement: SumoTrack owns momentum, basis tracking, scaling, LR, fallback semantics, and accounting.

## Validation expectations

Use the repo's `uv` environment:

```bash
uv run python -m unittest discover -s tests
uv run python experiments/<script>.py
```

Useful checks include projector shape/orthonormality, state-dict restart, bf16 behavior, optimizer state accounting, loss curves, peak VRAM, step time, tokens/sec, and retention curves.

Do not report optimizer work as done because code imports or a smoke run descends. Optimizers fail silently and convincingly. If validation cannot be run, say exactly what remains unverified.

## Reporting and style

Prefer signal over ceremony.

A useful report says what changed in system meaning: which geometric assumption became stronger or weaker, which bottleneck became visible, which product constraint bit, and what should be cut next.

Avoid reports that merely list files, commands, and green tests unless those details carry decision value.

Favor direct, readable PyTorch first. Once the math is right, move hot paths toward HeavyBall-style transforms and compiled utilities. Keep comments for non-obvious math, state-shape invariants, and performance traps. Do not narrate obvious tensor operations.

Maintainability is a feature, but product-shaped algorithm clarity outranks tidy scaffolding.
