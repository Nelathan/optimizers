# AGENTS.md

## Repo purpose

This repo is the SumoTrack lab: a place to design and test a memory-efficient optimizer for high-capacity continued pretraining on consumer GPUs.

SumoTrack should help a user move a pretrained model across a real distribution shift — clean reasoning data, books, stories, domain corpora — without needing full AdamW state, slow gradient accumulation, or adapter-only capacity limits.

The near-term target is Gemma 4 12B-class training on a single RTX 5090-class machine with high tokens per step, good throughput, precision, and minimal gradient accumulation. The optimizer must serve that product reality, not win tiny harness games.

## Current design thesis

Public optimizer name: **SumoTrack**.

Mainline algorithm:

- one-sided SubTrack/GaLore-style gradient subspace tracking,
- projected first moments for 2D matrices,
- SUMO/Muon-style orthogonalization of the projected moment,
- Aurora-style leverage-uniform rectangular orthogonalization as the default projected direction map,
- HeavyBall Newton-Schulz as the internal polar primitive inside Aurora,
- full-matrix Muon scale semantics via `orthogonalization_scale_mode="muon"`,
- Grassmann/Stiefel basis tracking as adaptation smoothing.

Interpret the current one-sided rectangular update as intentional:

> Adapt along selected representation / hidden-state directions, while distributing the update across the larger side with Muon/SUMO geometry.

This is especially relevant for transformer MLP and attention matrices. Do not casually flatten this into “low-rank optimizer” or “Muon but smaller.” The hidden-state-facing axis is part of the product bet.

## What matters now

The next value is algorithm design, not smoke-test accumulation.

High-leverage questions:

- Module-role projection side beats shape `AUTO` at identical state; make it the harness default.
- Size-rank under budget saved ~10 MB with a marginal quality delta vs uniform; uniform is simpler and close enough.
- How should the current rank policy evolve toward spectrum/calibration-aware allocation?
- Does burst Grassmann smoothing provide useful retention behavior; should refresh interval vary by layer role?
- How much does Aurora's per-step overhead actually cost at product-scale token counts (32k+)?

Low-leverage traps:

- repeating broad topology smokes after accounting already works,
- polishing LR brackets before the geometry is right,
- treating AdamW as the opponent rather than an obvious anchor,
- adding HeavyBall-native/ECC plumbing before the algorithm earns it,
- implementing projected-gradient hooks before the ordinary-gradient design is proven,
- expanding harness ceremony without a sharper algorithm question.

## Competitors and anchors

Serious reference points include LoRA/DoRA/QLoRA/Unsloth-style adapter training, GaLore, SubTrack, SUMO/Muon-family methods, and other low-state full-finetuning approaches.

AdamW is a quality and sanity anchor. It is not the product target and not the memory story.

Do not spend a session proving SumoTrack is smaller than AdamW. Everyone in the room knows that. Spend the session finding whether SumoTrack's geometry buys adaptation capacity per byte/second.

## Evaluation stance

SYNTH is valuable because it is clean and low-noise. Use it to test adaptation signal before moving to harder short-story data that breaks models.

A useful evaluation reports:

- target validation movement,
- source/retention movement when available,
- tokens per step and whether grad accumulation was needed,
- measured step time / tokens/sec,
- peak CUDA memory,
- matrix/fallback/total optimizer state bytes,
- update/param diagnostics,
- basis motion or residual diagnostics when testing subspace tracking.

Do not overvalue a 20-step run. A short run is allowed only when it falsifies a sharp assumption or validates a sensor needed for the next design cut.

## Implementation priorities

Keep the eager implementation as the algorithm rail for now. HeavyBall-native integration and ECC are important later, not the next place to hide uncertainty.

Current invariants:

- Matrix params do not store full-size first moments.
- Matrix params do not store full-size second moments in the main path.
- Non-2D params use boring fallback semantics or are frozen by task policy.
- Fallback state is accounted separately.
- Orthogonalization happens in projected space.
- Aurora with Muon scale semantics is the only forward projected direction after the 1k target result.
- Grassmann tracking is the only forward basis-update path after initialization; tune smoothing with refresh cadence and step size.
- Two-sided square-core projection was removed from the active code path after weaker target movement and no clear product pull.
- Burst refresh (all bases on interval steps) is the Grassmann schedule; round-robin was measured and removed as complexity rent.
- Module-role projection side and uniform rank allocation are the current best policy defaults; architecture-aware side/rank lives in the harness.
- Unsupported ECC/param-ECC fails loudly.
- Exact SVD remains a correctness rail and possible initialization choice, not the steady-state performance path.

Near-term implementation cuts should be one of:

1. default module-role side + uniform rank in the harness,
2. basis movement / residual diagnostics for tuning Grassmann smoothing,
3. measure realistic-token-step amortization of bucketed Aurora (32k+ tokens/step),
4. use retention/source validation output for a real medium adaptation run.

Do not add speculative abstractions. Every new knob should correspond to a named geometry question.

## HeavyBall and adjacent repos

Build on `../HeavyBall` directly. Use it as the primary optimizer substrate for compiled orthogonalization, future ECC/param-ECC, chainable machinery, clipping, MARS, caution, and API compatibility.

Do not fork HeavyBall unless explicitly asked. Do not vendor HeavyBall internals. Do not add a submodule casually.

Aurora is relevant because very rectangular projected moments may not be well served by ordinary NS. Use it as a projected direction map inside SumoTrack, not as a wholesale optimizer replacement: SumoTrack owns momentum, basis tracking, scaling, LR, fallback semantics, and accounting.

## Product discipline

The product is usable distribution adaptation under memory pressure. Keep asking:

- Does this let us train more tokens per step?
- Does this preserve enough source behavior while moving target loss?
- Does this spend state on the tensors that matter?
- Does this improve the optimizer geometry or merely make the harness prettier?
- Would this help Gemma 12B on a 5090, or only make LFM smoke tables look tidy?

If the work does not sharpen one of those answers, it is probably displacement activity wearing a lab coat.

## Validation expectations

Use the repo's `uv` environment:

```bash
uv run python -m unittest discover -s tests
uv run python experiments/<script>.py
```

Useful checks include projector shape/orthonormality, state-dict restart, bf16 behavior, optimizer state accounting, loss curves, peak VRAM, step time, tokens/sec, and retention curves.

Do not report optimizer work as done because code imports or a smoke run descends. Optimizers fail silently and convincingly.

If validation cannot be run, say exactly what remains unverified.

## Reporting

Prefer signal over ceremony.

A useful report says what changed in system meaning: which geometric assumption became stronger or weaker, which bottleneck became visible, which product constraint bit, and what should be cut next.

Avoid reports that merely list files, commands, and green tests unless those details carry decision value.

Do not silently work around the user's framing. If the user says the product wants no gradient accumulation, do not optimize an experiment around grad accumulation and call it equivalent. If the requested framing seems wrong, stop and argue the point explicitly.

## Naming

Use **SumoTrack** for the optimizer family and public optimizer class. Avoid churny renaming once code exists.

## Style

Favor direct, readable PyTorch first. Once the math is right, move hot paths toward HeavyBall-style transforms and compiled utilities.

Keep comments for non-obvious math, state-shape invariants, and performance traps. Do not narrate obvious tensor operations.

Maintainability is a feature, but product-shaped algorithm clarity outranks tidy scaffolding.
