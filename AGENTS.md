# AGENTS.md

This file is the operating contract for agents working in this repo. It is not the plan, lab notebook, or task ledger.

- Current direction and future leads live in `PLAN.md`.
- Durable empirical facts live in `INSIGHTS.md`.
- Brief experiment records live in `RESULTS.md`: why the run existed, what was tested, what moved, and what was observed.

## Repo purpose

This repo is the SumoTrack lab: a place to design and test a memory-efficient optimizer for high-capacity continued pretraining on consumer GPUs.

The product target is usable distribution adaptation under memory pressure: move a pretrained model across a real data shift without full AdamW state, slow gradient accumulation, or adapter-only capacity limits. The near-term hardware target is a single RTX 5090-class machine training Gemma 4 12B-class models with high tokens per step.

## Working posture

Act as a partner, not an autopilot. The job is not to complete the requested command at all costs; the job is to preserve the question we are trying to answer. If the route stops answering that question, stop and say so before spending more compute or writing more code.

A benchmark is only meaningful when the model, data, loss path, batch shape, attention path, optimizer scope, and measurement target match the intended claim. If any of those drift — a named model is unavailable, a cached substitute appears convenient, HF full-logits loss replaces CCE, padded batches create an `attention_mask`, an ablation sneaks into a benchmark, or a smoke test is being mistaken for product evidence — stop. State the mismatch and the consequence for the claim. Do not paper over it with warnings and continue producing benchmark-shaped garbage.

Input formatting is part of the benchmark contract. Before interpreting optimizer curves, verify the exact text stream the model sees: row-to-text conversion, BOS/EOS behavior, packing boundaries, separators between examples, chat/template absence, and decoded packed blocks. When training on SYNTH and measuring source behavior, say the streams precisely: target train is SYNTH train, target eval is held-out SYNTH, source eval is the external pretraining-style corpus. Do not use reporting language that implies the source corpus was the training data.

Do not overcorrect by building option gardens. First make the contract explicit in plain language. If the next move changes defaults, benchmark meaning, model choice, or user-facing workflow, propose the clean design cut and wait for review. Implement only after alignment. Prefer one clean path with explicit stop conditions over many escape hatches.

Every line must earn its place. Do not add flags, branches, adapters, warnings, compatibility layers, or “just in case” scaffolding to preserve momentum. When the system shape is wrong, prefer deletion, simplification, or a sharper boundary over shotgun accommodation. Code that makes invalid states easy to run is worse than missing code.

Keep talking when alignment matters. Short progress notes should expose changed assumptions, invalidated routes, and decisions needed from the user. Silence during a strategic mismatch is a failure mode, even if the code keeps running.

Stop at crossroads. If the exact model/cache/path is unclear, if pulling a model is the real choice, if the next probe changes scale, or if there are two plausible experiment contracts, ask before wandering. A five-minute download is cheaper than a thirty-minute detour through clever uncertainty.

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
- Basis initialization uses stable side-Gram `eigh`: fp32 finite check, Frobenius normalization, symmetric Gram, and trace-scaled jitter retry on backend failure. Exact SVD was removed from the default/init path after being slower and less robust for this one-sided subspace contract.

## Harness defaults and coordinate convention

The LLM harness defaults to broad no-embedding training, uniform rank 64, stable `eigh` basis init, residual-facing projection side, faithful SYNTH right-padded no-mask batches, `batch_size=4`, `seq_len=1024`, and CCE loss. Override only the axis being tested; do not cargo-cult long CLI invocations that restate defaults. EOS-packed no-mask remains available, but it is now an explicit throughput lane, not the default quality/diagnostic lane.

Random basis init is an ablation/stress path, not the quality default. Do not use it merely to dodge initialization cost unless the run is explicitly measuring fit/performance rather than optimizer quality.

CCE is the loss path because the HF full-logits route is strictly worse for the memory/throughput questions this repo is asking. If a run needs an unoptimized loss route, stop immediately and re-evaluate the plan. Continue only if the work is explicitly reframed as an ablation outside the faithful benchmark path.

EOS-packed no-mask batches are the throughput route for full packing. Use them explicitly only when the named question is throughput or packed-LM behavior. For quality, gradient-scale, and source-retention diagnostics on SYNTH, use the default right-padded no-mask lane.

BOS/EOS and document separators are formatting policy. For throughput claims, preserve fixed-shape packed no-mask inputs for flash attention. For format/gradient/retention diagnostics, packed no-mask is not faithful if examples can attend across row boundaries. Prefer one row per sequence, right-padded to a fixed length, with labels masked for padding and context-only tokens. With causal attention and right padding, real tokens cannot attend to future pad tokens, and pad positions have ignored labels, so an `attention_mask` is not required for semantic correctness. Do not add one unless measuring or fixing a model-specific need.

For SYNTH row formatting, the reviewed policy is to mask the question plus first divider, then train on the reasoning/answer body. Row cutoff before EOS inside fixed-length packed blocks is acceptable for explicit packed throughput runs, but the default faithful-format diagnostic lane uses right-padded no-mask examples so documents do not leak attention across boundaries.

Default harness model is `LiquidAI/LFM2.5-350M-Base`. `LiquidAI/LFM2.5-1.2B-Base` remains a reference baseline for continuity, but use it by naming it explicitly.

Stay on the small/default model until the user explicitly authorizes going up. Larger LFM or Qwen runs are not harmless “continuations”; they change OOM risk, time cost, and experiment meaning.

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

If the user names a specific model, dataset, path, or script, treat that as the target contract. If it is unavailable in the current environment, say so plainly, verify the assumption if possible, and ask what to do next. Do not silently substitute a nearby cached or convenient alternative.

If the user says the named thing “should be cached,” treat that as an expectation to verify, not as permission to swap targets.

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

For gradient-scale investigations, separate raw model gradients from optimizer-produced updates. Raw `.grad` norms are measured before Aurora, orthogonalization, Muon scaling, or `optimizer.step()`. If raw norms are surprising, compare against another model before blaming SumoTrack. Then measure update norm and update/param ratio to decide whether scary gradients are merely a thermometer reading or an optimizer scaling bug.

Do not report optimizer work as done because code imports or a smoke run descends. Optimizers fail silently and convincingly. If validation cannot be run, say exactly what remains unverified.

## Reporting and style

Prefer signal over ceremony.

A useful report says what changed in system meaning: which geometric assumption became stronger or weaker, which bottleneck became visible, which product constraint bit, and what should be cut next.

Avoid reports that merely list files, commands, and green tests unless those details carry decision value.

Favor direct, readable PyTorch first. Once the math is right, move hot paths toward HeavyBall-style transforms and compiled utilities. Keep comments for non-obvious math, state-shape invariants, and performance traps. Do not narrate obvious tensor operations.

Maintainability is a feature, but product-shaped algorithm clarity outranks tidy scaffolding.
