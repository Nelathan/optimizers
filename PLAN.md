# SumoTrack Plan

SumoTrack is a memory-efficient optimizer line for high-capacity continued pretraining on consumer GPUs. The public optimizer class is `SumoTrack`.

The goal is not to make AdamW cheaper. The goal is to make real distribution adaptation possible where full optimizer state, gradient accumulation, or adapter-only training would otherwise define the ceiling. The target use case is moving a pretrained model across a meaningful data distribution shift — for example web/knowledge model behavior toward clean reasoning, books, stories, or other target corpora — while preserving enough of the source model to avoid crude overwrite.

The near-term hardware target is a single RTX 5090-class workstation training Gemma 4 12B-class models with as many tokens per step as practical, no gradient accumulation if avoidable, high throughput, and enough update quality to actually change the model rather than flavor it.

## Product thesis

For a fixed consumer-GPU memory budget, SumoTrack should deliver more useful model movement per byte and per second than adapter-only or low-rank-gradient methods when the user wants high-capacity distribution adaptation, not just style steering.

The relevant competitors are LoRA/DoRA/QLoRA/Unsloth-style adapter training, GaLore, SubTrack, SUMO/Muon-family optimizers, and adjacent low-state full-finetuning methods. AdamW is an obvious quality and sanity anchor, not the thing SumoTrack is trying to beat on memory.

Success looks like a Pareto point:

- enough trainable capacity to materially shift distribution,
- optimizer state small enough to preserve tokens/step on consumer hardware,
- stable updates that do not trash the source model faster than they learn the target,
- step time that does not lose the memory gain to launch spray or decompositions,
- a configuration story a serious user could operate without occult LR roulette.

## Main algorithm thesis

SumoTrack combines:

- **SubTrack/GaLore-style one-sided gradient subspace tracking** for memory-shaped state,
- **projected first moments** rather than full-size matrix moments,
- **SUMO/Muon-style orthogonalization of the projected moment** for update geometry,
- **HeavyBall Newton-Schulz / polar machinery** for the practical orthogonalization path,
- **Grassmann/Stiefel basis tracking** for smooth adaptation of the active subspace.

The current mainline is not “projected momentum plus optional ortho.” Early LFM/SYNTH evidence already showed projected moment orthogonalization is useful in this setting. Treat orthogonalized projected momentum as the main SumoTrack path; keep no-ortho only as a diagnostic when the geometry itself is under suspicion.

The core interpretation for a tall matrix gradient `G ∈ R^{m×n}` is:

```text
Q ∈ R^{r×n}
M_hat = G Qᵀ      # [m, r]
O_hat = orthogonalize(M_hat)
update = O_hat Q  # [m, n]
```

For a wide matrix, the symmetric left-side form is used.

This means SumoTrack currently says:

> Adapt along selected representation / hidden-state directions, while distributing the update across the larger side with Muon/SUMO geometry.

That is a deliberate and valuable bias, especially for transformer MLP and attention matrices. The backbone-facing hidden-state axis carries stability; the larger expanded side has room to shape a useful manifold. The shape is rectangular because the active subspace is one-sided, not because the optimizer accidentally became quadratic.

## Geometry questions that actually matter

The important design work is not broad smoke tests or broad LR polishing. It is deciding which geometry gives the best adaptation capacity under memory and speed constraints.

### 1. One-sided projection policy

Current `AUTO` chooses the smaller ambient side. This is simple and often sensible, but it is still a crude proxy for transformer semantics.

Open design questions:

- Should the chosen side always be the hidden-state / backbone-facing axis when module structure is known?
- Should attention projections, MLP up/gate/down projections, and output projections use different side policies?
- Should side choice be architecture-aware for Gemma/LFM/Qwen rather than purely shape-aware?
- Are we throwing away task-vital gradient components because they did not appear in the tracked/SVD subspace early enough?

### 2. Rank allocation

Uniform rank is a baseline, not a product design.

Rank should become budget-driven:

```text
optimizer_state_budget_mb = ...
rank_policy = "uniform" | "size" | "spectrum" | "module_role"
```

Promising policies:

- larger rank for high-leverage MLP and attention matrices,
- role-specific rank caps,
- gradient-spectrum-aware rank after a short calibration window,
- global byte budget allocation rather than rank-per-tensor habit,
- minimum rank for small-but-semantically-important matrices.

Default rank remains `32` for library sanity, but real Gemma-scale evaluation should not pretend fixed rank `64` is a design conclusion.

### 3. Subspace tracking as adaptation smoothing

SVD initialization and SVD refresh are not merely “better basis” operations. They can also be sharper interventions that rapidly redirect the optimizer toward the current batch distribution. Grassmann tracking is slower and smoother: it limits adaptation to a rotating subspace rather than a list of unrelated spaces over time.

That smoothing may be central for distribution shift without total forgetting.

Open design questions:

- exact SVD init vs random init vs short calibration SVD,
- continuous Grassmann tracking vs periodic SVD refresh,
- moment transport strength across basis changes,
- basis update rate by layer depth or module role,
- whether basis motion should be slower in backbone-stability-critical modules.

### 4. Orthogonalization quality for rectangular projected moments

Projected moments are often rectangular. HeavyBall therefore uses its standard Newton-Schulz path rather than square-only PolarExpress-style machinery. That is fine and expected.

But very tall rectangular NS may not spread information as well as desired. Aurora-style rectangular orthogonalization is directly relevant because SumoTrack's rank-64 projected moments commonly have extreme aspect ratios: `2048×64`, `8192×64`, `9728×64`, and their transposes. Ordinary polar/NS makes the small side orthonormal but can leave large-side row leverage uneven; Aurora adds diagonal preconditioning so the large-side update energy approaches the Stiefel target.

SumoTrack should treat Aurora as a **projected direction map**, not as a replacement optimizer. Momentum, basis tracking, full-matrix Muon scaling, LR, fallback semantics, and state accounting remain SumoTrack's responsibility.

Current status: `orthogonalization="aurora"` exists as an eager projected orthogonalization mode. It applies Aurora-style leverage-uniform polar to the projected moment, then uses the same `orthogonalization_scale_mode` semantics as the HeavyBall path. `--log-norms` also reports projected leverage diagnostics so an Aurora run can answer whether the large-side energy actually became more uniform.

Open design questions:

- HeavyBall Newton-Schulz vs Aurora-style rectangular balancing on real gradients,
- scale and norm grafting semantics after rectangular orthogonalization,
- whether projected tensor aspect ratio should influence rank allocation,
- whether some modules should use two-sided square cores for cleaner orthogonalization.

### 5. Two-sided projection as an optional branch

Two-sided projection exists as an experimental branch, but it is not the mainline:

```text
G_hat = Q_Lᵀ G Q_R      # [r_l, r_r]
O_hat = orthogonalize(G_hat)
update = Q_L O_hat Q_Rᵀ
```

It costs roughly `r(m+n) + r²` state instead of one-sided `r(m+n)`, so the byte penalty is modest for large matrices. It may trade raw adaptation capacity for smoother, more stable, square-core SUMO geometry.

Current status: `projection_mode="two_sided"` stores left/right bases plus a square projected moment. It supports random/SVD initialization and either fixed bases (`subspace_update_method="none"`) or explicit SVD refresh. It rejects Grassmann tracking because the coupled left/right smoothing update has not been designed. A short fixed-basis LFM/SYNTH check found two-sided stable and state-cheap but weaker than one-sided rectangular updates on immediate target movement. Keep it as a stability/retention/rank-budget branch, not as the default.

Evaluate this only when it answers a real design question: capacity vs stability, forgetting vs target movement, or rectangular orthogonalization limits. Do not add it as abstraction ornament.

## Engineering direction

Keep the eager implementation as the algorithm rail until the geometry is clearer. HeavyBall-native/ECC integration matters later, but premature integration will hide algorithmic questions behind framework work.

Current implementation invariants:

- Public class is `SumoTrack`.
- Matrix parameters must not store full-size first moments.
- Matrix parameters must not store full-size second moments in the main path.
- Non-2D params use a boring fallback path or are frozen by task policy.
- Fallback state must be accounted separately from matrix state.
- Orthogonalization happens in projected space.
- HeavyBall Newton-Schulz with `orthogonalization_scale_mode="muon"` is the practical mainline orthogonalization path.
- Unsupported ECC/param-ECC must fail loudly until implemented honestly through HeavyBall state hooks.

Do not optimize kernel launches before the algorithmic shape is worth optimizing. Once the design is stable enough, the performance ladder is:

1. bucket same-shape projected moments for batched orthogonalization,
2. evaluate Aurora-style rectangular orthogonalization,
3. move compatible pieces toward HeavyBall-native transforms,
4. add ECC/param-ECC,
5. consider projected-gradient hooks only after the ordinary-gradient optimizer has proven worth.

Projected-gradient hooks remain dangerous and later. Avoiding full-gradient materialization could be a real Gemma-scale memory frontier, but it must be isolated and proven equivalent against ordinary full-gradient projection on tiny models.

## Evaluation philosophy

SYNTH is not a toy here. Clean, low-noise target data is useful for measuring adaptation because optimizer noise is less hidden by dataset noise. Ace the easy data before claiming robustness on a short-story dataset that breaks models.

The evaluation target is distribution adaptation with retention, not merely short-run loss descent.

Useful evaluation signals:

- target validation loss curve,
- source/retention validation loss curve,
- tokens per optimizer step without gradient accumulation,
- wall-clock tokens/sec,
- peak CUDA memory,
- optimizer state bytes by matrix/fallback/total,
- update/param and grad/param diagnostics,
- basis motion / orthonormality diagnostics when testing tracking,
- qualitative generation only after quantitative movement is real.

The next serious evaluation harness should support periodic validation and structured output. But harness work is only justified when it lets us answer an algorithm question faster.

## Near-term design cuts

The next work should improve the algorithm under our feet:

1. **Retention-aware Aurora ablation.** Compare `orthogonalization="heavyball"` vs `"aurora"` on the same one-sided SumoTrack configuration. Report target movement, source/retention movement, step time, peak CUDA, update/param, and projected leverage CV/min/max. The question is no longer “does Aurora balance rows?” — it does. The question is whether leverage-uniform projected updates reduce thrash/forgetting enough to justify a small target-loss and step-time cost.
2. **Architecture-aware side/rank inspection.** Inspect Gemma/LFM/Qwen module shapes and map which axis is hidden-state-facing for MLP up/gate/down and attention projections. Decide whether `AUTO` is aligned with the intended stability axis or merely lucky.
3. **Budget-driven rank policy.** Add a minimal rank-policy layer that can allocate rank by module role or tensor size under a target state budget. Uniform rank remains available but should stop being the only serious mode.
4. **Tracking smoothness diagnostics.** Instrument basis movement and projected-gradient residual so Grassmann vs SVD can be reasoned about as smoothing/adaptation-area control, not just speed.
5. **Medium SYNTH adaptation run.** After one geometry cut above, run enough SYNTH to see curve shape and retention, not another 20-step smoke.

## Next session contract

Start with Aurora/retention, because the evidence now points to a concrete pressure point in the current mainline: extreme-aspect projected moments have uneven large-side leverage under ordinary NS, and Aurora fixes that mechanically. The unknown is whether that smoothing helps product behavior or only slows target descent.

Minimum useful next session:

1. Run a short but nontrivial LFM/SYNTH comparison:
   - `--param-scope broad-no-embeddings`,
   - rank `64`,
   - `--subspace-init random` for performance-path timing unless explicitly testing initialization quality,
   - HeavyBall NS + Muon scale at the current center LR,
   - Aurora + Muon scale at the same LR, then only one LR adjustment if update/param or loss says the scale changed materially,
   - `--log-norms` so projected leverage diagnostics are present.
2. Interpret Aurora by three signals together:
   - target loss movement,
   - projected leverage CV/min/max,
   - measured step time / tokens/sec.
3. Include a source/retention validation split if available. Aurora's plausible value is less thrashy distribution movement, not prettier row norms.
4. If Aurora improves leverage but hurts both target and retention, demote it. If it hurts short target movement but preserves source behavior, schedule a longer run rather than discarding it.
5. Keep two-sided square-core out of the mainline unless it shows a retention or rank-budget advantage. Its short fixed-basis target movement was weaker than one-sided.
6. Do not spend the session on AdamW, broad topology proof, ECC, or projected-gradient hooks.

Falsifier: if Aurora's better leverage uniformity does not improve target movement, retention, or stability after a fair scale check, keep HeavyBall NS as the practical mainline and move to rank/side policy. Do not worship prettier row norms.

## Stop conditions

Stop and rethink if:

- SumoTrack cannot exploit higher tokens/step or larger model scope than serious alternatives on the target GPU,
- orthogonalized one-sided updates improve target loss but cause unacceptable retention collapse,
- rank allocation shows most state is being spent on low-leverage tensors,
- Grassmann tracking is too sluggish to adapt or too fast to smooth,
- rectangular orthogonalization quality becomes the bottleneck,
- the code starts serving old gates instead of the product thesis.

The product is not a checklist of optimizer features. The product is a usable path to high-capacity distribution adaptation under consumer-GPU constraints.
