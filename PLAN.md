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
- **Aurora-style rectangular polar machinery** for practical projected orthogonalization,
- **Grassmann/Stiefel basis tracking** for smooth adaptation of the active subspace.

The current mainline is not “projected momentum plus optional ortho.” Early LFM/SYNTH evidence already showed projected moment orthogonalization is useful in this setting, and the 1k Aurora result made the direction concrete. Treat Aurora-orthogonalized projected momentum as SumoTrack's forward path.

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

Current status: a 200-step broad-no-embeddings LFM/SYNTH evaluation at 4096 tokens/step, rank 64, equal ~89.9 MB state budget showed residual-facing side policy beating shape `AUTO` decisively on target validation (`1.600` vs `1.810` final val). This is evidence that aligning projection sides to the hidden-state / backbone-facing axis improves adaptation at identical byte cost.

Important convention: optimizer side names are **PyTorch weight-storage coordinates**. `nn.Linear.weight` is `[out_features, in_features]`. Therefore MLP up/gate and attention q/k/v weights have the residual stream on the forward input / activation side, which is storage `right` / columns. MLP down and attention output weights return to the residual stream on the output / top side, which is storage `left` / rows. The harness policy is named `residual-facing` to avoid making callers think in storage coordinates.

### 2. Uniform rank policy

Uniform rank is the active harness policy. Keep it boring until a real product-shaped budget or retention failure earns another allocation cut.

Default rank remains `32` for the library optimizer constructor, but the LLM harness defaults to rank `64` because that is the current useful broad-model evaluation baseline.

Current status: a 200-step broad-no-embeddings LFM/SYNTH evaluation at 4096 tokens/step, rank 64, residual-facing side, ~89 MB state budget compared several rank-allocation ideas. Size-rank saved ~10 MB vs uniform (`79.7 MB` vs `89.9 MB`) with a marginal quality delta (`1.591` vs `1.600`, likely noise at 200 steps). Role-based rank overshot the intended budget. Spectrum-calibrated allocation was attempted but blocked by HeavyBall compile issues; initial-gradient R99 was measured as ~35 mean, drifting down to ~9 after 200 steps as gradient structure concentrates during training. The decision from that noise pile is not “add knobs”; it is residual-facing side + uniform rank 64 as the default.

### 3. Subspace tracking as adaptation smoothing

SVD initialization is a useful quality/correctness rail. Grassmann tracking then controls smoothing: burst refresh on a cadence interval. Equal-cadence burst-vs-round-robin measurements showed identical peak VRAM (~8.18 GB at 4096 tokens/step, ~5.03 GB at 64 tokens/step) — state is ~90 MB and activation/model memory dominates. Burst is the simpler default; refresh cadence is tuned via `basis_refresh_interval` / `--basis-refresh-interval`.

That smoothing may be central for distribution shift without total forgetting.

Open design questions:

- exact SVD init vs random init vs short calibration SVD,
- refresh interval by layer depth or transformer matrix role,
- moment transport strength across basis changes,
- whether basis motion should be slower in backbone-stability-critical modules.

### 4. Orthogonalization quality for rectangular projected moments

Projected moments are often rectangular. HeavyBall therefore uses its standard Newton-Schulz path rather than square-only PolarExpress-style machinery. That is fine and expected.

But very tall rectangular NS may not spread information as well as desired. Aurora-style rectangular orthogonalization is directly relevant because SumoTrack's rank-64 projected moments commonly have extreme aspect ratios: `2048×64`, `8192×64`, `9728×64`, and their transposes. Ordinary polar/NS makes the small side orthonormal but can leave large-side row leverage uneven; Aurora adds diagonal preconditioning so the large-side update energy approaches the Stiefel target.

SumoTrack should treat Aurora as a **projected direction map**, not as a replacement optimizer. Momentum, basis tracking, full-matrix Muon scaling, LR, fallback semantics, and state accounting remain SumoTrack's responsibility.

Current status: Aurora-style leverage-uniform polar is the projected direction map. Muon scale semantics are fixed, not a user-facing scale-mode knob. `--log-norms` reports projected leverage diagnostics so a run can answer whether the large-side energy actually became more uniform. Packed SDPA batches, the CCE harness path, and Aurora cycle knobs are now wired through the smoke harness; the remaining question is whether they matter at realistic token counts rather than tiny smokes. Qwen3.5-2B-Base plus `flash-linear-attention` is now present, and `causal-conv1d` is now buildable in this repo environment. The repo floor is pinned to Python 3.13 via `.python-version`.

The Aurora question has moved. A 1k-step broad-no-embeddings LFM/SYNTH target run at rank 64, random init, Grassmann tracking, 768 tokens/step, LR `0.0025`, and Muon scale showed Aurora beating HeavyBall NS on target validation (`1.590971` vs `1.663156`) at matched update/param and identical peak memory, while reducing mean projected leverage CV from `1.2257` to `0.0197`. The measured `~21%` step-time cost happened at an unrealistically small token count, where per-step orthogonalization overhead is least amortized. Treat this as evidence that leverage-uniform rectangular orthogonalization is behaviorally useful, not merely prettier telemetry.

Aurora is now the default projected direction map. Same-shape one-sided projected moments are bucketed before Aurora/HeavyBall orthogonalization, using a local batched Newton-Schulz rail because HeavyBall's compiled helper does not currently handle the leading bucket dimension SumoTrack needs. The remaining question is product behavior: retention/source movement, overhead amortization at realistic tokens per step, and state allocation.

Open design questions:

- Aurora retention/source behavior after its target-movement win,
- whether Muon scale semantics continue to hold under larger target/source runs,
- whether transformer matrix role should influence Grassmann refresh cadence.

### 5. Two-sided projection: removed from active path

Two-sided projection was implemented as an experimental branch:

```text
G_hat = Q_Lᵀ G Q_R      # [r_l, r_r]
O_hat = orthogonalize(G_hat)
update = Q_L O_hat Q_Rᵀ
```

It costs roughly `r(m+n) + r²` state instead of one-sided `r(m+n)`, so the byte penalty is modest for large matrices. It may trade raw adaptation capacity for smoother, more stable, square-core SUMO geometry.

Current status: removed from the active optimizer and harness. A short fixed-basis LFM/SYNTH check found two-sided stable and state-cheap but weaker than one-sided rectangular updates on immediate target movement. Without Grassmann tracking or a retention-specific win, it was complexity rent. Keep the result as historical evidence; recover from git only if a future retention/rank-budget question demands it.

Do not revive it unless a real retention/rank-budget result demands the complexity.

## Engineering direction

Keep the eager implementation as the algorithm rail until the geometry is clearer. HeavyBall-native/ECC integration matters later, but premature integration will hide algorithmic questions behind framework work.

Current implementation invariants:

- Public class is `SumoTrack`.
- Matrix parameters must not store full-size first moments.
- Matrix parameters must not store full-size second moments in the main path.
- Non-2D params use a boring fallback path or are frozen by task policy.
- Fallback state must be accounted separately from matrix state.
- Orthogonalization happens in projected space.
- Aurora with fixed Muon scale semantics is the projected direction. HeavyBall Newton-Schulz remains an internal polar primitive.
- One-sided same-shape projected orthogonalization is bucketed. Two-sided projection has been removed.
- Grassmann basis tracking refreshes all bases on a burst cadence controlled by `basis_refresh_interval`. Equal-cadence round-robin was measured and removed as complexity rent — state is too small to affect peak VRAM.
- Architecture-aware side policy lives in the LLM harness via named-parameter groups. Rank allocation is uniform. The optimizer core remains responsible for projected state and updates, not model taxonomy.
- Unsupported ECC/param-ECC must fail loudly until implemented honestly through HeavyBall state hooks.

The harness should encode the current boring-best defaults so experiment commands stay lean. The default route is SVD basis init, `batch_size=4`, `seq_len=1024`, EOS-packed no-mask batches plus CCE; HF/full-logits loss or non-packed/padded batches are stop conditions for faithful throughput/memory work. Random basis init is only for performance/fit measurements where SVD cold-start cost is explicitly not the claim. Continue only when the work is explicitly reframed as an unoptimized ablation outside the faithful benchmark path. Prefer:

```bash
uv run python experiments/llm_synth_smoke.py --measure-steps 200
```

over restating defaults such as `--optimizers sumotrack --rank 64 --projection-side-policy residual-facing --param-scope broad-no-embeddings`. Pass an argument only when the run is intentionally changing that axis.

Do not optimize kernel launches before the algorithmic shape is worth optimizing. Once the design is stable enough, the performance ladder is:

1. measure realistic-token-step amortization of bucketed projected orthogonalization overhead,
2. tighten bucket implementation only where measured overhead remains material,
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

## Future leads

This is the active lead list distilled from the old phase checklist. It is not a museum of every knob ever implemented; if a lead no longer serves the current product thesis, cut it instead of polishing it.

### Must carry forward

1. **Retention/source validation is the next truth sensor.** Aurora has enough target-only evidence. The next meaningful run needs a distinct source corpus and periodic retention validation so target movement can be judged against forgetting.
2. **Basis movement and residual diagnostics.** Add minimal per-step or per-refresh metrics for basis motion, projected-gradient residual, and transported-moment drift. Grassmann smoothing should be tuned by observed adaptation area, not by the pleasant scent of differential geometry.
3. **Realistic-token throughput.** Measure bucketed Aurora at 32k+ tokens/optimizer step with no gratuitous norm logging. The useful signal is tokens/sec, peak CUDA, and whether gradient accumulation was actually avoided.
4. **Fallback policy under broad training.** Non-2D fallback state is small in LFM, but embeddings, norms, biases, and tiny conv kernels need a principled policy before Gemma-scale claims. Decide what trains, what freezes, and what fallback optimizer is acceptable.
5. **State and resume invariants.** Keep tests for matrix projected-state shape, fallback accounting, bf16 behavior, and mixed state-dict resume. Optimizer bugs love reload boundaries.
6. **HeavyBall-native path only after geometry earns it.** Move hot paths toward HeavyBall transforms/ECC when Aurora + tracking + retention look worth productizing. Do not hide algorithm uncertainty behind framework plumbing.
7. **Projected-gradient hooks remain isolated and late.** Avoiding full-gradient materialization may be the real memory frontier, but it needs equivalence tests against `project(full_gradient)` on tiny models before it touches the main optimizer.
8. **Full backward-time gradient release is a future memory frontier.** Once `G_hat = project(G)` is materialized, SumoTrack can release the full gradient for that parameter. The current safe path can consume grads during optimizer step after projection, but flashoptim-style post-accumulate hooks need a bucket-aware/deferred design so immediate per-parameter release does not destroy same-shape Aurora batching or change basis-refresh semantics.

### Conditional leads

- **Role/depth refresh cadence:** worth testing after basis-motion diagnostics exist. Without those sensors it is knob gardening.
- **Two-sided square-core projection:** recover from git only if one-sided Aurora shows retention damage that square-core geometry plausibly addresses.
- **Stochastic rounding:** possible low-complexity precision win before full ECC/param-ECC. Worth testing for fp32 projected/Aurora math cast back to bf16 params or low-precision projected state, but only if it has a direct bf16 stability/quality comparison; do not let it become pseudo-ECC ceremony.
- **Nesterov/perpendicular recovery/extra momentum knobs:** hold unless a named failure mode demands them. They are currently more likely to reintroduce noise and API clutter than product leverage.

### Maintenance leads

- Keep `experiments/llm_synth_smoke.py` as a lean lab harness, not a trainer framework. Defaults should represent the current best baseline; examples should override only the axis being tested.
- Keep `RESULTS.md` as a brief experiment log: why the run existed, what was tested, what moved, and what was observed. Put exhaustive command/output detail in external logs when needed; keep current direction in this file.
- Prefer deleting unsupported branches over keeping dormant flags. If an experiment is not active, preserve it in git history and docs, not in the public path.
- Add tests for any policy default that encodes a design decision, especially backbone-facing side grouping and uniform-rank reporting.

## Near-term design cuts

The next work should improve the algorithm under our feet:

1. **Tracking smoothness diagnostics.** Instrument basis movement and projected-gradient residual so Grassmann smoothing can be reasoned about as adaptation-area control, not just speed.
2. **Medium SYNTH adaptation run with retention/source validation.** Use the current defaults and a real source corpus to see curve shape and preservation behavior. The target-only Aurora question has enough evidence; the product question is whether that movement preserves useful source behavior.
3. **Realistic-token Aurora amortization.** Run 32k+ tokens/optimizer step without `--log-norms` unless diagnostics are the point. Measure tokens/sec, peak CUDA, and whether no-gradient-accumulation is practical. The packed 256-token smoke shows cycle knobs matter at tiny shapes, but it does not answer the amortized product question.
4. **Qwen3.5 fast-path throughput follow-up.** The memory check says `batch=1, seq=1024` fits with random basis init, but `batch=4, seq=1024` is still a near-edge fit/OOM depending on GPU occupancy. The next question is whether batch 3 is the practical ceiling on this card under the faithful CCE/no-mask route.

## Next session contract

Start from the 1k Aurora result and the side evaluation. Residual-facing side beats AUTO decisively, uniform rank 64 is the harness policy, CCE is the default loss path, and Aurora's per-step overhead amortizes with token count. The unknown has narrowed to retention/source behavior and basis smoothing.

Default harness model is `LiquidAI/LFM2.5-350M-Base`; `LiquidAI/LFM2.5-1.2B-Base` remains a continuity/reference baseline. Use the 1.2B only by naming it explicitly when preserving comparison continuity matters. Do not let cache convenience choose the model.

Minimum useful next session:

1. Add minimal per-step basis-motion instrumentation so Grassmann smoothing can be tuned by refresh cadence rather than belief.
2. Use the retention/source validation output with a real source corpus, not SYNTH-as-retention.
3. Measure one realistic-token run using defaults unless deliberately testing one axis; on the local 12GB card, use random basis init if exact SVD cold-start OOMs again.
4. If pursuing Qwen3.5, solve the `causal-conv1d` toolchain first; `flash-linear-attention` alone does not clear the target ceiling.
5. Do not spend on AdamW, topology proof, ECC, projected-gradient hooks, HeavyBall-vs-Aurora reruns, two-sided revival, schedule/budget archaeology, or rank-allocation horse races.

Falsifier: if Aurora's target advantage disappears when source/retention is measured, or if side-rank wins are noise at longer horizons, revisit the direction map and allocation.

## Stop conditions

Stop and rethink if:

- SumoTrack cannot exploit higher tokens/step or larger model scope than serious alternatives on the target GPU,
- orthogonalized one-sided updates improve target loss but cause unacceptable retention collapse,
- uniform rank clearly spends most state on low-leverage tensors,
- Grassmann tracking is too sluggish to adapt or too fast to smooth,
- rectangular orthogonalization quality becomes the bottleneck,
- the code starts serving old gates instead of the product thesis.

The product is not a checklist of optimizer features. The product is a usable path to high-capacity distribution adaptation under consumer-GPU constraints.
