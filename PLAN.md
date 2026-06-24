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
- Are we throwing away task-vital gradient components because they did not appear in the tracked spectral subspace early enough?

Current status: a 200-step broad-no-embeddings LFM/SYNTH evaluation at 4096 tokens/step, rank 64, equal ~89.9 MB state budget showed residual-facing side policy beating shape `AUTO` decisively on target validation (`1.600` vs `1.810` final val). This is evidence that aligning projection sides to the hidden-state / backbone-facing axis improves adaptation at identical byte cost.

Important convention: optimizer side names are **PyTorch weight-storage coordinates**. `nn.Linear.weight` is `[out_features, in_features]`. Therefore MLP up/gate and attention q/k/v weights have the residual stream on the forward input / activation side, which is storage `right` / columns. MLP down and attention output weights return to the residual stream on the output / top side, which is storage `left` / rows. The harness policy is named `residual-facing` to avoid making callers think in storage coordinates.

### 2. Uniform rank policy

Uniform rank is the active harness policy. Keep it boring until a real product-shaped budget or retention failure earns another allocation cut.

Default rank remains `32` for the library optimizer constructor, but the LLM harness defaults to rank `64` because that is the current useful broad-model evaluation baseline.

Current status: a 200-step broad-no-embeddings LFM/SYNTH evaluation at 4096 tokens/step, rank 64, residual-facing side, ~89 MB state budget compared several rank-allocation ideas. Size-rank saved ~10 MB vs uniform (`79.7 MB` vs `89.9 MB`) with a marginal quality delta (`1.591` vs `1.600`, likely noise at 200 steps). Role-based rank overshot the intended budget. Spectrum-calibrated allocation was attempted but blocked by HeavyBall compile issues; initial-gradient R99 was measured as ~35 mean, drifting down to ~9 after 200 steps as gradient structure concentrates during training. The decision from that noise pile is not “add knobs”; it is residual-facing side + uniform rank 64 as the default.

### 3. Subspace tracking as adaptation smoothing

Stable side-Gram `eigh` initialization is the basis-init path. It computes the one-sided dominant singular subspace SumoTrack actually uses, without carrying full SVD as hot-path ceremony. Grassmann tracking then controls smoothing: burst refresh on a cadence interval. Equal-cadence burst-vs-round-robin measurements showed identical peak VRAM (~8.18 GB at 4096 tokens/step, ~5.03 GB at 64 tokens/step) — state is ~90 MB and activation/model memory dominates. Burst is the simpler default; refresh cadence is tuned via `basis_refresh_interval` / `--basis-refresh-interval`.

That smoothing may be central for distribution shift without total forgetting.

Open design questions:

- whether `eigh` init plus Grassmann tracking needs a short calibration phase,
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

The harness should encode the current boring-best defaults so experiment commands stay lean. The default quality/diagnostic route is stable `eigh` basis init, `batch_size=4`, `seq_len=1024`, faithful SYNTH right-padded no-mask batches plus CCE. EOS-packed no-mask batches are still the explicit throughput route; use them only when the claim is packed throughput or packed-LM behavior. HF/full-logits loss remains a stop condition unless the work is explicitly reframed as an unoptimized ablation. Random basis init is an explicit ablation/stress path, not a default-quality shortcut. Prefer:

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

The current faithful stream contract is: train on SYNTH train, evaluate target movement on held-out SYNTH, and evaluate source behavior on a separate broad pretraining-style corpus. The accessible source corpus is `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled`, using the first parquet shard when a small local source sensor is needed. If source loss improves during SYNTH adaptation, report that literally: this source shard did not expose forgetting. Do not call it retention solved.

Input formatting is a first-class evaluation risk. The harness now has a faithful SYNTH diagnostic lane by default: one row per sequence, right-padded fixed length, explicit first divider, labels masked with `-100` for padding plus question/divider, and training on the reasoning/answer body. Before treating an optimizer curve as meaningful, still inspect the exact text stream when the format changes: SYNTH row conversion, HF row conversion, tokenizer BOS/EOS behavior, document separators, decoded blocks, and absence of chat/template contamination. A base model that greedily emits ordinary document/wiki prose while the harness trains on query + markdown reasoning traces is a format-contact fact, not decoration. BOS can be present as context while its label is masked with `-100`; this does not require an `attention_mask` when examples are right-padded under causal attention.

Gradient norms are also a first-class sensor, but only when their definition is explicit. Raw dense `.grad` norms are not optimizer update norms. Aurora, orthogonalization, and Muon scaling happen after raw gradients are formed. If raw norms look wrong, compare CCE to ordinary CE on the same batch, break the norm down by parameter scope/class, compare another model on the same packed text, then measure update/param. Do not blame SumoTrack geometry for a pre-step gradient until the update path is measured.

Useful evaluation signals:

- target validation loss curve,
- source/retention validation loss curve,
- exact decoded train/eval packed blocks, tokenizer special-token behavior, and document boundary policy,
- raw gradient norm by data stream, model, and parameter scope,
- update/param and grad/param diagnostics,
- peak CUDA memory when scale changes,
- optimizer state bytes by matrix/fallback/total,
- basis motion / orthonormality diagnostics when testing tracking,
- tokens per optimizer step and wall-clock tokens/sec only when throughput is the named question,
- qualitative generation only after quantitative movement is real.

The harness should support periodic validation and structured output, but harness work is only justified when it lets us answer an algorithm question faster. Output sample text artifacts to files when investigating format; do not force the user to infer stream shape from summary prose.

## Future leads

This is the active lead list distilled from the old phase checklist. It is not a museum of every knob ever implemented; if a lead no longer serves the current product thesis, cut it instead of polishing it.

### Must carry forward

1. **Input stream faithfulness remains a truth sensor, now encoded in the harness default.** Save and inspect actual decoded examples whenever the row format, tokenizer, or batching mode changes. The old packed artifacts explain why the previous curves were not retention evidence; the current default faithful lane avoids row-boundary attention and masks the SYNTH question/divider.
2. **Gradient-scale and update-scale separation.** Current probes show LFM2.5-350M has high raw dense gradients on SYNTH contact while CCE and ordinary CE agree. A Qwen3.5-2B-Base probe on the same text showed much lower raw norms but still above the user's normal pretraining/LoRA intuition. A faithful `bs8 + checkpointing` LR sweep found the useful knee below the old default: `1e-4` is the clean balance point and `3e-4` is the aggressive target-winning point, while `1e-3` is already past the knee on the source sensor. Do not reason from absolute update norm alone; let target/source curves choose.
3. **Retention/source validation under the faithful lane.** Aurora has enough target-only evidence, but the earlier packed SYNTH/source curves are not retention evidence. Meaningful retention runs need the explicit stream contract: train SYNTH train, target eval held-out SYNTH, source eval external pretraining-style corpus, with boundary policy visible in decoded samples. Loss descent on a malformed stream is adaptation to mismatch, not product signal.
4. **Basis movement and residual diagnostics remain useful, but later.** Add minimal per-refresh energy-capture/basis-motion sensors when they answer a specific failure mode. Do not revive broad projected-residual telemetry as metric gardening, but do not discard the lead; it remains relevant after format and scale are trustworthy.
5. **Fallback policy under broad training.** Non-2D fallback state is small in LFM, but embeddings, norms, biases, and tiny conv kernels need a principled policy before Gemma-scale claims. Decide what trains, what freezes, and what fallback optimizer is acceptable.
6. **State and resume invariants.** Keep tests for matrix projected-state shape, fallback accounting, bf16 behavior, and mixed state-dict resume. Optimizer bugs love reload boundaries.
7. **HeavyBall-native path only after geometry earns it.** Move hot paths toward HeavyBall transforms/ECC when Aurora + tracking + retention look worth productizing. Do not hide algorithm uncertainty behind framework plumbing.
8. **Projected-gradient hooks remain isolated and late.** Avoiding full-gradient materialization may be the real memory frontier, but it needs equivalence tests against `project(full_gradient)` on tiny models before it touches the main optimizer.
9. **Full backward-time gradient release is a future memory frontier.** Once `G_hat = project(G)` is materialized, SumoTrack can release the full gradient for that parameter. The current safe path can consume grads during optimizer step after projection, but flashoptim-style post-accumulate hooks need a bucket-aware/deferred design so immediate per-parameter release does not destroy same-shape Aurora batching or change basis-refresh semantics.

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
- Keep experiment reports linguistically faithful. “Source eval” means an evaluation stream, not training data. Bad labels can make a correct run look like it answered the wrong question.

## Near-term design cuts

The next work should improve the algorithm under our feet:

1. **Refine the faithful `bs8 + checkpointing` LR/update-scale knee.** The broad 1k sweep across `1e-5` to `1e-3` says `1e-4` is the balanced point, `3e-4` is target-best with extra source cost, and `1e-3` is past the knee. Stay in the `1e-4–3e-4` neighborhood unless a longer run changes the tradeoff.
2. **Keep update sanity visible on every scale run.** Log raw grad norm, update norm, and update/param at cadence. The old `~0.35` update norm was not intrinsically disqualifying, but the faithful lane achieved better target/source tradeoff at lower update norms around `0.04–0.07`.
3. **Use Qwen3.5-2B-Base only as a calibration probe unless explicitly promoted.** OOM-safe CCE probes at `batch=1`, short sequence length are enough to answer whether LFM is uniquely high-gradient on the same text. Do not turn Qwen into the main benchmark lane by accident.
4. **If `bs16`-equivalent signal is needed, use grad accumulation or real memory cuts.** Checkpointing got `batch_size=8`, but the card remains effectively full; do not assume another batch-size rung exists.
5. **Throughput can wait, but remains a product lead.** Realistic-token Aurora amortization remains useful; it is just not the next cut unless the user explicitly reopens throughput. Current uncertainty is update-scale/source tradeoff under the faithful lane, not launch overhead.

## Next session contract

Start from the faithful-format 1k comparison and LR sweep, not from throughput. Residual-facing side, uniform rank 64, stable `eigh` basis init, CCE, faithful right-padded SYNTH batches, and Aurora remain the quality/diagnostic harness mainline. EOS-packed no-mask remains a throughput lane. The active unknown is now the exact source-tolerable point in the `1e-4–3e-4` LR neighborhood, not whether frequent refresh or packed batching fixes the issue.

Default harness model is `LiquidAI/LFM2.5-350M-Base`; stay there until the user explicitly authorizes going up. `LiquidAI/LFM2.5-1.2B-Base` remains a continuity/reference baseline. `Qwen/Qwen3.5-2B-Base` is a calibration model for gradient-scale/model-specific behavior, but use OOM-safe probes and do not let it become the main benchmark by cache convenience.

Minimum useful next session:

1. Run a narrower or longer `LiquidAI/LFM2.5-350M-Base` faithful `bs8 + activation_checkpointing` LR/update-scale check around `1e-4–3e-4`, with source eval and wandb cadence sensors.
2. Keep default `basis_refresh_interval=100` unless a targeted tracking metric proves refresh cadence is the bottleneck.
3. If more token mass is needed, choose grad accumulation or a memory-cut experiment explicitly; do not pretend `batch_size=16` is free on this card.
4. If comparing Qwen, keep it to OOM-safe CCE gradient calibration unless the user explicitly asks for a training run.
5. Do not spend on throughput, AdamW, topology proof, ECC, projected-gradient hooks, HeavyBall-vs-Aurora reruns, two-sided revival, schedule/budget archaeology, or rank-allocation horse races unless the user changes the question.

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
