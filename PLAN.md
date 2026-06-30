# SumoTrack Plan

SumoTrack is a memory-efficient optimizer line for high-capacity continued pretraining on consumer GPUs. The public optimizer class is `SumoTrack`.

The goal is not to make AdamW cheaper. The goal is usable distribution adaptation under memory pressure: move a pretrained model across a meaningful data shift while avoiding the ceilings imposed by full AdamW state, slow gradient accumulation, or adapter-only capacity.

Near-term target: a single RTX 5090-class workstation training Gemma 4 12B-class models with as many tokens per optimizer step as practical, no gradient accumulation if avoidable, high throughput, and enough update quality to actually change the model rather than merely flavor it.

## Documentation map

- `PLAN.md`: current direction, active invariants, and next cuts.
- `INSIGHTS.md`: durable empirical facts and local terrain.
- `RESULTS.md`: brief experiment records with setup, run links, metrics, and interpretation.
- `AGENTS.md`: operating contract for agents in this repo.
- `whiteflow/README.md` and `muonfactor/README.md`: historical optimizer sketches, not active SumoTrack specs.

## Product thesis

For a fixed consumer-GPU memory budget, SumoTrack should deliver more useful model movement per byte and per second than adapter-only or low-rank-gradient methods when the user wants high-capacity distribution adaptation, not just style steering.

Relevant comparators are LoRA/DoRA/QLoRA/Unsloth-style adapter training, GaLore, SubTrack, SUMO/Muon-family optimizers, and adjacent low-state full-finetuning methods. AdamW is a quality and sanity anchor, not the memory target.

Success is a Pareto point:

- enough trainable capacity to materially shift distribution,
- optimizer state small enough to preserve tokens/step on consumer hardware,
- stable updates that do not trash the source model faster than they learn the target,
- step time that does not lose the memory gain to launch spray or decompositions,
- a configuration story a serious user can operate without occult LR roulette.

## Main algorithm thesis

SumoTrack combines:

- SubTrack/GaLore-style one-sided gradient subspace tracking for memory-shaped state,
- projected first moments rather than full-size matrix moments,
- SUMO/Muon-style orthogonalization of the projected moment for update geometry,
- Aurora-style rectangular polar machinery for practical projected orthogonalization,
- Grassmann/Stiefel basis tracking for smooth adaptation of the active subspace.

The current mainline is Aurora-orthogonalized projected momentum. Treat that as the forward path, not as optional decoration around projected momentum.

For a tall matrix gradient `G ∈ R^{m×n}`:

```text
Q ∈ R^{r×n}
M_hat = G Qᵀ      # [m, r]
O_hat = aurora(M_hat)
update = O_hat Q  # [m, n]
```

For a wide matrix, SumoTrack uses the symmetric left-side form.

Interpretation:

> Adapt along selected representation / hidden-state directions, while distributing the update across the larger side with Muon/SUMO geometry.

This is a deliberate bias, especially for transformer MLP and attention matrices. The backbone-facing hidden-state axis carries stability; the expanded side has room to shape a useful manifold. The update is rectangular because the active subspace is one-sided, not because the optimizer accidentally became quadratic.

## Current invariants

- Public class is `SumoTrack`.
- Matrix parameters must not store full-size first moments.
- Matrix parameters must not store full-size second moments in the main path.
- Non-2D params use fallback AdamW semantics or are frozen by task policy.
- Fallback state is accounted separately from matrix projected state.
- Orthogonalization happens in projected space.
- Aurora with fixed Muon scale semantics is the projected direction.
- HeavyBall Newton-Schulz is an internal polar primitive inside Aurora.
- One-sided same-shape projected orthogonalization is bucketed.
- Grassmann basis tracking refreshes all bases on a burst cadence controlled by `basis_refresh_interval`.
- Stable side-Gram `eigh` is the quality/default basis-init path.
- Random basis init is an explicit ablation/stress path.
- Two-sided square-core projection has been removed from the active path.
- Unsupported ECC/param-ECC fails loudly until implemented through HeavyBall state hooks.

## Harness contract

Default quality/diagnostic route:

- model: `LiquidAI/LFM2.5-350M-Base`,
- scope: broad no embeddings,
- rank: uniform `64`,
- projection side: `residual-facing`,
- basis init: stable `eigh`,
- batching: faithful SYNTH right-padded no-mask,
- shape: `batch_size=4`, `seq_len=1024`,
- loss: CCE,
- attention: Transformers `sdpa`,
- basis refresh: burst every `100` steps,
- orthogonalization: Aurora `pp=2`, Newton-Schulz `ns=5`, pending a cleaner cycle-economy result.

Override only the axis being tested. Prefer:

```bash
uv run python experiments/llm_synth_smoke.py --measure-steps 200
```

over restating defaults such as `--optimizers sumotrack --rank 64 --projection-side-policy residual-facing --param-scope broad-no-embeddings`.

Important lanes:

- Faithful SYNTH right-padded no-mask is the quality, gradient-scale, and source-retention diagnostic lane.
- EOS-packed no-mask is the throughput / packed-LM lane.
- HF full-logits loss is an unoptimized ablation route, not the benchmark path.
- `LiquidAI/LFM2.5-1.2B-Base` is a continuity/reference model and should be named explicitly.
- `Qwen/Qwen3.5-2B-Base` is a calibration model for gradient-scale/model-specific behavior unless explicitly promoted.

## Coordinate convention

Optimizer side names use PyTorch weight-storage coordinates: `nn.Linear.weight == [out_features, in_features]`.

- MLP up/gate and attention q/k/v map the residual stream on the forward input side, storage `right` / columns.
- MLP down and attention output return to the residual stream on the output side, storage `left` / rows.

The harness policy is named `residual-facing` so callers do not need to think in storage coordinates.

## Evaluation philosophy

SYNTH is not a toy here. Clean, low-noise target data is useful for measuring adaptation because optimizer noise is less hidden by dataset noise. Ace the easy data before claiming robustness elsewhere.

The evaluation target is distribution adaptation with source behavior measured, not merely short-run loss descent.

Current faithful stream contract:

- target train: SYNTH train,
- target eval: held-out SYNTH,
- source eval: external pretraining-style corpus, currently the first parquet shard of `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` for small local runs.

If source loss improves during SYNTH adaptation, report that literally: this source shard did not expose forgetting. Do not call retention solved.

Input formatting is a first-class evaluation risk. Before treating an optimizer curve as meaningful after any format change, inspect row conversion, tokenizer BOS/EOS behavior, separators, decoded blocks, attention-mask policy, and absence of chat/template contamination. A base model that emits ordinary document/wiki prose while the harness trains on query + markdown reasoning traces is a format-contact fact, not decoration.

Gradient norms are also first-class sensors, but only with explicit definitions. Raw dense `.grad` norms are not optimizer update norms. Aurora, orthogonalization, and Muon scaling happen after raw gradients are formed. If raw norms look wrong, compare CCE to ordinary CE, break down by parameter scope/class, compare another model on the same text, then measure update/param.

Useful evaluation signals:

- target validation loss curve,
- source validation loss curve,
- exact decoded train/eval examples when stream format changes,
- raw gradient norm by stream/model/scope,
- update norm and update/param,
- peak allocated and reserved CUDA memory when scale changes,
- optimizer state bytes by matrix/fallback/total,
- basis rotation chordal when testing tracking,
- tokens/step and wall-clock tokens/sec only when throughput is the named question.

## Active leads

1. **Faithful `bs8 + activation_checkpointing`, LR `2e-4` as the current small-model quality baseline.** The broad 1k LR sweep put the useful knee around `1e-4–3e-4`; the direct `2e-4` lane is the current working baseline while other axes are tested.
2. **Retention/source validation under the faithful lane.** Aurora has target-movement evidence, but older packed SYNTH/source curves are not retention evidence. Meaningful retention runs need the explicit stream contract and decoded samples.
3. **Cycle economy without default churn.** The `2e-4` cycle sweep found `pp=1, ns=1` too cheap and made `pp=1, ns=2` interesting. Keep `pp=2, ns=5` for baseline continuity until a clean comparison earns a default change.
4. **Basis movement as a small sensor.** Keep chordal basis rotation. Do not revive projection-capture, leverage, or broad projected-residual telemetry unless a sharper failure mode demands them.
5. **Fallback policy under broad training.** Non-2D fallback state is small in LFM, but embeddings, norms, biases, and tiny conv kernels need a principled train/freeze/fallback policy before Gemma-scale claims.
6. **State and resume invariants.** Keep tests for projected-state shape, fallback accounting, bf16 behavior, and mixed state-dict resume.
7. **HeavyBall-native path only after geometry earns it.** Move hot paths toward HeavyBall transforms/ECC when Aurora + tracking + retention look worth productizing.
8. **Projected-gradient hooks remain isolated and late.** Avoiding full-gradient materialization may be a real memory frontier, but it needs equivalence tests against `project(full_gradient)` on tiny models before touching the main optimizer.
9. **Backward-time gradient release is a future memory frontier.** Once `G_hat = project(G)` is materialized, SumoTrack could release the full gradient for that parameter, but hook timing must preserve same-shape Aurora batching and basis-refresh semantics.
10. **First-basis quality from accumulated gradients.** Test “initialize `eigh` bases from several microbatches, then train normally” separately from ordinary gradient accumulation.

## Conditional leads

- Role/depth refresh cadence: test only after basis-motion diagnostics suggest cadence matters.
- Two-sided square-core projection: recover from git only if one-sided Aurora shows retention damage that square-core geometry plausibly addresses.
- Stochastic rounding: possible low-complexity bf16 stability win, but only with a direct stability/quality comparison.
- Nesterov, perpendicular recovery, extra momentum knobs: hold unless a named failure mode demands them.
- Throughput: still product-relevant, but not the next default cut while faithful-lane quality/source behavior is the active unknown.

## Maintenance rules

- Keep `experiments/llm_synth_smoke.py` a lean lab harness, not a trainer framework.
- Keep `RESULTS.md` as a brief experiment log. Put exhaustive command/output detail in external logs when needed.
- Prefer deleting unsupported branches over keeping dormant flags. Preserve old experiments in git/docs, not in public code paths.
- Add tests for policy defaults that encode design decisions, especially residual-facing side grouping and uniform-rank reporting.
- Keep experiment language faithful. “Source eval” means an evaluation stream, not training data.
- Do not optimize kernel launches before the algorithmic shape is worth optimizing.

Performance ladder, once geometry and retention justify productization:

1. Measure realistic-token-step amortization of bucketed projected orthogonalization overhead.
2. Tighten bucket implementation only where measured overhead remains material.
3. Move compatible pieces toward HeavyBall-native transforms.
4. Add ECC/param-ECC.
5. Consider projected-gradient hooks only after ordinary-gradient SumoTrack has proven worth.

## Next session contract

Start from the faithful-format 1k comparison, LR sweep, direct `2e-4` baseline, grad-accumulation check, and basis-rotation cleanup. Do not drift back to throughput, topology proof, or metric sprawl unless the user changes the question.

Mainline for quality/diagnostic work: residual-facing side, uniform rank `64`, stable `eigh` basis init, CCE, faithful right-padded SYNTH batches, SDPA attention, targeted tensor-kernel compile when useful, and Aurora. EOS-packed no-mask remains a throughput lane.

Minimum useful next work:

1. Use `--torch-compile` for future 1k small-model quality runs unless it changes the experiment contract; it compiles model forward/backward and SumoTrack tensor kernels, not Python optimizer bookkeeping.
2. Keep `basis_refresh_interval=100` unless basis-rotation telemetry proves refresh cadence is the bottleneck.
3. Keep wandb to target/source validation loss, train loss, raw grad norm, update norm, update/param, and basis rotation chordal.
4. If comparing Qwen, keep it to OOM-safe CCE gradient calibration unless the user explicitly asks for a training run.
5. Do not spend on throughput, AdamW, topology proof, ECC, projected-gradient hooks, HeavyBall-vs-Aurora reruns, two-sided revival, schedule/budget archaeology, or rank-allocation horse races unless the user changes the question.

Falsifier: if Aurora's target advantage disappears when source/retention is measured, or if side/rank wins vanish at longer horizons, revisit the direction map and allocation.

## Stop conditions

Stop and rethink if:

- SumoTrack cannot exploit higher tokens/step or larger model scope than serious alternatives on the target GPU,
- orthogonalized one-sided updates improve target loss but cause unacceptable retention collapse,
- uniform rank clearly spends most state on low-leverage tensors,
- Grassmann tracking is too sluggish to adapt or too fast to smooth,
- rectangular orthogonalization quality becomes the bottleneck,
- the code starts serving old gates instead of the product thesis.

The product is not a checklist of optimizer features. The product is a usable path to high-capacity distribution adaptation under consumer-GPU constraints.
