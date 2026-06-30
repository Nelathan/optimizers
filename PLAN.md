# SumoTrack Plan

SumoTrack is a memory-efficient optimizer line for high-capacity continued pretraining on consumer GPUs. The goal is usable distribution adaptation under memory pressure: move a pretrained model across a meaningful data shift while avoiding the ceilings imposed by full AdamW state, slow gradient accumulation, or adapter-only capacity.

This file is the live direction map: current claims, durable empirical facts, and next leads. Repo operating rules, harness defaults, coordinate conventions, and benchmark discipline live in `AGENTS.md`. Chronological experiment records live in `RESULTS.md`.

## Product thesis

For a fixed consumer-GPU memory budget, SumoTrack should deliver more useful model movement per byte and per second than adapter-only or low-rank-gradient methods when the user wants high-capacity distribution adaptation, not just style steering.

Relevant comparators are LoRA/Unsloth-style adapter training, GaLore, SubTrack, SUMO/Muon-family optimizers, and adjacent low-state full-finetuning methods. AdamW is a quality anchor where it fits, not the memory target.

Success is a Pareto point:

- enough trainable capacity to materially shift distribution,
- optimizer state small enough to preserve tokens/step on consumer hardware,
- stable updates that do not trash source behavior faster than they learn the target,
- step time that does not lose the memory gain to launch spray or memory shuttle,
- a configuration story a user can operate without LR roulette.

## Algorithm thesis

SumoTrack combines:

- SubTrack/GaLore-style one-sided gradient subspace tracking for memory-shaped state,
- projected first moments rather than full-size matrix moments,
- SUMO/Muon-style orthogonalization of the projected moment for update geometry,
- Aurora-style rectangular polar machinery for practical projected orthogonalization,
- Grassmann/Stiefel basis tracking for smooth adaptation of the active subspace.

For a tall matrix gradient `G ∈ R^{m×n}`:

```text
Q ∈ R^{r×n}
M_hat = G Qᵀ      # [m, r]
O_hat = aurora(M_hat)
update = O_hat Q  # [m, n]
```

For a wide matrix, SumoTrack uses the symmetric left-side form. The intended bias is: adapt along selected representation / hidden-state directions while distributing the update across the larger side with Muon/SUMO geometry.

## Current readout

The small-model faithful SYNTH lane has mostly done its job. It established the current shape: residual-facing one-sided projection, uniform rank, stable side-Gram `eigh`, burst refresh `100`, Aurora `pp=2/ns=5`, CCE, `bs8 + activation_checkpointing`, and an LR band around `1e-4–3e-4`.

Treat the 350M faithful SYNTH setup as a regression/continuity harness now, not the main discovery engine. Use it when a new code path, scale step, or comparator needs a known-good control.

## Durable empirical facts

### What worked

- **Projected first moments are the viable state shape.** Broad no-embedding LFM runs kept SumoTrack optimizer state around tens of MB (`~48.5 MB` on LFM-350M, `~89.9 MB` on LFM-1.2B) where AdamW used GB-scale state.
- **Orthogonalized projected momentum beats plain projected momentum in tested regimes.** Early matrix-only runs and later broad runs both favored SUMO/Muon-style geometry over no-orthogonalization at comparable state budget.
- **Aurora earned the forward path.** It first proved the leverage fix mechanically, then won a 1k broad LFM/SYNTH target run against HeavyBall NS at matched update/param and identical peak memory.
- **Residual-facing projection side beat shape-only side choice.** Transformer semantics mattered more than the smaller-side heuristic.
- **Uniform rank 64 is a good boring baseline, but rank is a real byte/quality lever.** Size/role/spectrum allocation policies did not earn their complexity; keep rank uniform. A later uniform sweep showed higher rank improved SYNTH target loss with roughly flat source cost, so tune rank before inventing allocation policy.
- **Stable side-Gram `eigh` replaced exact SVD for basis init.** It computes the one-sided subspace SumoTrack needs, was faster on all measured shapes, and had acceptable projection agreement.
- **Faithful SYNTH formatting changed the quality question.** Right-padded one-row batches with masked question/divider are the meaningful diagnostic lane. Packed SYNTH is throughput history, not retention evidence.
- **More tokens/update helped more than frequent refresh.** `bs8 + activation_checkpointing` improved signal; refresh interval `20` did not replace `100`.
- **Dirty Sugar Quill was stable but lower-contact.** Training on `Nelathan/synthetic-sugar-quill` with profile masked and text supervised made the model learn less and forget less at the same update scale. That is encouraging stability evidence, not a reason to tune dirty data now.

### What did not work

- **Gradient accumulation did not improve quality at equal token budget.** `grad_accum=2` reduced raw grad norm but lost target/source quality against the non-accumulated baseline.
- **First-step basis accumulation was a near-null.** A `4×` first-gradient basis estimate did not materially improve SYNTH or Sugar Quill, so the harness flag was removed.
- **Round-robin basis refresh did not buy peak-memory relief.** Burst stayed because it is simpler and matched peak VRAM.
- **Two-sided square-core projection did not beat one-sided rectangular updates.** Leave it in git history unless retention evidence specifically calls it back.
- **Cheap Aurora/NS cycles are not free.** `pp=1/ns=1` was too weak. Default remains `pp=2/ns=5` for quality continuity.
- **Qwen3.5-2B did not solve the local 12GB ceiling.** Activations/temporaries dominated after the fast short-conv path was available.

## Active leads

1. **Scale transfer.** Test whether the established default survives a meaningful scale step while preserving the memory advantage, starting with `LiquidAI/LFM2.5-1.2B-Base` when memory-safe. Change one scale axis at a time.
2. **Projected-gradient/backward hook path.** Can we project gradients as they appear during backward so full gradients do not need to remain stored? This needs tiny-model equivalence tests against `project(full_gradient)` before touching the main optimizer path, and it must preserve same-shape Aurora batching and basis-refresh semantics.
3. **HeavyBall-native final shape.** SumoTrack does **not** currently use HeavyBall ECC/param-ECC. The next engineering task is migrating the final fast implementation into `../HeavyBall` for compiled transforms, ECC/param-ECC, clipping, chainable optimizer machinery, and API compatibility. This repo remains the accessible experiment/testbed lane.
4. **Performance cleanup after the shape is fixed.** There may still be speedup in better batching, fewer synchronizations, less CPU/GPU memory shuttle, and tighter orthogonalization buckets. Measure first; do not optimize by vibes.
5. **Matched-memory comparators.** Compare against alternatives users would actually run under the same memory pressure: LoRA/Unsloth-style adapters, GaLore/SubTrack-like low-rank-gradient routes, and AdamW only as a quality anchor where it fits.
6. **Clean product-shaped data.** SYNTH proved low-noise mechanics. The next target distribution should be product-shaped but clean enough that loss means optimizer/data-fit behavior rather than data-quality archaeology.

## Stop conditions

Stop and rethink if:

- SumoTrack cannot exploit higher tokens/step or larger model scope than serious alternatives on the target GPU,
- orthogonalized one-sided updates improve target loss but cause unacceptable retention collapse,
- uniform rank clearly spends most state on low-leverage tensors,
- Grassmann tracking is too sluggish to adapt or too fast to smooth,
- rectangular orthogonalization quality becomes the bottleneck,
- the code starts serving old gates instead of the product thesis.

The product is not a checklist of optimizer features. The product is a usable path to high-capacity distribution adaptation under consumer-GPU constraints.
