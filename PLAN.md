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
- **Projected-activation backward is the first real peak-VRAM win.** Parameter hooks were too late. At LFM-350M `bs4 × seq1024`, the same-shape 10-step baseline without activation projection peaked at `~5.24 GB`; custom LFM MLP backward with activation-facing/storage-right bases lowered steady peak to `~3.99 GB`. Layer-staggered refresh brought the 200-step refresh-crossing MLP-only peak from `~5.26 GB` down to `~4.09 GB`. Extending the same `lfm` activation-projection mode to standard-attention q/k/v/out and short-conv in/out Linear boundaries lowered the 10-step peak further to `~3.80 GB` and the 200-step staggered peak to `~3.91 GB` while leaving FA/SDPA and causal-conv kernels untouched. Recomputing fused MLP `gate_pre`/`up` in backward instead of saving those full `[tokens, intermediate]` tensors lowered full `lfm`, no-checkpoint `bs4 × seq1024` peak to `~2.75 GB` for 10 steps and `~2.93 GB` across a 200-step refresh-crossing smoke. Tiny wrapper tests now show the emitted projected gradients equal ordinary full weight gradients projected by the stored right-side bases; quality differences are therefore about activation-facing/right-side geometry, not a basic backward-math bug. LFM2's advertised HF activation checkpointing was not actually wired through decoder layers; a narrow harness repair that checkpoints decoder layers drops full `lfm`, `bs8 × seq1024` one-step monitor peak from `~4.51 GiB` no-checkpoint / `~4.56 GiB` stock-checkpoint to `~1.61 GiB` while keeping queued projected grads around `27 MiB` and retained full grads around `0.1 MiB`. A corrected-LR 200-step run with repaired checkpointing peaked at `1.80 GB` allocated and reached target/source `1.861770 / 3.004450`, with slower `0.5709s` steps.
- **Repaired checkpointing moved the useful token-mass regime.** On LFM-350M residual-facing `off`, repaired checkpointing makes `bs32 × seq1024` a practical continuity lane. At 200 steps, LR `4e-4` reached target/source `1.770638 / 3.011879` at `~14.6k tok/s` and `~4.87 GB` allocated; LR `6e-4` improved target to `1.748466` but raised source to `3.034460`; LR `8e-4` barely improved target (`1.743787`) while source rose to `3.062698`. At the same token budget, `bs64` 100-step runs were not better: LR `4e-4` reached `1.805528 / 3.009495`, LR `6e-4` reached `1.766816 / 3.034950`, and used `~8.81 GB`. More tokens/update fits, but fewer optimizer updates still matter; `bs32` currently looks like the better quality/walltime point than `bs64` for this 350M SYNTH lane.
- **The faithful LR default is `2e-4`, not the old `0.0025` prior.** The rank-ablation/continuity lane explicitly used `--sumotrack-lr 2e-4`. Running current controls at the stale parser default `0.0025` inflated update norms by roughly `6×` and caused source-loss blowout. The harness default has been corrected to `2e-4`; do not trust quality comparisons that omit the LR unless they were run after that fix.

### What did not work

- **Gradient accumulation did not improve quality at equal token budget.** `grad_accum=2` reduced raw grad norm but lost target/source quality against the non-accumulated baseline.
- **First-step basis accumulation was a near-null.** A `4×` first-gradient basis estimate did not materially improve SYNTH or Sugar Quill, so the harness flag was removed.
- **Round-robin basis refresh did not buy peak-memory relief.** Burst stayed because it is simpler and matched peak VRAM.
- **Two-sided square-core projection did not beat one-sided rectangular updates.** Leave it in git history unless retention evidence specifically calls it back.
- **Cheap Aurora/NS cycles are not free.** `pp=1/ns=1` was too weak. Default remains `pp=2/ns=5` for quality continuity.
- **Qwen3.5-2B did not solve the local 12GB ceiling.** Activations/temporaries dominated after the fast short-conv path was available.
- **Parameter backward hooks do not solve peak VRAM.** Hook-time projection can clear retained full matrix `.grad` buffers and keep only tiny projected gradients, but PyTorch still materializes full weight gradients before parameter hooks run. On LFM-350M this removed `~548 MiB` of retained bf16 matrix grads yet barely moved peak at the faithful `bs4/bs8` shapes because activations/backward/loss temporaries dominate.

## Active leads

- **Build on projected activation by using checkpointing as a measured substrate, not a slogan.** At `2e-4`, repaired-checkpoint full `lfm` activation projection and a no-activation all-right control match closely at 200 steps, separating custom backward math from side-policy geometry. Under real layer checkpointing, projected activation still cuts peak from `~2.07 GB` to `~1.80 GB` at `bs8 × seq1024`, but eager wrappers slow steps from `~0.524s` to `~0.571s`. The next practical question is whether that extra headroom matters at larger shapes or can be recovered with low-risk wrapper/perf cleanup, not whether the math works.
- **Tune around the repaired-checkpoint `bs32` residual-facing lane.** The first larger-token LR sweep says `bs32 × seq1024`, residual-facing, no activation projection, repaired checkpointing, and LR around `4e-4–6e-4` is the strongest current 350M lane. `4e-4` is the conservative target/source balance; `6e-4` is the aggressive target side; `8e-4` is probably past the useful source tradeoff. A longer run around `4e-4`/`6e-4` is more product-shaped than pushing maximum batch size for its own sake.
- **Use `bs64` as a capacity tool, not the default assumption.** Full `lfm` projected activation with repaired checkpointing fits short `seq1024` smokes up to `bs80`, and residual-facing `off` fits `bs64` quality sensors. But at equal token budget/walltime, `bs64` underperformed `bs32` on target. Use `bs64` when memory headroom is needed for scale experiments or when a later quality curve proves it, not because bigger batches look heroic.
- **Extend refresh smoothing evidence beyond smokes.** Opt-in layer-staggered refresh preserved most of the memory win across a 200-step smoke. Next check longer quality runs and activation-checkpointing interaction before making it a default.
- **Compile after the checkpoint substrate is stable, but expect modest gains.** A repaired-checkpoint residual-facing `bs32`, LR `4e-4` compile smoke improved step time from `~2.248s` to `~2.144s` (`~5%`) and slightly lowered peak. Useful, but the main cost is real checkpoint recompute/model math, not Python alone.
- **Performance cleanup after the shape is fixed.** There may still be speedup in better batching, fewer synchronizations, less CPU/GPU memory shuttle, and tighter orthogonalization buckets. Measure first; do not optimize by vibes.
- **Scale transfer.** Test whether the established default survives a meaningful scale step while preserving the memory advantage, starting with `LiquidAI/LFM2.5-1.2B-Base` when memory-safe. Change one scale axis at a time.
- **Clean product-shaped data.** SYNTH proved low-noise mechanics. The next target distribution should be product-shaped but clean enough that loss means optimizer/data-fit behavior rather than data-quality archaeology.
- **HeavyBall-native final shape.** SumoTrack does **not** currently use HeavyBall ECC/param-ECC. The next engineering task is migrating the final fast implementation into `../HeavyBall` for compiled transforms, ECC/param-ECC, clipping, chainable optimizer machinery, and API compatibility. This repo remains the accessible experiment/testbed lane.
- **Matched-memory comparators.** Compare against alternatives users would actually run under the same memory pressure: LoRA/Unsloth-style adapters, GaLore/SubTrack-like low-rank-gradient routes, and AdamW only as a quality anchor where it fits.
