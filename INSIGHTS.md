# SumoTrack Insights

Durable lessons only. Current direction, defaults, and next work live in `PLAN.md`; run details live in `RESULTS.md`.

## What worked

- **Projected first moments are the viable state shape.** Broad no-embedding LFM runs kept SumoTrack optimizer state around tens of MB (`~48.5 MB` on LFM-350M, `~89.9 MB` on LFM-1.2B) where AdamW used GB-scale state. The main product lever is real.
- **Orthogonalized projected momentum beats plain projected momentum in the tested regimes.** Early matrix-only runs and later broad runs both favored SUMO/Muon-style geometry over no-orthogonalization at comparable state budget.
- **Aurora earned the forward path.** It first proved the leverage fix mechanically, then won a 1k broad LFM/SYNTH target run against HeavyBall NS at matched update/param and identical peak memory. Treat rectangular leverage-uniform projected orthogonalization as useful, not cosmetic.
- **Residual-facing projection side beat shape-only side choice.** In the side-policy check, residual-facing beat `AUTO` at equal state budget. Transformer semantics mattered more than the smaller-side heuristic.
- **Uniform rank 64 is good enough as the boring baseline.** Size/role/spectrum rank ideas did not earn their complexity. Initial gradient-spectrum rank shrank during training, so first-step spectrum is a conservative sensor, not a default allocation policy.
- **Stable side-Gram `eigh` replaced exact SVD for basis init.** It computes the one-sided subspace SumoTrack needs, was faster on all measured shapes, and had acceptable projection agreement. Full SVD was ceremony for this contract.
- **Faithful SYNTH formatting changed the quality question.** Right-padded one-row batches with masked question/divider are the meaningful diagnostic lane. The older packed stream was useful for throughput but not for retention or clean source-behavior claims.
- **More tokens/update helped more than more frequent refresh.** The `bs8 + activation_checkpointing` lane improved the small-model signal; refresh interval `20` did not earn replacing the default `100`.
- **The useful LR/update-scale knee under the faithful lane is much lower than the old packed/random-init habit.** The current small-model quality baseline belongs around `1e-4–3e-4`, with `2e-4` as the working lane.
- **Raw gradient norms are a stream/model sensor, not an optimizer verdict.** CCE and ordinary CE agreed; LFM had much higher raw dense gradients than Qwen on the same text. Judge SumoTrack by update/param and target/source curves, not raw `.grad` panic.

## What did not work

- **Packed SYNTH curves did not answer retention.** They trained on a malformed/unfaithful stream with row-boundary ambiguity. Keep those runs as plumbing/throughput history only.
- **Round-robin basis refresh did not buy peak-memory relief.** Equal-cadence burst and round-robin had the same peak VRAM; burst stayed because it is simpler.
- **Two-sided square-core projection did not beat one-sided rectangular updates.** It was stable and nearly state-neutral, but weaker on immediate target movement. Leave it in git history unless retention evidence specifically calls it back.
- **Gradient accumulation did not improve quality at equal token budget.** `grad_accum=2` reduced raw grad norm but lost target/source quality against the non-accumulated baseline. Use accumulation only when memory/token budget forces it.
- **Cheap Aurora/NS cycles are not free.** `pp=1, ns=1` was too weak. `pp=1, ns=2` remains interesting, but the conservative default stays until a cleaner comparison earns a change.
- **Qwen3.5-2B did not solve the local 12GB ceiling.** The fast short-conv path helped plumbing, but batch/sequence limits were still dominated by activations/temporaries rather than optimizer state.

## Standing interpretation

SumoTrack's promising shape is: one-sided residual-facing projected first moments, stable `eigh` basis init, Grassmann burst tracking, Aurora/Muon projected direction, uniform rank 64, CCE, and faithful SYNTH diagnostics. The unresolved product question is not whether the optimizer can move target loss cheaply; it is whether that movement preserves enough source behavior at useful scale.
