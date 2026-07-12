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

The small-model faithful SYNTH lane is the current clean product-shaped dataset lane for optimizer and algorithm work. Repaired LFM2 checkpointing reopened the useful token/rank regime without changing that target contract. The current 350M mainline is residual-facing one-sided projection, stable side-Gram `eigh`, burst refresh `100`, Aurora `pp=2/ns=5`, CCE, repaired activation checkpointing, LR warmup `50`, `adafactor_ema` projected moment (Adafactor row/col factored second-moment dampening on the full gradient before basis refresh/projection, feeding a `beta=0.9` first-moment EMA), a wide/loose safety-only projected-grad clip (`norm=2000`, ratio disabled) sized for that mode's native scale, and larger rank/token settings than the old `bs8/rank64` continuity lane.

Treat the 350M faithful SYNTH setup as both a regression harness and the current optimizer-shape tuner. Use it when a new code path, rank/token setting, performance change, or comparator needs a known-good control. Do not casually move to another model or dataset: that resets the ablation stack and changes the question.

### Current default lane (copy-pasteable)

The CLI's own flag defaults already match this lane on every axis except batch size, rank, LR, and checkpointing (see table below). Per `AGENTS.md`, override only the axis under test — the block below is the full best-candidate shape, not a template to restate for one-axis ablations:

```bash
uv run python experiments/llm_synth_smoke.py \
  --batch-size 16 --rank 256 --sumotrack-lr 3e-4 \
  --activation-checkpointing --measure-steps 1000
```

Divergences from CLI defaults, and why:

| flag | CLI default | this lane | why |
| --- | --- | --- | --- |
| `--batch-size` | `4` | `16` | more tokens/optimizer-update outperformed maximizing batch size alone (see Durable facts) |
| `--rank` | `64` | `256` | rank256 is the current best 350M candidate, not the old `bs8/rank64` continuity lane |
| `--sumotrack-lr` | `2e-4` | `3e-4` | tuned for the rank256/bs16 shape; `4e-4`/`6e-4` hurt source too much for `ema`, and the same held for `adafactor_ema` at `6e-4` (see moment-mode finding below) |
| `--activation-checkpointing` | off | on | repaired LFM2 checkpointing is what reopened this token/rank regime under memory pressure |
| `--measure-steps` | `3` | `1000` | default `3` is a fast correctness smoke, not a training run; use `1000` only when the question needs a real convergence signal |

Everything else (`param-scope`, `seq-len`, `lr-warmup-steps`, `beta`, `moment-mode`, `adafactor-beta2`, `projection-side-policy`, `basis-init`, `basis-refresh-interval/schedule`, `projected-grad-clip-norm/ratio`) is already the CLI default — do not restate it. `moment-mode` defaults to `adafactor_ema` as of commit `db62ca2` (see Durable facts); `--moment-mode ema` remains available as the pre-ablation comparator but needs `--projected-grad-clip-norm 2.0 --projected-grad-clip-ratio 6.0` to match its previously-tuned clip scale, since the shared defaults now target `adafactor_ema`'s native ~1800-2000 projected-grad scale.

## Durable empirical facts

### What worked

- **Projected first moments are the viable state shape.** Broad no-embedding LFM runs kept SumoTrack optimizer state around tens of MB (`~48.5 MB` on LFM-350M, `~89.9 MB` on LFM-1.2B) where AdamW used GB-scale state.
- **Orthogonalized projected momentum beats plain projected momentum in tested regimes.** Early matrix-only runs and later broad runs both favored SUMO/Muon-style geometry over no-orthogonalization at comparable state budget.
- **Aurora earned the forward path.** It first proved the leverage fix mechanically, then won a 1k broad LFM/SYNTH target run against HeavyBall NS at matched update/param and identical peak memory.
- **Residual-facing projection side beat shape-only side choice.** Transformer semantics mattered more than the smaller-side heuristic.
- **Uniform rank remains the policy; rank 64 is no longer the best default candidate.** Size/role/spectrum allocation policies did not earn their complexity; keep rank uniform. Earlier rank sweeps already showed rank 256 as the quality winner with acceptable state. Repaired checkpointing and larger token batches make that more relevant: rank64 shows capacity/refresh symptoms, while rank256 gives smoother train loss and stronger target movement. At LFM-350M broad no-embeddings, rank256 matrix state is `191,889,408` bytes, about `0.334×` one full bf16 moment and `0.167×` two full bf16 moments, so it remains firmly in the memory-saving regime.
- **Stable side-Gram `eigh` replaced exact SVD for basis init.** It computes the one-sided subspace SumoTrack needs, was faster on all measured shapes, and had acceptable projection agreement.
- **Faithful SYNTH formatting changed the quality question.** Right-padded one-row batches with masked question/divider are the meaningful product-shaped diagnostic lane for this lab: clean enough that loss reflects optimizer/data-fit behavior rather than data archaeology, structured enough to test real distribution movement. Packed SYNTH is throughput history, not retention evidence.
- **More tokens/update helped more than frequent refresh.** `bs8 + activation_checkpointing` improved signal; refresh interval `20` did not replace `100`. In the current repaired-checkpoint rank256 lane, interval `20` overlapped interval `100` through step 300 (`1.725838 / 3.036959` vs interval100 `1.725741 / 3.040125`) and was stopped rather than spending more compute on basis churn.
- **Dirty Sugar Quill was stable but lower-contact.** Training on `Nelathan/synthetic-sugar-quill` with profile masked and text supervised made the model learn less and forget less at the same update scale. That is encouraging stability evidence, not a reason to tune dirty data now.
- **Projected-activation backward is the first real peak-VRAM win.** Parameter hooks were too late. At LFM-350M `bs4 × seq1024`, the same-shape 10-step baseline without activation projection peaked at `~5.24 GB`; custom LFM MLP backward with activation-facing/storage-right bases lowered steady peak to `~3.99 GB`. Layer-staggered refresh brought the 200-step refresh-crossing MLP-only peak from `~5.26 GB` down to `~4.09 GB`. Extending the same `lfm` activation-projection mode to standard-attention q/k/v/out and short-conv in/out Linear boundaries lowered the 10-step peak further to `~3.80 GB` and the 200-step staggered peak to `~3.91 GB` while leaving FA/SDPA and causal-conv kernels untouched. Recomputing fused MLP `gate_pre`/`up` in backward instead of saving those full `[tokens, intermediate]` tensors lowered full `lfm`, no-checkpoint `bs4 × seq1024` peak to `~2.75 GB` for 10 steps and `~2.93 GB` across a 200-step refresh-crossing smoke. Tiny wrapper tests now show the emitted projected gradients equal ordinary full weight gradients projected by the stored right-side bases; quality differences are therefore about activation-facing/right-side geometry, not a basic backward-math bug. LFM2's advertised HF activation checkpointing was not actually wired through decoder layers; a narrow harness repair that checkpoints decoder layers drops full `lfm`, `bs8 × seq1024` one-step monitor peak from `~4.51 GiB` no-checkpoint / `~4.56 GiB` stock-checkpoint to `~1.61 GiB` while keeping queued projected grads around `27 MiB` and retained full grads around `0.1 MiB`. A corrected-LR 200-step run with repaired checkpointing peaked at `1.80 GB` allocated and reached target/source `1.861770 / 3.004450`, with slower `0.5709s` steps.
- **Repaired checkpointing moved the useful token-mass regime.** On LFM-350M residual-facing `off`, repaired checkpointing makes `bs32 × seq1024` a practical continuity lane. At 200 steps, LR `4e-4` reached target/source `1.770638 / 3.011879` at `~14.6k tok/s` and `~4.87 GB` allocated; LR `6e-4` improved target to `1.748466` but raised source to `3.034460`; LR `8e-4` barely improved target (`1.743787`) while source rose to `3.062698`. At the same token budget, `bs64` 100-step runs were not better: LR `4e-4` reached `1.805528 / 3.009495`, LR `6e-4` reached `1.766816 / 3.034950`, and used `~8.81 GB`. More tokens/update fits, but fewer optimizer updates still matter; `bs32` currently looks like the better quality/walltime point than `bs64` for this 350M SYNTH lane.
- **Rank256 plus smaller batches is the current best 350M default candidate.** A rank256 `bs32`, LR `6e-4`, 500-step interrupted/partial run produced strong smooth target movement (`1.749001 → 1.699121` from step 100 to 500) but high source (`3.102894` at step 500), consistent with “more signal and too much update scale.” A completed rank256 `bs16`, LR `3e-4`, 1k run reached target/source `1.683542 / 3.076128`, train loss `1.670848`, state `192.4 MB`, peak `3.23 GB`, and `~14.2k tok/s`. Adding per-matrix projected-gradient clipping at `2.0` improved the same shape to `1.675010 / 3.059717`, train `1.660258`, with mean pre-clip max projected tensor norm `1.088935` and unchanged state/memory class. Extending that to dual-rail clipping (`norm=2`, projected-grad/moment ratio probe `5`) reached `1.674212 / 3.059725`, train `1.660581`, mean p90 ratio `4.693210`, and the same state/memory class. Ratio `5` did not move the loss needle but was a little tight in telemetry; harness default is ratio `6` for breathing room. Adding LR warmup `50` to the 200-step dual-rail sensor improved source/pressure clearly (`1.756553 / 3.017035`, update norm `0.088776`, mean p90 ratio `3.918923`) versus the no-warmup 200-step lane (`1.747635 / 3.022103`, update norm `0.097216`, p90 ratio `4.397594`), while target lag is expected from the first 50 throttled steps. LR warmup is accepted default hygiene; it does not need a separate 1k convergence proof unless a future baseline run already needs that shape. This suggests rank capacity and optimizer-update count matter more than simply maximizing batch size.
- **Grassmann refresh is now scale-controlled like basis init.** Basis init already normalized the full gradient before side-Gram `eigh`; Grassmann refresh now normalizes the spectral input before tangent formation too. Unit tests assert right- and left-side refreshes are invariant to multiplying the refresh gradient by `1000×`. A rank256 `bs16`, LR `3e-4`, 200-step smoke after the change reached target/source `1.748007 / 3.020345`, essentially matching the previous same-lane step200 `1.748073 / 3.022887`; longer-run history showed a small quality improvement and healthy chordal behavior. Faster subspace convergence from lower refresh interval did not translate into a loss win once rank was high enough, so there is no active refresh-rate lead.
- **Projected-gradient clipping belongs after projection, before the projected EMA.** A stressed rank64 `bs32`, LR `4e-4` lane showed raw/projected gradient spikes without a chordal/basis event: one diagnostic run hit pre-clip projected tensor max `109486`, raw grad norm `773670`, and then poisoned the projected moment tail. Per-matrix norm clipping catches absolute blowouts, but tensor norms are not homogeneous across the model. The better health signal is projected-grad norm relative to that tensor's pre-update projected moment. The harness now uses dual rails: absolute norm cap `2` plus p90-observed ratio cap `6`, both applied after projection and before the EMA. A tighter ratio `5` was a useful pressure-test rail and did not move the 1k rank256 loss needle; ratio `6` matches the observed “warm but bounded” p90 telemetry better. Do not clip raw full gradients before projection.
- **High beta is not a current tuning path.** A `beta=0.98`, dual-rail rank256 run was killed after step 300: step100 was already bad (`1.825120 / 3.134735`), p90 projected-grad/moment ratio started around `47×`, and later rows climbed into thousands with a step194 spike at p90 ratio `30933`. Disabling ratio clipping did not rescue beta `0.98`: the no-ratio control was source-bad from step100 (`1.804987 / 3.094521`) and still worse through step400 (`1.715256 / 3.104252`) than beta `0.9`. A running-average beta warmup backtest (`beta_t=min(beta,t/(t+1))`) plus LR warmup `50`, norm clip `2`, and no ratio rail also failed: `1.779715 / 3.182923` at step200 with final p90 ratio `10.33`. Slow projected moments plus Aurora/Muon remain too laggy under high beta. The beta-schedule branch was removed; keep beta `0.9` now.
- **Fp32 projected moments did not earn the memory cost in the first sensor.** Migrating `projected_exp_avg` to fp32 while keeping bases at param dtype raised rank256 LFM-350M matrix state from `191,889,408` to `335,544,320` bytes. The matching 200-step `bs16`, LR `3e-4` smoke reached target/source `1.748452 / 3.023449`, slightly worse than the bf16-moment normalized-refresh reference `1.748007 / 3.020345`. Reverted to bf16 projected moments; revisit only if a longer high-beta run shows accumulator precision symptoms.
- **The faithful LR default is `2e-4`, not the old `0.0025` prior.** The rank-ablation/continuity lane explicitly used `--sumotrack-lr 2e-4`. Running current controls at the stale parser default `0.0025` inflated update norms by roughly `6×` and caused source-loss blowout. The harness default has been corrected to `2e-4`; do not trust quality comparisons that omit the LR unless they were run after that fix.
- **Adafactor-dampened gradient feeding a real EMA (`adafactor_ema`) beats plain `ema` on both target and source, not just a tradeoff.** A five-arm moment-mode ablation at the rank256/bs16/LR3e-4/warmup50/checkpointed default-candidate lane tested: `ema` (plain first-moment EMA, the old default), `none` (no moment, clipped projected grad straight into Aurora), `second_moment` (RMSprop-style elementwise `v`-dampened grad replacing the moment slot, no EMA), `adafactor` (Adafactor-style row/col factored second-moment dampening on the full gradient before basis refresh/projection, replacing the moment slot, no EMA), and `adafactor_ema` (same Adafactor dampening, but still feeding a `beta=0.9` EMA instead of replacing the slot). `none`/`second_moment`/`adafactor` all lose target speed for retention monotonically with how much first-moment smoothing they remove (200-step target/source: `ema` `1.756/3.011`; `none` `1.823/2.986`; `second_moment` `1.840/2.965`; `adafactor` `1.847/2.954`) — the *mechanism* (v vs. factored row/col) barely matters once `m` is gone; only its absence does. `adafactor_ema` breaks that pattern: at 1000 steps, matched LR/rank/shape against the existing `ema` 1k reference (`1.683542/3.076128`), `adafactor_ema` reached `1.671/3.022` — better target *and* better source simultaneously, at unchanged state/memory/throughput class (`193.4 MB` state, `3.25 GB` peak, `14.8k tok/s`). A `beta2` sweep (`0.95` vs `0.99`) showed Adafactor's factorization is insensitive to that range (identical loss/alignment/erank both ways); `0.99` is the accepted default. Pushing `adafactor_ema`'s LR to `6e-4` (matching the standard LR-pressure pattern already documented for `ema`) reproduced the same known tradeoff — faster target, worse source (`1.699/3.084` at 500 steps) — confirming better Aurora conditioning does not exempt an arm from the usual LR/source cost; `3e-4` remains the well-behaved point, not a hotter LR. New `aurora_alignment`/`aurora_erank` diagnostics (Frobenius cosine similarity between the pre-Aurora tensor and Aurora's orthogonalized output; spectral effective rank of the pre-Aurora tensor via `eigh` on the smaller-side Gram) were added because `update_norm` is flat across every arm by construction — Newton-Schulz/polar orthogonalization normalizes the output's scale away regardless of input quality, so it cannot distinguish a healthy moment from a noisy one. Read final-step values, not the mean over a run: alignment/erank climb during warmup, so the run-mean underestimates steady-state conditioning. `beta=0.98`'s known rejection (high moment beta) is confirmed by source loss alone (`3.118` vs `3.011` at 200 steps); its alignment/erank end up close to `beta=0.9`'s by the final step (both `~0.586`), so the failure is a worse *trajectory*/convergence property, not a persistently different final conditioning. `moment_mode` is now `adafactor_ema` by default (constructor and CLI); `ema` remains available as the pre-ablation comparator. The losing arms (`none`, `second_moment`, plain `adafactor`) were stripped from the live tree after this finding landed; the full ablation code (all five modes, both diagnostics) is preserved at commit `db62ca2f202aa3214f66b97b5abc0d8618c35c93` for reproduction.

### What did not work

- **Gradient accumulation did not improve quality at equal token budget.** `grad_accum=2` reduced raw grad norm but lost target/source quality against the non-accumulated baseline.
- **First-step basis accumulation was a near-null.** A `4×` first-gradient basis estimate did not materially improve SYNTH or Sugar Quill, so the harness flag was removed.
- **Round-robin basis refresh did not buy peak-memory relief.** Burst stayed because it is simpler and matched peak VRAM.
- **Two-sided square-core projection did not beat one-sided rectangular updates.** Leave it in git history unless retention evidence specifically calls it back.
- **Cheap Aurora/NS cycles are not free.** `pp=1/ns=1` was too weak. Default remains `pp=2/ns=5` for quality continuity.
- **Qwen3.5-2B did not solve the local 12GB ceiling.** Activations/temporaries dominated after the fast short-conv path was available.
- **Parameter backward hooks do not solve peak VRAM.** Hook-time projection can clear retained full matrix `.grad` buffers and keep only tiny projected gradients, but PyTorch still materializes full weight gradients before parameter hooks run. On LFM-350M this removed `~548 MiB` of retained bf16 matrix grads yet barely moved peak at the faithful `bs4/bs8` shapes because activations/backward/loss temporaries dominate.

## Active leads

- **Current default-candidate lane is clean enough to use as the baseline.** The old `bs8/rank64` lane is continuity history. The best current 350M shape is rank256 with enough optimizer updates, not maximum batch size. Current best candidate is `bs16 × seq1024`, rank256, LR `3e-4`, LR warmup `50`, repaired checkpointing, residual-facing, burst refresh `100`, beta `0.9`, `moment_mode=adafactor_ema` (Adafactor row/col dampening feeding the EMA, `adafactor_beta2=0.99`), and a wide safety-only projected-gradient clip (`norm=2000`, ratio off) sized for that mode's native scale. This supersedes the earlier `ema`-only/dual-rail (`norm=2`,`ratio=6`) description: `adafactor_ema` beat plain `ema` on both target and source at this exact shape (see Durable facts). Do not spend a 1k run merely to prove warmup: warmup is accepted industry-standard hygiene and the 200-step source/pressure result was decisive enough. Run this lane long only when a baseline/control is needed for another claim.
- **Moment-mode is a closed ablation, not an open lever.** `adafactor_ema` won outright against `ema`/`none`/`second_moment`/`adafactor` at the default-candidate shape; do not reopen this question without a new, concrete reason (a different model/scale, a different rank, or a specific failure mode in `adafactor_ema`). If the ablation needs to be rerun or extended, the full five-mode code plus the `aurora_alignment`/`aurora_erank` diagnostics live at commit `db62ca2f202aa3214f66b97b5abc0d8618c35c93`; do not rebuild it from scratch.
- **LR/source balance is the only obvious small tuning lead left in the mainline.** High-rank updates should move target down while source rises; do not pretend both can always improve. The current `3e-4` rank256 lane may be a little hot, `4e-4` hurt source too much in the pressure sensor, and `2e-4` has not yet been tested in the repaired-checkpoint rank256 dual-rail/warmup lane. A narrow `2e-4` versus current `3e-4` check is plausible if the question is source/target tradeoff. Do not reopen a broad LR sweep without a new reason.
- **Grassmann subspace tracking is an active sidequest — see [`SUBSPACE_TRACKING.md`](SUBSPACE_TRACKING.md), which is run as a QUESTION LEDGER, not a task list.** This work answers questions, not ships features; the discipline is never losing a question and updating each answer as evidence arrives (different from software engineering — read the doc's ledger section). Settled this pass: the QR retraction was replaced with a faithful Grassmann geodesic port of SubTrack's `track_the_subspace`; a real frame bug was found and fixed (tangent projected off the wrong side of the basis → non-orthonormal rotation frame; passing tests had only exercised the no-op regime); rotation is now **k=1 (drift), confirmed against SubTrack source** — full-spectrum rotation was an unacknowledged divergence and is gone (measured: ~30× the rotation of k=1, churns the basis, buys no loss); the motion diagnostic is now the honest **`rotation_angle` in radians** (superseded `rotation_energy`=`Σ|sin|`, which aliased a wrapping rotation as small); `step_size = 5` chosen (sweep closed — no breaking point in range, σ self-anneals, adafactor keeps σ stable). **Key open question (Q5): does basis quality bind on loss at all at these scales?** Both the drift-vs-spin and rank-32 sweeps showed *train* loss unmoved while subspace metrics moved a lot — but no *eval* loss yet, bs8 not bs16, ≤200 steps, so it's a soft-negative not settled. The next step is an honest **baseline (eigh, rank32, step5, bs16, eval on, 200 steps)** as the ranking reference. The real open work is the **accumulation design space** (a lattice of candidate forms: sum vs EMA of the tangent, rank-1 vs rank-n reductions, reset-buffer vs transported EMA, and rotation-averaging on the manifold via Karcher) — to be *evaluated across*, not collapsed to one arm. All axes, questions, evaluation methods, and current evidence live in the sidequest doc. Interval `20` still did not beat `100` on loss at high rank — that older finding stands.
- **Leave projected moments bf16 unless a specific failure reopens the question.** The first fp32 `projected_exp_avg` migration did not improve the 200-step default-candidate sensor and cost `~144 MB` extra matrix state at rank256. Beta `0.98` failed because the ratio denominator was under-calibrated, not because bf16 moments obviously lacked precision. Basis storage should also remain param dtype: init/refresh already use fp32 work internally, while fp32 bases would double basis memory/matmul traffic and answer a different question.
- **Projected-gradient Linear is a correctness primitive, not the speed path.** At `2e-4`, repaired-checkpoint full `lfm` activation projection and a no-activation all-right control matched closely at 200 steps, separating custom backward math from side-policy geometry. Under real layer checkpointing, projected activation cut peak from `~2.07 GB` to `~1.80 GB` at rank64 `bs8 × seq1024`, but eager wrappers slowed steps from `~0.524s` to `~0.571s`. Compiling pure projected-backward tensor helpers later made short rank64 repaired-checkpoint smokes smaller and modestly faster, but audit found the initial `torch.compile(model)` path missed the CCE transformer hot path because CCE unwraps the CausalLM wrapper. The full all-right rank256 path was not a speed win once true-hot-path compile and fixed measurement were used. The current cleanup restores residual-facing side policy for wrapped params and makes the generic projected `Linear` side-aware: right side saves projected activations; left side projects `grad_out` in custom backward and avoids full `weight.grad`, but has different memory economics. Fair rank256 `bs64 × seq1024` smokes with expandable CUDA segments made the call: baseline `off` ran `20,089 tok/s`, `3.262s/step`, peak `8.42 GB`; side-aware `lfm` ran `14,564 tok/s`, `4.500s/step`, peak `9.56 GB`. End the per-Linear try. The side-aware fused MLP primitive was then built and run through its pre-registered synthetic microbench gate at LFM-350M MLP shapes (`hidden=1024`, `intermediate=6656`, T=16k/64k, rank 64/256, bf16): fp64-exact math, step time tied with the compiled ordinary baseline within noise, but peak memory lost to it at every shape because the down-left contraction keeps ~5 `[T, intermediate]` transients alive while Inductor's own save/recompute scheduling on the plain baseline does better. The gap is structural, not a fixable schedule. Projected-gradient backward is now a closed speed lane at 350M shapes: compiled ordinary backward plus repaired checkpointing is the performance path, and the fused/side-aware primitives remain tested correctness artifacts only.
- **Use `bs64` as a capacity tool, not the default assumption.** Full `lfm` projected activation with repaired checkpointing fits short `seq1024` smokes up to `bs80`, and residual-facing `off` fits `bs64` quality sensors. But at equal token budget/walltime, `bs64` underperformed `bs32` on target. Use `bs64` when memory headroom is needed for scale experiments or when a later quality curve proves it, not because bigger batches look heroic.
- **Compile should be normal policy for expensive 1k runs.** A repaired-checkpoint residual-facing `bs32`, LR `4e-4` compile smoke improved step time from `~2.248s` to `~2.144s` (`~5%`) and slightly lowered peak. It is not a strategy by itself, but long quality runs should use compile unless the run is explicitly measuring eager behavior or compile breaks the contract. Record compile state in results.
- **Performance cleanup is the main near-term work.** The optimizer-quality lane is no longer mushy: rank256/residual-facing/checkpointed/warmup/dual-rail is a coherent baseline. Remaining high-value work is measured speed and memory work: compile coverage, fewer synchronizations, better batching, less CPU/GPU memory shuttle, tighter orthogonalization buckets, and projected-activation overhead removal. Measure first; do not optimize by vibes.
- **Do not scale-transfer yet.** Moving from LFM-350M to LFM-1.2B or another model resets the ablation stack. Save scale transfer for after the local implementation/performance story is sharper or when the explicit question is transfer.
- **SYNTH remains the clean product-shaped dataset lane.** Do not replace it because it is “synthetic.” It already demonstrated target-format adoption across ranks, with rank256 changing language and structure more deeply than low-rank style-cover behavior. New product data can be useful later, but it is not the next optimizer lead.
- **HeavyBall-native final shape.** SumoTrack does **not** currently use HeavyBall ECC/param-ECC. The next engineering task is migrating the final fast implementation into `../HeavyBall` for compiled transforms, ECC/param-ECC, clipping, chainable optimizer machinery, and API compatibility. This repo remains the accessible experiment/testbed lane.
- **Matched-memory comparators are release evidence, not the next local optimization task.** Compare against alternatives users would actually run under the same memory pressure: LoRA/Unsloth-style adapters, GaLore/SubTrack-like low-rank-gradient routes, and AdamW only as a quality anchor where it fits. A serious Unsloth/LoRA comparator is non-trivial because the harness and memory contract must be comparable; it matters near the end because if LoRA reaches much lower target loss at the same memory/time, users will judge accordingly. The expected SumoTrack advantage is full-parameter distribution movement without LoRA's adapter-rank ceiling/forgetting tradeoff, closer to GaLore capacity but with cleaner projected/Aurora updates.

## Path to a trustworthy implementation (publication later)

Do not optimize for publication yet. The last two days exposed design errors, not
ordinary implementation errors: internally consistent formulas answered the wrong
geometric question, normalization hid poisoned magnitudes, and mechanisms that
looked active had no useful control authority. A reference implementation of those
designs would have reproduced the bugs perfectly.

The current product is therefore **design confidence**, not test coverage. No
finite campaign can promise that future work will reveal no new conceptual error.
The honest goal is to make each assumption explicit, derive its consequences, and
try to falsify it through an observation that is not defined by the same assumption.

### P-1 — design falsification campaign

Build this before performance work or comparator runs. For every load-bearing
mechanism, keep a short contract with five fields:

```text
purpose       what product problem this mechanism solves
assumption    the geometric/statistical claim that makes it plausible
intervention  what can be changed while holding neighbors fixed
observable    what must move if the mechanism has useful control authority
rival         the simplest competing explanation or design
```

Then attack the design from four directions.

**1. Coordinate, unit, and limiting-case derivation.** Before a run, derive both
projection sides separately and follow scale through the entire chain—not merely
through the local function. Ask what happens at zero angle, full rank, rank one,
90-degree rotation, constant gradients, pure noise, and multiplication of the raw
gradient by a scalar. These checks would have exposed wrong-frame inactivity and
the sigma/Adafactor scale path without relying on training loss. They do not choose
between valid momentum semantics; that still requires a product-level argument.

**2. Intervention with mechanism-local telemetry.** A knob or mechanism earns its
name only if changing it causes the predicted intermediate quantity to move:

```text
tracking step       -> principal-angle motion, not merely final capture
dampening           -> boundary-gradient scale and target stability
transport choice    -> lifted ambient history and pre-Aurora singular spectrum
clipping            -> the exact tensor and stage it claims to bound
Aurora              -> input spectrum, polar residual, and lifted update direction
```

Use deliberately strong interventions first. If `0` versus `1` produces no local
effect, the mechanism is inactive, cancelled downstream, or measured incorrectly.
A final loss curve is a late and confounded sensor.

**3. Rival designs in hostile synthetic worlds.** Construct tiny sequences where
the desired behavior is defined independently of SumoTrack: a stationary signal
plus noise, a smoothly rotating dominant subspace, an abrupt regime change, and
cutoff-plane eigenvalue crossings. Compare position control, tangent control,
frozen basis, reset, fixed-ambient reprojection, and moving-fiber transport only on
the worlds that distinguish their assumptions. This cannot prove which semantics
real language-model training wants; it can show where each design necessarily
wins, loses, or becomes undefined.

**4. Real-training falsification.** Promote only surviving distinctions into the
faithful SYNTH lane. Measure both the mechanism's predicted intermediates and the
product outcomes: capture, target loss, source retention, update/parameter ratio,
state, and walltime. Prefer paired interventions around a causal claim over broad
hyperparameter sweeps. If the local observable moves as predicted but the product
outcome does not, the mechanism may be true and still not worth having.

Implementation tests remain necessary but subordinate. Exact fp64 oracles,
gauge/transpose properties, state-machine coverage, save/resume, and eager/compile
parity protect a design after it survives scrutiny; they do not validate its
premises. Historical bug mutations are useful regression tests, never evidence
that the next unknown design error would have been caught.

When a design error is found, record the failed assumption, the observation that
falsified it, and the broader family of mechanisms sharing that assumption. The
campaign has no magical “bug-free” exit. Publication becomes reasonable when a
fresh adversarial review yields no unresolved contradiction in the main path and
the remaining assumptions and support boundaries can be stated without bluffing.

### P0 — make the generic PyTorch contract true

The optimizer itself no longer depends on activation-projection wrappers, so the
core should work with an ordinary training loop:

```python
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Only after P-1, build a compatibility matrix and test it. The minimum integration
gate is:

- arbitrary rectangular and square dense matrix parameters, both projection
  sides, mixed matrix/non-matrix groups, tied parameters, missing gradients, and
  multiple parameter groups;
- fp32 and bf16 parameters, autocast/GradScaler usage, gradient accumulation,
  closure semantics, save/resume across devices and dtypes, and deterministic
  continuation after `state_dict` reload;
- DDP as the first distributed boundary; explicitly mark sparse gradients,
  FSDP/flattened parameters, differentiable/capturable optimizers, and exotic
  tensor subclasses unsupported until tested;
- a clear policy for embeddings and output heads. `p.ndim == 2` currently makes
  them matrix-path parameters automatically, while the quality harness excludes
  embeddings by policy. That difference must be visible in the public API and
  README rather than hidden behind the word “automatic.”

There is also an honest side-policy boundary. Shape-based `side="auto"` is generic
and model-agnostic; the measured quality default, `residual-facing`, requires
module-role knowledge supplied by the harness. Do not imply that a bare optimizer
can infer transformer semantics from a tensor shape. A small optional parameter-
group helper can make residual-facing setup convenient later; it should not be
baked into the optimizer core.

### P1 — one release-shaped evidence table, when confidence holds

Do not reproduce a conference benchmark suite. Run one controlled table on the
existing LFM-350M faithful SYNTH contract, with the same model, formatting, token
budget, compile state, optimizer scope, and eval cadence. Report:

- target eval loss and source retention;
- peak allocated VRAM, optimizer-state bytes, tokens/sec, and walltime to the
  named target loss;
- SumoTrack current default, AdamW quality anchor where it fits, one credible
  GaLore/SubTrack-class implementation, and one LoRA/Unsloth-style matched-memory
  user alternative;
- both “same token budget” and “same memory ceiling” views. Either view alone can
  make the wrong optimizer look heroic.

Three seeds are not the first use of this machine. First run one seed per arm and
repeat only comparisons whose margin is close enough that variance changes the
decision. Publish curves and exact commands. That is afternoon-scale rigor:
selective replication, not ritual replication.

Avoid an unsupported “state of the art” claim for orthogonalization. The honest
claim today is “Aurora/Muon-family projected polar updates, selected by measured
quality in this optimizer.” If runtime or approximation quality is compared
against current polar methods, name the exact methods, shapes, dtype, error metric,
and hardware; then strengthen the wording only as far as that table supports.

### P2 — profile the optimizer step after correctness, then cut launches

The model can compile while Python optimizer bookkeeping remains eager. Profile a
real rank256 step with refresh and non-refresh iterations separated. Attribute
walltime and launches to these stages:

```text
sanitize + raw clip
adafactor row/column reduction and dampening
gradient projection
projected EMA
Aurora / polar iterations
project back + parameter update
basis target Gram + eigh + geodesic refresh
Python grouping, stacking, diagnostics, and device synchronizations
```

The first cheap audit target is synchronization, not Triton. Grassmann refresh
currently converts `last_tangent_sigma_max` and `last_rotation_angle` to CPU
floats inside the projector even when basis diagnostics are disabled. Keep those
as device scalars or avoid producing them outside diagnostics. Search the entire
ordinary step for the same pattern before writing kernels.

Next, preserve bucket structure across steps rather than rebuilding Python shape
maps, and measure the cost of `torch.stack`/`unbind` around same-shape Aurora
batches. Transformer layers provide repeated shapes; a compiled tensor function
over a stable bucket is the natural unit. Do not compile `optimizer.step()` as an
object graph again—the earlier failure already showed that Parameter identity and
fallback bookkeeping are the wrong compilation boundary.

### Triton candidates — gated by a profiler

Good candidates are memory-bound chains with several reads/writes, not GEMMs that
cuBLAS already owns:

1. **Adafactor preparation.** Fuse finite sanitization, raw-norm accumulation,
   clipping, squared-gradient row/column reductions, and as much dampening setup
   as the reduction dependency permits. This currently walks every full gradient
   several times. A two-stage Triton reduction may earn its keep.
2. **Projected EMA and small tensor bookkeeping.** Clip/EMA can fuse when enabled,
   though the default rails are off and the likely prize is modest.
3. **Project-back epilogue.** A custom low-rank matmul whose epilogue applies LR,
   weight decay, and writes directly into the parameter could avoid materializing
   a full update tensor. This is the most interesting memory-traffic candidate,
   but it must beat cuBLAS plus `add_` on real transformer shapes before adoption.
4. **Aurora elementwise polynomial stages.** Fuse normalization and polynomial
   combinations between matrix multiplies. Do not replace the matrix multiplies
   themselves with handwritten Triton unless the small projected dimensions leave
   cuBLAS visibly launch-bound; HeavyBall's compiled kernels are the first bar.

Basis Gram construction, projection, and most Newton–Schulz products are GEMMs;
`eigh` is a library decomposition. They are poor first Triton targets. A custom
kernel there would mostly become a slower numerical library maintained by one
person, which is a grim hobby.

### P3 — isolate the final semantic risks cheaply

The geometry is now coherent, but two compact experiments would make the release
more honest:

- **Transport semantics:** deterministic rotating-subspace toy first, then a
  narrow SYNTH comparison of identity parallel transport, fixed-ambient
  reprojection, and moment reset at refresh. The toy must check the lifted Aurora
  direction, not just moment norm. This tests which momentum contract helps the
  optimizer; it is not another attempt to prove the already-exact no-op math.
- **Tracking value:** current position control versus frozen-eigh on a horizon
  long enough for drift to bite, reading capture, target loss, and source loss.
  If position control wins there, stop tuning tracking. If it does not, static
  eigh is a major simplification and should become a serious release candidate.

The tracking sidequest has earned a stopping rule: no new aim, transport, or
schedule mechanism without a failure in capture or loss that the current position
controller cannot explain. Curiosity remains welcome; mechanism inventory does
not belong in the default path.

### P4 — package the small thing, archive the laboratory

The public package should contain the optimizer, projector, tested defaults, and
short examples. Harness-specific model surgery and the closed projected-activation
lane should remain research artifacts, not dependencies or advertised fast paths.
Keep advanced knobs—users really do have different ranks, drift rates, and memory
ceilings—but classify them:

- normal user controls: LR, rank, weight decay, refresh interval;
- advanced geometry controls: side, aim, rotation fraction, Aurora iterations;
- ablation/debug controls: tangent aim, random init, diagnostic probes.

Many knobs are acceptable. An undifferentiated constructor is not. Stable defaults
plus named tiers preserve experimentation without making every user reconstruct
the lab's history.

### What the project needs most

The strongest next move is **not publication and not more optimizer invention**.
It is the adversarial correctness campaign: exact tiny oracles, metamorphic tests,
forced state transitions, and step-by-step differential trajectories. Only when
those stop producing surprises should generic-PyTorch compatibility and compiled
walltime become the main cuts. The matched-memory table is last; benchmarking a
still-misunderstood optimizer only produces precise folklore.
