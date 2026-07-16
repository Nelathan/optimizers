# UsuiTrack Plan

UsuiTrack is a memory-efficient optimizer for high-capacity continued pretraining
on consumer GPUs. It should move a pretrained model across a real distribution
shift without full AdamW matrix state, slow gradient accumulation, or
adapter-only capacity limits.

This is the live direction map. Read `SPEC.md` for the current update,
`SUBSPACE_TRACKING.md` for the open geometry ledger, and `QUANTIZATION.md` for the
scoped rank-state quantization sidequest. Superseded runs and design arcs are
preserved under `archive/`; they are provenance, not guidance.

## Product contract

The target is a useful Pareto point under a fixed consumer-GPU budget:

- enough trainable capacity for distribution adaptation, not only style steering;
- matrix optimizer state small enough to buy more tokens per step;
- target movement without needlessly destroying source behavior;
- walltime that does not repay saved memory in launches or memory traffic;
- ordinary PyTorch use with defaults that do not require reconstructing the lab.

AdamW is the quality anchor where it fits. LoRA/Unsloth is the user alternative
under memory pressure. GaLore/SubTrack is the nearest low-rank-gradient family.

The primary product gate is UsuiTrack versus LoRA trained with AdamW at
comparable VRAM and training conditions: match or beat target convergence while
retaining source behavior comparably. LoRA keeps a fixed low-rank
parameterization; UsuiTrack makes rank-limited per-step updates whose moving
frame does not impose the same fixed final displacement subspace. GaLore,
SubTrack, and older projected optimizers are secondary comparisons for later
state-of-the-art positioning, not prerequisites for a useful open-source result.

Do not target a 12B model on a 32GB RTX 5090 yet. Bf16 weights plus full bf16
gradients already exceed that budget before activations or optimizer machinery.
The near-term systems question is how far full-gradient continued pretraining can
be expanded on consumer cards, likely beginning around the 4B class. Full
gradients, not UsuiTrack's rank state, are expected to be the dominant VRAM tax.
The current post-backward preparation already consumes each full gradient after
forming its rank-sized pending update, which shortens optimizer-step residency but
cannot lower the peak where backward has materialized all gradients. A HeavyBall-
inspired `register_post_accumulate_grad_hook` prototype moved the same preparation
into backward, released each completed matrix gradient, and retained only the
rank-space work for batched application. Exact update and optimizer-state parity
held. At checkpointed LFM-1.2B, release removed about 1.97 GB of live state after
backward. Eager peak allocated fell 13.3%, but the compiled peak did not move:
AOTAutograd materialized its complete gradient-output pile before the first leaf
post-accumulate callback. The mechanism is therefore a real eager memory lane, not
a compiled-path memory optimization under the current graph boundary.

## Current design

The forward path is:

```text
finite raw gradient
  -> per-tensor raw clip
  -> Adafactor row/column SNR conditioning with RMS restoration
  -> one-sided residual-facing projection
  -> projected first-moment EMA
  -> Aurora / HeavyBall Newton-Schulz polar direction
  -> full-parameter Muon scale
  -> lift and parameter update
```

The basis starts from stable side-Gram `eigh`, then one-state Oja moves the live
frame from every conditioned full gradient with harmonic steps `1/2, 1/3, ...`
down to `.01`. Its exact
rank-space geodesic rotates every tracked plane and stores no second frame. The
geodesic's rigid frame rotation parallel-transports projected momentum with
identity coordinates, so tracking does not brake and rebuild the moment. Fixed
`.25` boundary EIGH position control and tangent velocity control remain explicit
ablations; cadence, boundary step, and rotation-rank controls govern only them.

Current redesignable decisions and their reasons live in `SPEC.md`. Do not infer
the design from old commands or old result tables.

## Current evidence

These claims survived the experiments that produced the present design:

- Projected first moments reduced broad no-embedding matrix state from the
  full-moment class to tens of MB at rank 64 on LFM-350M and LFM-1.2B.
- Orthogonalized projected momentum beat plain projected momentum in the tested
  lanes; Aurora then beat the HeavyBall-NS-only forward map at matched update
  scale and memory.
- Residual-facing projection beat shape-only side selection.
- Uniform rank beat the tested allocation policies on complexity-adjusted value.
  Rank remains a cost/quality control. Rank 256 carried the strongest historical
  350M quality lane; after Oja repaired tracking, rank 128 is the current harness
  balance rather than assuming the older capacity optimum remains necessary.
  Rank also changes update norm, so results across ranks describe different
  capacity/pressure choices and are not optimizer-control baselines for one another.
- Stable side-Gram `eigh` was faster and more robust than exact SVD for the
  required one-sided initialization.
- Adafactor-conditioned EMA beat plain EMA, no momentum, and second-moment-only
  variants in the matched rank-256 SYNTH ablation.
- Raw clipping must precede Adafactor, target construction, and projection. A
  downstream projected clip cannot protect upstream state from a blip.
- Tangent tracking was a noisy velocity controller. Fractional full-spectrum
  `eigh` aim converged as position control.
- A rank-32 transformer-gradient diagnostic showed why the EIGH controller still
  leaves headroom: a raw single-gradient EIGH target predicted the next interval
  worse than held `Q`, while one-state geodesic Oja tracked a more predictive
  blind-spot frame with far less churn and no second basis state. Oja then
  led EIGH and a separate Oja-target controller at every checkpoint of a matched
  compiled 500-step run, finishing at `1.890727` versus `1.895432` and `1.891082`.
  The separate target was deleted, and the surviving mechanism is now simply
  named `oja`. At rank 128 and 1k steps, Oja then finished slightly better on
  target loss (`1.732211` versus `1.733106`) with slightly worse source retention,
  materially better capture, and clean 50-step basis convergence. Its current
  original per-gradient implementation was 36.9% slower. Profiling traced that
  tax to per-matrix SVD/QR/SVD decompositions; exact rank-space geodesics,
  Polar Express correction, and batched equal-rank EIGH reduced synthetic
  optimizer-only Oja time from `616.9 ms` to `122.1 ms` against an `83.0 ms`
  EIGH non-refresh baseline. The subsequent full-path pass removed a full-size
  Adafactor temporary and batched equal-shape Oja geometry and Aurora health
  diagnostics, reducing the telemetry-weighted synthetic optimizer path to
  `100.5 ms`. Its rank-128 500-step replay reached `1.783078 / 2.973868`
  target/source at `.8040 s/step`, versus EIGH's `1.783133 / 2.972688` at
  `.9229 s/step`. Oja is 12.9% faster under the real training contract, with the
  same small target/source trade seen before. The implementation-efficiency
  promotion gate is cleared.
- Reprojecting momentum across basis refresh charged a principal-plane cosine
  tax before Aurora. Identity coordinates are exact transport along the selected
  rigid frame rotation and preserve the projected singular spectrum.
- Round-robin refresh and two-sided square-core projection did not earn their
  complexity and were removed.
- Projected-activation backward was correct but lost to compiled ordinary
  backward plus repaired checkpointing at the measured LFM-350M shapes. That
  lane is closed, not a current performance recommendation.

Exact runs, false starts, and historical defaults remain in
`archive/RESULTS.md`, `archive/SUBSPACE_TRACKING_ARC.md`, and
`archive/PROJECTED_ACTIVATION.md`.

## Confidence before publication

Recent failures were design failures: correct code implementing the wrong control
law or preserving the wrong geometric object. More conventional tests alone
would canonize those mistakes. Publication is downstream of design confidence.

Every load-bearing mechanism needs this contract:

```text
purpose       product problem it solves
assumption    geometric or statistical claim it relies on
intervention  change that isolates its causal role
observable    local quantity that must move if it is active
rival         competing design or explanation
```

Attack each contract in four ways:

1. **Derive coordinates, units, and limits.** Work both projection sides and test
   zero angle, full rank, rank one, 90-degree rotation, constant signal, pure
   noise, and gradient rescaling.
2. **Force local interventions.** Compare strong settings first and inspect the
   claimed local effect before reading loss. An inactive mechanism can produce a
   green training run.
3. **Use hostile synthetic worlds.** Stationary noise, smooth rotation, abrupt
   replacement, eigenvalue crossings, and near-orthogonal targets distinguish
   rival semantics cheaply.
4. **Falsify on faithful training.** Only survivors enter SYNTH, where local
   telemetry is read beside target loss, source retention, state, and walltime.

Fp64 oracles, gauge/transpose properties, state transitions, resume, and
eager/compile parity then pin the surviving design. They are regression guards,
not evidence that its premise is true.

## Research map

This is a space, not an ordered queue. The user chooses traversal.

| question | cheapest discriminating evidence | possible consequence |
|---|---|---|
| Does tracking improve the product over frozen `eigh`? | long-enough frozen vs Oja run with capture, target, and source | keep tracking or delete it |
| Does moving-frame momentum beat fixed-ambient history? | rotating toy reading lifted pre-Aurora direction, then narrow SYNTH survivor | keep identity, reproject, or reset |
| Is the chosen frame path stable at cutoff churn and near 90 degrees? | gauge/sign and hostile-target stress | stabilize target/path or accept boundary |
| Does tracking cadence cost meaningful walltime? | profile ordinary and telemetry Oja steps; separate EIGH boundary/non-boundary steps only for that ablation | retain per-gradient Oja unless a measured sparse estimator preserves quality |
| Did Oja clear promotion over noisy boundary EIGH? | one-state Oja won rank-32 replay loss and the rank-128 1k target endpoint; optimized rank-space geometry preserved loss and beat EIGH walltime by 12.9% in a matched rank-128 500 replay | yes: Oja is released; retain EIGH as the position-control ablation and keep the slight source trade visible |
| Can Oja close its early post-EIGH lag without losing its settled stability? | rank-128 sensor `ytqnw1ip` changed only the schedule to `1/2, 1/3, ...` down to `.01`; target loss improved at every checkpoint and settled geometry also improved | yes: mature Oja is the released schedule; fixed `.01` remains the steady-step ablation |
| Does the well-aligned projected moment still need Aurora `pp=2/ns=5`? | `pp=1/ns=2` under-stepped badly; `pp=1/ns=5` preserved update norm but adapted more gently at 200, and the chosen rank-128 `pp=1/ns=5`, LR `3e-4` configuration reached `1.693917 / 3.010227` target/source at 1k | use one Aurora pass with five NS steps; do not infer the separate effect of pass count from the chosen configuration run |
| How much adaptation pressure should the rank-128 product lane use? | hold mature Oja and `pp=1/ns=5` fixed, then compare LR or horizon one axis at a time against the `3e-4` 1k reference `cqokmxft` | choose target/source operating point without laundering rank-256 capacity evidence into rank-128 tuning |
| Does the repaired moving-frame moment benefit from longer memory? | strict beta `.95` sensor `5lkubxig` was nearly tied through 200; the 1k replay `6gkm9k8h` remained healthy and monotonic, ending at `1.685367 / 3.022996` target/source versus beta `.9` control `cqokmxft` at `1.693917 / 3.010227` | yes: `.95` wins about as much target as it loses source, and its longer stable memory is now the constructor and harness default |
| Can projected second-moment conditioning replace full-gradient Adafactor? | projected Adam m1/m2 runs `op5xujxd` (`3e-4`) and `t5wbbf9s` (`2e-4`) show a consistently stronger learner with more source loss: at `2e-4`, `1.763509 / 3.004049` target/source versus historical Adafactor `dpiwqydb` at `1.783078 / 2.973868`; raw-gradient Oja captures more energy (`.740349`), while Aurora alignment and moment rank are slightly lower and Oja step angle is unchanged. Matrix state rises from the current control's `92.93 MiB` to `160 MiB` because the third slot is a wide projected m2 while the basis is narrow. Compiled walltime did not establish the required win: one current replay tied Adafactor (`.7927` vs `.7919` sec/step), while the `2e-4` arm reached `.7800`; the historical `.8040` Adafactor comparator still included the deleted basis-lag diagnostic. | no: lower LR does not remove the stronger-adaptation trade, state rises substantially, and the expected compiled speedup was not demonstrated; implementation removed, evidence retained |
| Do unstable cutoff planes harm useful planes? | per-plane target stability and capture | keep full spectrum or rotate a measured stable prefix |
| Where does compiled optimizer walltime go? | launch and synchronization profile by stage | stable buckets, compiled tensor cuts, or a fused kernel |
| Can full gradients be released during backward? | exact parameter/state parity held. At checkpointed LFM-1.2B, 92 matrix gradients total `2,071,986,176` bytes. Compiled AOT materialized the entire pile before the first traced leaf callback: callback residency was flat at `4,752,924,160` bytes in control, while release drained it to `2,780,337,664`; nevertheless peak stayed flat (`7,022,023,168` vs `7,025,431,040`). Eager callbacks interleaved with backward: release moved the peak from late layer-0 `w3` to the second matrix in backward, cutting allocated peak from `7,202,254,336` to `6,242,306,048` bytes (13.3%) and reserved from `7,503,609,856` to `6,698,303,488`. Post-backward current fell from `4,752,398,848` to `2,780,860,928`. Single traced-step timing was `2.512` vs `2.581` seconds compiled and `3.121` vs `3.221` eager; synchronized probe timing is directional, not a throughput benchmark. | keep optional as an eager, no-accumulation memory lane; do not promote it under the compiled quality default. A compiled win requires moving preparation inside the AOT backward graph or changing the compile boundary, not merely testing a larger model. Retain the non-transactional failure contract. |
| Can rank-side rotation make basis or moment state safely low-bit? | rank-64 outlier anatomy, then subspace/Aurora fidelity | quantize a proven target or close the sidequest |

Tracking work stops unless it deletes state or machinery, reduces measured tracking
cost, repairs demonstrated faithfulness, or improves capture and evaluation under
drift. New mechanism inventory is not progress.

## Engineering gates

### Generic PyTorch contract

Before release, verify ordinary `loss.backward(); optimizer.step()` use across:

- rectangular and square matrices, both sides, mixed fallback tensors, tied and
  missing gradients, and multiple parameter groups;
- fp32/bf16, autocast and GradScaler, accumulation, closure behavior, and
  deterministic state-dict continuation across devices and dtypes;
- DDP as the first distributed boundary;
- an explicit embedding/output-head policy and explicit unsupported boundaries
  for sparse gradients, FSDP-flattened parameters, capturable/differentiable mode,
  and exotic tensor subclasses.

Bare `side="auto"` sees shape, not transformer semantics. Residual-facing setup
requires module-role information and must remain an explicit parameter-group
policy rather than a claim of architectural inference.

### Performance contract

Profile optimizer steps at the product-candidate rank, with per-gradient Oja and
explicit EIGH refresh/non-refresh ablation iterations separated where applicable:

```text
sanitize + raw clip
Adafactor reductions and dampening
projection
projected EMA
Aurora / polar iterations
lift + parameter update
target Gram + eigh + geodesic
Python bucketing, diagnostics, and synchronizations
```

Remove accidental host synchronization first. Reuse stable shape buckets and
compile pure tensor stages rather than the `optimizer.step()` object graph.
Only profiler evidence may nominate Triton work. Plausible candidates are fused
Adafactor preparation, projected EMA bookkeeping, and a project-back GEMM
epilogue that writes the parameter directly. cuBLAS GEMMs, Newton-Schulz products,
Gram construction, and vendor `eigh` are not first targets.

The current mechanism is still moving; do not begin Triton work while Oja startup
and Aurora/NS depth remain live algorithmic questions. Sparse Oja cadence,
stable-prefix rotation, and a new rank sweep are not active leads: per-gradient
Oja is already fast, useful rank energy is broadly distributed, and rank remains
an explicit cost/quality choice.

Current diagnostic cost is measured rather than assumed. On the rank-128
LFM-350M topology, ordinary basis-step telemetry is negligible, while the old
fixed-50-step basis-lag probe cost about `233 ms` when it fired (`~4.7 ms`
amortized per optimizer step) and stored a second full basis snapshot. Its
startup question closed with mature Oja, so the probe and snapshot state were
deleted. Core optimizer diagnostics cost about `20 ms` per logged step, Aurora
health about `10 ms`, and harness gradient/parameter scans about `8.5 ms`.
Keep alignment/effective rank only while they answer the live Aurora/moment
questions, then reassess the eigensolve rent rather than making every sensor
permanent.

### Release evidence

When the design and integration gates hold, first run the primary controlled
LFM-350M LoRA+AdamW comparison with identical formatting, token budget, compile
state, and evaluation cadence. Report the target/source Pareto curve, state
bytes, peak allocated VRAM, tokens/sec, and training-only walltime. A clear
result here is sufficient to justify the core open-source release and its
whitepaper-style repository document.

Broader optimizer positioning can then add AdamW where it fits and one credible
GaLore/SubTrack-class route. Show both equal-token and equal-memory views. Repeat
only close decisions rather than performing ritual seed multiplication.

The public artifact should contain the optimizer, projector, tested defaults, and
short examples. Keep harness surgery and archived research out of the package.
Classify controls as normal (`lr`, rank, weight decay), advanced (side, aim,
Aurora iterations), or ablation/debug. Refresh interval, boundary rotation
fraction, and rotation rank are ablation-only under the released Oja path.

## Benchmark contract

The harness defaults to the current 1k quality contract: `LiquidAI/LFM2.5-350M-Base`,
broad no-embedding training, uniform rank 128, residual-facing projection, stable
`eigh` init, right-padded no-mask SYNTH rows, `batch_size=16`, `seq_len=1024`, CCE,
mature Oja, matrix LR `3e-4` with 50-step warmup, projected-moment beta `.95`, raw
per-tensor clip `1`, Aurora `pp=1/ns=5`, source retention, `torch.compile`, target
and source evaluation every 100 steps, telemetry every 25, and a final qualitative
sample. Non-2D fallback tensors use a separate fp32-state AdamW at half matrix LR,
betas `.9/.99`, epsilon `1e-8`, and zero weight decay.
`experiments/llm_synth_smoke.py` is authoritative for CLI defaults; the
retained interval-10 burst settings apply only when an explicit EIGH/tangent
boundary ablation is selected.

The practical run ladder is 200, 500, and at most 1k steps. A 200-step health
sensor keeps the 1k algorithm/training contract unchanged except for the named
axis, sets `--max-steps 200 --eval-every 50 --no-final-sample --no-torch-compile`,
and retains telemetry every 25. Compile startup is not worth paying for a short
mechanism sensor; enable it only when the sensor explicitly measures compiled
throughput. A 500-step replay uses evaluation every 100, normally omits the final
sample, and follows the compiled quality contract. The 1k default is still simple fine-tuning where LoRA
remains a credible alternative. The product claim points toward a conceptual
10k continued-pretraining/full-finetuning regime, where LoRA can reach capacity
limits and UsuiTrack's broader trainable capacity should matter. Interpret
shorter evidence for transfer toward that regime, but do not let a short sensor
silently redefine the 1k defaults or make an expensive 10k run a routine gate.

The completed Oja promotion lane used rank 128, now the harness default. Historical
evidence showed quality scaling through rank 256, but improved tracking changes
that cost/quality balance; rank remains a user control, not a universal optimum.
Any future rank comparison must keep source retention, `torch.compile`,
training-only elapsed time, and final qualitative sampling in contract.

Use `torch.compile` for expensive quality runs unless compile itself is under test
or breaks the contract. Packed no-mask inputs are the explicit throughput lane,
not source-retention evidence. Stay on the small model until scale transfer is the
named question. Do not tune against an old command copied from the archive.
