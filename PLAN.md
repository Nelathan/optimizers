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

The basis starts from stable side-Gram `eigh`. At refresh boundaries, a transient
`eigh` target steers every principal plane a fraction `0.25` along the Grassmann
geodesic. This is position control, not accumulated tangent velocity. The
geodesic's rigid frame rotation parallel-transports projected momentum with
identity coordinates, so refresh does not brake and rebuild the moment.

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
  Rank remains a user control; rank 64 is the generic harness default, while
  rank 256 carried more quality in the historical 350M candidate lane.
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
  named `oja`. Fixed `.25` EIGH remains the released default until Oja clears a
  matched rank-128 1k target/source promotion run.
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
| Does tracking improve the product over frozen `eigh`? | long-enough frozen vs position-control run with capture, target, and source | keep tracking or delete it |
| Does moving-frame momentum beat fixed-ambient history? | rotating toy reading lifted pre-Aurora direction, then narrow SYNTH survivor | keep identity, reproject, or reset |
| Is the chosen frame path stable at cutoff churn and near 90 degrees? | gauge/sign and hostile-target stress | stabilize target/path or accept boundary |
| Does refresh cadence cost meaningful walltime? | profile refresh and non-refresh steps separately | leave fixed cadence or test a monotone schedule |
| Can Oja replace noisy boundary EIGH? | one-state Oja won every checkpoint through a matched compiled rank-32 500-step replay; rank-128 `.02` and `.01` sensors expose the freshness/stability tradeoff before the matched 1k | promote one-state online tracking or retain fixed `.25` EIGH |
| Do unstable cutoff planes harm useful planes? | per-plane target stability and capture | keep full spectrum or rotate a measured stable prefix |
| Where does compiled optimizer walltime go? | launch and synchronization profile by stage | stable buckets, compiled tensor cuts, or a fused kernel |
| Can rank-side rotation make basis or moment state safely low-bit? | rank-64 outlier anatomy, then subspace/Aurora fidelity | quantize a proven target or close the sidequest |

Tracking work stops unless it deletes state or machinery, reduces measured refresh
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

Profile optimizer steps at the product-candidate rank, with EIGH refresh and
non-refresh iterations separated where applicable:

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
Classify controls as normal (`lr`, rank, weight decay, refresh interval), advanced
(side, aim, rotation fraction, Aurora iterations), or ablation/debug.

## Benchmark contract

The default diagnostic lane is `LiquidAI/LFM2.5-350M-Base`, broad no-embedding
training, uniform rank 64, residual-facing projection, stable `eigh` init,
right-padded no-mask SYNTH rows, `batch_size=16`, `seq_len=1024`, CCE loss, and
position-controlled burst refresh every 10 steps. `experiments/llm_synth_smoke.py`
is authoritative for CLI defaults; `UsuiTrack` constructor defaults may differ.

The practical run ladder is 200, 500, and at most 1k steps: 200 is a
geometry/warmup sensor, 500 is replay-scale evidence, and 1k is still simple
fine-tuning where LoRA remains a credible alternative. The product claim points
toward a conceptual 10k continued-pretraining/full-finetuning regime, where LoRA
can reach capacity limits and UsuiTrack's broader trainable capacity should
matter. Interpret shorter evidence for transfer toward that regime, but do not
make an expensive 10k run a routine evaluation gate.

The current Oja promotion lane uses rank 128. Historical evidence showed clear
quality scaling through rank 256, but improved tracking may move that Pareto
frontier and rank 128 is the chosen balance for the next 1k. The comparison must
use source retention, `torch.compile`, training-only elapsed time, and a final
qualitative sample. Evaluate target and source loss every 100 steps. W&B training
telemetry every 25 steps preserves useful geometry resolution without making
validation part of the timed training path.

Use `torch.compile` for expensive quality runs unless compile itself is under test
or breaks the contract. Packed no-mask inputs are the explicit throughput lane,
not source-retention evidence. Stay on the small model until scale transfer is the
named question. Do not tune against an old command copied from the archive.
