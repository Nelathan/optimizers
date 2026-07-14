# Subspace Tracking: Current Ledger

This is the live geometry ledger. The complete chronological investigation,
including superseded defaults and failed mechanisms, is preserved in
`archive/SUBSPACE_TRACKING_ARC.md`. Read `SPEC.md` for the canonical update.

## Settled current mechanism

For a canonical column frame `Q:[d,r]`, each refresh constructs a target frame
`T` from the Adafactor-conditioned boundary gradient's side Gram. The basis moves
one fraction `eta=0.25` toward `T` along every principal plane of the selected
Grassmann geodesic.

```text
T   = eigh_target(conditioned_gradient)
Q_+ = geodesic(Q, T, eta=0.25, all_planes=True)
```

This is position control. Consecutive noisy targets scatter around the signal
subspace, so target error decays rather than integrating as angular velocity.
Tangent aim remains only as a SubTrack-faithful single-gradient ablation.

The mature-EMA/Karcher schedule (`1/2, 1/3, …`, floor `0.1`) improved basis and
moment health but did not improve target loss against the fixed `eta=0.25`
controller. A replay-stable 500-step comparison was effectively neutral/slightly
worse on loss; fixed `0.25` remains the default. At 200 steps, projecting the
refresh boundary gradient with either the held frame or the just-updated frame
was indistinguishable (`2.011840` vs `2.011878` target loss), while injecting the
full instantaneous target was worse (`2.012180`). The fractional moving frame is
the useful mixture; boundary-gradient frame selection is not a product axis and
was deleted.

The geodesic implementation supplies a horizontal rigid frame lift:

```text
Q_+ = RQ
```

For projected moment coordinates `m`, tautological-bundle parallel transport
along that selected path is:

```text
R(Qm) = Q_+m
```

Therefore stored coordinates remain exactly unchanged. This is not an
approximation and not the same operation as preserving a fixed ambient vector.
Overlap reprojection computes `Q_+^T Qm`, contracts principal-plane components by
cosines, and can hand Aurora weak, noise-selected directions that its polar map
restores to full authority.

At a near-90-degree principal angle, shortest paths can be non-unique. Transport
is exact along the overlap SVD's selected path; target/path stability remains a
separate question.

## Why the previous controller failed

The tangent tracker observed local residual escape and integrated another turn.
After its frame bug was fixed, it was mathematically active but remained a noisy
velocity controller with no destination. Small steps wandered, large steps
thrashed, and smoothing reduced noise without creating an aim. Full-spectrum
motion amplified this failure; the same full spectrum became useful under
position control.

The original tall/right port also projected a column-frame tangent through a
row-frame formula and could return zero. Passing stability tests had exercised an
inactive mechanism. This is why local authority must be forced and measured.

## Current design contract

```text
purpose
    follow a drifting useful gradient subspace without storing full history

assumption
    instantaneous spectral targets are noisy positions around a useful center;
    moving-frame momentum is the useful history semantics for Aurora

intervention
    frozen vs position aim; identity vs overlap vs reset transport; target/path
    stress; refresh cadence

observable
    principal-angle motion, capture, target stability, pre-Aurora moment spectrum,
    lifted update direction, target loss, and source retention

rival
    frozen basis; tangent velocity control; fixed-ambient overlap reprojection;
    moment reset; abrupt refit
```

## Open question map

These are independent axes whose signs are not assumed.

### R1. Is tracking worth keeping?

Compare frozen `eigh` initialization with current position control over a horizon
long enough for capture to decay. Read capture, target loss, source loss, and
walltime. A null product result makes frozen `eigh` a serious simplification, not
an embarrassing outcome.

### R2. Which momentum semantics help training?

First use a deterministic rotating-subspace world where the desired lifted update
is specified independently. Compare identity-coordinate moving-frame transport,
fixed-ambient overlap reprojection, and reset. Inspect lifted pre-Aurora direction
and singular spectrum, not only moment norm. Promote only a real distinction into
a narrow SYNTH comparison.

### R3. Is the selected path stable?

Stress gauge/sign changes, cutoff eigenvalue crossings, and near-orthogonal target
frames. Distinguish a correct transport along an unstable selected path from an
incorrect transport formula.

### R4. Does cadence deserve intelligence?

Measure refresh and non-refresh walltime first. If fixed cadence is material,
compare fixed 10, fixed 100, and a monotone `10 -> 20 -> 50 -> 100` schedule before
building feedback control. A controller without an open-loop trace is ceremony.

### R5. Can online tracking improve the target aim?

Yes at rank 32. Direct one-state geodesic Oja (`Q` itself, fixed step `.04`) is
the surviving challenger to the released fixed `.25` boundary-EIGH controller.
The evaluated separate Oja target was deleted after replay-scale evidence: it
added a second frame and a second low-pass controller without beating direct
tracking.

A rank-32, 200-step transformer-gradient diagnostic established the reason for
the comparison. Mean next-interval predictive capture over mature steps 110-200 was
`.4588` for the raw boundary EIGH target, `.5077` for live `.25`-EIGH `Q`, `.5299`
for separate Oja `.0625`, and `.5274` for direct Oja `.04`. Oja therefore found a
more predictive, much lower-churn blind-spot frame; this was estimator evidence,
not evidence that applying Oja improves training loss.

Warm block was useful early evidence—it won predictive capture through step 100—
but it adds another `[d,r]` state, depends on held `Q`, and has an exact orthogonal
replacement blind spot. Keep the idea as a fallback clue if Oja initialization
fails, not as code or a fourth training arm. The random-factor sketch and
synthetic estimator harness did not earn continued maintenance and were removed.

#### Precision and repeated-update health

A 500-update left/right covariance probe compared persistent bf16 and fp32 Oja
frames at rank 32. The discarded separate Euler/QR target differed by about `.13`
radians total and had indistinguishable capture (`~.93077` versus `~.93080`).
Direct bf16 Oja differed from fp32 by about `.16-.17` radians with capture
differing by at most `.00011` after its numerical invariant was repaired.

That repair matters: the existing boundary geodesic's tiny orthogonality error
compounded under per-gradient use, reaching catastrophic error in both fp32 and
bf16. Direct Oja now projects each raw geodesic result back to Stiefel and
Procrustes-registers the corrected frame to the raw geodesic gauge before storage.
Over 500 updates fp32 orthogonality error stayed around `4e-6` and bf16 around
`.007`, rather than diverging. The Oja ascent sign is separately pinned: the
generic retraction steps along the negative of its supplied cost tangent, so the
covariance ascent tangent must be negated at that boundary.

Direct Oja reuses held `GQ` both for its covariance action and as the projected
gradient coordinates carried through the rigid frame move. It therefore needs
only the covariance backprojection plus the stabilized small-frame retraction,
not a second full projection or second basis state.

#### Training gate

Fresh matched rank-32 health checks first established finite behavior. Final
target loss at step 200 was `2.010188` for EIGH, `2.009062` for the separate Oja
target, and `2.007628` for direct Oja. Direct Oja led at every 50-step evaluation
and used the same matrix state as EIGH. The separate target added `6,029,312`
bf16 bytes.

The decisive replay used the same LFM2.5-350M, rank-32, broad-no-embedding,
right-padded SYNTH contract for 500 steps with `torch.compile`, evaluation every
50 steps, and W&B project `pink-marker/usuitrack`:

| step | EIGH `.25` | separate Oja target | direct Oja `.04` |
|---:|---:|---:|---:|
| 50 | 2.206577 | 2.207120 | **2.206353** |
| 100 | 2.109085 | 2.109607 | **2.108134** |
| 150 | 2.050799 | 2.050696 | **2.048889** |
| 200 | 2.010339 | 2.009210 | **2.007335** |
| 250 | 1.979776 | 1.977259 | **1.975824** |
| 300 | 1.955225 | 1.952096 | **1.950991** |
| 350 | 1.935433 | 1.931883 | **1.930990** |
| 400 | 1.919583 | 1.915692 | **1.915053** |
| 450 | 1.907060 | 1.902743 | **1.902104** |
| 500 | 1.895432 | 1.891082 | **1.890727** |

Runs: EIGH `o2prop3k`, separate target `nm7jcgbc`, direct Oja `r50bh8jw`.
Direct Oja led at every checkpoint, beat EIGH by `.004705` at step 500, and beat
the separate target by `.000355`. Its final held-frame capture was `.5081`
versus `.4820` EIGH and `.5114` separate-target; Aurora alignment was `.8363`
versus `.8285` and `.8330`; moment effective rank was `27.73` versus `27.41` and
`27.64`; projected-gradient/moment ratio was `4.26` versus `4.93` and `4.36`.
This is target-loss evidence only: no retention corpus was configured.

The separate target's initially rising lag and later catch-up were the expected
two serial low-pass filters—the online estimator followed by the `.25` boundary
controller—not useful warmup. Direct Oja adapted immediately and consistently
won loss. The separate estimator/controller arm was therefore deleted rather
than retained as a user-facing option.

One diagnostic bug was found while closing the comparison: basis lag was measured
over five basis moves. That meant 50 optimizer steps for boundary arms but only
five steps for direct Oja, so the logged direct lag masses are not numerically
comparable to EIGH or the separate target. The implementation now uses a fixed
50-optimizer-step snapshot horizon for every aim. This does not affect training
or any reported loss.

Compiled measured throughput was 18,631 tokens/s for EIGH, 19,684 for the
separate target, and 17,929 for direct Oja. Direct Oja is therefore about 3.8%
slower than EIGH in this single run. Do not infer the exact source from aggregate
walltime: profile the reduced QR plus rank-space Procrustes SVD before changing
the geometry. If stabilization is the measured tax, test the existing Polar
Express primitive, beginning with one iteration because correction runs every
step; do not introduce another polar implementation. Replay any changed
stabilization before promotion.

Rank 32 was deliberate: it is a rank-starved tracking stress test, not a proposed
product rank. The promotion rank remains open. Rank 64 is only the current harness
default; historical 128/256 runs still showed quality gains and healthy rank use
on this small model. Choose the rank for the matched 1k promotion comparison
explicitly rather than laundering the harness default into a product decision.

Do not add a harmonic warm-start arm yet: applying `1/n`-style steps immediately
is hottest when observations are noisiest, while delaying all basis application
would add ceremony unsupported by the nearly identical warmup losses. Fixed
steps keep the causal comparison clean.

The current released/default mechanism remains fixed `.25` EIGH until direct Oja
survives stabilization-cost work (if needed) and a matched 1k promotion
comparison. Predictive sensors and 500-step target loss nominate the challenger;
the default changes only after the promotion run and documentation review.

### R6. Do cutoff planes poison stable planes?

Measure per-plane target stability and contribution to capture. A stable-prefix
rotation is interesting only if cutoff churn demonstrably harms useful planes.
Otherwise full-spectrum position control remains simpler.

### R7. Where is the compile boundary?

Audit CPU scalar conversions, diagnostics, Python bucketing, and repeated shape
construction before considering custom geodesic kernels. Most geometry work is a
small decomposition plus vendor GEMMs; profiler evidence must identify the actual
tax.

## Hostile worlds

Use worlds with independently stated desired behavior:

| world | distinction exposed |
|---|---|
| stationary signal plus noise | aim variance and long-run drift |
| smoothly rotating signal | tracking lag and transport semantics |
| abrupt replacement | adaptation speed versus stale history |
| eigenvalue crossing at rank cutoff | target/path continuity |
| near-orthogonal target | path ambiguity and overlap collapse |

Synthetic success does not prove language-model utility. Synthetic failure does
invalidate a mechanism that claims to handle that world.

## Stop condition

No new aim, transport, target, or schedule mechanism enters the main path unless
it does at least one of the following:

- deletes state or tracking machinery;
- reduces measured refresh walltime;
- repairs a demonstrated faithfulness failure;
- improves capture and evaluation under drift.

When evidence moves an answer, update this ledger first. Put full run records in
the archive only after the current conclusion has been distilled here.
