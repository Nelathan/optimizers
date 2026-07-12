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

### R5. Is target decomposition expensive enough to replace?

Exact library `eigh` is the reference. Only after profiling should it be compared
with warm-started block iteration, a randomized range finder, or less frequent
exact targets. Incremental decomposition is justified only if measured Gram
changes have exploitable structure.

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
