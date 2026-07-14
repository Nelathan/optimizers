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

### R5. Can an interval sketch improve the target aim?

The one boundary gradient gives a proper instantaneous `eigh` target: under a
stationary signal it is a noisy observation of the optimal basis, so fixed
fractional position control converges toward a destination. The old C1 tangent
accumulator did not have that property. Its observations were local velocity
commands rather than one destination plus noise, so averaging reduced its noise
floor without making stochastic descent converge to the optimal basis.

The next question is therefore specifically **target estimation**, not tangent
smoothing or a new controller. For conditioned gradients `G_t` in one refresh
interval, define the equal-step covariance target

```text
K_bar = mean_t(G_t^T G_t / ||G_t||_F^2)  # right
K_bar = mean_t(G_t G_t^T / ||G_t||_F^2)  # left
T_*   = top-r eigh frame of K_bar
```

Per-step normalization preserves the scale invariance of the current single-grad
target instead of allowing one high-norm step to set the interval target. This is
an explicit estimator choice to test, not a claim that raw covariance weighting
cannot work. The main path always forms the full gradient, applies Adafactor, then
uses that conditioned gradient for tracking and projection; this investigation
does not require projected-activation or custom-backward machinery.

#### Estimator forms to keep open

| form | interval state | target semantics | primary risk |
|---|---:|---|---|
| exact covariance oracle | `O(d^2)` | exact finite-window `T_*` | forbidden product state; synthetic reference only |
| warm block range | `O(d*r)` | `Y=sum_t K_t Q`, `T=orth(Y)` | target depends on held `Q`; exact orthogonal replacement is invisible |
| random gradient-factor sketch | `O(d*s)` | form thin `B` with `E[B^T B]=K_bar`, then ordinary `eigh` | sketch variance and extra thin projection work |
| per-step target mean | `O(d*r)` plus one `eigh` per step | average already-extracted target frames | expensive and averages after cutoff noise acts |
| Oja/online subspace iteration | `O(d*r)` | stochastic covariance iteration | introduces another controller and step size |

The random-factor form sketches the opposite gradient axis independently each
step and concatenates the thin factors over the interval:

```text
B_t = R_t (G_t / ||G_t||_F)   # right, B_t:[k,d]
B_t = (G_t / ||G_t||_F) R_t   # left,  B_t:[d,k]
B   = concatenate_t(B_t)
T   = eigh_target(B)
```

With correctly scaled independent `R_t`, the sketched side covariance is an
unbiased estimate of `K_bar`. Candidate total widths are `s=r`, `2r`, and `4r`;
`2r` is the initial product-shaped budget because `s=r` has no oversampling.
A dense random orthoprojector is the clean simulator reference: its width ladder
has an exact endpoint when the projected width reaches the contacted gradient
axis. Rademacher, CountSketch, structured projections, lower precision, and
kernel/state optimization come only after the direction earns value.

Warm block range remains in the comparison despite not directly estimating a
fixed target. It uses the held projection cheaply:

```text
Z_t = G_t Q                         # right
K_t Q = G_t^T Z_t

Z_t = Q^T G_t                       # left
K_t Q = G_t Z_t^T
```

It is a normalized position target rather than C1's magnitude-bearing tangent,
and repeated block-power steps can converge to the dominant eigenspace whenever
the held frame has contact with it. It may also act as a useful prior that rejects
finite-window noise. But its `Q`-dependent bias and orthogonal blind spot must be
measured rather than laundered into "better averaging."

Oja first enters as a separate online estimator frame `S:[d,r]`: it receives
every conditioned gradient while the live projection basis `Q` still moves only
at the interval boundary. This isolates estimator quality from basis cadence and
the known `.25` controller. That separation is not a permanent architectural
commitment. If Oja proves both substantially better and stable, compare using `S`
directly as `Q`; carrying two basis-sized frames and a second controller would
then be complexity without meaning. Direct promotion must re-measure per-step
frame motion, moment behavior, and walltime rather than being rejected merely
because the first clean experiment kept the roles separate.

#### First feasibility evidence

A disposable fp64 stationary-covariance probe used `d=64`, `r=8`, signal
eigenvalues `8 -> 3` over an isotropic noise floor `1`, ten finite-sample
covariance observations per interval, fixed `eta=0.25`, 40 intervals, and 40
trials. Against the population covariance, repeated warm block-range updates
converged from initial principal angles of 15, 45, and 75 degrees, but remained
stuck under an exactly orthogonal noiseless replacement, as predicted. With
finite-window observations, random cross-contact let it escape even the
orthogonal start. Its final mean principal-angle mass was about `0.69` radians
versus `0.81` for exact finite-window EIGH position targets. This is feasibility
evidence only: in that stationary world the warm prior suppressed sample noise,
but it does not show rotating-signal tracking, language-model utility, or safety
under abrupt replacement. A one-window probe also showed its error relative to
the exact interval target growing sharply as the held frame moved farther away.

The reusable simulator now lives at
`experiments/interval_target_estimators.py`; arithmetic checks live in
`tests/test_interval_target_estimators.py`. Its first map used `d=64`, `r=10`,
interval 10, 32 Gaussian covariance-contact rows per gradient, 40 intervals,
eight trials, and the production `.25` boundary controller. Random-factor rows
use dense random orthoprojectors, giving the width ladder an honest exact endpoint
when all 32 rows per gradient are retained. Oja is a persistent separate frame
with a Rayleigh-normalized Grassmann-gradient step; the simulator calibrates its
single step size before any optimizer API exists.

First-map evidence, all numbers mean principal-angle mass in radians:

- In the stationary world, exact interval EIGH left the controlled basis at
  `0.88`; warm block improved it to `0.75`. Oja reached `0.69` at step `.0625`
  and `0.81` at `.125`. The persistent prior can beat the finite-window target by
  rejecting sample noise.
- Over a smooth 60-degree rotation, exact interval EIGH reached `1.21`, warm
  block `1.33`, and Oja `.125/.1875` about `1.27`. The coldest Oja step lagged at
  `1.44`: stationary denoising and tracking speed are a real trade, not one scalar
  ranking.
- After an exactly orthogonal replacement, exact interval EIGH reacted fastest
  immediately. Warm block and Oja retained their expected prior lag. By the final
  interval the basis errors were `0.87` exact, `0.98` warm, and `0.82` Oja `.125`;
  Oja's online contact recovered faster than warm block after the first shocked
  windows, but did not erase the abrupt-change cost.
- At a rank-cutoff crossing, exact interval EIGH finished at `0.88`, warm block
  at `1.61`, and Oja `.1875/.25` near `1.00`. Hotter Oja steps replaced stale
  membership faster, while colder steps preserved more stationary denoising.
- The random-factor estimator was state-inefficient in this world. Controlled
  stationary basis error was `3.85` at `2r`, `2.88` at `4r`, `1.92` at `8r`, and
  `1.26` at `16r`; only `32r`, which retained the complete gradient factor,
  matched exact interval EIGH at `0.88`. This does not prove every structured
  covariance sketch bad, but the proposed `2r` factor has not earned optimizer
  integration.

This first map warms Oja substantially and keeps warm block alive. It does not
select an Oja step: `.0625` is the stationary winner, `.125-.1875` is the current
rotation/abrupt compromise, and `.1875-.25` handles cutoff replacement better.
Before controller integration, sensitivity must vary SNR/eigengap, rotation rate,
contact rows, and rank ratio so one synthetic world does not silently tune the
estimator. The JSON output includes mean per-interval trajectories so recovery
speed remains visible rather than being flattened into final scalars.

#### Direct-Oja calibration and noise sensitivity

If the Oja estimator frame itself becomes the live `Q`, its update must use the
horizontal Grassmann geodesic rather than QR. QR is harmless for a target
subspace `S`, but can arbitrarily change frame gauge while projected momentum
coordinates are retained. A geodesic Oja step preserves the same rigid-frame,
identity-coordinate transport contract as the current controller.

For a Rayleigh-normalized Oja tangent, local target-error contraction per gradient
is approximately

```text
gamma * rho
rho = (lambda_r - lambda_{r+1}) / mean(lambda_1:r)
```

The separate estimator receives ten Oja updates and then `Q` receives only `.25`
of the resulting displacement. In the small-motion limit this maps separate Oja
steps `.125-.1875` to direct steps around `.031-.047`. Asking direct `Q` to close
`0.1` of a stationary target over ten gradients gives per-gradient contraction
`1 - 0.9^(1/10) = 0.01048`; the first simulator's gap ratio `rho ~= .364` maps
that independently to `gamma ~= .029`. Both estimates therefore center the
direct calibration at `.03`, without claiming transformer gradients share the
synthetic eigengap.

Direct-geodesic Oja at `.02/.03/.04/.05` confirmed the estimate. In the original
stationary world `.03` reached basis error `0.87`; under smooth 60-degree motion,
`.04-.05` reached `1.20-1.17`, slightly better than the separate estimator. It
was much slower after orthogonal replacement and cutoff membership changes. This
is the expected price of a cold direct tracker, and those hostile worlds are
deliberately faster than the presumed LLM target drift.

Three slower, noisier sensitivity regimes then varied contact rows and signal
gap:

- Moderate noise (`32 -> 16` contact rows, signal `6 -> 2`, 20-degree total
  rotation): direct `.03-.04` was the stable center (`1.68-1.65` rotating basis
  error), while separate `.0625` reached `1.33`.
- Weak signal (signal `4 -> 1.5`): direct `.03-.04` reached `2.44-2.36`; warm
  block and separate `.0625` were both about `1.94`.
- Very weak signal (`8` contact rows, signal `3 -> 1.25`, 10-degree rotation):
  direct `.02-.03` reached `4.58-4.59`; separate `.03125` reached `4.19` and
  warm block `4.27`. All estimators were poor in absolute terms, but the colder
  online estimator retained the most signal.

The robust direct-Oja center is therefore approximately `.03`, but direct `Q` is
not yet a quality winner under high observation noise. The separate frame buys a
second temporal filter: `S` can learn online while the `.25` boundary controller
rejects its high-frequency motion. That benefit costs one additional `[d,r]`
state tensor, an independent `G S` projection plus covariance backprojection,
and per-step estimator orthogonalization. Direct Oja instead reuses the actual
`G Q` projected gradient, needs only the covariance backprojection, deletes `S`,
and makes every small frame move the live basis. Warm block also reuses `G Q` and
orthogonalizes only at the boundary, but was consistently slower than the best
separate Oja under replacement.

This is still Gaussian covariance evidence, not a transformer-gradient model.
Synthetic worlds are mechanism filters and calibration tools; actual conditioned
LLM gradients must decide whether the extra filter earns its state and compute.
The next integration decision is a narrow comparison among warm block, separate
Oja near `.05-.0625`, and direct geodesic Oja near `.03`, not a broad step sweep.

#### Staged evaluation contract

1. **Estimator simulator.** Compare boundary EIGH, exact interval covariance,
   warm block range, a simple Rayleigh-normalized Oja estimator, and random-factor
   widths `r`, `2r`, `4r` in stationary, smoothly rotating, abrupt-replacement,
   and cutoff-crossing worlds. Read both error to the population-optimal subspace
   and error to the exact finite-window target: fidelity and useful prior bias are
   different questions. Cheap simulator work may calibrate Oja's one step-size;
   do not mistake an untuned miss for a mechanism verdict.
2. **Controller integration.** Integrate only estimators that survive the
   simulator. Hold `eta=0.25`, full-spectrum geodesic motion, interval 10,
   Adafactor conditioning, and identity-coordinate moment transport fixed. Use a
   direct, legible implementation with honest state and walltime accounting.
3. **Narrow SYNTH direction run.** Promote the smallest faithful sketch budget
   and warm block range only if each still answers a live distinction. Measure
   target loss first; target-angle demand, applied step mass, lag mass, capture,
   and moment health explain the result. Do not multiply expensive arms merely
   because the simulator had cheap rows.
4. **Long comparison.** A promising direction must survive a replay-scale run
   before the final 1k comparison against the last default. Short-horizon basis
   health alone is not promotion evidence because it has previously failed to
   bind loss.
5. **Optimize afterward.** Only after direction evidence should work move to
   projection family, precision, allocation reuse, compilation, or cadence
   optimization. Clean science precedes making the losing mechanism fast.

No estimator is selected yet. Exact library `eigh` remains the extraction
reference wherever the estimator produces a gradient factor or covariance;
replacing the decomposition is outside this question.

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
