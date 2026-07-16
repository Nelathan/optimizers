# Subspace Tracking: Current Ledger

This is the live geometry ledger. The complete chronological investigation,
including superseded defaults and failed mechanisms, is preserved in
`archive/SUBSPACE_TRACKING_ARC.md`. Read `SPEC.md` for the canonical update.

## Settled current mechanism

For a canonical column frame `Q:[d,r]`, one-state Oja reads every
Adafactor-conditioned full gradient, forms its horizontal covariance tangent at
the live frame, and moves along every tangent plane's exact Grassmann geodesic.
After EIGH initialization, `eta_t=max(0.01, 1/t)` gives harmonic early adaptation
and a steady `.01` floor. One Polar Express correction restores the Stiefel invariant.

```text
Delta = normalized_oja_tangent(conditioned_full_gradient, Q)
eta_t = max(0.01, 1 / optimizer_step)
Q_+   = polar_express(geodesic(Q, Delta, eta=eta_t, all_planes=True))
```

This is the released tracker. It stores no second frame and requires a full
matrix gradient every step. Fixed `.25` boundary EIGH position control and
tangent velocity control remain explicit ablations. Their refresh interval,
schedule, step, and rotate-rank controls are inert under Oja.

In the retained EIGH ablation, the mature-EMA/Karcher schedule (`1/2, 1/3, …`,
floor `0.1`) improved basis and moment health but did not improve target loss
against the fixed `eta=0.25` controller. A replay-stable 500-step comparison was
effectively neutral/slightly worse on loss; fixed `0.25` remains that ablation's
setting. At 200 steps, projecting the
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
    online conditioned covariance action tracks a useful drifting subspace;
    moving-frame momentum is the useful history semantics for Aurora

intervention
    frozen vs Oja vs boundary EIGH; identity vs overlap vs reset transport;
    path stress; Oja cadence

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

Compare frozen `eigh` initialization with current Oja over a horizon
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

### R4. Does Oja cadence deserve intelligence?

Current Oja deliberately consumes every gradient; boundary refresh cadence does
not govern it. Sparse Oja with a correspondingly larger step is a later
estimator-variance comparison, not an equivalent free speedup. Measure that axis
directly before building a schedule or feedback controller.

### R5. Can online tracking improve the target aim?

Yes. One-state geodesic Oja (`grassmann_aim="oja"`, using `Q` itself) is the
released tracker; fixed `.25` boundary EIGH is now the position-control ablation.
The deleted two-frame arm supplied the temporary need for the name "direct Oja";
the surviving mechanism is simply Oja.
The evaluated separate Oja target was deleted after replay-scale evidence: it
added a second frame and a second low-pass controller without beating one-state
tracking.

A rank-32, 200-step transformer-gradient diagnostic established the reason for
the comparison. Mean next-interval predictive capture over mature steps 110-200 was
`.4588` for the raw boundary EIGH target, `.5077` for live `.25`-EIGH `Q`, `.5299`
for separate Oja `.0625`, and `.5274` for one-state Oja `.04`. Oja therefore found a
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
One-state bf16 Oja differed from fp32 by about `.16-.17` radians with capture
differing by at most `.00011` after its numerical invariant was repaired.

That repair matters: the existing boundary geodesic's tiny orthogonality error
compounded under per-gradient use, reaching catastrophic error in both fp32 and
bf16. The first repair used QR plus Procrustes registration. The promoted hot path
computes the same geodesic from the rank-space tangent Gram and applies one
near-identity Polar Express correction before storage, removing both
decompositions while preserving the intended subspace and gauge to replay
precision. Over 500 updates fp32 orthogonality error stayed around `4e-6` and
bf16 around `.007`, rather than diverging. The Oja ascent sign is separately
pinned against the generic cost-tangent retraction.

Oja reuses held `GQ` both for its covariance action and as the projected
gradient coordinates carried through the rigid frame move. It therefore needs
only the covariance backprojection plus the stabilized small-frame retraction,
not a second full projection or second basis state.

#### Training gate

Fresh matched rank-32 health checks first established finite behavior. Final
target loss at step 200 was `2.010188` for EIGH, `2.009062` for the separate Oja
target, and `2.007628` for one-state Oja. Oja led at every 50-step evaluation
and used the same matrix state as EIGH. The separate target added `6,029,312`
bf16 bytes.

The decisive replay used the same LFM2.5-350M, rank-32, broad-no-embedding,
right-padded SYNTH contract for 500 steps with `torch.compile`, evaluation every
50 steps, and W&B project `pink-marker/usuitrack`:

| step | EIGH `.25` | separate Oja target | one-state Oja `.04` |
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

Runs: EIGH `o2prop3k`, separate target `nm7jcgbc`, one-state Oja `r50bh8jw`.
Oja led at every checkpoint, beat EIGH by `.004705` at step 500, and beat
the separate target by `.000355`. Its final held-frame capture was `.5081`
versus `.4820` EIGH and `.5114` separate-target; Aurora alignment was `.8363`
versus `.8285` and `.8330`; moment effective rank was `27.73` versus `27.41` and
`27.64`; projected-gradient/moment ratio was `4.26` versus `4.93` and `4.36`.
This is target-loss evidence only: no retention corpus was configured.

The separate target's initially rising lag and later catch-up were the expected
two serial low-pass filters—the online estimator followed by the `.25` boundary
controller—not useful warmup. One-state Oja adapted immediately and consistently
won loss. The separate estimator/controller arm was therefore deleted rather
than retained as a user-facing option.

One diagnostic bug was found while closing the comparison: basis lag was measured
over five basis moves. That meant 50 optimizer steps for boundary arms but only
five steps for Oja, so the logged Oja lag masses are not numerically
comparable to EIGH or the separate target. The implementation now uses a fixed
50-optimizer-step snapshot horizon for every aim. This does not affect training
or any reported loss.

Compiled measured throughput was 18,631 tokens/s for EIGH, 19,684 for the
separate target, and 17,929 for one-state Oja. Oja was therefore about 3.8%
slower than EIGH in this single run. Do not infer the exact source from aggregate
walltime: profile the reduced QR plus rank-space Procrustes SVD before changing
the geometry. If stabilization is the measured tax, test the existing Polar
Express primitive, beginning with one iteration because correction runs every
step; do not introduce another polar implementation. Replay any changed
stabilization before promotion.

Rank 32 was deliberate: it is a rank-starved tracking stress test, not a proposed
product rank. Rank 128 is the chosen promotion lane. Historical 128/256 runs
showed quality gains and healthy rank use, but better tracking may move the rank
Pareto frontier; returning directly to 256 would assume the old frontier survived
the new controller.

#### Rank-128 promotion sensors

The first compiled bs16, 200-step sensor used Oja step `.02`, raw clip `2.0`, and
source retention (W&B `mcib2cti`). Target loss moved `2.296923 -> 1.887331` and
source `2.917480 -> 2.946691`. Final capture was `.645365`, alignment `.778973`,
moment effective rank `104.30/128`, step angle mass `1.663318`, and corrected
50-step lag mass `20.969606`. The frame was healthy and adaptive but visibly
churny. No tensor crossed the raw `2.0` clip at the final logged step; median
per-tensor raw norm was `.424750`.

A second sensor used Oja step `.01` and raw clip `1.0` (W&B `tjsljczt`). Target
finished slightly worse at `1.889800`; source was effectively identical at
`2.946424`. Step mass nearly halved to `.860838`, while 50-step lag fell only
about 35% to `13.624978`. Capture, alignment, and effective rank were slightly
lower at `.633374`, `.767387`, and `102.79/128`. The median raw norm remained
`.425403`, and `9.59%` of tensors crossed the tighter clip.

The pair shows the intended freshness/stability tradeoff: `.01` is steadier and
slightly slower to fit at 200 steps. It does not isolate step size because the
raw clip changed simultaneously. Total angle mass also sums over all principal
planes, so its growth with rank does not by itself establish a rank scaling law.
An implicit step or LR scale such as `1/sqrt(rank)` remains a hypothesis, not a
current mechanism.

The mature-step Oja start won its rank-128 sensor (`ytqnw1ip`). After EIGH
initialization it used harmonic steps `1/2, 1/3, 1/4, ...` down to the steady
`.01` floor, changing no other training axis. Against fixed `.01` (`tjsljczt`),
target loss improved at every checkpoint: `2.162642` vs `2.164869` at step 50,
`2.010792` vs `2.015143` at 100, `1.932781` vs `1.936936` at 150, and
`1.886524` vs `1.889800` at 200. Final capture improved `.633374 -> .643801`,
50-step lag fell `13.624978 -> 11.620014`, alignment rose
`.767387 -> .778025`, and effective rank rose `102.79 -> 104.21`; source loss
was effectively level/slightly worse (`2.947005` vs `2.946424`). The harmonic
start is therefore released; fixed `.01` remains the steady-step ablation.

Aurora depth was then reopened under mature rank-128 Oja without treating old
rank-256 results as a baseline. One Polar Express pass with only two NS steps
was decisively underpowered: `pp=1/ns=2` reached `1.956012` at step 200 versus
`1.886524` for `pp=2/ns=5`, with a 16% smaller update norm. Restoring five NS
steps isolated the pass count. `pp=1/ns=5` kept the same update norm but reached
`1.903529 / 2.943642` target/source at 200: slower target adaptation and slightly
better source retention than `pp=2/ns=5` (`1.886524 / 2.947005`). Its lower
alignment/effective rank therefore describes a gentler leverage map, not a
smaller step. Runs: `o2jt5qsc` (`pp=1/ns=2`), `wbjmebwo` (`pp=1/ns=5`), and
`ytqnw1ip` (`pp=2/ns=5`).

The chosen product-shaped rank-128 configuration then ran `pp=1/ns=5` for 1k
steps at LR `3e-4` (`cqokmxft`). Target/source finished at
`1.693917 / 3.010227`, with target loss falling monotonically at every 100-step
evaluation, capture `.674401`, alignment `.860838`, effective rank
`113.46 / 128`, and `.795854 s/step`. Against the earlier rank-128
`pp=2/ns=5`, LR `2e-4` Oja run, this chosen configuration gains `.038294`
target loss for `.019536` source loss. Because Aurora pass count and LR changed
together, this is evidence for the configuration, not attribution to either
axis. Historical rank-256 results remain capacity references only; rank 128 is
the current cost/quality choice and future beta or conditioning tests must hold
it fixed.

A strict projected-moment beta `.95` sensor then changed only beta relative to
that chosen contract (`5lkubxig`). At step 100 it was slightly worse on both
target/source (`1.922924 / 2.944701` versus `1.920719 / 2.942897`). By step 200
it had moved farther on both: target improved `1.815352 -> 1.808193`, while
source worsened `2.965921 -> 2.972707`. This is a small target/source trade, not
the catastrophic failure of the old `.98` lane and not a promotion result.
Beta `.9` remains the default because it owns the 1k evidence; any longer beta
test must preserve every other axis.

The fixed-50-step lag probe was removed after this question closed. It cost
about `233 ms` per event, amortized to `~4.7 ms` per optimizer step, and its
snapshot rented a second full basis frame. Removing it reduced reported state in
the rank-128 1k run from the prior `146.2 MB` class to `98.0 MB`. Step angle,
capture, alignment, and moment spectrum remain available for the live questions.

The matched compiled rank-128 1k promotion comparison showed a real but slight
target/source trade, not Pareto dominance. Oja crossed EIGH around step 400 and
finished slightly better on target loss (`1.732211` versus `1.733106`) while slightly worse
on source loss (`2.990691` versus `2.989094`). Final held-frame capture was
materially better (`.681027` versus `.653297`); alignment and moment effective
rank converged to the same endpoint. Over a comparable 50-step horizon, Oja lag
mass fell cleanly to `8.1480` while EIGH remained at `56.6568`, and Oja's step mass
declined to `.7193` rather than chasing EIGH's noisy boundary targets.

Runs: EIGH `0vfocafl`; Oja `0knxstwb`. Oja paid `1.292748 s/step` versus
`.944404 s/step`, a 36.9% training-time tax (`12,674` versus `17,349` tokens/s).
Profiling located the tax in 92 tall tangent SVDs plus 92 QR/Procrustes
stabilizations per step. The semantics-preserving replacement computes the exact
geodesic from the `r x r` tangent Gram, applies one near-identity Polar Express
correction in the same gauge, and batches equal-rank eigendecompositions within
each parameter group. On the rank-128 product topology, synthetic optimizer-only
time fell from `616.9 ms` to `122.1 ms`; the non-refresh EIGH baseline was
`83.0 ms`, so Oja's incremental tax fell from `533.9 ms` to `39.1 ms` without
changing cadence, step size, initialization, or moving-frame coordinates.

The matched rank-128 500-step replay confirmed both semantics and walltime. The
optimized Oja trajectory reproduced the old implementation at step 500 to
`0.0000017` target loss and `0.0000076` source loss. Against a fresh EIGH control,
Oja reached `1.782755 / 2.974067` target/source versus EIGH's
`1.783133 / 2.972688`, while training at `.8318 s/step` versus `.9229 s/step`
(`19,697` versus `17,752` tokens/s). Runs: Oja `pc69yp41`; EIGH `ji7sf72c`.
Per-gradient Oja was therefore 9.9% faster in the matched training contract, not
merely faster in the synthetic optimizer profile. The avoidable implementation
tax was no longer a promotion blocker.

A full-path profiler then separated ordinary from telemetry steps. It identified
three remaining accidental costs: a full-size Adafactor reconstruction temporary,
per-matrix Aurora health eigensolves, and serial post-EIGH Oja geometry. Separable
Adafactor scaling plus batched equal-shape health and Oja geometry reduced the
9:1 telemetry-weighted synthetic optimizer path from `131.5 ms` before this pass
to `100.5 ms`. The final replay `dpiwqydb` reached `1.783078 / 2.973868` at
`.803951 s/step` (`20,379` tokens/s), versus fresh EIGH control `ji7sf72c` at
`1.783133 / 2.972688` and `.922948 s/step`. It also stayed within `0.000323`
target and `0.000199` source loss of the preceding optimized Oja replay, while
capture (`.66399`), 50-step lag (`9.6092`), alignment (`.83559`), and effective
rank (`110.30 / 128`) held. Oja is therefore 12.9% faster than EIGH in the final
matched training contract, and the Phase 1 promotion gate is satisfied. This
does not erase the slight source-retention cost. Sparse Oja with a larger step is
a later estimator-variance trade, not a free equivalent of per-gradient Oja.

### R6. Do cutoff planes poison stable planes?

Measure per-plane target stability and contribution to capture. A stable-prefix
rotation is interesting only if cutoff churn demonstrably harms useful planes.
Otherwise all-plane Oja remains the simpler released policy.

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
- reduces measured tracking walltime;
- repairs a demonstrated faithfulness failure;
- improves capture and evaluation under drift.

When evidence moves an answer, update this ledger first. Put full run records in
the archive only after the current conclusion has been distilled here.
