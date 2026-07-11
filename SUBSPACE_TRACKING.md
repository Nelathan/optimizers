# Subspace Tracking Sidequest

Deep-dive companion to `PLAN.md`. Scope: how SumoTrack's projection basis is
refreshed over training — the Grassmann geodesic tracker, what we fixed, what we
measured, and the masterplan for making tracking actually *earn* its place. This
started as "audit basis-update fidelity vs SubTrack" and grew large enough to
deserve its own file.

`PLAN.md` remains the top-level direction map; this file owns the tracking arc.
Read the Masterplan section for where we're going; the rest is the terrain we
already crossed so we don't re-derive it.

---

## Why this exists

The projection basis decides *which* subspace of each gradient SumoTrack keeps.
Init picks a basis once (side-Gram `eigh`); tracking rotates it over time so it
follows the moving gradient signal instead of going stale (LoRA-style adapter
saturation is the analogy). SubTrack's `track_the_subspace` is the reference
mechanism. The question was whether our port was faithful and whether tracking
was doing anything at all. Answer: it was neither faithful nor active until this
sidequest fixed both.

---

## Durable facts (what we learned, load-bearing)

### The geodesic retraction, done right
- **Basis refresh is a Grassmann geodesic retraction**, ported from SubTrack's
  `track_the_subspace` (`../SubTrack/low_rank_torch/low_rank_projector.py`). It
  replaced a first-order QR retraction (`QR(Q − step·tangent)`). The geodesic
  rotates the `[Q@V, U]` principal-angle frame by `cos/sin(step_size·σ)` where
  `σ` are the tangent's singular values. It is **exactly orthonormal by
  construction** — no trailing QR/re-orthonormalization needed — *once the frame
  is built correctly*.
- **k=1 — drift, not spin.** SubTrack rotates the *single* dominant tangent
  direction per refresh (`rank_k_matrix_estimation(..., k=1)`); the rest of the
  frame is carried unrotated by the `(I − V Vᵀ)` term. That is the intended
  behaviour and it is what we now do. An earlier pass of *this* port had quietly
  set `k=eff_rank` (rotate the whole spectrum at once) and back-justified it in
  this doc as a "faithful spectral generalization." It was not a shared decision
  and not the heading — it was an unacknowledged divergence introduced by the
  port. Full-spectrum rotation *spins* the basis on the near-isotropic noise tail
  and thrashes (this is the step-5/step-50 fragility we measured). Rank-1 lets the
  basis **drift**: one controlled correction per refresh, noise averaged out
  *through* the motion across refreshes rather than tracked. The Touge image is
  exact — drift the line, don't spin the tires.
  - Multi-rank rotation is **parked, not dead.** It may be worth revisiting, but
    it is not the priority and was never signed off. The heading is drift.
  - k=1 vs *stacked* rank-1 (accumulate per-step top-1 tangents, then rotate) is
    the genuinely open axis — see the Masterplan. That is a real research
    question; k=1-vs-full-spectrum is not.

### The frame bug (the real bug tests hid)
- **SubTrack's geodesic is defined for a COLUMN-orthonormal basis `Q:[dim,rank]`**
  with tangent `T = (I − QQ.T)·partial` (partial projected off Q's *column*
  space, so the left-singular vectors `U ⊥ Q` and `[Q@V, U]` is a valid
  orthonormal 2k-frame). Our port projected the tangent off the *wrong* side of
  the basis on both RIGHT and LEFT branches, so `U` was **not** orthogonal to the
  basis. The `[Q@V, U]` frame was not orthonormal, and every real rotation
  corrupted the basis (orthonormality error `~3e-3`).
- **Fix:** canonicalize both sides to column-orthonormal `[dim,rank]` form (RIGHT
  is stored row-orthonormal `[rank,n]`, so transpose it and the gradient into
  canonical form, run one shared geodesic, transpose back). LEFT is already
  canonical. After the fix, orthonormality holds at `~1e-6` through 20 real
  rotations on both sides.
- **Why tests missed it:** raw `σ` on a well-fit normalized basis is `~1e-8`, so
  `sin/cos(step·σ) ≈ no-op`. Orthonormality "held" only because the basis wasn't
  rotating at all. The k=1 formula test and the orthonormality test both passed
  in the no-op regime. **Passing tests over a mechanism that never fires is not
  evidence.** Only forcing a real rotation exposed the bug.

### `step_size` units and the σ-normalization detour (rejected)
- SubTrack runs `track_the_subspace` on the **raw gradient**, so their `σ` carries
  gradient magnitude and their default `st_step_size=1e4` is tuned to it. We
  normalize the spectral input to unit norm (needed for a neutral basis at init,
  see `fit_eigh`) **and** run adafactor upstream (inflates grad norm ~7→~2k), so
  our raw `σ` is both tiny (`~1e-4`) and, before the frame fix, *drifting*.
- We briefly normalized the rotation angle by `σ_max` to make `step_size` a
  direct top-angle in radians. **Rejected and reverted.** Two reasons: (1) the
  `σ` "drift" that motivated it was the frame bug injecting garbage into the
  tangent — with the frame fixed, `σ` is stable on its own; (2) normalizing by
  `σ_max` forces a constant top-angle every refresh, which **re-inflates the
  residual tail and prevents convergence** (the tracker can never take *smaller*
  steps as it settles). The faithful raw-`σ` form is **self-annealing**: big
  residual → big `σ` → big step; well-fit → small `σ` → small step → settles.
  That self-annealing is the point; don't remove it.
- **`σ_max` vs `‖tangent‖_F` for any future normalization:** if you ever
  normalize again, `σ_max` beats Frobenius norm. Frobenius couples the top angle
  to the spectrum's *shape* (shrinks it for high-rank/flat tangents),
  reintroducing exactly the drift we're removing. But the current answer is: **do
  not normalize; use raw `σ`.**
- *(Superseded description above:* "we normalize the spectral input to unit norm"
  *was true until this was fixed — see next entry. The input normalization is GONE.)*

### The σ poison and the adafactor RMS inflation (two coupled fixes)
Two bugs sat on top of each other; fixing one exposed the other. Both are fixed.
- **Bug 1 — per-step Frobenius normalization poisoned σ.** `update_grassmann`
  divided the gradient by its own norm each refresh (`work_matrix / norm`). This
  made σ a *scale-free ratio*: structurally blind to residual magnitude, so it
  could not self-anneal. Measured smoking gun: on the real harness (rank32, bs16),
  σ read **flat ~0.066 for BOTH eigh and random init** — identical, no anneal. A
  good basis and a garbage basis looked the same. Removed the normalization.
- **Bug 2 — adafactor dampening inflates magnitude 261×.** Removing Bug 1 made σ
  explode to **~250,000** (angle ~1.3M rad → `cos/sin` of noise → basis flung to a
  random orientation each step, step time 0.53s→1.25s). Root cause: our simplified
  Shazeer-form `_adafactor_dampen_full_grad` outputs a gradient with **RMS≈1 per
  element**, whose Frobenius norm is `√(m·n)` — e.g. ~2610 for a 6656×1024 MLP grad,
  ~261× a norm-10 raw grad. σ scales *quadratically* with magnitude, so it blew up.
  This is *why* the projected-grad norm was ~1200–2000 (it appeared *with* adafactor,
  as the user recalled) — not projection geometry (orthonormal projection can only
  *shrink* norm), an RMS-vs-norm unit mismatch. **The old Frobenius normalization
  had been masking this all along; EMA and Newton-Schulz don't care about input
  scale, so nothing complained.**
- **Fix:** rescale the dampened gradient to the raw grad's **RMS** (`dampened *
  grad.rms()`), preserving adafactor's SNR *direction* while restoring *magnitude*.
- **Result (measured, fixed code):** σ sane and **annealing** (eigh 0.229→0.036,
  random 0.134→0.040, ~9× decay as the basis fits — see Q6). projected-grad norm
  back to the **pre-adafactor ~2–4.6 regime**, so `projected_grad_clip_norm`
  reverted `2000 → 2` (the old 2 was calibrated to exactly this scale; 2000 was a
  symptom of the blowup, not a real threshold). Clip ratio left as-is; observe.
- *Open — `step_size = 5` is the PERSISTED-BUT-TEMPORARY default.* It was tuned
  against the *inflated/normalized* σ scale; σ is now ~10× smaller and annealing, so
  5 is **not recalibrated** for the fixed regime. It is persisted as the default in
  both `SumoTrack(grassmann_step_size=5.0)` and the harness `--grassmann-step-size`
  because it is the best current estimate and beats the old placeholder `1e4` — but
  it is explicitly a placeholder, marked `# TEMPORARY` in code. **Recalibrate on the
  fixed/clipped code before trusting any result that depends on it.** (Feeds
  Q-step-size.)

### The three clips, and what each does (and does not) protect
Three independent guards, each protecting a different consumer. They are NOT
interchangeable — the session learned this the hard way.
- **σ clip (`grassmann_sigma_clip`, default 0.1).** Caps the tangent singular value
  → bounds the *rotation angle* so one noisy batch can't fling the basis far. Fires
  during the early warmup (σ > 0.1), inert once σ anneals below it. Verified it caps
  the reported σ (test). **Safety cap on step magnitude — NOT denoising and NOT a
  fix for basis quality decay** (measured: clipping σ made erank decay slightly
  *worse*, not better — so the decay is an *aim* problem, not a magnitude one).
- **Raw-grad clip (`grad_clip_norm`, default 8).** Clips the raw gradient BEFORE
  adafactor/basis/projection. This is the load-bearing one. A blip batch (grad norm
  spiking ~180× to 1481) otherwise poisons adafactor's row/col second moment:
  `grad_sq` of the blip is ~30000× normal, and at `beta2=0.99` that spike decays
  over ~100 steps, over-dampening whole gradient directions for the entire window →
  basis tracks a *starved* residual → **alignment collapses for the whole back half
  of the run** (measured: a single step-54 blip tanked alignment 0.71→~0.5 over 45
  steps). Clipping the raw grad to 8 stops it at the source: same seed/blip batch,
  grad_norm now capped 1481→8.8 at step 54, **alignment holds 0.722 instead of
  collapsing**, erank recovers to the healthy 23.7. (max grad_norm 16.3 over the run
  → the clip also catches smaller spikes, doing steady work.)
- **Projected-grad clip (`projected_grad_clip_norm`, now default 0/OFF).** Clips the
  *projected* grad, downstream, protecting only the moment EMA. **Turned off:** raw
  clipping bounds the gradient upstream of everything, making this redundant. And it
  could *never* have stopped the blip collapse — the damage is in adafactor's
  second-moment state (upstream), which a moment-only clip does not touch. The old
  2000 value was a symptom of the adafactor RMS inflation (now fixed), not a real
  threshold.
- **Separation of concerns proven:** the blip collapse (adafactor poison, fixed by
  raw clip) and the *slow* alignment/erank decay (noisy single-batch aim, needs the
  averaging lattice) are TWO DIFFERENT problems. The raw clip fixed the first; the
  second persists (alignment still gently declines 0.834→0.722, same as a healthy
  no-blip run) and is the lattice's job.

### Diagnostics
- **`rotation_angle`** (radians) = `step_size·σ₁`, the geodesic angle of the one
  rotated direction, computed inside `update_grassmann`. 0 = basis static (no-op /
  well-fit direction), grows without a ceiling = a bigger turn. **Monotonic in how
  far the basis turned.** Superseded `rotation_energy` = `Σ|sin(step_size·σ_i)|`,
  which was dishonest: `sin` peaks at 90° and *comes back down*, aliasing a large
  wrapping rotation as a small one. Both avoid the chordal float-noise floor; only
  the radian never lies about magnitude. Threaded projector → optimizer
  (`mean_rotation_angle`) → harness/wandb. (Historical run tables below still quote
  the old `rotation_energy` — that is what was logged at the time; kept accurate.)
- **`chordal` was removed.** `basis_rotation_chordal` (principal-angle chordal
  distance between consecutive bases via `svdvals(old.mT @ new)`) has a **noise
  floor that grows with rank** — at rank 256 it reported `~0.52` for a basis that
  had not moved at all (raw diff `7e-8`), because it sums `√Σ(1−σ²)` over 256
  near-degenerate angles of float noise. It silently mismeasured the exact regime
  we care about. Do not bring it back for high-rank tracking.
- **`tangent_sigma_max`** (`last_tangent_sigma_max`) is logged as the tangent
  scale sensor. Stable `σ_max` = healthy geodesic; rising/runaway `σ_max` = the
  frame bug or a divergent step. After the frame fix it is stable across
  `step_size` from 0.1 to 500.

### Step 0 calibration result (rank 64, bs16, 200 steps) — NOTE: pre-k=1 (spin)
> **Superseded framing.** This entire sweep was run under the old full-spectrum
> rotation (`k=eff_rank`, "spin"). Its `rotation_energy` column is the *spin*
> energy (16 at step 50 = 64 directions each rotating), not the drift energy. The
> "step_size≈50 optimum" is therefore a spin artifact — under k=1 the honest
> step_size is far smaller (see the drift-vs-spin subsection below). Kept for the
> record; do not use its step_size number for the drift path.

The first masterplan step: find a `step_size` where tracking is neither an
idle no-op nor thrashing into noise. Coarse sweep `{0.1, 5, 50, 500}`:

| step_size | rotation_energy | σ_max      | erank (max /64) | alignment (max) | target eval | retention |
|-----------|-----------------|------------|-----------------|-----------------|-------------|-----------|
| 0.1 (base)| 0.04 flat       | 0.07 stable| ~40             | ~0.72           | 1.875       | 2.956     |
| 5         | 2.2             | 0.08 stable| 50.8            | 0.79            | 1.878       | 2.962     |
| **50**    | **16**          | **0.08 stable** | **57.8**   | **0.883**       | **1.871**   | 2.956     |
| 500       | 47              | 0.07 stable| 54.7            | 0.820           | 1.912       | 2.952     |

- **`σ_max` is stable at every `step_size`, including 500.** The frame fix made
  the geodesic *unconditionally* stable across four orders of magnitude. The old
  blowup (run `d9q9vh1o`: σ 9e-5→0.18, chordal→6.8) was the frame bug, not an
  inherent instability.
- **`step_size≈50` is a genuine optimum, not monotonic.** erank and alignment
  *peak* at 50 and *regress* at 500 — over-rotation past 50 chases each batch's
  noise and degrades the subspace. This is single-batch fragility: no averaging
  to pull back a rotation toward a spurious direction.
- **Better subspace, but loss barely moves.** step 50 beats the 0.1 baseline by
  `0.004` target. Subspace metrics move enormously (erank +45%, alignment +23%);
  loss does not follow. At rank 64 the bottleneck is too loose — erank maxes at
  57.8/64, never saturating the budget — so subspace quality is **not binding on
  loss** at this rank/horizon. Consistent with the standing PLAN fact that basis
  size affects *convergence*, not loss directly, on 1k-step runs.
- A real projected-grad spike around step 170 (projected grad norm → ~3500)
  transiently hurt erank/alignment. A blip event, not a step-size property; the
  harder-rotating step-50 run recovered from it, the gentle step-5 run lingered.
- **Working value: `grassmann_step_size ≈ 50`** for rank 64. (Constructor default
  remains `1e4` — SubTrack's raw-σ number — pending a decision to change it; the
  50 is the calibrated value for *our* normalized+adafactor σ scale at this rank.
  Not yet set as the default; that's a crossroads we stopped at.)

### Drift vs spin, measured (rank 64, bs8, interval 20, 100 steps, synth)
After switching the port to `k=1` (drift), a 3-arm run isolates what the rank of
rotation actually changes. Same seed, same everything but the rotation:

| arm | rotation | `step_size` | rotation_energy | erank /64 | alignment | last train loss |
|-----|----------|-------------|-----------------|-----------|-----------|-----------------|
| spin | full-rank | 50 | **16.6** | 54.1 | 0.825 | 2.113 |
| hot drift | k=1 | 50 | 0.55 | 50.4 | 0.764 | 2.116 |
| gentle drift | k=1 | 5 | 0.35 | 48.8 | 0.733 | 2.116 |

- **`rotation_energy` cleanly separates spin from drift (~30×).** Full-rank sums
  `|sin(step_size·σ_i)|` over all 64 directions; with the tail near-degenerate
  noise, every one contributes, so the basis is churned across 64 axes each
  refresh. k=1 rotates one axis → energy drops below 1. The diagnostic does its
  job: it *names* the spin.
- **k=1 makes `step_size` honest.** At σ_max≈0.07, `step_size=50` gives an angle
  `50·0.07 ≈ 3.5 rad` — past 90°, wrapping — even for a *single* direction. That
  is why 50 was never a "gentle" number; the old calibration was tuning the spin.
  Under k=1 the drift regime (small angle, a few degrees per refresh) wants
  `step_size` on the order of `1–5`, not 50. The knob only becomes meaningful once
  the algorithm is honest — which is exactly why we did not promote 50 to default.
- **The uncomfortable part: loss does not move.** All three arms land within
  `0.003` train loss, and spin's erank/alignment are marginally *higher*. At
  rank 64 over 100 steps the basis churn neither helps nor hurts loss. This
  re-confirms the standing fact — **basis quality does not bind loss at loose
  rank/short horizon** — and means the drift fix is an *honesty* win (correct
  mechanism, honest diagnostic, meaningful knob), **not yet a demonstrated loss
  payoff.** Proving payoff still needs the parked experiment: a real bottleneck
  (tighter rank) or a long enough horizon for staleness to bite. Drift is the
  precondition for that experiment to mean anything; it is not itself the result.

---

## What did not work

- **`σ_max` rotation-angle normalization.** Reverted; see detour above. It fought
  a drift that was really the frame bug, and it destroyed self-annealing
  convergence.
- **Chordal as the rotation metric at high rank.** Rank-dependent noise floor;
  replaced by `rotation_energy`.
- **SubTrack's full-rank grad accumulation buffer — DEAD.** SubTrack accumulates
  the raw `[m,n]` gradient across the refresh interval (`accumulated_grad +=
  full_rank_grad`), then feeds `mean = accumulated_grad / interval` to
  `track_the_subspace`. This is a real architectural contradiction in their
  design: the whole pitch of low-rank optimization is *never materialize
  full-rank state*, and they reintroduce a full-`[m,n]` buffer (same size as one
  moment tensor) to steer the basis. **We will not copy this.** The *goal* of the
  buffer — averaging the gradient to sharpen the residual's low-rank structure so
  k=1 has a dominant direction to rotate toward — is valid; the full-rank storage
  is not. See Masterplan approach B for how to get the averaging benefit with
  low-rank storage.

---

## Masterplan (where we're going)

Step-0 (step_size calibration) is done. This is the arc it was step 0 *of*.

Framing insight that drives everything below: our per-refresh *single-batch*
tangent spectrum is **nearly flat** (measured: σ[1]/σ[0]=0.994, top-1 holds 0.8%
of energy). At the loose over-provisioned rank we measured it, that flatness is
the success signature of a well-fit basis (residual left over is near-isotropic
noise). But that same flatness is the whole problem for tracking: **the failure
of the residual is mostly noise, and that noise is the thrashing.** Rotating the
top-1 direction of a *single noisy batch* drifts toward a direction that is
partly garbage. SubTrack sidesteps this by averaging: they accumulate the grad
over the interval and take the top-1 of the *averaged* residual, which is
genuinely low-rank. We reject their full-`[m,n]` accumulation buffer (see below),
so the masterplan's job is to get the same noise-averaging without full-rank
state. The candidate forms live in **The accumulation design space** (the lattice)
below, and the questions they must answer live in the **Question ledger**. The
answer is drift + averaging, never full-spectrum spin — but *which* averaging is
an open space to evaluate, not a settled pick.

### Contact vs storage (the governing principle)
We always need **full-rank contact** with the gradient — the out-of-subspace
residual only exists there, and the EMA (which lives inside the current subspace)
is blind to it. But contact ≠ storage. The raw grad exists transiently in the
backward pass; we can extract a low-rank tracking signal from it without ever
*storing* `[m,n]`. Every approach below touches full-rank grad, none stores it.

### Session findings: random-init rank-32 step_size sweep (bs8, per-step, 200 steps)
Stress test — random init + tight rank 32 + per-step update, to make tracking do
real work (eigh init was too good/un-stale for loss to move). Swept step_size
{5,10,50}. Read on wandb; do not re-tabulate.
- **Decision: `step_size = 5`, sweep closed.** Best p90 projected-grad ratio
  (least noise surviving), σ *annealing* (0.092→0.077), still climbing at 200 →
  should give a strong basis over 1k with a smoothed direction + better init.
- **σ is ~stable across a 10× step_size range (~0.07–0.09).** Best hypothesis:
  **adafactor** — the basis/residual is SNR-shaped (adafactor was chosen so
  selection tracks signal-to-noise, not magnitude), so the tangent's top σ is a
  *ratio* that neither inflates nor collapses. Consequence: step_size is a pure
  angle knob multiplying a near-fixed σ. Not a "spectrum shape" story.
- **No breaking point in the useful range.** step_50 did *not* tumble (I predicted
  it would): σ still annealed, loss still fell. Because σ self-anneals *faster*
  under a big step (fast fit → small residual → small σ → small angle). But
  step_50's rank is suspect — likely noise-inflated erank (EMA thrashing, p90 up),
  not real utilization. Higher step_size not worth testing; 50 already looks worse.
- **Loss is basis-independent even here.** All arms ~2.0 train loss at a real
  rank-32 bottleneck. Confirms the lever is not "rotate harder" — it is **aim**
  (smooth the direction), which is what accumulation buys. Caveat: bs8 (default is
  bs16, noisier), train-loss only (no eval), single seed — the ranking is soft
  until the eval baseline exists.

### Measured baseline behavior — per-step top-1, no accumulation (lattice N=1)
This is the degenerate corner of the lattice (rank-1 tangent, no buffer, rotate
every step) — what the rank-32 sweep actually ran. Findings that constrain the
space:
- **"A_cheap" (interval > 1, no accumulation) is DEAD.** Doing single-batch updates
  *less often* can't beat doing them every step — it just samples the same noise
  more sparsely. Every-step or accumulate; nothing between.
- **Per-step-no-accumulation converges the basis, but slowly and never sharply**
  (rotation flat, basis wobbles toward but doesn't settle) — because the
  single-batch top-1 direction is partly noise (see Q9). This is the empirical case
  *for* accumulation, and the reason the lattice exists.

### Accumulation geometry — can tangents even be smoothed? (leads, to eval)
SubTrack never smoothed *tangents* (it averages the raw grad, then computes one
tangent). We are off their map here, so these are open and must be measured.
- **SubTrack top-1 confirmed from source.** `track_the_subspace` builds the full
  `[m,rank]` tangent, then `rank_k_matrix_estimation(tangent, k=1)` keeps only the
  top singular triple and rotates by `cos/sin(step_size·σ₁)`. Our port matches.
  So "rank-1 SVD of the tangent → one `[m]` vector" is *exactly* what SubTrack
  already computes each step — accumulation just buffers/EMAs that vector instead
  of consuming it immediately. Clean insertion point.
- **Reducing to rank-1 first avoids Karcher.** If we keep only the top-1 `[m]`
  vector, we accumulate *ambient vectors in `[m]`*, not points on the Grassmannian
  → linear sum/EMA is legal, no Riemannian barycenter (Karcher) needed. Karcher
  only enters if we average rotations/subspaces *directly*. Rank-1-then-accumulate
  sidesteps it by construction. (mergekit's Karcher = the correct tool *if* we ever
  averaged subspaces; we don't. mergekit's SLERP/Multi-SLERP = spherical vector
  averaging, relevant only if per-step directions swing widely → normalize the EMA.)
- **The parallel-transport question (the real geometric risk).** Each step rotates
  Q→Q', so the tangent space moves: a tangent from step 1 lives in `T_{Q₁}`, not
  `T_{Q₂}`. Averaging tangents across *different* tangent spaces is the geometric
  sin. It is only a good approximation when the per-step angle is *small* (frames
  ≈ coincide, error is 2nd-order). At step_size 5 the angle is ~20–30°, which is
  **not obviously small** — transport error might be real. Same condition both
  ways: the drift regime (gentle enough to converge) is also the regime where
  naive tangent-EMA is *approximately* valid. **Must measure, not assume:** after a
  Q→Q' rotation, log `‖(I−Q'Q'ᵀ)·old_tangent − old_tangent‖ / ‖old_tangent‖` — how
  much the old direction sticks out of the new tangent space. Small → naive EMA
  fine. Large → need parallel transport (Karcher-flavored, the cost SubTrack dodged
  by never smoothing tangents).
- **CAVEAT for the baseline: eigh ≠ the tracked basis's target.** eigh picks
  directions by side-Gram magnitude; adafactor upstream reweights directions by
  SNR. So the eigh baseline and a tracked basis are optimizing *different* notions
  of "important direction" — eigh is not the ground truth the tracker should match,
  just a strong static reference. Don't read "tracked ≈ eigh" as success/failure
  of tracking per se.

### The accumulation design space (the lattice — evaluate, do not collapse)
This is a *space of candidates to evaluate*, not a ranked list with a chosen
winner. The axes interact and the signs are unknown; picking one arm early throws
away the comparisons that tell us which axis dominates. Do not prune before
measuring. The candidate forms on the table:

| # | What is accumulated | How | Reset on rotate? | Storage | Karcher? | Transport? |
|---|---|---|---|---|---|---|
| C1 | full tangent `[m,rank]` | running **sum** → /N | yes (clear buffer) | O(m·rank) | no (ambient) | none (single rotate) |
| C2 | full tangent `[m,rank]` | **EMA** | no | O(m·rank) | no | **needed** (EMA spans frames) |
| C3 | rank-1 of tangent `[m]` | sum or EMA | sum:yes / ema:no | O(m) | no | ema:needed |
| C4 | rank-n of tangent `[m,n]` | sum or EMA | " | O(m·n) | no | ema:needed |
| C5 | stack of rank-1 `[m,N]` | stack, SVD at refresh | yes | O(m·N) | no | none (stack in one frame) |
| C6 | **rotations** (solved on manifold) | buffer + Riemannian mean | yes | O(rotations) | **YES** | n/a (already on manifold) |

Two orthogonal design axes cut across the table, and a third:
- **What we reduce to before accumulating** (full tangent → rank-n → rank-1):
  smaller = cheaper + keeps us in ambient vector space (linear averaging legal,
  no Karcher). Richer = more of the residual retained but risks re-introducing the
  noise the reduction was meant to strip.
- **Sum-then-reset vs EMA-no-reset:** a *reset buffer* (C1/C3-sum/C5) accumulates
  within one frame `T_Q`, rotates once, clears — **no transport needed**, clean
  geometry, but rotates only every N steps (not true per-step drift). An *EMA*
  (C2/C3-ema/C4-ema) rotates every step (true drift) but its running average spans
  *changing* frames `T_{Q_t}` — **transport is required or the average is a
  geometric sin.** This is the core tension: per-step drift ⟺ transport cost.
- **Accumulate directions vs accumulate rotations (C6):** everything C1–C5 averages
  in tangent/ambient space then exponentiates *once*. C6 instead solves each step's
  rotation and averages the *rotations* on the manifold (Karcher/Riemannian mean).
  This is the only form that needs Karcher — and the only one that never commits
  the average-across-frames sin, because it works intrinsically. Most expensive,
  most correct-by-construction. Keep it in the space; do not dismiss it.

Retired framing: the old "Approach A (per-step, no buffer) / Approach B (stack +
SVD)" split was two points in this lattice (A = C3-immediate at N=1; B = C5).
A_cheap (interval>1 with no accumulation) is dead. The lattice supersedes them.

### C1 measured: sum-of-tangents removes the noise wall (2026-07-10)
C1 is implemented and is now the default (`grassmann_accumulate=True`;
`--no-grassmann-accumulate` is the ablation arm, slated for retirement). Form:
every step computes the canon tangent at the frozen Q and rolls it into an fp32
`[dim, rank]` buffer in optimizer state; at the refresh boundary the basis
retracts once from `buffer / true_count` and the buffer clears. Faithful to
SubTrack's `use_acc_grad` except (a) we buffer the *tangent*, not the full
`[m,n]` grad (full-size buffers violate the state invariant), and (b) we divide
by the true count, not the interval (SubTrack off-by-ones: it accumulates
interval+1 grads and divides by interval).

Geometry note: mean-of-tangents ≠ tangent-of-mean-grad (the tangent is quadratic
in the grad), but the tangent operator `(I−QQᵀ)·E[ggᵀ]·Q` annihilates an
*isotropic* noise covariance exactly (`(I−QQᵀ)Q = 0`), so isotropic noise cancels
in expectation either way; only anisotropic noise covariance survives, and
persistent anisotropic structure outside the subspace is arguably signal.

**A/B evidence (LFM2.5-350M, eigh, rank 64, step_size 5, clip 0.1, eval on):**
- interval 100, bs4, 200 steps (2 retractions): σ at boundary 0.141 (on) vs
  0.213 (off); rotation 0.274 vs 0.410 rad; final val loss identical (1.9697 vs
  1.9703).
- interval 10, bs16, 200 steps (20 retractions): **σ converges 0.014 (on) vs
  0.054 (off)** — the single-grad arm hits a noise wall by step ~200 while the
  accumulate arm keeps annealing; visually cleaner σ curve; erank and loss
  unchanged. The floor ratio 0.054/0.014 ≈ 3.9 ≈ √10 (window size) is exactly the
  noise-cancellation prediction for a noise-dominated tangent at convergence — so
  the wall is *choosable*: window N buys a √N lower floor at zero extra state.
- Converged rotations land at 5×0.014 ≈ 0.07 rad ≈ 4°, just under the 5–30°
  working target; early retractions still hit the 0.1 σ-clip (28.6°). Rail plan:
  hold step_size 5; if aim stays clean, later raise toward clip×step ≈ π/2 (90°).
- **rank 32 stress pairs (clip 2.5, bs16, interval 10, 200 steps):**
  - step_size 10 pair was confounded (rank+step+clip changed at once) and ran hot:
    rotation 3× the rank-64 numbers where doubling step predicts 2× — the extra
    1.5× is rank-32's larger residual raising the σ floor, not step instability
    per se. Recorded, not load-bearing. step_size stays 5.
  - Aurora alignment's U shape is **rank-bound** (user, from the curves): rank 64
    shows the U, rank 32 is flat — not step, not clip. At starvation rank the
    moment apparently never gets the slack that produces the mid-run alignment
    dip-and-recover.
  - step_size 5 pair (clean, single-axis vs the hot pair): **accumulated σ
    converges to 0.0175** vs a visibly noisier off-arm — converged. Grad/moment
    ratio is the lowest measured, i.e. the 10-step window is coherent (denoised
    aim shows up in the moment path, not just the tracker). erank lowest of all
    runs (rank starvation, expected) but **climbing** at 200 steps. Eval loss
    identical on vs off.
- *Lead (user):* interval 20 is probably better (window 20 → √20 floor, still
  frozen-frame legal); staying at interval 10 for iteration speed. Revisit when a
  run wants the lower floor.

### Tracking is maintenance, not acquisition (random-init stress, 2026-07-10)
Random-init rank-32 A/B (step 5, clip 2.5, 200 steps) falsified subspace
*acquisition*: projected grad norm stays low (the basis never learns to capture
the gradient), erank pinned at ~56% and flat, aurora alignment stuck at 0.51 with
a messy spectrum, eval loss worse than the eigh pair and on/off within noise.
Accumulation still wins on its own terms (only the acc arm's σ converges; moment
ratio recovers) — **we aim with less noise, but the aim does not improve the
basis** (user's conclusion, adopted).

Mechanism: **SOLVED in vitro (2026-07-10) — step_size 5 is ~10× above the
stability boundary, and the σ-clip masks the divergence as a plateau.** A
starvation story (tangent multiplicative in capture) was proposed first and
falsified by the run itself (σ starts ~0.2 same as eigh); an eviction-cost story
was proposed next and never needed: the fixed-grad ratchet probe (random init,
rank 32, n=2048, iterate `update_grassmann` on ONE grad) showed capture climbing
0.25→0.61 in 5 iterations then freezing forever with σ pinned at ~0.48 and
rotation pinned at exactly step×clip = 0.5 rad. While σ > clip, rotation is a
CONSTANT angle — self-annealing is dead and the geodesic orbits the target
instead of converging (the same constant-top-angle failure documented for
σ-normalization; the clip reintroduces it whenever σ sits above the rail).
Step/clip sweep, same problem, 400 iters (eigh reference capture 0.926):

| step | clip | capture | final σ | final rot |
|---|---|---|---|---|
| 5.0 | 0.1 | 0.596 frozen | 0.52 | 0.500 limit cycle |
| 5.0 | none | 0.583 | 1.26 | 6.28 (2π — wraps!) |
| 1.0 | none | 0.504 | 1.34 | 1.34 orbit |
| **0.5** | none | **0.871** | 0.037 | 0.018 **converged** |
| 0.2 | none | 0.843 | 0.072 | still annealing |

**The update math is correct; self-annealing works in the stable regime.** Eigh
runs never exposed this because drift-scale σ is small (step 5 × small σ = stable);
acquisition-scale σ is large (orbit regime). step_size 5 must be retired;
step 0.5 is the in-vitro stable candidate, with the clip re-scoped to genuine
blip safety (≈1.0) rather than a rail σ lives above.

*Resolution (user, 2026-07-10):* **σ-clip DELETED** (no blips observed with the
per-tensor raw-grad clip at 2.5 upstream; the clip's failure mode — killing
self-annealing whenever σ > clip — outweighed its safety value). **step_size
default = 1.0** (user lean; smaller step visibly preserved aurora alignment in
the stable-regime run). Rotation is now purely `step_size × σ`, un-capped.

**In-vivo verdict (random init, rank 32, step 0.5, clip 1.0, 200 steps): even in
the stable regime, acquisition is dead — by signal starvation, not dynamics.**
Both arms plateau `basis_capture` at ~0.22 (random-chance territory) after the
first retractions; σ anneals to 0.029 (acc) / 0.060 (single), rotations ~0.014
rad, alignment high-flat ~0.72, final val 2.09. The in-vitro/in-vivo gap is the
explanation: in vitro the residual was 100% persistent (one fixed grad, 400
retractions → capture 0.87); in vivo the mean tangent keeps only the residual
that persists across the window, and at random init the uncaptured mass is
diffuse churn — each window's coherent component is thin, so tiny σ/rotations
are the tracker *honestly reporting* there is no persistent direction to
acquire. 20 retractions × 0.014 rad ≈ 0.3 rad total budget vs ~32×π/2 needed.
Third meaning of σ recorded: small σ can mean well-fit (eigh), no-contact
(orthogonal), or weak *persistent* residual (churn-dominated window).
**Geodesic rank-1 tracking is maintenance-only — established in both the hot
regime (limit cycle) and the stable regime (starvation). Acquisition is closed
terrain; acquisition belongs to decomposition (eigh init / arm-A refit), which
needs no persistence.**

Two audit findings alongside (both 2026-07-10):
- **SubTrack's published code never tracks tall/square matrices.** Its tangent
  projector is `(eye(shape[0]) − QQᵀ)partial`; for the tall/right branch the
  basis is `[r, n]` ROW-orthonormal, so `QQᵀ = I_r` exactly and the tangent is
  identically zero — Σ=0, cos=1, sin=0, basis unchanged. Only the wide/left
  branch runs real Grassmann math. Our port canonicalizes both sides into the
  column-orthonormal frame first, so we run the correct geodesic everywhere —
  but the paper's empirical support for tracking on transformer-shaped (tall/
  square) weights is far weaker than assumed.
- **σ is a contact meter, not a fit meter, with an exact fixed point:** on the
  SAME grad, the eigh basis gives tangent ≡ 0 (σ = 0 algebraically — residual and
  capture live on orthogonal singular directions). Live σ on an eigh basis is
  pure drift signal; live σ on a random basis is residual×capture product. Same
  number, different meanings — never read σ without capture next to it. Capture
  (`||Qᵀg||/||g||`) is now logged as `basis_capture` (the "fit" metric).

Consequences:
- **eigh init is load-bearing, permanently** — not init ceremony. Random stays a
  stress/ablation path only.
- σ convergence must never be read as basis quality on its own; pair it with
  capture (projected grad norm) and erank. σ says "still fitting?", capture says
  "fitting the right thing?".
- The payoff regime for tracking is **maintenance under distribution drift** (the
  product case: eigh basis goes stale as continued pretraining shifts the data),
  not recovery from a bad basis. Q8's horizon run should be read on capture decay:
  off-arm capture should sag as the basis stales; on-arm should hold it.

### Approach 4b (deferred) — EMA-weighted rank replacement
Project the EMA back to see which basis dimensions carry little moment energy →
those are the cheap-to-replace ones → weight the rotation to preferentially move
low-moment-energy directions and preserve high-moment ones. This is a
*weighting on the rank update*, distinct from the tangent-source question above.
Evaluate after A/B give a trusted tracker.

## Question ledger (the real artifact — do not let these fall out of view)
This work is not about reaching a goal; it is about **answering questions**. The
most important discipline here, unlike software engineering: never forget a
question, always note new ones, and update each answer as evidence arrives. Status
tags: 🔴 open · 🟡 partial evidence · 🟢 answered (with the evidence) · ⚫ retired.
When a run produces evidence, update the relevant row — that is the point.

**Q1 — Can tangent spaces be smoothed at all, geometrically?** 🟡
- *Why it matters:* gates the entire EMA/no-reset half of the lattice (C2/C3-ema/C4).
- *Evaluate:* after a Q→Q' rotate, log `‖(I−Q'Q'ᵀ)·old_tangent − old_tangent‖ /
  ‖old_tangent‖` — the fraction of the old direction that leaves the new tangent
  space. Small → naive EMA legal; large → transport (or reset-buffer) required.
- *Evidence so far:* none measured. At step_size 5 the per-step angle is ~20–30°,
  which is **not obviously small** — the user flags 30° as possibly a real sin.
  Unresolved until the probe runs.

**Q2 — Sum-reset vs EMA: does per-step drift beat every-N drift, net of transport
cost?** 🟡 (reset side pinned)
- *Update 2026-07-10:* the reset-buffer side is now measured (C1, window 10, see
  "C1 measured"): clean geometry, σ floor drops √N, converges without transport.
  The EMA side remains unmeasured; the open half is whether per-step drift buys
  anything *on top of* a C1 that already anneals cleanly — and it now has to beat
  a working baseline, not a noise wall.
- *Why:* the central lattice tension. EMA gives continuous drift but pays
  transport; reset-buffer is clean but coarser-grained.
- *Evaluate:* run a reset-buffer arm (C3-sum, N≈10) against an EMA arm (C3-ema)
  at matched step_size, read convergence (σ anneal, alignment climb) and eval loss.
- *Evidence:* none. Do not pre-rank; measure.

**Q3 — What reduction before accumulating: full tangent / rank-n / rank-1?** 🟡
- *Update 2026-07-10:* C1 (full tangent, reduce-once-at-boundary) measured and
  works (√N σ-floor drop). Pre-reduction buffering was chosen deliberately:
  reducing each batch to rank-1 *before* averaging commits to one noisy direction
  per batch and forfeits the pre-SVD noise cancellation. rank-n/rank-1 arms remain
  unmeasured but now need a reason to exist (they save memory only below
  O(dim·rank), which C1 already meets).
- *Why:* smaller stays in ambient space (no Karcher) and is cheaper, but may throw
  away structure. rank-1 is what SubTrack already computes per step (confirmed from
  source), so it's the cheapest insertion point — but "cheapest" is not "best."
- *Evaluate:* compare C3 (rank-1) vs C4 (rank-n) vs C1 (full tangent sum) on basis
  convergence + eval loss at matched everything.
- *Evidence:* none directly. Indirect: single-batch top-1 is noisy (drift wobbles,
  doesn't settle) — motivates *accumulating* the rank-1, not abandoning it.

**Q4 — Is Karcher (C6, averaging rotations on the manifold) worth its cost?** 🔴
- *Why:* it's the only form that never commits the average-across-frames sin, but
  it's the most expensive and needs Riemannian machinery.
- *Evaluate:* only after Q1/Q2 — if naive/reset forms converge well, C6 is
  unnecessary; if transport error is large and reset is too coarse, C6 is the
  principled fallback. Gate on Q1's transport measurement.
- *Evidence:* none. Held as the correct-by-construction backstop.

**Q5 — Does a better basis bind on loss at all, at these scales?** 🟡
- *Why:* the whole payoff question. If loss is basis-independent everywhere we can
  run, tracking is a dead lever and the effort belongs elsewhere.
- *Evaluate:* eval loss (not train loss) across basis-quality-varying arms; tighten
  rank / lengthen horizon to make quality bind.
- *Evidence:* drift-vs-spin (rank64/100 steps) and the rank-32 sweep (bs8/200)
  both showed **train loss unmoved** while subspace metrics moved a lot. But: no
  *eval* loss yet, bs8 not bs16, ≤200 steps. Soft-negative, not settled. The
  baseline (eigh, rank32, step5, bs16, eval on, 200 steps) is the first honest read.
- *Update 2026-07-10:* the C1 A/B ran **with eval loss, bs16, interval 10, 200
  steps**: subspace σ moved 3.9×, eval loss and erank did not move at all
  (1.9697 vs 1.9703 at interval 100; "loss is same" on the int10/bs16 curves).
  Soft-negative strengthened: a measurably better-aimed basis still does not bind
  on loss at rank 64 / 200 steps. Remaining levers before calling it: tighter
  rank, longer horizon (Q8).
- *Rank semantics (user, 2026-07-10):* rank 32 is a deliberate **stress test** —
  the regime where basis aim must bind on loss if it ever will. Rank 256 is the
  convergence config with no rank bottleneck at all (aim binds least there). So
  Q5 is decided at rank 32, not by raising rank: if the C1 on/off A/B shows no
  eval-loss gap under rank starvation, aim is not the bottleneck anywhere
  reachable and geometry refinements (EMA/transport/Karcher) have no prize.
- *Update (rank-32 stress pair, step 5, 2026-07-10):* **eval loss identical on vs
  off even under rank starvation at 200 steps.** Strong soft-negative for the
  horizon we can see. The one live counter-signal: erank is still *climbing* at
  step 200 in the accumulated arm — the basis is still improving when the run
  ends, so the loss question may only open on a longer horizon (Q8). Aim quality
  itself is no longer in doubt (σ converged, coherence up); what's unproven is
  that it *pays*.
- *Update (random-init pairs, 2026-07-10):* **Q5 splits, and half of it is now
  POSITIVE.** Basis quality binds on loss when the gap is large: random basis
  (capture ~0.22) costs ~0.12 nats vs eigh at 200 steps (2.09 vs 1.97 final val).
  What does not bind is the *marginal* improvement tracking adds on top of eigh
  init at these horizons. The lever is real; eigh init already harvests most of
  it. Tracking's remaining case is holding that harvest under drift (Q8).

**Q6 — Does σ track fit, or is it flat?** 🟢 (answered — with a corrected story)
- *Why:* if σ is a fixed artifact, step_size is a pure angle knob and σ carries no
  "am I fit yet" signal. If σ *anneals* as the basis fits, σ is a trustworthy
  convergence readout.
- *Two bugs had to be fixed before σ could answer anything (see "The σ poison and
  the adafactor RMS inflation" in Durable facts):*
  1. A **per-step Frobenius normalization** in `update_grassmann` made σ a
     scale-free ratio — measured *flat at ~0.066 for BOTH eigh and random init*, no
     anneal, structurally quality-blind. (This falsified my earlier handwave that
     "eigh reads 0.014, random 0.08" — that was a different run/regime; under
     normalization σ cannot distinguish init at all.) Removed.
  2. Removal exposed σ **exploding to ~250,000** (rotation angle ~1.3M rad, pure
     numeric garbage): the Shazeer-adafactor dampening outputs RMS≈1 per element,
     whose Frobenius norm is √(m·n) ≈ 261× the raw grad norm. σ scales
     *quadratically* with magnitude, so it blew up. Fixed by rescaling the dampened
     grad to the raw-grad RMS (keeps adafactor's SNR *direction*, restores *scale*).
- *Answer, measured on the fixed code (rank32, bs16, step5, 100 steps):* **σ now
  anneals ~9× as the basis fits** — eigh 0.229→0.036, random 0.134→0.040 (both:
  warmup bump while LR ramps, then decay to ~0.04). Under normalization this was
  impossible (flat 0.066). So **σ is a trustworthy *convergence-rate* readout.**
- *Corrected nuance (do not overclaim):* σ tracks *convergence*, not *quality*.
  Both inits anneal to nearly the same σ (~0.04); the *quality* difference shows up
  in **erank** (eigh holds 23.7, random collapses to 18.4), not σ endpoint. σ = "am
  I still fitting"; erank = "how good is what I fit." Complementary, not redundant.
  My earlier "σ tracks quality" was wrong; the evidence says convergence.

**Q7 — eigh-init vs adafactor-update vs loss: three notions of "important
direction."** 🔴
- `fit_eigh` selects by *magnitude* (side-Gram eigenvectors); adafactor-fed update
  selects by *SNR*; *loss* cares about *massive singular values*. Three different
  targets. Whether the mismatch matters, and whether tracking should be biased back
  toward high-singular-value directions, is open. Also means: **do not read
  "tracked ≈ eigh" as tracking success/failure** — they optimize different things.

**Q8 — Staleness horizon: does the basis go stale enough that tracking pays,
within a runnable horizon?** 🔴 — now THE payoff question
- LoRA-saturation analogy is *why* we track. May only appear on long runs. Evaluate
  by tightening rank (bind sooner) or lengthening (let staleness bite). Tied to Q5.
- *Sharpened 2026-07-10:* acquisition is falsified (see "Tracking is maintenance,
  not acquisition") and maintenance doesn't bind at 200 steps, so staleness-driven
  maintenance is the only regime left where tracking can pay. Read the horizon run
  on **capture decay**: does the off-arm's projected grad norm sag as the eigh
  basis stales, and does the on-arm hold it — and does that gap reach eval loss.

**Q9 — k=1 under a flat single-batch spectrum: coin-flip direction?** 🟢
- k=1 rotates one direction of ~64 near-equal ones; on a single noisy batch that
  direction is partly arbitrary. Accumulation (the lattice) exists to sharpen the
  spectrum so top-1 is real. Full-spectrum rotation is *not* the fix (measured: 30×
  the rotation of k=1, churns the basis, buys no loss — see drift-vs-spin).
- *Answered 2026-07-10:* yes, the single-batch top-1 is noise-dominated at
  convergence — its σ floor (0.054) sits ~√10 above the window-10 mean's (0.014),
  which is the signature of averaging zero-mean noise, and the single-grad arm
  walls there while the accumulated arm keeps annealing. Accumulation is the fix,
  and the floor scales as √window.

**Q10 — How noisy is a single-batch eigh target?** 🟢 (answered 2026-07-11:
churn, exactly at the cutoff, exactly as predicted — but *benign* churn)
- *Answer:* consecutive T1 targets disagree by ~90° at the top principal angle
  (`eigh_target_self_angle` = 1.562 rad ≈ 89.5° at the end of the A′1 run) and
  the cutoff is degenerate (`eigh_target_cutoff_ratio` = 0.976 — the dropped
  33rd eigenvalue is 97.6% of the kept 32nd). T1's largest-angle aim is pure
  rank-cutoff membership churn; `tangent_sigma_max` = 1.561 confirms the basis
  fully snapped a ~90° swap *every boundary*. The mitigating nuance: the churn
  is energy-degenerate, so each swap trades a direction for a near-equivalent
  one — capture landed at 0.438, *between* C1 rank-1 (0.429) and full-spectrum
  (0.442), despite the musical chairs. The prediction on record (stable core,
  churning cutoff) is confirmed on the cutoff half; the "stable core" half is
  untested — the top-1 probe only sees the largest angle, which churn saturates.
- *Consequence for the fork:* T1+R1+largest-angle is aimed at noise. But
  cutoff_ratio ≈ 0.98 also caps the upside of fixing it: when the spectrum is
  that flat at r, boundary membership is genuinely arbitrary and even a
  denoised T2 target must make an arbitrary cutoff call. Denoising stabilizes
  *which* arbitrary call gets made (fewer wasted rotations), it does not create
  a signal that isn't there. Eigenvalue-weighted angle selection has the same
  ceiling.
- *Record correction (2026-07-11):* the target frame is built from the
  **adafactor-dampened** boundary grad, not the raw grad — dampening runs
  before `_refresh_projector` by design ("same dampened gradient feeds both
  consumers"). A chat report claimed raw; the code was right. So the flat
  cutoff is flatness of the *SNR-normalized* spectrum: at rank 32 many
  directions carry near-equal signal-to-noise. Rank-starved in the strongest
  sense — consistent with what we already knew, no rerun needed.
- *Probe demotion (user, 2026-07-11):* worst-case-angle and #33/#32 are
  steepness-at-cutoff reads (one in angle, one in magnitude) and carry little
  ongoing value; overall fit (`basis_capture`) is the read that matters. The
  probe stays wired but is no longer a decision input; do not extend it with
  spectrum plots unless a question demands one.
- *Evaluate:* the gating probe in the A′ design-space section — boundary logs of
  Q-vs-target angle, consecutive-target angle, cutoff eigengap.

**Q11 — Does decomposition aim beat residual-persistence aim on maintenance?** 🟢
(answered 2026-07-11 via Q14: YES, once smoothed — fractional full-spectrum eigh
aim beats buffered tangent tracking on loss and capture with zero state. The 🟡
history below records how the first A/B misled and was corrected.)
- *Why:* the A′ premise. Tangent tracking needs a persistent residual and drift
  outruns it; eigh aim needs no persistence. If A′ holds capture under drift
  better than C1 and frozen, decomposition-aim becomes the forward path.
- *Evidence (A′1 = T1+R1, step 1.0, rank 32, 200 steps, vs the Q13 pair):*
  final val 2.0267 vs C1 rank-1 2.0253 vs C1 full-spectrum 2.0212; capture
  0.438 vs 0.429 vs 0.442; grad p90/moment ratio worse (2.97); alignment/erank
  a touch higher (0.746 / 24.3) but those are rotation-biased reads. Every
  margin within run-to-run noise. Frozen control confirmed clearly worse
  (drift result replicated). User read, concurring: leans original algo.
- *Interpretation:* this run does not test a *clean* decomposition aim — Q10
  shows every A′1 rotation was a full-snap 90° cutoff swap, and the arm still
  tied C1. That is oddly robust (energy-degenerate swaps are cheap), but it
  means the honest comparison needs A′2 (windowed target) or
  eigenvalue-weighted selection — and Q10's cutoff_ratio ≈ 0.98 says the
  achievable margin over C1 is small.
- *Correction (user, 2026-07-11):* the comparison above crosses state classes.
  C1 carries a persistent fp32 `[d,r]` accumulator per matrix; A′1 carries
  nothing. **Within the zero-state class, decomposition aim wins:** A′1 (capture
  0.438) clearly beats the unbuffered single-grad tangent rank-1-per-10 arm
  that drift outran last arc. The honest statement: decomposition aim matches
  buffered tangent tracking *without the buffer*, while snap-rotating pure
  churn. The open question is whether smoothing it (Q14) beats buffered C1
  outright — which would delete C1's `[d,r]` state from the product path.
- *Evaluate (next):* Q14 (stateless manifold EMA) first; A′2 only if Q14 stalls.

**Q14 — Does fractional full-spectrum eigh aim (stateless manifold EMA) beat
both A′1-snap and buffered C1?** 🟢 (answered 2026-07-11: YES, decisively — the
first non-noise-level win of the arc)
- *Answer:* both α beat every anchor on loss AND capture with zero persistent
  state. α=0.5: val 2.0126, capture 0.621, p90 grad/moment 4.25 (hot),
  alignment 0.742, erank 24.2. α=0.25: val **2.0103**, capture 0.554,
  grad/moment 3.35, alignment **0.764**, erank **25.1** (both best-ever).
  Anchors: C1 full-spectrum 2.0212/0.442, C1 rank-1 2.0253/0.429, A′1 snap
  2.0267/0.438. That is −0.011 nats and +0.11–0.18 capture over the best
  *buffered* arm. Both pre-registered predictions held: probe values identical
  across α (~89.5°, 0.976 — they measure the targets; the basis found a stable
  center the target stream orbits), and rotation_angle scales with α
  (16.1 vs 7.8 rad summed over 32 planes ≈ 0.50/0.24 rad per plane) without
  instability (σ_max churn-pinned at ~π/2 in both).
- *The α tension:* higher α chases the current target harder → higher
  instantaneous capture (0.621) but hotter grad/moment ratio (4.25) — rotation
  violence taxes adafactor's second-moment validity. Lower α wins where it
  counts: loss, alignment, erank, calmer moments. Bias–variance in the EMA
  constant, and variance is the binding side.
- *α floor located (2026-07-11):* α=0.125 is worse — val 2.0128, capture 0.517
  (alignment/erank kept rising, 0.769/25.2, but loss is the boss). **Knee at
  α≈0.25.** The rotation_angle trends across the sweep are the orbit read:
  flat = the basis orbits the target stream at a stable radius (it can NOT
  converge to zero while targets churn — the floor is α·Σθ(target noise), so
  flat is healthy, not stuck); *rising* (α=0.5, mildly α=0.25) = the basis is
  being kicked hard enough that its displacement grows, self-inflated orbit.
  α=0.125 flat, α=0.5 rising brackets the same knee the loss found.
- *Interval axis measured (α=0.5 @ interval 20, 2026-07-11):* loses — val
  2.0139 vs 2.0126 (same α @ 10) and 2.0103 (matched EMA rate, 0.25@10).
  Capture (0.622), grad/moment (4.24), and rotation_angle (16.2) are clones of
  0.5@10: **α alone sets the rotation violence; the interval sets nothing but
  the sample count.** User's corrected mechanism: the eigh aim has no buffer,
  so a longer interval denoises nothing — every boundary target carries the
  same single-batch noise, that noise floor is what caps usable α, and more
  boundaries per horizon simply feed the manifold EMA more samples to average.
  The counter-tax on *more* boundaries is the moment-transfer loss (below);
  interval 10 is where those currently balance. Note the earlier "fewer,
  bolder, buffered corrections" framing was wrong for this aim precisely
  because nothing buffers between boundaries.
- *Moment tax of a rotation (mechanism confirmed in code):* on every non-fit
  refresh the first moment is transferred exactly — lifted through the old
  basis, re-projected onto the new (`project(project_back(m))`). Each rotated
  plane keeps cos(αθ) of its moment component; sin(αθ) is dropped
  irrecoverably. For the churned ~90° plane: α=0.5 drops sin²(45°) = 50% of
  that plane's moment energy per boundary, α=0.25 drops 14.6%, C1's mrad
  rotations ~1e-5. This is the mechanism behind aurora_alignment ordering
  monotonically against α (0.769/0.764/0.742). Adafactor row/col state is
  full-size and basis-free — untouched by rotation. New refresh-only metric
  `moment_transfer_retained` (= ||m_after||/||m_before||, 1.0 lossless)
  measures the *total* tax, which code alone cannot predict (depends on how
  much moment lives in the rotated planes).
- *Logging cadence (user, 2026-07-11):* logging at the refresh interval sampled
  basis_capture at a fixed phase of the rotation cycle and hid its sawtooth.
  Harness default `--wandb-log-every` now 5; keep it at ≤ half the refresh
  interval.
- *Still open inside Q14:* does the win hold over a longer horizon (1k steps —
  drift slows after adafactor settles ~100 steps, the knee may move). Default
  promotion: RESOLVED — approved and shipped 2026-07-11 (see the design cut
  below); the C1 accumulator is deleted.

### Position control vs velocity control — the arc's centerpiece (2026-07-11)

Why eigh aim can rotate 22.5° planes every 10 steps without thrashing while
tangent tracking crawled at milliradians and still lost to drift. The two aims
are different control systems:

- **Tangent tracking is open-loop velocity control.** Each boundary applies a
  displacement derived from a gradient — a velocity command. Noise in velocity
  *integrates*: the basis random-walks with no restoring force when a step was
  wrong. The only defense is tiny steps, which is why the honest tangent
  step_size landed at milliradians and drift outran it. Velocities sampled at
  different times point different ways; their average cancels toward zero
  (the user's "all paths lead to Rome, average them and you go nowhere").
- **Eigh aim is closed-loop position control.** Each boundary names a
  *position* (the target frame) and the geodesic contracts toward it by factor
  α. Errors *decay geometrically* instead of integrating: overshoot toward a
  noisy target is pulled back by the next rotation, because targets scatter
  around the same attractor. Positions average to their mean; velocities
  average to nothing. The noise floor limits α but never accumulates.

Corollaries now explained rather than mysterious: C1's σ "converged" because
window-averaging killed tangent *noise* (√N), not because the basis approached
anything — σ measured the thermometer, not the patient (second confirmed
instance of "σ is contact, not fit"). And the Karcher/rotation-averaging lead
is subsumed: the manifold EMA *is* streaming Karcher averaging of positions,
which is the form of averaging that converges.

### Approved design cut (user-approved and shipped 2026-07-11)

1. **Position control promoted to default**: `grassmann_aim="eigh"`,
   `grassmann_rotate_rank=None` (all planes), `grassmann_step_size=0.25`,
   interval 10. The C1 tangent accumulator is DELETED (fp32 `[d,r]` buffer,
   per-step launches, reset/seed logic, `grassmann_accumulate` flag).
   `aim="tangent"` survives only as the SubTrack-faithful single-grad
   reference/ablation arm.
2. **Moment transport is exact and free**: the retraction is a rigid frame
   rotation (`Q_new = R @ Q_old`, pinned by test), so parallel transport of the
   projected moment is the IDENTITY in projected coordinates — the honest
   transfer is a no-op. The old project-back/re-project transfer (orthogonal
   projection, cos-tax) is deleted along with the one-session
   `moment_transfer_retained` metric it motivated. This matters specifically
   for Aurora: NS re-amplifies cos-shrunken moment directions to full strength
   with noise-dominated direction estimates — the sin-drop was worse under
   polar orthogonalization than it would be under plain momentum.
3. **Convergence metric added** (`basis_lag_mean_angle` / `basis_lag_top_angle`):
   principal angles between the basis and its own snapshot 5 refreshes back
   (sine-based, well-conditioned at small angles). Mean = settling read (→0 iff
   every plane stops), top = orbit-radius read (churn planes hold it up).
   Neither rotation_angle nor target self-angle can show convergence — both are
   floored by target noise by construction. The lag angle is also the future
   signal for adaptive intervals: on a 10k run the basis should settle and
   SubTrack's interval-100 becomes legitimate *after* the decay, not before.

**Q17 — How much to turn, when: is there a step size that is both stable and
fast self-decaying?** 🔴 — the open question the cut leaves behind
- *Why:* α=0.25 is the measured knee for THIS drift rate (early adaptation,
  adafactor still settling, rank-starved SYNTH). A fixed α cannot be right at
  both step 10 (fast drift, want big α) and step 10k (settled, want tiny α and
  long intervals). Stochastic-approximation theory wants a decaying schedule;
  the basis_lag_angle decay is the natural controller input.
- *Candidate mechanisms (user, on record):* (a) "mature EMA" schedule — after
  init the boundaries take α = 1/2, 1/3, 1/4, ... down to a floor (~0.1): the
  basis becomes the TRUE running Karcher mean of all targets during the decay,
  then the floor keeps it tracking drift. Polyak-style averaging on the
  manifold, zero state (α derived from the refresh count). (b) Steady α may
  still win when drift is fast — the schedule trades tracking speed for noise
  averaging exactly like any EMA warmup. (c) Prefer deriving α from a
  stateless signal over lag-state feedback if possible.
- *Evaluate:* first just WATCH basis_lag_angle on a longer run at fixed α=0.25
  — if it decays on its own (self-annealing via shrinking target angles), no
  schedule is needed; if it plateaus while loss stalls, the mature-EMA schedule
  is the next arm. Do not build the controller before the open-loop trace.
- *Evidence (open-loop watch, 500 steps, defaults, 2026-07-11):* val 2.2968 →
  1.8956, best-yet loss trajectory (already best at step 200). basis_lag decays
  but slowly and noisily: mean angle ~halved over the run to 0.474 rad, top to
  0.802 — the basis is still moving substantially at step 500, not settled;
  consistent with churn planes orbiting while signal planes drift. Target
  thermometers unchanged (self-angle 1.561, cutoff 0.978 — rank starvation is
  a property of the problem, not the tracker). **aurora_alignment and erank
  rose MONOTONICALLY all run (0.828 / 27.4 = 86% of rank at step 500)** —
  under the old projection transfer, alignment degraded with rotation; with
  transport the moment survives every boundary. Not a controlled A/B, but the
  first run where constant rotation and monotone moment health coexist. The
  capture sawtooth is now visible at log-every 5, confirming the aliasing fix.
  Verdict: partial self-annealing, no plateau-with-stalled-loss yet — no
  controller needed at this horizon; revisit at 1k+ or on a real drift task.
- *Why:* `Q ← geodesic(Q, α, target_t)` at each boundary is stochastic gradient
  descent on the Grassmann sum-of-squared-distances to the target stream — an
  *incremental Karcher mean* whose accumulator is the basis itself (GROUSE uses
  exactly this fractional-step mechanism against noisy observations). This is
  the temporal smoothing A′1 lacked, with no `[d,r]` buffer, no `k×k` state,
  and no new code: `--grassmann-aim eigh --grassmann-rotate-rank 32
  --grassmann-step-size α`. Note a `k×k` buffer could never substitute: the aim
  lives in `(I−QQᵀ)`-space, which Q-coordinates cannot express — stateless
  manifold EMA is the *only* smoothing shape that fits the no-big-state
  constraint, not a compromise.
- *Prediction on record:* the churn angle (~90° at the cutoff, Q10) bounces —
  α·90° toward an arbitrary energy-degenerate direction each boundary,
  capture-neutral, so `rotation_angle` stays ≈ α·π/2 and should NOT be read as
  progress. The other 31 angles are the signal: persistent drift directions get
  consistently closed while churn averages out on the manifold. Expect capture
  ≥ A′1-snap; the interesting outcome is capture ≥ buffered C1, which would
  delete C1's fp32 `[d,r]` state from the product path.
- *Evaluate:* eigh aim, rotate_rank 32, α ∈ {0.5, 0.25}, standard contract
  (rank 32, bs16, interval 10, 200 steps, eval on) vs A′1-snap and the Q13
  pair. Q10 probe stays on — self-angle should be unchanged (~90°, it measures
  the targets, not the basis); Q-vs-target σ_max at steady state is the new
  read: lower than π/2 means the basis found a stable center the targets orbit.
- *Evidence:* none.

**Q13 — Does full-spectrum rotation of the MEAN tangent beat rank-1 drift?** 🟢
(soft-positive, 2026-07-11) — the pre-A′ baseline
- *Answer:* full-spectrum (rotate_rank 32) modestly beats rank-1 on every axis
  that matters, and is stable: capture 0.442 vs 0.429, final val 2.0212 vs
  2.0253, σ_max LOWER (0.0150 vs 0.0166 — converging better), alignment 0.734
  vs 0.730, erank 24.0 vs 23.9. Total rotation 0.099 vs 0.017 rad — but that is
  ~0.003 rad *per plane*: the mean tangent turns "spin" into 32 gentle
  corrections. **The old spin verdict was about single-batch tail noise, not
  about multi-rank rotation per se; C1's window denoises the tail and makes
  full-spectrum rotation safe and mildly better.** Margins are small (0.004
  val, 0.013 capture) and the pair is unseeded — soft, but coherent across all
  metrics in the same direction.
- *Consequences:* (1) rank-1-only is no longer a discipline, it was a noise
  guard — on denoised signals, rotate more; (2) A′'s "largest principal angle
  only" should be evaluated against multi-angle rotation toward the eigh target
  (the `grassmann_rotate_rank` knob applies to A′'s synthesized tangent for
  free); (3) default stays rotate_rank=1 pending the user's call — this was one
  soft pair, not a promotion.
- *Why:* the spin verdict (drift-vs-spin, rank 64) was passed on *single-batch*
  tangents, before C1 and before `basis_capture` existed. Spin thrashed because
  the single-batch tail is isotropic noise — but the window-mean tangent is
  precisely the object whose tail is denoised, so the verdict does not
  automatically transfer. SubTrack averaged ~100 grads and still rotated only
  rank 1; if full-spectrum rotation of a 10-window mean wins on capture, that
  measures how much structure the window actually recovers (and would reshape
  A′'s rank-1-only discipline too).
- *Evaluate:* `--grassmann-rotate-rank {1, 32}` pair (new flag, threaded through
  `update_grassmann_from_tangent`), eigh init, rank 32, bs16, interval 10,
  200 steps, eval on. Read basis_capture primary, eval loss; erank/alignment
  secondary (frozen-biased). Fresh rank-1 arm in the same pair — the harness has
  no seed control, so cross-session curve comparison is not matched.
- *Evidence:* none under the mean tangent. Historical spin evidence (single
  batch): 30× rotation energy, no loss gain, marginally higher erank/alignment.

**Q12 — Rotations vs signal: is averaging rotations (closed-form chordal mean,
C6/R3) better than averaging the signal before one rotation?** 🔴
- *Why:* the only form that never averages across frames *and* never averages
  the signal; the Markley/Moakher closed form removes C6's cost excuse (no
  Karcher iteration — mean of skew logs in one frozen frame, one small SVD).
- *Evaluate:* A′3 vs A′2 at matched window; per-step eigh cost makes this a
  diagnostic-tier run, not a default candidate.
- *Evidence:* none. Subsumes Q4's practical half; Q4 stays open only for the
  cross-frame (transport-paying) variant.

### Drift is real and tracking pays (frozen-basis control, 2026-07-10)
Eigh-init rank-32, 200 steps, step 1, no clip: tracked capture fell 0.60→0.43 —
but the frozen-basis control (interval 100000, no refresh ever) is **worse on
capture and worse on loss** (live-curve read). So "the update fights the eigh
basis" is falsified: capture falls because the *target drifts* (~0.33 rad of
total rank-1 correction cannot explain a 0.17 capture drop; drift does), and
tracking claws back a measurable part of it — **the first positive tracking-
binds-on-loss evidence**, at 200 steps. The frozen arm is *better* on erank and
aurora alignment (every rotation perturbs the moment spectrum; a frozen basis
never does), so those two metrics must not be read as tracking quality alone.

Drift source instinct (user, adopted as hypothesis): after ~100 steps adafactor's
second-moment state settles and the *dampened*-gradient stream it emits differs
substantially from step-0's — the best basis at step 100+ is very different from
the init basis. Part of the drift is optimizer-internal, not just data/model.

### Immediate next step (next session): arm A′ — decomposition-aimed rank-1
C1 accumulation, honest σ dynamics (no clip), step 1, and capture logging are in
place and committed. The tangent path is maintenance-only (needs persistent
residual); drift outruns it. The lead to build next: **eigh-target rank-1
tracking** — at each interval compute an eigh target frame from current gradient
info, take the principal-angle decomposition `SVD(Qᵀ Q_target)` (cheap, r×r),
and rotate along only the largest principal angle, rank-1 geodesic. Decomposition
supplies the aim (no persistence needed); rank-1 supplies bounded, moment-
friendly motion (user's EMA-protection instinct). Open design constraint: what
the eigh sees — boundary grad is free but single-batch noisy; a window Gram is
denoised but side-Gram is `[n,n]` (full-size state, forbidden); sketch/low-rank
alternatives to be designed. Evaluate against: C1 tangent tracking (this arc's
default) and frozen basis, on capture + eval loss.

### Arm A′ design space (mapped 2026-07-10 — evaluate, do not collapse)

**Shared mechanics (all arms).** Given current canon basis `Q:[d,r]` and a
target frame `Q_t:[d,r]` (both in the canon column-orthonormal frame; canon
ambient `d` = the smaller matrix side), the principal-angle decomposition is
`SVD(Qᵀ Q_t) = U cosΘ Vᵀ` — an `[r,r]` SVD, genuinely cheap. The largest
principal angle `θ` pairs `u = Q U[:,i]` (our most-disagreeing direction) with
`w = normalize((I − QQᵀ) Q_t V[:,i])` (where the target says it should be).
Synthesizing the rank-1 canon tangent **`T = θ · w uᵀ`** and feeding it to the
existing `update_grassmann_from_tangent` reproduces exactly the rank-1 geodesic
toward the target: its top singular triple is `(w, θ, u)`, rotation
`step_size·θ`. **No new retraction code; A′ is a new tangent *source*, and the
whole C1 boundary plumbing is reusable.** Two semantic shifts vs the tangent
path, both structural wins: (1) the "σ" is now a true angle in radians —
scale-free by construction, the entire σ-units saga does not apply; (2)
self-annealing is geometric (`Q → Q_t ⟹ θ → 0`) rather than
residual-magnitude-driven. The noise floor moves from σ into the *target*: if
`Q_t` jitters batch-to-batch, θ floors at target-noise level. Same disease, new
thermometer — which is why the target-source fork is load-bearing.

**Transient vs state, clarified.** `fit_eigh` already builds an `[n,n]` Gram
*transiently* at every fit; the invariant forbids full-size **state**, not
full-size transient compute. So a boundary eigh target costs one extra
fit-sized computation (`m·n²` Gram + `n³` eigh) per matrix per interval —
amortized over interval 10 that is real but acceptable. What stays forbidden is
a *persistent* window Gram (`Σ gᵢᵀgᵢ` as state).

**`step_size` semantics change.** `step_size` becomes a fraction of the angle
to the target: 1.0 = snap the top disagreement direction fully onto the target.
Under a noisy target (T1 below), fractional step is the *only* noise filter the
arm has — step_size is load-bearing there, an implicit EMA on aim. Under a
denoised target it is nearly free. Calibrate per target source; do not carry 1.0
across sources on faith.

**Axis 1 — what the eigh sees (the target source):**

| src | signal | state | per-step cost | boundary cost | persistence needed |
|---|---|---|---|---|---|
| T1 | boundary single grad | none | none | Gram `m·n²` + eigh `n³` | none |
| T2 | **sketched window-mean grad**: fixed `Ω:[k,m]`, accumulate `S += Ω@gᵢ`, state `[k,n]`, `k ≈ r+8` | `[k,n]` ≈ 1.25× basis | `k·m·n` ≈ 0.4× tangent path | SVD of `[k,n]` (`k²n`, cheaper than eigh) → top-r right-singular vectors | within-window (same as C1) |
| T3 | per-step eigh target (each batch grad → its own `Q_t`) | none (rotations buffered, see R3) | Gram+eigh **every step** ≈ `n/(3r)`× tangent (~10× at n=1024, r=32) | tiny | none — each target is instantaneous |
| T4 | persistent Gram / second-moment EMA `[n,n]` | **forbidden state** | — | — | ⚫ dead on arrival |

T2 notes: sketching is *exact* for the averaging (`Σ Ωgᵢ = Ω Σgᵢ` — sketch of
sum is sum of sketches); only the range-finding is approximate
(Halko–Martinsson–Tropp randomized range finder, good for decaying spectra).
Design details to decide at build time: redraw `Ω` per window vs fixed (fixed
has a permanent blind spot), and optionally Q-augmented sketch `[Q; Ω]` to
guarantee the current subspace stays visible to the target.

T1↔T2 is also a *conceptual* trade, not just noise: window-averaging
reintroduces a within-window persistence requirement — the very thing
decomposition-aim was meant to escape. But the requirement is much weaker than
the tangent path's: eigh ranks by **magnitude over the whole grad** (dominated
by captured signal, which is batch-stable), while the tangent sees only the
residual (churn-dominated at convergence). Expectation, to be measured, is that
even T1's target is far more stable than a single-batch tangent.

**Axis 2 — how the rotation is smoothed:**

| rot | form | frame legality | gates |
|---|---|---|---|
| R1 | one rank-1 rotation per boundary, `step_size·θ` | single frame, clean | none |
| R2 | cross-boundary rotation EMA: keep a running mean of plane-rotation logs `θ(wuᵀ − uwᵀ)` across boundaries, apply its top plane | **spans frames — transport sin** | Q1 |
| R3 | within-window rotation mean at frozen Q (C6 instantiated): per-step target (T3) → per-step plane triple `(uᵢ, wᵢ, θᵢ)` all computed at the SAME Q → boundary **closed-form chordal mean** (mean of skew logs = mean skew, top singular plane of a `[d,2N]` structured stack; Markley/Moakher lineage — no Karcher iteration needed at small angles) → one rank-1 retraction | one frame, clean — the frozen-frame trick that made C1 legal makes this legal too | needs T3's per-step targets (hot) |

R3 *is* lattice row C6 made concrete, with the closed form replacing Riemannian
iteration: within one tangent frame the chordal mean of small rotations is just
the arithmetic mean of their skew logs projected back to a single plane —
"accumulate a linear object, solve back to the manifold once with a small
eigh/SVD" (Markley's quaternion averaging is the SO(3) special case; Moakher's
projected-arithmetic-mean is the general statement). Note the deep symmetry: R3
averages *rotations-to-target*, C1 averages *tangents* — same reset-buffer
geometry, different signal. R2 is the rotation-space mirror of C2 (tangent EMA)
and inherits exactly Q1's transport question.

**Predicted hazard — rank-cutoff churn (measure before trusting T1).**
*[MEASURED 2026-07-11: fired. Self-angle ≈ 89.5°, cutoff ratio 0.976 — see Q10.]* The
largest principal angle between Q and a fresh eigh target likely lives at the
rank cutoff: when the grad spectrum is near-degenerate around eigenvalue r, the
target's membership at the boundary swaps batch-to-batch, producing spurious
~90° principal angles of pure churn — and rank-1-largest-angle selection aims
at *precisely* that direction. Mitigations live in the space (windowed target
T2 stabilizes the cutoff; selection weighted by target eigenvalue instead of
raw angle), but measure first. Gating probe (diagnostic-only, one extra
`[r,n]` buffer per matrix to hold the previous target): at each boundary log
(a) top principal angle Q vs target, (b) top principal angle between
*consecutive* targets (target self-consistency — the direct noise read),
(c) the eigengap at the rank cutoff. If (b) sits near 90°, T1's aim is
cutoff churn and the fork resolves toward T2/T3 selection-weighting.

**Arms and what each answers** (controls: C1 tangent tracking = this arc's
default, and frozen basis):
- **A′1 = T1+R1** (zero state, boundary-only cost): does decomposition aim beat
  residual-persistence aim at all? (vs C1 on capture + eval loss)
  *[RUN 2026-07-11: tied C1 within noise while snap-rotating pure cutoff churn
  — see Q10 (probe fired) and Q11 (evidence). Built as `--grassmann-aim eigh`.]*
- **A′2 = T2+R1** (~C1's cost class): does denoising the target pay? (vs A′1)
- **A′3 = T3+R3** (hot, diagnostic-tier): is averaging *rotations* better than
  averaging the *signal*? (C6/Q4's question, now concretely answerable)
- **A′4 = R2 on any source**: per-step-drift mirror; gated on Q1, do not build
  before Q1's transport probe runs.

**Evaluation contract** (per the standing next-step): eigh init, rank 32, bs16,
interval 10, 200 steps, eval on; primary reads **basis_capture** (does the arm
hold capture against drift better than frozen, and than C1?) and **eval loss**;
erank/aurora-alignment secondary only (both known biased toward frozen — every
rotation perturbs the moment spectrum). A′ needs its own small step_size
calibration ({0.25, 0.5, 1.0} on T1) since the knob's meaning changed.

---

## Reference
- SubTrack source: `../SubTrack/low_rank_torch/low_rank_projector.py`
  (`track_the_subspace` at ~line 140, `rank_k_matrix_estimation` at ~line 261,
  the accumulation dispatch at ~line 106–131).
- Our implementation: `sumotrack/projector.py::update_grassmann`.
- Diagnostics plumbing: `sumotrack/optimizer.py` (`_refresh_projector`,
  `_accumulate_basis_diagnostics`, `_new_diagnostics`, `_finalize_diagnostics`)
  and `experiments/llm_synth_smoke.py` (`optimizer_rotation_angle`, per-step
  logging, summary fields). Note: the basis-motion diagnostic is now the raw
  geodesic **angle in radians** (`mean_rotation_angle`), not `|sin(angle)|` —
  sin aliases a wrapping rotation as a small one; the radian never lies.
- Tests: `tests/test_projector.py` (geodesic formula rank-1, orthonormality under
  real rotation, gradient-scale invariance, step-size load-bearing).
