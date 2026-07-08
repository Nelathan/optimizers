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
- **k=rank, not k=1.** SubTrack always calls it with `k=1` (rotate the single
  dominant direction). We use `k=eff_rank` (rotate the whole tracked spectrum at
  once). This is the faithful spectral generalization, not a shortcut — see the
  Masterplan for why k=1-vs-k=rank is actually a live research axis given our
  flat tangent spectrum.

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

### Diagnostics
- **`rotation_energy`** = `Σ_i |sin(step_size·σ_i)|` over all tracked directions,
  computed inside `update_grassmann`. 0 = basis static (no-op step), larger =
  more total subspace rotation this refresh. Read from the quantity that actually
  drives the geodesic, so unlike a chordal distance it has **no rank-dependent
  float-noise floor**. Threaded projector → optimizer (`mean_rotation_energy`) →
  harness/wandb.
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

### Step 0 calibration result (rank 64, bs16, 200 steps)
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

Framing insight that drives everything below: our per-refresh tangent spectrum is
**nearly flat** (measured: σ[1]/σ[0]=0.994, top-1 holds 0.8% of energy). That is
*not* a failure — it is the success signature of a well-fit, deliberately
over-provisioned rank (we inflated rank until the bottleneck vanished, so the
residual left over is near-isotropic noise). The flatness is *why* SubTrack's k=1
doesn't transfer directly (their averaged residual is low-rank; our single-batch
residual is flat) and *why* the payoff of tracking is invisible at loose rank.
The masterplan is about making tracking matter and doing it without full-rank
state.

### Contact vs storage (the governing principle)
We always need **full-rank contact** with the gradient — the out-of-subspace
residual only exists there, and the EMA (which lives inside the current subspace)
is blind to it. But contact ≠ storage. The raw grad exists transiently in the
backward pass; we can extract a low-rank tracking signal from it without ever
*storing* `[m,n]`. Every approach below touches full-rank grad, none stores it.

### Approach A — per-step top-1 geodesic (basis as accumulator)
Fire `update_grassmann` *every step* on the single-batch tangent, rank-1 rotation,
no buffer. Noise averages out **across steps** because the basis itself becomes
the accumulator — each step rotates a little toward that step's top-1 direction.
- **Cost-cleared:** step time at rank 64 with per-refresh SVD is identical to
  baseline (1.09s); per-step tangent SVD is affordable.
- Sub-variants to test: fire *every step* vs *interval-only on the latest batch*.
- Risk: single-batch top-1 may be too noisy at tight rank (step-5/step-50
  fragility is evidence for this).

### Approach B — stack rank-1 tangents, then SVD (the real improvement)
Instead of buffering the full-rank `[m,n]` grad (SubTrack's dirty move), buffer
the per-step **rank-1 tangent vectors**: each step contributes one `[m]`-vector
(top-1 tangent, scaled by σ). Stack `interval` of them → `[m, interval]` buffer
(`interval/n` the memory of the full grad buffer). SVD/eigh the stack at refresh →
dominant directions of the *accumulated* tangent → rotate toward them.
- **Strictly better than SubTrack:** same noise-averaging benefit (top directions
  of stacked tangents ≈ top of averaged residual), full-rank contact each step,
  low-rank storage only.
- A is B with interval=1 and immediate rotate. **Build A's per-step-tangent
  machinery first — it's the substrate for both** — get it trusted, then B is a
  small extension (stack + SVD instead of immediate rotate).
- Open sub-question (throughput, measure don't guess): is per-step
  residual+top-1-SVD cheap enough to run every step (A-immediate), or should we
  stack and pay one SVD at refresh (B)? The step-0 timing (SVD is free at rank 64)
  is the first data point.

### Approach 4b (deferred) — EMA-weighted rank replacement
Project the EMA back to see which basis dimensions carry little moment energy →
those are the cheap-to-replace ones → weight the rotation to preferentially move
low-moment-energy directions and preserve high-moment ones. This is a
*weighting on the rank update*, distinct from the tangent-source question above.
Evaluate after A/B give a trusted tracker.

### Open design questions (not yet experiments)
- **eigh-init vs adafactor-update basis mismatch.** `fit_eigh` init selects
  dimensions by *magnitude* (side-Gram eigenvectors, ignores variance/SNR). The
  adafactor-fed *update* selects dimensions by *signal-to-noise* (adafactor
  dampening favors good-SNR directions, not massive singular values). But *loss*
  cares about the massive singular values. So init, update, and loss optimize
  three different notions of "important direction." Whether this mismatch matters
  — and whether tracking should be biased back toward high-singular-value
  directions — is an open thread worth its own investigation.
- **Staleness horizon.** The basis should go stale over training the way a LoRA
  adapter saturates, which is *why* we track at all. Not clearly measurable in
  1k-step runs — the payoff of tracking may only appear on longer horizons. Test
  either by tightening rank (make subspace quality bind on loss sooner) or
  lengthening the run.
- **k=1 vs k=rank under a flat spectrum.** Neither is obviously right when the
  tangent is flat: faithful k=1 rotates one arbitrary direction out of 256
  near-equal ones (near-useless); k=rank rotates the whole block roughly rigidly
  (drifts). Averaging (A/B) sharpens the spectrum and may make k=1-ish behavior
  meaningful again. Resolve alongside A/B, not before.

### Immediate crossroads (where we stopped)
Two ways to make subspace quality bind on loss, to be picked *with the captain*:
1. **Tighten to rank 32 at step 50** — force the bottleneck to pinch (erank can't
   hide near the ceiling), the direct test of whether tracking *matters*.
2. **Extend to 500+ steps at rank 64 / step 50** — give the better subspace time
   to pay off.

Lean: (1) — cheaper per run, and 200 steps already showed the subspace-quality
signal cleanly. But this is a walk-together decision, not a solo call.

---

## Reference
- SubTrack source: `../SubTrack/low_rank_torch/low_rank_projector.py`
  (`track_the_subspace` at ~line 140, `rank_k_matrix_estimation` at ~line 261,
  the accumulation dispatch at ~line 106–131).
- Our implementation: `sumotrack/projector.py::update_grassmann`.
- Diagnostics plumbing: `sumotrack/optimizer.py` (`_refresh_projector`,
  `_accumulate_basis_diagnostics`, `_new_diagnostics`, `_finalize_diagnostics`)
  and `experiments/llm_synth_smoke.py` (`optimizer_rotation_energy`, per-step
  logging, summary fields).
- Tests: `tests/test_projector.py` (geodesic formula rank-1, orthonormality under
  real rotation, gradient-scale invariance, step-size load-bearing).
