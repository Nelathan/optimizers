# Subspace Tracking: Current Ledger

This is the live geometry ledger. Historical mechanisms, measurements, and
superseded defaults live in `archive/SUBSPACE_TRACKING_ARC.md`. `SPEC.md`
defines the selected update.

## Settled mechanism

For a canonical column frame `Q:[d,r]`, UsuiTrack consumes every
Adafactor-conditioned full gradient. It projects in the held frame, uses that
projection to form Oja's horizontal covariance tangent, and stores projected EMA
coordinates in the moving frame.

```text
condition every full gradient
  -> held-frame projection Z
  -> Oja tangent Delta, when the basis update is due
  -> projected EMA M
  -> release the full gradient
  -> batched exact tangent-Gram geodesic and Polar Express correction
  -> Aurora(M), lift through the moved frame, update W
```

`basis_update_interval` controls only basis motion and defaults to `1`.
Conditioning, projection, and the projected EMA happen on every matrix step. The
geodesic step is

```text
eta_t = max(0.01, 1 / t)
```

where `t` counts basis updates, including EIGH initialization. Thus initialization
uses `1`, the first geodesic uses `1/2`, and a sparse cadence retains the same
sequence of actual basis moves. One Polar Express correction restores the Stiefel
invariant. The tracker stores no target frame or second basis.

The selected transport is identity in moving-frame coordinates. If `Q+ = RQ`,
stored coordinates `M` lift as `M Q+^T` on the right or `Q+ M` on the left.
Overlap reprojection is deliberately excluded: it applies a cosine tax to rotated
directions and gives Aurora weakened coordinates that its polar map can amplify.

## Current questions

### R1. Is online basis tracking worth its cost?

Compare the selected tracker against a frozen EIGH-initialized frame under the
same model, data, rank, cadence, and state contract. Read target/source loss,
capture, peak memory, and walltime.

### R2. Is the selected moving-frame path stable?

Stress sign/gauge changes, cutoff crossings, and near-orthogonal motion. Separate
an unstable selected path from an incorrect transport law.

### R3. Does `basis_update_interval` earn a value other than one?

The default is per-gradient motion. A larger interval is a distinct
estimator-variance and systems trade: it leaves phase-one conditioning intact but
uses fewer geodesics. It must be measured directly; a larger harmonic move is not
automatically equivalent.

### R4. Can lower-precision rank state preserve the selected geometry?

See `QUANTIZATION.md`. Basis and moment codecs must be evaluated separately before
their errors are composed.

## Closed design decisions

- Stable side-Gram EIGH is the only initialization path.
- Oja names the covariance tangent and geodesic primitive; UsuiTrack names the
  full two-phase basis-tracking optimizer.
- The basis uses one state; the projected moment follows it as identity
  coordinates.
- Boundary controllers, target frames, overlap/reset transport, random
  initialization, and projected-activation ingress are historical, not active
  design arms.
