# UsuiTrack specification

Current matrix-update design. Direction and unresolved questions live in
`PLAN.md` and `SUBSPACE_TRACKING.md`; superseded evidence lives under `archive/`.
This document specifies only the selected basis-tracking path.

## Read these traps first

- **Basis != subspace.** `Q` and `QH` span the same subspace, but coordinates in
  those frames differ.
- **Left != right with renamed shapes.** Derive each projection and lift.
- **Overlap reprojection != parallel transport.** Reprojection preserves the
  least-squares part of a fixed ambient vector. UsuiTrack carries momentum with
  its moving frame, so its stored coordinates do not change at refresh.
- **A healthy polar output can hide a sick input.** Newton--Schulz restores
  semi-orthogonal scale after weak directions have already become noise-dominated.
- **A downstream operation cannot protect upstream state.** Raw clipping must
  precede Adafactor, held-frame projection, and Oja tangent construction.
- **`side="auto"` knows shape, not architecture.** It cannot infer a transformer's
  residual-facing axis.
- **Exact implementation of a formula does not validate the formula's premise.**

## Coordinates

Let a matrix parameter and gradient be

$$W,G\in\mathbb R^{m\times n},\qquad 1\le r\le\min(m,n).$$

The configured rank must satisfy this bound for every optimized matrix; the
specified path does not define per-tensor rank truncation.

`Q` is always the canonical column frame, `Q^T Q = I`:

| side | canonical frame | stored basis | project $\Pi_Q(G)$ | lift $\Lambda_Q(Z)$ |
|---|---|---|---|---|
| right | $Q:[n,r]$ | $Q^\top:[r,n]$ | $GQ:[m,r]$ | $ZQ^\top:[m,n]$ |
| left | $Q:[m,r]$ | $Q:[m,r]$ | $Q^\top G:[r,n]$ | $QZ:[m,n]$ |

`auto` chooses right for `m >= n`, otherwise left. The harness uses an explicit
residual-facing policy instead. Spectral work promotes fp16/bf16 inputs to fp32.

## One matrix step

```text
raw gradient G
  -> sanitize and raw clip
  -> Adafactor SNR conditioning
  -> stable EIGH initialization on the first step
  -> held-frame projected gradient Z
  -> Rayleigh-normalized one-state Oja tangent
  -> projected EMA M
  -> release full gradient; retain rank-sized pending work
  -> exact Oja frame move with identity moment coordinates
  -> Aurora leverage balance + HeavyBall polar map
  -> full-parameter Muon scale
  -> lift through the moved frame Q+
  -> parameter update
```

### 1. Sanitize and raw clip

Let `S(G)` replace each non-finite entry with zero. With threshold `c=1` by
default,

$$G_c=S(G)\min\left(1,\frac{c}{\max(\|S(G)\|_F,10^{-12})}\right).$$

This occurs before all matrix consumers. Disabling the threshold leaves only
sanitization.

### 2. Adafactor SNR conditioning

The selected path maintains row and column means of squared
full-gradient entries (`beta2=0.99`, `epsilon_a=1e-30`):

$$R_t=\beta_2R_{t-1}+(1-\beta_2)\operatorname{mean}_j(G_{c,ij}^2+\epsilon_a),$$
$$C_t=\beta_2C_{t-1}+(1-\beta_2)\operatorname{mean}_i(G_{c,ij}^2+\epsilon_a).$$

Bias-correct both, then reconstruct

$$\widehat V_{ij}=\frac{\widehat R_i\widehat C_j}
{\operatorname{mean}(\widehat R)}.$$

The gradient consumed by tracking and projection is

$$\widetilde G=\frac{G_c}{\sqrt{\widehat V}}\operatorname{RMS}(G_c).$$

RMS restoration retains relative SNR weighting without feeding an RMS-one matrix
into the tracker. Reconstruction denominators and square roots are floored by
`epsilon_a`.

### 3. Initialize the frame

Normalize `A = G_tilde / ||G_tilde||_F` and form the symmetrized side Gram:

$$K=A^\top A\quad\text{(right)},\qquad K=AA^\top\quad\text{(left)}.$$

`Q` is the top-`r` eigenvector frame. On backend failure, retry `eigh` with

$$K\leftarrow K+10^{-6}\max(\operatorname{tr}(K)/d,10^{-12})I.$$

An exactly zero gradient produces the deterministic EIGH frame of the zero side
Gram.

### 4. Project in the held frame

Using the held frame before the current Oja move,

$$Z_t=\Pi_{Q_t}(\widetilde G_t).$$

### 5. Track with one-state Oja

After EIGH initialization, the live frame updates on the configured basis-update
cadence (default every conditioned full gradient). Reuse the held-frame
projection to form the covariance action:

$$A=\widetilde G^\top Z\quad\text{(right)},\qquad
A=\widetilde GZ^\top\quad\text{(left)},$$

and form the symmetrized Rayleigh matrix and horizontal tangent

$$R=\operatorname{sym}(Q^\top A),\qquad
\Delta=\frac{A-QR}{\operatorname{mean}(\operatorname{diag}R)}.$$

The denominator is floored at `1e-12`. With

$$\Delta^\top\Delta=V\operatorname{diag}(\sigma_i^2)V^\top,$$

the exact full-rank Grassmann step uses

$$\eta_t=\max(0.01, 1/t),$$

where `t` counts basis updates, including EIGH initialization. Thus the first
geodesic uses `1/2`. With `basis_update_interval=k`, phase one still runs for
every matrix gradient while geodesics occur only on matrix steps divisible by
`k`; `t` still counts only basis updates. The released harmonic schedule reaches
its steady `.01` floor at basis update 100. The frame update is

$$Q_{raw}=\left[(QV)\operatorname{diag}(\cos(\eta_t\sigma_i))
+(\Delta V)\operatorname{diag}
\left(\frac{\sin(\eta_t\sigma_i)}{\sigma_i}\right)\right]V^\top.$$

The zero-singular-value limit is `sin(eta_t sigma) / sigma -> eta_t`.
Equal-rank tangent-Gram eigendecompositions are batched. With
`S=Q_raw^T Q_raw`, one near-identity Polar Express step using the final stable
HeavyBall coefficient triple retracts before storage:

$$Q_+=Q_{raw}(aI+bS+cS^2).$$

The Oja tangent rotates every tracked plane. UsuiTrack stores no target frame or
second tracker state and requires a full matrix gradient on every step.

### 6. Accumulate momentum

$$M_t=\beta M_{t-1}+(1-\beta)Z_t,\qquad \beta=0.95.$$

There is no EMA bias correction.

### 7. Transport momentum through frame motion

The geodesic chooses an ambient rotation `R` with `Q_+ = RQ`. UsuiTrack defines
momentum as moving with that frame:

$$MQ^\top\mapsto MQ_+^\top\quad\text{(right)},$$
$$QM\mapsto Q_+M\quad\text{(left)}.$$

Thus its stored coordinates and singular spectrum are unchanged:

$$M_+=M.$$

This is parallel transport along the selected lifted Oja path. Multiplication by the
old/new frame overlap would answer a different question: represent the surviving
projection of a fixed old ambient vector. It contracts each rotated plane by a
principal-angle cosine before Aurora.

### 8. Aurora direction

Aurora acts only on `M`. For a rectangular tensor, orient it as
`A:[p,q]`, `p >= q`, transposing if needed. Initialize row scaling

$$D_{0,ii}=1/\|A_{i,:}\|_2.$$

Run one leverage-balancing/polar iteration:

$$P_k=\operatorname{NS}(D_kA),$$
$$D_{k+1,ii}=D_{k,ii}
\left(\frac{q/p}{\|P_{k,i,:}\|_2^2}\right)^{1/2}$$

with the diagonal update omitted after the iteration. Transpose back. Square
tensors skip leverage balancing and use `NS(M)` directly.

`NS` first divides by Frobenius norm and orients its input with rows no greater
than columns. Five default HeavyBall polynomial steps apply

$$H_k=X_kX_k^\top,\qquad
X_{k+1}=(a_kI+b_kH_k+c_kH_k^2)X_k.$$

with

```text
k    a         b         c
0    4.0848   -6.8946    2.9270
1    3.9505   -6.3029    2.6377
2    3.7418   -5.5913    2.3037
3    2.8769   -3.1427    1.2046
4    2.8366   -3.0525    1.2012
```

The result `O_t` is an approximate leverage-balanced polar direction, not an
exact SVD polar factor.

**Decision — projected Aurora:** Aurora chooses direction inside the retained
update space. UsuiTrack, not Aurora, owns momentum, basis motion, scale, and LR.

### 9. Scale, lift, and update

Muon scale uses the original parameter shape, not the projected shape:

$$\widehat U_t=O_t\sqrt{\max(1,m/n)},
U_t=\Lambda_{Q_{t+}}(\widehat U_t).$$

The selected contract has zero weight decay. Apply the learning rate:

$$W_t=W_{t-1}-\alpha U_t.$$

Matrix parameters retain no full-size first or second moment.

## Two-phase execution

For each matrix, phase one runs exactly once after its full gradient is complete:

1. sanitize and clip `G`;
2. update Adafactor state and form `G_tilde`;
3. read `Z_t` and, after initialization, form `Delta_t` in the held frame `Q_t`;
4. update `M_t`;
5. retain `M_t`, the optional `Delta_t`, and the frame reference, then release `G`.

Phase one does not move the basis or update the parameter. Phase two runs in
`step()`: it batches tangent-Gram eigendecompositions, moves each frame to
`Q_{t+}`, applies Aurora to `M_t`, lifts through `Q_{t+}`, and updates `W`.

`prepare(param)` exposes phase one explicitly. The selected no-accumulation
harness invokes it from a post-accumulate gradient hook. Preparation mutates
Adafactor, projected-moment, initialization, and step state, so pending work is
not transactional: it must be consumed by `step()`. Calling `zero_grad()` while
work is pending would only discard the retained tangent after persistent state
has advanced and is therefore rejected. Repeated preparation, gradient
accumulation before `step()`, and optimizer closures with pending work are
unsupported.

## Fallback path

UsuiTrack accepts only 2D matrix parameters. A separate AdamW owns selected
non-2D parameters, matching the standard Muon split. Its first and second
moments are fp32, with LR `1e-4`, `betas=(0.9,0.99)`, `eps=1e-8`, and zero
weight decay in the selected harness contract. Matrix and fallback state are
accounted separately. Sparse gradients are unsupported; ECC and parameter ECC
fail explicitly.

## Persistent state

| matrix state | right shape | left shape | dtype |
|---|---|---|---|
| basis | `[r,n]` | `[m,r]` | parameter dtype |
| projected EMA | `[m,r]` | `[r,n]` | gradient dtype |
| Adafactor row variance | `[m]` | `[m]` | fp32 |
| Adafactor column variance | `[n]` | `[n]` | fp32 |
| update counts and resolved side | scalar | scalar | Python |

Oja tangents are transient pending work. No target frame, second basis, or lag
snapshot is stored. Fallback parameters retain AdamW's fp32 first moment, second
moment, and step tensor in their separate optimizer.

## Decisions and reasons

These choices define the current design; they are redesignable.

1. **One-sided projection:** retains a rank-limited update without a square core
   or second basis.
2. **Residual-facing harness policy:** up/gate and q/k/v use storage-right; down
   and attention-output use storage-left. Generic `auto` remains shape-only.
3. **Side-Gram `eigh` initialization:** directly solves the one-sided target and
   has explicit fp32, finite-input, symmetrization, and jitter behavior.
4. **One-state full-gradient basis tracking:** the live frame follows the Oja
   covariance action on its configured cadence, with harmonic geodesic motion
   `1/2, 1/3, ...` down to `0.01` and no second basis.
5. **Moving-frame momentum:** identity coordinates preserve the projected
   moment's spectrum through the chosen frame rotation.
6. **Adafactor before tracking and projection:** both consumers see the same
   SNR-conditioned signal; RMS restoration preserves tracker scale units.
7. **Raw clipping before all consumers:** protects persistent Adafactor state and
   Oja tangent construction, not just momentum.
8. **Aurora plus full-shape Muon scale:** direction belongs to projected geometry;
   scale remains tied to parameter geometry.
9. **Separate AdamW fallback:** non-matrix tensors remain trainable under a
   separate optimizer, with their full state exposed rather than hidden in the
   matrix claim.

## Identities and edge behavior

| condition | consequence required by this design |
|---|---|
| gauge change `Q -> QH` with matching coordinate change | same ambient update |
| transpose problem and swap left/right | transposed projected and ambient update |
| Oja covariance action lies inside the current subspace | zero frame motion |
| Oja tangent is rank-deficient | zero singular planes remain fixed |
| full rank | projection/lift loses no component; Aurora can still alter direction |
| gradient rescaling below raw clipping and away from epsilon floors | Rayleigh-normalized Oja tangent unchanged; pre-Aurora magnitude follows the stated conditioning |
| geodesic frame rotation | stored moment coordinates and singular values unchanged |
