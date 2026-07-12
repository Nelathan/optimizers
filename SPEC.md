# SumoTrack specification

Current matrix-update design. Direction and unresolved questions live in
`PLAN.md` and `SUBSPACE_TRACKING.md`; superseded evidence lives under `archive/`.

## Read these traps first

- **Basis != subspace.** `Q` and `QH` span the same subspace, but coordinates in
  those frames differ.
- **Left != right with renamed shapes.** Derive each projection and lift.
- **A tangent != a position target.** Tangent integration is velocity control;
  fractional motion toward an `eigh` frame is position control.
- **Overlap reprojection != parallel transport.** Reprojection preserves the
  least-squares part of a fixed ambient vector. SumoTrack carries momentum with
  its moving frame, so its stored coordinates do not change at refresh.
- **A healthy polar output can hide a sick input.** Newton--Schulz restores
  semi-orthogonal scale after weak directions have already become noise-dominated.
- **A downstream clip cannot protect upstream state.** Projected clipping does
  not protect Adafactor or the basis target.
- **`side="auto"` knows shape, not architecture.** It cannot infer a transformer's
  residual-facing axis.
- **Exact implementation of a formula does not validate the formula's premise.**

## Coordinates

Let a matrix parameter and gradient be

$$W,G\in\mathbb R^{m\times n},\qquad r\le\min(m,n).$$

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
  -> Adafactor SNR conditioning (default)
  -> initialize or refresh frame Q
  -> projected gradient Z
  -> optional projected clip
  -> projected EMA M
  -> Aurora leverage balance + HeavyBall polar map
  -> full-parameter Muon scale
  -> lift through Q
  -> decoupled weight decay and parameter update
```

### 1. Sanitize and raw clip

Let `S(G)` replace each non-finite entry with zero. With threshold `c=2.5` by
default,

$$G_c=S(G)\min\left(1,\frac{c}{\|S(G)\|_F}\right).$$

This occurs before all matrix consumers. Disabling the threshold leaves only
sanitization.

### 2. Adafactor SNR conditioning

Default `moment_mode="adafactor_ema"` maintains row and column means of squared
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
`epsilon_a`. In plain `moment_mode="ema"`, `G_tilde = G_c` and this state is absent.

### 3. Initialize the frame

Normalize `A = G_tilde / ||G_tilde||_F` and form the symmetrized side Gram:

$$K=A^\top A\quad\text{(right)},\qquad K=AA^\top\quad\text{(left)}.$$

`Q` is the top-`r` eigenvector frame. On backend failure, retry `eigh` with

$$K\leftarrow K+10^{-6}\max(\operatorname{tr}(K)/d,10^{-12})I.$$

An exactly zero gradient uses a random QR frame. Random initialization otherwise
remains an ablation.

### 4. Refresh by position control

After initialization, burst-refresh eligible matrix parameters every
`basis_refresh_interval` group steps (`100` API default). Compute a transient
top-`r` `eigh` target frame `T`, then

$$Q^\top T=U\operatorname{diag}(c_i)V^\top,$$
$$h_i=Tv_i-c_iQu_i,\qquad s_i=\|h_i\|_2,\qquad
\theta_i=\operatorname{atan2}(s_i,c_i),\qquad w_i=h_i/s_i.$$

The zero-angle limit uses `theta_i / s_i -> 1`. For fractional step
`eta=0.25` by default,

$$Q_+=\sum_i
\left(Qu_i\cos(\eta\theta_i)+w_i\sin(\eta\theta_i)\right)u_i^\top
+Q(I-UU^\top).$$

The default rotates every principal plane. A configured `rotate_rank` keeps the
largest-angle planes. `eta=1` reaches the target subspace when all planes are used
and the selected shortest path is unique.

**Decision — position control:** the target says where the signal subspace is;
fractional motion geometrically forgets target noise. Integrating a noisy tangent
instead accumulates velocity error without a restoring position.

### 5. Project and accumulate momentum

Using the refreshed frame,

$$Z_t=\Pi_{Q_t}(\widetilde G_t).$$

Optional projected clipping limits `||Z_t||_F` by an absolute threshold, a ratio
times `||M_{t-1}||_F`, or their minimum. Both controls default off. It protects
only the projected EMA:

$$M_t=\beta M_{t-1}+(1-\beta)Z_t,\qquad \beta=0.9.$$

There is no EMA bias correction.

### 6. Transport momentum through refresh

The geodesic chooses an ambient rotation `R` with `Q_+ = RQ`. SumoTrack defines
momentum as moving with that frame:

$$MQ^\top\mapsto MQ_+^\top\quad\text{(right)},$$
$$QM\mapsto Q_+M\quad\text{(left)}.$$

Thus its stored coordinates and singular spectrum are unchanged:

$$M_+=M.$$

This is parallel transport along the selected lifted path. Multiplication by the
old/new frame overlap would answer a different question: represent the surviving
projection of a fixed old ambient vector. It contracts each rotated plane by a
principal-angle cosine before Aurora.

### 7. Aurora direction

Aurora acts only on `M`. For a rectangular tensor, orient it as
`A:[p,q]`, `p >= q`, transposing if needed. Initialize row scaling

$$D_{0,ii}=1/\|A_{i,:}\|_2.$$

Run two leverage-balancing iterations:

$$P_k=\operatorname{NS}(D_kA),$$
$$D_{k+1,ii}=D_{k,ii}
\left(\frac{q/p}{\|P_{k,i,:}\|_2^2}\right)^{1/2}$$

with the diagonal update omitted after the final iteration. Transpose back.
Square tensors skip leverage balancing and use `NS(M)` directly.

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
update space. SumoTrack, not Aurora, owns momentum, basis motion, scale, LR, and
weight decay.

### 8. Scale, lift, and update

Muon scale uses the original parameter shape, not the projected shape:

$$\widehat U_t=O_t\sqrt{\max(1,m/n)},
U_t=\Lambda_{Q_t}(\widehat U_t).$$

Apply decoupled weight decay and learning rate:

$$W_t=(1-\alpha\lambda)W_{t-1}-\alpha U_t.$$

Matrix parameters retain no full-size first or second moment.

## Fallback path

Every non-2D parameter uses ordinary AdamW with fp32 first and second moments,
the group LR and weight decay, `eps=1e-8`, and default betas `(0.9,0.99)`. Matrix
and fallback state are accounted separately. Sparse gradients are unsupported;
ECC and parameter ECC fail explicitly.

## Persistent state

| matrix state | right shape | left shape | dtype |
|---|---|---|---|
| basis | `[r,n]` | `[m,r]` | parameter dtype |
| projected EMA | `[m,r]` | `[r,n]` | gradient dtype |
| Adafactor row variance | `[m]` | `[m]` | fp32 |
| Adafactor column variance | `[n]` | `[n]` | fp32 |
| update counts and resolved side | scalar | scalar | Python |

Adafactor state is absent in plain EMA mode. Target frames and tangents are
transient. Basis-target and lag snapshots exist only when their diagnostics are
enabled. Fallback parameters retain AdamW's fp32 first moment, second moment, and
step tensor.

## Decisions and reasons

These choices define the current design; they are redesignable.

1. **One-sided projection:** retains a rank-limited update without a square core
   or second basis.
2. **Residual-facing harness policy:** up/gate and q/k/v use storage-right; down
   and attention-output use storage-left. Generic `auto` remains shape-only.
3. **Side-Gram `eigh` initialization:** directly solves the one-sided target and
   has explicit fp32, finite-input, symmetrization, and jitter behavior.
4. **Full-spectrum fractional position control:** the instantaneous spectral
   target supplies displacement; fractional motion supplies temporal averaging
   without state.
5. **Moving-frame momentum:** identity coordinates preserve the projected
   moment's spectrum through the chosen frame rotation.
6. **Adafactor before tracking and projection:** both consumers see the same
   SNR-conditioned signal; RMS restoration preserves tracker scale units.
7. **Raw clipping before all consumers:** protects persistent Adafactor state and
   transient basis targets, not just momentum.
8. **Aurora plus full-shape Muon scale:** direction belongs to projected geometry;
   scale remains tied to parameter geometry.
9. **Burst refresh:** one explicit boundary is simpler than round-robin state and
   scheduling.
10. **AdamW fallback:** non-matrix tensors remain trainable, with their full state
    exposed separately rather than hidden in the matrix claim.

## Identities and edge behavior

| condition | consequence required by this design |
|---|---|
| gauge change `Q -> QH` with matching coordinate change | same ambient update |
| transpose problem and swap left/right | transposed projected and ambient update |
| target equals current subspace | zero frame motion |
| conceptual refresh fraction `eta=0` | frozen frame and unchanged moment coordinates |
| `eta=1`, all planes | target subspace along the selected shortest path |
| full rank | projection/lift loses no component; Aurora can still alter direction |
| gradient rescaling below raw clipping and away from epsilon floors | basis target unchanged; pre-Aurora magnitude follows the stated conditioning |
| geodesic frame rotation | stored moment coordinates and singular values unchanged |
| disable Adafactor | row/column state vanishes; clipped raw gradient feeds tracking |

At a principal angle near 90 degrees, a shortest Grassmann path may be non-unique.
Transport is exact along the overlap SVD's selected path; that does not guarantee
the noisy target selects a temporally stable path.
