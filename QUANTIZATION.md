# Rank-State Quantization Sidequest

Status: scoped, not implemented, no outlier captures or quantized runs yet.

This ledger asks whether UsuiTrack's rank-limited matrix state can be stored below
bf16 without damaging the geometry that makes it useful. `SPEC.md` remains
authoritative for the current update.

The first targets are the tracked basis and projected first moment. Quantized
master weights are a different project: they alter model storage and update
application, need a separate numerical and kernel contract, and are out of scope.
HeavyBall ECC and parameter ECC are not prerequisites for this sidequest.

## Heading

Test a fixed group-wise regular Hadamard rotation before symmetric low-bit
quantization, always along the rank axis. The worked codec below was scoped at
rank `64`; the live harness now defaults to rank `128`, so block grouping must be
re-reviewed before this sidequest becomes active rather than silently inherited.

The order of evidence is:

1. establish that rank-coordinate outliers exist and that the rotation flattens
   them;
2. measure offline quantization damage in the state tensor's own geometry;
3. measure propagation through the one-state Oja basis update and
   Aurora/HeavyBall Newton--Schulz;
4. run faithful SYNTH loss and source-retention comparisons only for survivors;
5. count realized state bytes and walltime before calling a quality-neutral arm
   a product win.

Rotation is not the goal. It earns a place only if it improves a low-bit storage
format over the same format without rotation. If the current state has no useful
outlier structure, this branch stops cheaply.

## Current UsuiTrack contract

For a parameter gradient `G:[m,n]`, let `Q:[d,r]` be the canonical
column-orthonormal basis, where `d=n` on the right projection side and `d=m` on
the left. Define a canonical projected-coordinate tensor `C:[s,r]`:

| projection side | stored basis | stored moment | canonical `Q` | canonical `C` | lifted update |
|---|---|---|---|---|---|
| right | `Q^T:[r,n]` | `M:[m,r]` | `Q:[n,r]` | `C=M:[m,r]` | `C Q^T` |
| left | `Q:[m,r]` | `M:[r,n]` | `Q:[m,r]` | `C=M^T:[n,r]` | `Q C^T` |

Stable EIGH initializes the basis. One-state Oja then moves it from every
Adafactor-conditioned full gradient with harmonic steps down to `.01`, exact all-plane
rank-space geometry, and Polar Express correction. There is no second frame.
The projected first moment is an elementwise linear EMA in the moving frame.
Aurora then leverage-balances that moment and applies
five HeavyBall polynomial Newton--Schulz steps before full-parameter Muon scaling
and lift.

Quantization must preserve this meaning. In particular, the basis is not merely
a matrix of values: it represents a subspace and is required to be orthonormal.
The moment is not independent of its basis: it is expressed in that frame.

## Rank-gauge math

Let `H:[r,r]` be an orthogonal block-diagonal regular Hadamard transform, with
blocks confined to the rank axis. At rank `64`, the primary experiment uses one
normalized `H_64` block, matching the reference transform's power-of-four rule.
Do not rotate an ambient/model dimension.

Apply the same canonical rank gauge to basis and coordinates:

```text
Q_H = Q H
C_H = C H
```

Then both projection sides preserve the exact ambient update:

```text
right: C_H Q_H^T = C H H^T Q^T = C Q^T
left:  Q_H C_H^T = Q H H^T C^T = Q C^T
```

Projection is covariant in the same coordinates. On the right,
`G Q_H = (G Q) H`; on the left, canonicalizing the stored projection gives
`(Q_H^T G)^T = G^T Q H`. A first-moment EMA therefore commutes with the fixed
rotation.

The current Aurora map is also rank-gauge equivariant. It orients every projected
moment as the tall canonical `C`, computes its leverage diagonal from row norms,
and applies polynomial Newton--Schulz. Right multiplication by `H` preserves row
norms, so the leverage diagonal is unchanged, and every polynomial NS iterate
satisfies

```text
Aurora(C H) = Aurora(C) H.
```

This is stronger than the imported proposal's bare-polar argument. Aurora's
nonlinearity does not obstruct this particular rank-side gauge.

The Oja update is gauge-invariant as a Grassmann operation and
should be equivariant as the selected horizontal frame lift. That is a
mathematical expectation, not yet evidence about the implementation around SVD
gauge choices, repeated principal angles, and finite precision. It must be pinned
for both projection sides before a persistent rotated-gauge implementation is
considered.

## Two distinct uses of rotation

Do not blur these designs.

### Storage codec — first study

Canonicalize a target tensor to `[ambient, rank]`, rotate, quantize, dequantize,
and rotate back:

```text
X_hat = Qdecode(Qencode(X H)) H^T
```

The live optimizer continues to see its native gauge. This isolates whether
rotation improves representation and allows basis-only, moment-only, and both
targets to be compared honestly. It costs transforms at codec boundaries and is
not yet a performance design.

### Persistent rotated gauge — later implementation question

Keep `Q_H` and `C_H` as the live state and exploit equivariance to avoid rotating
back on every use. This couples basis and moment semantics, reaches into refresh,
projection, Aurora, state dicts, and restart, and cannot be inferred safe from an
offline RMSE win. It is downstream of the storage-codec and geometry evidence.

## What “outlier” means here

ConvRot was designed for model weights and activations. Its premise does not
automatically transfer to orthonormal bases or Adafactor-conditioned projected
moments.

All captures are viewed in canonical `[ambient,64]` layout. For each ambient row
`x`, record at least

```text
rank peak ratio = sqrt(r) * ||x||_inf / ||x||_2
```

which is `1` for perfectly flat rank coordinates and `sqrt(r)` for a one-hot row.
Also record absolute-max/RMS, high magnitude quantiles, scale inflation, and the
fraction of quantization range used, before and after `H_64`. Aggregate by target,
module role, projection side, tensor shape, and training phase; do not let a few
large tensors hide a systematically bad module family.

For the basis, separate two phenomena:

- `||Q[i,:]||_2` is row leverage and is invariant under `Q -> QH`; rank rotation
  cannot remove an ambient high-leverage row;
- concentration within that row can change under `H` and is the only basis
  outlier shape this transform can flatten.

This distinction is a stop condition, not bookkeeping. A global basis absmax may
look alarming while being caused by invariant leverage that rank rotation cannot
fix. Conversely, if the eigenvector frames are already dense and approximately
isotropic, regular Hadamard rotation should have little to offer.

## Initial quantizer contract

The first simulator should be deliberately boring:

- symmetric signed integers;
- bit widths `8` and `4`;
- one absmax scale per canonical ambient row and rank group;
- nearest rounding for the static study;
- regular normalized Hadamard blocks, group size `64` at rank `64`;
- fp32 transform and scale calculation;
- explicit packed-byte accounting rather than dtype-based guesses.

Per-row scaling is the product-shaped default: its metadata is small at rank 64
and it does not force unrelated ambient rows to share one outlier. MSE clipping is
not in the first comparison because it can hide whether rotation fixed the range
problem; it becomes a rival only after the absmax mechanism is understood.
Stochastic rounding and error feedback are separate temporal-accumulation axes,
not prerequisites for testing ConvRot. UsuiTrack currently implements neither
for matrix state.

For `N` canonical rows at rank 64, bf16 payload is `128N` bytes. Ignoring packing
headers, int8 plus one fp16/fp32 scale is `66N`/`68N` bytes; packed int4 plus a
scale is `34N`/`36N` bytes. These are component estimates, not whole-optimizer
savings: Adafactor row/column state and fallback AdamW state remain unchanged.

Packed four-bit state does not imply four-bit optimizer arithmetic. RTX 4070
SUPER (Ada, SM 8.9) already supports signed/unsigned INT4 tensor-core MMA; RTX
5080 (Blackwell, SM 12.0) supports INT4 and additionally native FP4 tensor-core
formats. Neither generation automatically accelerates this codec: packing,
scaling, elementwise EMA, basis refresh, and Aurora are not an eligible INT4 GEMM
merely because state is stored in nibbles. A specialized kernel and compatible
layout would be required. Therefore the product gate is measured codec walltime
and memory traffic on the target GPU, not advertised low-bit tensor-core TOPS.

## Evidence ladder

### A. Exact math and implementation oracles

Before quantization noise, force both left and right projection paths and verify:

- `H_64^T H_64 = I` and codec round-trip identity without quantization;
- projection covariance and lift invariance under `(Q,C) -> (QH,CH)`;
- EMA covariance over multiple updates;
- Oja update equivariance, including zero tangent, ordinary angles, repeated
  tangent singular values, and near-90-degree motion;
- Aurora covariance after each leverage/NS stage, not merely final output;
- transpose/side equivalence and state-dict restart in the chosen gauge.

Failure here is a coordinate bug. Do not continue to loss.

### B. Static outlier anatomy

Capture current bf16 basis and projected moment tensors from the faithful rank-64
LFM-350M lane across initialization and ordinary Oja steps. No
quantized optimizer is needed. Compare identity against fixed regular `H_64` on
the rank-coordinate metrics above.

The premise survives only where rotation consistently reduces range inflation on
a meaningful fraction of that target's bytes. Report distributions and worst
module families, not only a global mean.

### C. Offline quantization fidelity

For each surviving target, compare the same quantizer with rotation off and on.

Moment observables:

- relative Frobenius error and cosine to bf16;
- row-wise error distribution and singular-spectrum/effective-rank drift;
- Aurora output cosine to the bf16 Aurora output;
- lifted ambient-update cosine and relative error for both projection sides;
- intrinsic pre/post-Aurora alignment as a secondary health read, not a
  substitute for agreement with the bf16 update.

Basis observables:

- orthonormality defect of the decoded frame;
- principal angles, chordal distance, and projector error against bf16;
- capture delta on the same conditioned gradient;
- projected-gradient and lifted-update error with a matched bf16 moment.

Elementwise basis RMSE is insufficient. An optional fp32 QR of the decoded basis
may be measured as a diagnostic upper bound: it restores orthonormality but does
not restore the original subspace, and doing it on every decode may erase the
memory win in walltime. It is not the assumed implementation.

The target matrix remains open:

| target stored low-bit | question isolated |
|---|---|
| basis only | can a quantized frame preserve subspace and projection quality? |
| moment only | can Aurora tolerate compressed history without amplifying weak-direction noise? |
| basis and moment | do individually acceptable errors compose in the lifted update? |

Each low-bit row has a no-rotation control. “Both” is not inferred from two solo
wins.

### D. Dynamic geometry before LLM loss

Static decoding misses accumulation and feedback. Exercise repeated moment EMA
updates and repeated quantize/dequantize cycles, plus stationary, smoothly
rotating, abrupt-replacement, cutoff-crossing, and near-orthogonal basis targets.
Read the quantization-induced noise floor in:

- moment cosine, spectrum, and Aurora/lifted-update direction;
- basis path distance from bf16, orthonormality, capture, and settling/orbit size;
- one-step error versus accumulated drift.

This stage decides whether low precision merely adds bounded observation noise or
changes the Oja tracker and moving-frame history semantics.

### E. Faithful training

Only geometry survivors enter the default right-padded no-mask SYNTH lane on
`LiquidAI/LFM2.5-350M-Base`, broad no-embedding scope, rank `128`, residual-facing
projection, per-gradient Oja, CCE, and the current compile policy. Change only the
state codec. The retained interval `10` is an EIGH/tangent ablation control and is
inert under this default lane.

Read target loss and source retention beside pre-Aurora moment health, Aurora
output agreement where affordable, lifted update/parameter ratio, state bytes,
peak allocated VRAM, tokens/sec, and per-gradient tracker walltime. Explicit EIGH
comparators additionally separate boundary from non-boundary cost. Loss parity is
judged against matched bf16 run noise, not an invented decimal threshold.

The first training table contains only offline survivors and their direct
no-rotation low-bit controls. Four-bit is not required to justify an int8 result,
but it does not run merely because int8 ran.

## Stop and promotion rules

- Stop a target if `H_64` does not materially improve its outlier/range
  distribution over identity.
- Stop a format if rotation does not beat the same bit width and scale granularity
  without rotation on the target's native geometry.
- Stop basis quantization if useful subspace or orthonormality cannot be preserved
  without a retraction whose measured walltime repays the state saving.
- Stop moment quantization if Aurora turns bounded storage error into unstable or
  noise-selected lifted directions.
- Do not promote a loss-neutral arm until packed state bytes and walltime show a
  useful product trade.
- Do not generalize rank-64 evidence to other ranks or group sizes without a new
  measurement.

The likely asymmetry is worth stating before measurement: moment quantization is
the cleaner candidate because it has no orthonormality constraint, while basis
quantization risks changing the represented subspace and the Oja
trajectory. On the other hand, a side-Gram eigenbasis may already be too dense for
Hadamard flattening to help. The outlier capture, not taste, decides both.

## Non-goals

- quantized master weights, model parameters, gradients, activations, Adafactor
  row/column variances, or fallback AdamW state;
- ECC, parameter ECC, or bit-corruption protection;
- adding error-feedback machinery before plain low-bit storage is shown useful;
- changing tracking aim, refresh cadence, Adafactor conditioning, Aurora, Muon
  scale, rank allocation, or benchmark formatting;
- kernel work before a quality-preserving storage format exists;
- importing Comfy model classification, inference GEMM, or MSE-clipping policy.

## References and local authority

- Current optimizer mathematics: [`SPEC.md`](SPEC.md)
- Product direction and benchmark contract: [`PLAN.md`](PLAN.md)
- Current Oja/transport ledger:
  [`SUBSPACE_TRACKING.md`](SUBSPACE_TRACKING.md)
- ConvRot paper: [ConvRot: Rotation-Based Plug-and-Play 4-bit Quantization for
  Diffusion Transformers](https://arxiv.org/abs/2512.03673)
- Comfy model-tools int8 adaptation:
  [`quant_int8_convrot.py`](https://github.com/Comfy-Org/comfy-model-tools/blob/main/quant_int8_convrot.py)
- Comfy regular-Hadamard construction and rank-group rotation:
  [`int8_utils.py`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/int8_utils.py)
- Comfy W4A4 layout:
  [`convrot_w4a4.py`](https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/convrot_w4a4.py)
- CUDA PTX integer MMA target requirements:
  [`mma.sync`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)
- CUDA GPU compute capabilities:
  [RTX 40/50-series table](https://developer.nvidia.com/cuda-gpus)
- Blackwell FP4 support:
  [cuBLAS narrow-precision types](https://docs.nvidia.com/cuda/cublas/index.html#narrow-precision-data-types-usage)

The external code is a reference for the normalized regular Hadamard transform,
power-of-four grouping, and concrete quantizer mechanics. It is an inference
weight path, not an optimizer-state design. No external repository is required
for the scoped measurement plan; clone one under `~/code` only if implementation
work needs backend or packing internals.
