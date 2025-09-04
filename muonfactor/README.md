# StellaStiefel: Muon + AdaFactor Hybrid for bs=1 Full Finetuning

Status: v1.1 (2025-09-04)
Owner: Daniel & Lena
Scope: TRL/Transformers-compatible optimizer for memory-efficient, stable, and token-efficient full finetuning at batch size 1.

---

## 1) Intent

- Combine Muon’s orthogonalized update geometry (batched Newton–Schulz) with AdaFactor-style factored second-moment on hidden matrices to:
  - Keep optimizer VRAM close to AdaFactor for 2D weights.
  - Stabilize step directions and scale with layer shape at bs=1.
  - Avoid heavy preconditioners (Shampoo/Apollo) and keep loops compile-friendly and fast

Small, well-shaped steps over aggressive ones.

---

## 2) Preconditions and Assumptions

- Hardware: single RTX 4090 (24 GB) baseline.
- Regime:
  - bs=1, no grad accumulation
  - context length 2k–8k tokens (packed)
  - full finetuning on 0.6B–8B params
  - model params bf16; optimizer state fp32
  - gradient checkpointing; FlashAttention v2/3 available
- Stability goals:
  - Converge without LoRA/GA
  - Memory efficiency near AdaFactor on 2D weights
  - Token efficiency via Muon geometry on 2D

Constraints:

- Muon applies to 2D trainable matrices (linears, attn q/k/v/o) — not embeddings or 1D.
- Weight decay is disabled on the Muon path.
- Decoupled WD only on the fallback (embeddings/1D) path.

---

## 3) Why this hybrid

- Factored second-moment (AdaFactor) preconditions rectangular matrices with O(m+n) memory, capturing anisotropy cheaply.
- Muon orthogonalizes the update (Newton–Schulz), stabilizing geometry across rectangular layers with a shape-aware LR adjust.
- At bs=1, gradient noise is high; Muon path is momentum-free (no first moment). For embeddings/1D, AdamW moments give stable progress.

---

## 4) Algorithm (operational spec)

Two param groups:

- muonable
  - 2D matrices suitable for Muon: MLP linears and attention q/k/v/o when per-head metadata is available.
  - Update = AdaFactor factored preconditioning (row/col; head-wise for attention) → Muon (batched Newton–Schulz).
  - No weight decay on this path.
- fallback_adamw
  - Embeddings, lm_head, all 1D tensors (LayerNorm scales, biases), and any attention lacking head meta.
  - Update = AdamW with decoupled weight decay.
  - Per-param LR: embeddings/1D use `lr_embed_1d`; other fallback tensors use `lr_hidden`.

Step flow:

1) Cast grads to fp32.
2) Muonable 2D:
   - Compute AdaFactor denominators (tensor-wise for MLP; head-wise for attention).
   - Compute preconditioned grad g_pre; optionally clamp its RMS.
   - Apply Muon via batched Newton–Schulz, using a shape-aware LR adjust.
   - Write back; bf16 uses compile-friendly stochastic rounding.
3) Fallback (embeddings/1D):
   - AdamW moments (m, v) with token half-life β2 schedule and decoupled WD.
   - Use `lr_embed_1d` for embeddings/1D, `lr_hidden` otherwise.
   - bf16 write-back with stochastic rounding.

Notes:

- No momentum buffer on Muon path. AdamW moments only on fallback.
- Schedulers: operate on `param_groups` externally (Trainer).
- Global grad norm clip recommended at the Trainer level.

---

## 5) Token-based β2 scheduling for bs=1

We specify β2 by a target token half-life. Define:

- tokens_per_step = effective tokens processed per optimizer step (≈ context length at bs=1)
- H_steps = max(1, floor(H_tokens / tokens_per_step))
- β2_target = exp(-ln 2 / H_steps)

Warmup/cap:

- Linear ramp: from β2_target − 0.01 → β2_target over the first 256 steps
- Cap: β2 ≤ 0.9999

Defaults:

- muonable (hidden) half-life H_tokens_hidden ∈ {2M, 4M}; start 2M
- fallback half-life H_tokens_other ∈ {4M, 8M}; start 4M

Interpretation:

- Larger half-life (β2→1) = smoother denominators, smaller/fewer reactive updates; good for noisy bs=1.
- Smaller half-life (β2 lower) = more reactive/adaptive steps.

---

## 6) LR, half-life, and shape adjust: how they interact

You set two base LRs and two token half-lives:

- lr_hidden: base LR for muonable and non-1D fallback tensors
- lr_embed_1d: base LR for embeddings and 1D tensors
- H_tokens_hidden, H_tokens_other map to β2 via tokens_per_step above

Muon path LR is adjusted by tensor shape:

- adjusted_lr = lr_hidden × adjust_lr_fn(shape)
- adjust_lr_fn="match_rms_adamw" = 0.2 × sqrt(max(m, n))
- Denominators from AdaFactor scale the gradient; their responsiveness is controlled by β2.

Practically:

- Too slow? increase lr_hidden first; if still slow, reduce half-life (more reactive β2).
- Too noisy/jerky? increase half-life (β2 closer to 1) or lower lr_hidden.
- Embeddings/1D should be anchored: lr_embed_1d = 0.1–0.25 × lr_hidden.

---

## 7) Data types and stochastic rounding

- Params: bf16. Optimizer state: fp32.
- All updates computed in fp32 and written back with compile-friendly stochastic rounding (randomize 16 LSBs and mask to bf16). No graph breaks.

---

## 8) Weight decay

- Muon path: WD = 0.0 (disabled).
- Fallback AdamW: decoupled WD (default 2e-3).

---

## 9) Newton–Schulz settings

- ns_steps: 3–5. ns=3 is faster but yields higher orth residual than ns=5 at the same LR.
- Coefficients: (3.4445, −4.775, 2.0315)
- Implementation supports batched inputs (per-head), uses transpose(-1,-2) and batched matmul.

Rule of thumb:

- If orth residual > 0.6 consistently, either lower lr_hidden or bump ns_steps to 5.

---

## 10) Lightweight diagnostics (optional, no history)

Disabled by default; enable by setting:

- opt.metrics_enabled = True
- Use compile_loops=False if you want metrics during the same run.

Metrics per step (muonable group):

- rms_pre: RMS of preconditioned grad before clipping (aggregated across muonable params)
- orth_residual: ||NS(g_pre_clipped) − g_pre_clipped|| / ||g_pre_clipped||

FactoredEMA “means”:

- External, occasional read of mean(vr) and mean(vc) across `af_factored.state` gives a proxy for denom scale/responsiveness. Do this in a Trainer callback every K steps.

No gradient history is stored. Rademacher sketches for step-to-step correlation can be added later.

---

## 11) Trainer integration

- Optimizer subclass: torch.optim.Optimizer compatible.
- Resume: let Trainer own global step. On resume, call `opt.set_step(state.global_step)` so β2 schedules pick up correctly. We do not serialize step in the optimizer.
- Warmup: implement LR warmup (e.g., 5% of steps) in the Trainer scheduler.

Example callback:

```python
from transformers import TrainerCallback

class SyncStepWithTrainer(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        opt = kwargs.get("optimizer", None)
        if opt is not None and hasattr(opt, "set_step"):
            opt.set_step(state.global_step)
```

---

## 12) VRAM budget (4090, 24 GB, bf16, bs=1)

Weights (bf16): ~2 bytes/param. 8B → ~16 GB.

Optimizer states:

- Muonable (factored EMA on 2D): row+col vectors per matrix, fp32 → ~2–4% of weight size for those matrices.
- Fallback (AdamW moments):
  - Embeddings dominate. Example 150k × 2k ≈ 300M params → m+v ≈ 2.4 GB.
  - 1D tensors negligible in aggregate.

Activations:

- With FlashAttention + full checkpointing: ~6–8 GB (bs=1, 2k–8k context).
- Without full checkpointing: ~10–14 GB.

Total:

- Typically 22–26 GB peaks at 8k context on 8B; trim context to 4k if fragmentation is high.
- Note: Unsloth can offload embeddings, but not optimizer moments; plan VRAM accordingly.

Guardrails:

- Enable checkpointing and FlashAttention.
- Use torch.compile where stable; enable TF32 on Ampere+.

---

## 13) Hyperparameters (bs=1 starting points)

Conservative code defaults:

- lr_hidden = 1e-5
- lr_embed_1d = 1e-6
- H_tokens_hidden = 2M
- H_tokens_other = 4M
- ns_steps = 5
- clip_update_rms = 1.0
- wd_other = 2e-3

Recommended starting grid for sweeps (2k hidden, vocab 150k, 4–8k tokens/step):

- lr_hidden: {5e-6, 1e-5, 2e-5}
- lr_embed_1d: {0.1, 0.2, 0.25} × lr_hidden
- H_tokens_hidden: {2M, 4M}
- H_tokens_other: {4M, 8M}
- ns_steps: start 3; if orth_residual > 0.6 persistently, try 5
- clip_update_rms: 1.0
- Warmup: 5% of total steps (Trainer)

---

## 14) Baselines

- SGD (lr ~ 1e-2, no momentum)
- AdamW (lr ~ 3e-4–1e-3, betas=(0.9, 0.999), wd=0–2e-3)
- AdaFactor (HF; lr ~ 3e-4–1e-3; scale_parameter=False, relative_step=False)
- Muon (momentum=0, wd=0), standalone
- Apollo/SPlus for reference if available

---

## 15) Compatibility notes

- TRL/Transformers: uses standard Optimizer API; schedulers operate on `param_groups`.
- Distributed: FSDP/ZeRO compatible; factored states are small. Ensure optimizer state_dict broadcast.
- AMP: bf16 preferred.
- TF32: Recommended on Ampere+ for throughput.
  - torch.backends.cuda.matmul.allow_tf32 = True
  - torch.backends.cudnn.allow_tf32 = True
  - torch.set_float32_matmul_precision("high")
- Checkpointing: Save model weights + optimizer state_dict.
  - Our optimizer state stores:
    - `af_factored`: tensor-wise and head-wise states by parameter name
    - `adamw`: m and v by parameter name
  - We do not store training step; Trainer should restore it via `set_step`.

---

## 16) Failure modes and mitigations

- Early spikes:
  - Ensure β2 ramp (first 256 steps) and EMA bootstrap are active.
  - Increase half-life 2× or reduce lr_hidden ×0.5.
  - Keep clip_update_rms=1.0.
- Slow progress but stable:
  - Increase lr_hidden ×1.5.
  - Slightly reduce half-life (more reactive β2).
  - If residual gets too high, bump ns_steps 3→5 (trade speed for geometry).
- 1D drift (norm scales):
  - Lower lr_embed_1d multiplier (0.1–0.2 × lr_hidden).
  - Keep WD_other=2e-3.
- OOM:
  - Checkpointing + FlashAttention.
  - Reduce context; embeddings offload in Unsloth helps weights but not moments.

---

## 17) Implementation summary

- Optimizer name: StellaStiefel
- Two groups: muonable (2D + per-head attention) and fallback_adamw (embeddings/1D and residuals).
- FactoredEMA with token half-life β2; tensor-wise and head-wise states with bootstrap on first use.
- Batched Muon (Newton–Schulz) update for 2D/heads; shape-aware LR adjust.
- AdamW fallback with per-param LR and decoupled WD.
- bf16 stochastic rounding write-back.
- Name-keyed optimizer state save/load.
- Step sync helpers: `set_step(int)`, `get_step()` for Trainer resume.
- Optional, zero-history metrics: pre-clip RMS and orth residual (off by default).

---

## 18) Open questions / future work

- Add a toggle to compute “pure-geometry” orth residual on an unclipped copy (extra NS pass; off by default).
- Per-layer LR nudges (attention vs MLP) if systematic differences appear.
- Optional unit-norm constraint for norm scales instead of WD.
- Rademacher sketches for step-to-step correlation with near-zero memory.
- Mixed-precision NS (bf16 core + fp32 stabilization) to trade accuracy for speed.

---

If you change dtype, scale to multiple GPUs, or use non-autoregressive objectives, revisit Sections 5–9 and 12. Keep AdaFactor stats slow but responsive (token half-life), keep Muon steps orthogonalized and shape-adjusted, and anchor embeddings/1D with AdamW and mild WD.
