# DESIGN: Muon + AdaFactor Hybrid for bs=1 Full Finetuning

Status: v1.0 (2025-09-01)
Owner: Daniel & Lena
Scope: TRL/Transformers-compatible optimizer for memory-efficient, stable, and token-efficient full finetuning at batch size 1.

---

## 1) Intent

- Marry Muon’s mathematically principled update geometry (Newton–Schulz orthogonalization) with AdaFactor’s factored second-moment to:
  - Reduce optimizer VRAM to near-AdaFactor levels for full finetuning.
  - Keep step directions well-conditioned and scale-invariant for better token efficiency at bs=1.
  - Avoid large momentum buffers and heavy preconditioners (Shampoo/Apollo).

This is deliberately “small, well-shaped steps,” not “large, aggressive steps.”

---

## 2) Preconditions and Assumptions

- Hardware: single RTX 4090 (24 GB) as baseline.
- Training regime:
  - batch size = 1, no gradient accumulation.
  - context length up to 8k tokens (packed, best-fit decreasing).
  - full finetuning on models ranging from 0.6B to 8B params.
  - bf16 model dtype; optimizer state in fp32.
  - gradient checkpointing enabled; FlashAttention 2/3 available.
- Objectives:
  - Stable convergence without LoRA/GA.
  - Memory efficiency comparable to AdaFactor.
  - Token/sample efficiency via Muon’s geometry.
- Constraints:
  - Muon is defined for 2D hidden layer parameters. Do not apply Muon to embeddings or 1D params.
  - Weight decay disabled for Muon path (orthogonalized steps make WD less meaningful on hidden layers).
  - Decoupled WD only for 1D/embeddings (gentle anchor).

---

## 3) Why this hybrid (first principles)

- AdaFactor’s factored second-moment preconditioning gives curvature-aware scaling with O(m+n) memory per matrix vs O(mn).
- Muon orthogonalizes updates to stabilize and normalize step geometry across rectangular layers (consistent RMS), and is well-behaved mathematically (Newton–Schulz operator, adjust_lr_fn).
- At bs=1, gradient noise is high; momentum buffers are expensive and redundant alongside high-β2 adaptivity. We set Muon momentum=0 and rely on AdaFactor β2 tuned by token half-life.
- Embeddings are “lists of vectors,” not matrices in the Muon sense. Treat them as embeddings: AdaFactor only, lower LR, small WD.

---

## 4) Algorithm (operational spec)

Partition parameters:
- Hidden 2D (non-embedding linears): AdaFactor preconditioning → Muon (NS) step.
  - Muon: momentum=0, nesterov=False, weight_decay=0.0, adjust_lr_fn="match_rms_adamw", ns_steps∈{3,5}.
- Embeddings and 1D (bias, norm scales, heads):
  - AdaFactor update only (factored on 2D embeddings, unfactored on 1D), lr scaled down, WD=2e-3 decoupled.

Update flow per step:
1) Backward pass → per-param gradients (bf16/fp16 allowed).
2) Precondition grads for 2D hidden params with AdaFactor (factored EMA of g^2), using fp32 accumulators.
3) Optional update RMS clamp (Adafactor-style).
4) Muon applies NS orthogonalization, shape-based LR adjust, and updates 2D hidden params (no WD).
5) For “other” params (embeddings/1D), apply AdaFactor update with decoupled WD and lr scaling.
6) For bf16 params, compute in fp32 and write back with stochastic rounding to bf16.

Notes:
- No first-moment buffers; no Muon momentum.
- LR schedulers can be applied on top (e.g., short warmup).
- Global grad clip (norm 1.0) recommended.

---

## 5) Token-based β2 scheduling (bs=1)

We define β2 by a target half-life in tokens (Goldblum et al., small batch rule), not by steps.

- tokens_per_step ≈ context length (bs=1).
- H_steps = max(1, floor(H_tokens / tokens_per_step))
- β2_target = exp(-ln 2 / H_steps)
- Apply linear ramp from β2_target − 0.01 up to β2_target over the first K steps (K≈256) to avoid early thrash.
- Cap β2 ≤ 0.9999.

Defaults for 8k context:
- Hidden 2D half-life H_tokens_hidden ∈ [1M, 4M] → β2 ≈ 0.9945–0.9986
  - Start at 2M (β2≈0.9972), sweep 1M/2M/4M.
- Embeddings/1D half-life H_tokens_other ∈ [2M, 8M] → β2 ≈ 0.997–0.999
  - Start at 4M (β2≈0.9986).

EMA bootstrap:
- Initialize v_row/v_col (and v for 1D) with first observed g^2 statistics (row/col means) instead of zeros to prevent underestimated denominators in steps 1–1000.

---

## 6) Data types and stochastic rounding

- Model params: bf16.
- Optimizer stats: fp32.
- Compute updates in fp32; write back to bf16 with stochastic rounding:
  - Quantize-to-bf16, compute residual to next-up bf16, flip with probability proportional to residual/ULP to preserve expectation and reduce bias at small step sizes.

---

## 7) Weight decay policy

- Hidden 2D (Muon path): weight_decay=0.0 (decoupled WD disabled).
- Embeddings/1D (AdaFactor path): decoupled WD=2e-3.
- If LayerNorm/RMSNorm scales drift, optionally replace WD with a gentle “pull to unit norm” (don’t use both).

---

## 8) Learning rate policy

- Muon adjust_lr_fn="match_rms_adamw" allows reuse of AdamW-tuned LRs.
- Hidden 2D: start with AdamW-like LR (e.g., 3e-4 to 5e-4 for 7–8B instruction FT). Sweep 0.5–1.0× base.
- Embeddings/1D: lr_other_scale 0.25–0.5 × hidden LR (start 0.5).
- Warmup: 200–400 steps linear/cosine.

---

## 9) Newton–Schulz steps

- ns_steps = 5 by default. If orthogonality residual is consistently high (>0.6), drop to 3 or reduce LR.
- Coefficients: (3.4445, −4.775, 2.0315) default.

---

## 10) Logging and diagnostics

We log to wandb at a fixed cadence (e.g., every 25 steps) the following:

- update_rms_mean (hidden 2D): RMS of preconditioned update; ideal 3e-4 to 3e-3. If >1e-2 persistently, increase β2 or reduce LR.
- grad_norm_mean and var_ratio = Var(||g_pre||)/E(||g_pre||)^2 over a sliding window (Kobs).
  - Keep var_ratio ≤ 0.5; if >1, increase half-life 2× or halve LR.
- cos_sim_mean of consecutive preconditioned grads: 0.1–0.5 healthy.
  - <0.05: too noisy → increase β2 or lower LR.
  - >0.6 and slow progress: LR might be too low; nudge up 1.2×.
- orth_residual = ||NS_k(g_pre) − g_pre|| / ||g_pre|| (optional if we hook NS output). Typical 0.1–0.4; >0.6 → lower LR or ns_steps.
- CUDA max memory: max_reserved and max_allocated bytes; track peak VRAM.

Kobs (window sizes):
- 128 steps default at 8k context (≈1M tokens). Range 64–256.

---

## 11) Tuning playbook

If loss is spiky early:
- Ensure EMA bootstrap is on and β2 ramp is active (K≈256).
- Increase half-life (β2) by 2×; keep clip_update_rms=1.0.
- Use 400-step warmup.

If updates are too small (stalled):
- Raise LR_hidden 1.2×.
- Decrease β2 slightly (e.g., 0.9972→0.996).
- Consider ns_steps=3→5.

If orthogonality residual is high:
- Lower LR_hidden by 2×; or ns_steps=5→3.

If 1D drift (norm scales):
- Lower lr_other_scale to 0.25–0.4.
- Keep WD_other=2e-3; optionally swap to a unit-norm pull on norm scales.

---

## 12) VRAM budget estimate (4090, 24 GB, bf16, bs=1, 8k)

- Weights (bf16): ~2 bytes/param. For 8B → ~16 GB.
- Optimizer states:
  - Factored EMA for 2D: row+col vectors per matrix, fp32 → O(m+n). Roughly ~2–4% of weight size → ~0.3–0.6 GB for 8B.
  - Unfactored EMA for 1D: negligible in aggregate.
  - No momentum buffers; Muon adds no per-param state with momentum=0.
- Activations:
  - With FlashAttention and full gradient checkpointing (attn+MLP), peak ~6–8 GB; without full checkpointing, ~10–14 GB.
- Total:
  - Tight but feasible with full checkpointing. Expect 22–25 GB peaks depending on kernel stack and fragmentation. Be ready to trim context to 4k for safety on some stacks.

Guardrails:
- Enable gradient checkpointing and FlashAttention.
- Consider torch.compile with CUDA graphing to reduce fragmentation.
- If OOM, first dial: reduce context length; second: offload activations for MLP blocks.

---

## 13) Baselines and sweep plan

Baselines at bs=1 on Qwen3-0.6B (or Qwen2.5-0.5B as readily available):
- SGD (lr ~ 1e-2, no momentum).
- AdamW (lr ~ 3e-4–1e-3, betas=(0.9, 0.999), wd=0).
- Adafactor (HF impl; lr ~ 3e-4–1e-3; scale_parameter=False, relative_step=False).
- Muon (momentum=0, wd=0, ns_steps=5).
- Apollo (if available in your env; purely for memory/step-size comparison).

Hybrid sweep with Optuna:
- lr_hidden ∈ [0.5, 1.0] × base (log-uniform).
- lr_other_scale ∈ [0.25, 0.5].
- H_tokens_hidden ∈ {1M, 2M, 4M}.
- H_tokens_other ∈ {2M, 4M, 8M}.
- ns_steps ∈ {3, 5}.
- clip_update_rms ∈ {1.0, 2.0}.
Pruning: median over 2–3 eval points; objective: dev loss or tokens-to-target proxy.

Log:
- wandb: loss, max_reserved, all noise metrics, and hyperparameters.
- Store max_reserved_bytes as Optuna user attr.

---

## 14) Compatibility notes

- TRL/Transformers: Hybrid exposes torch.optim.Optimizer-like API; schedulers work on param_groups.
- Distributed: FSDP/ZeRO are compatible; factored states are small. Ensure state_dict sync includes our af states.
- AMP: bf16 preferred; stochastic rounding on write-back implemented.
- Checkpointing: Save both optimizer state_dict and model weights; our state is under af_factored/af_unfactored + muon.

---

## 15) Relation to Harmonic Loss experiments

- Backbone: hybrid is a good match—orthogonalized, RMS-normalized updates align with cosine-distance geometry and scale invariance.
- Prototype/dist head: treat as embeddings (AdaFactor only), lower LR and WD=2e-3, keep vectors normalized (explicit re-norm or constraint). Don’t apply Muon to the prototype table.

---

## 16) Anti-goals and what we intentionally did not include

- Shampoo/SPlus/Apollo low-rank whitening/projection states: too heavy, overlaps with Muon’s geometry, adds complexity without expected gains in this small-step regime.
- Momentum buffers: unnecessary with high-β2 and NS; costs full-size state.
- WD on Muon path: avoided by design.

---

## 17) Defaults

For Qwen3-0.6B, bs=1, 8k context:
- lr_hidden = 5e-4
- lr_other_scale = 0.5
- H_tokens_hidden = 2M → β2≈0.9972
- H_tokens_other = 4M → β2≈0.9986
- ns_steps = 5
- clip_update_rms = 1.0
- wd_other = 2e-3
- warmup = 400 steps
- grad clip = 1.0
- stochastic bf16 rounding = on

For 8B target (edge of 24GB):
- Same hyperparams; ensure full checkpointing.
- Be prepared to cut context to 4k if your stack’s fragmentation is high.

---

## 18) Failure modes and mitigations

- Early loss spikes (first 100–1000 steps):
  - Confirm EMA bootstrap and β2 ramp active.
  - Increase H_tokens by 2×.
  - Ensure clip_update_rms=1.0.
- Divergence under Harmonic Loss:
  - Reduce lr_hidden ×0.5, keep 1D anchored at 0.25–0.5×.
  - ns_steps=5→3.
- Slow progress:
  - lr_hidden ×1.2, slight β2 decrease (H_tokens / 1.5).
- OOM:
  - Enable/verify gradient checkpointing on attn and MLP.
  - Reduce context; offload activations for MLP.

---

## 19) Implementation summary (what’s in the single file)

- HybridMuonAdaFactorBS1 optimizer:
  - Param partitioning (2D hidden vs embeddings/1D).
  - FactoredEMA (row/col) and UnfactoredEMA with token-half-life β2 schedulers and bootstrap.
  - Muon wrapper for 2D with step pre/post hooks.
  - bf16 stochastic rounding write-back.
  - Noise metrics and CUDA max VRAM helpers.
  - wandb logging hook (optional).
- Clean API: step, zero_grad, state_dict/load_state_dict, param_groups accessible for schedulers.

---

## 20) Open questions / future work

- Orth residual logging: expose Muon’s inner NS output to log orth_residual directly per group.
- Per-layer LR shaping: consider tiny LR nudges for attention vs MLP if we see systematic differences under Harmonic Loss.
- Unit-norm constraint for norm scales: optional module wrapper to regularize to 1 without fighting WD.
- Mixed precision NS: evaluate speed/accuracy trade-offs for NS in bf16 with fp32 stabilization.

---

If you deviate from these preconditions (e.g., different dtype, multi-GPU sharding, or non-autoregressive objectives), revisit Sections 5–9 and 12 first. The logic is simple: keep the AdaFactor stats slow but responsive (token half-life), keep Muon steps orthogonalized and scale-adjusted, and anchor the fragile 1D/embedding parameters with LR and mild WD.
