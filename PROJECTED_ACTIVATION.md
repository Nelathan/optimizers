# Projected activation backward

This is the design and experiment record for SumoTrack's projected-activation backward path.

The goal is not merely to save some VRAM. The goal is to replace the dominant full activation storage needed for matrix weight-gradient formation with small projected activations, then fold loss gradients onto those small tensors during backward while feeding SumoTrack's projected-moment/Aurora update path.

Checkpointing remains an empirical tool and may still be part of a practical implementation for some blocks, but it is not the product story by itself. A checkpointed run can validate projected-gradient geometry and, when actually wired, can be the right substrate for avoiding depth-wise activation buildup. It does not by itself prove that every useful full activation has been replaced by a projected save.

## Design intent from the original pitch

The original idea lived in `Activation_projection.md`. Its useful content is folded here; that duplicate is removed so this file is the single first-class record.

The durable values from the pitch are:

- **Single-GPU full-parameter adaptation.** The target is full-parameter training under consumer-GPU memory pressure, not LoRA/adapters that permanently constrain forward capacity.
- **Activation storage is the wall.** Optimizer state matters, but after SumoTrack's projected moments the dominant obstacle is full activation storage and backward temporaries scaling with `batch × sequence × depth × feature`.
- **Full-rank forward stays sacred.** The forward model should not be low-rank bottlenecked. Projection is for optimizer/update state and weight-gradient formation, not for reducing model expressivity.
- **Store small when it replaces large.** Projected activations are valuable when they replace a full activation that would otherwise need to survive from forward to backward for matrix weight-gradient formation.
- **Checkpointing is a tool, not the thesis.** Normal checkpointing may still be best for some blocks if it stores a single block/layer boundary activation and recomputes internals while projected saves reduce `dW` materialization. It becomes a dead end only when projected saves are redundant with full recompute and do not buy meaningful memory.
- **Avoid kernel heroics until forced.** Chunking, slicing, Triton, or custom attention kernels are possible later, but not the goal. First remove the biggest activation-storage hogs with clean PyTorch/custom-autograd boundaries and measure where the peak lands.
- **Clean progress beats clever increments.** Each phase should answer a design question. Do not keep scratch lanes, duplicate docs, option gardens, or milestone hacks as product surfaces after their evidence has been harvested.

Important parts of the original pitch are now superseded by implementation evidence:

- Basis init is stable side-Gram `eigh`, not random init or SVD.
- Refresh smoothing is layer-staggered phase offsets for peak control, not general round-robin optimizer activity.
- Projected gradients are queued into SumoTrack's optimizer side channel, not expanded or injected into full `.grad` slots.
- Activation-facing/storage-right projection is the actual coordinate contract for projected activation. Residual-facing remains the baseline geometry; all-right/activation-facing is a measured tradeoff, not assumed free.

## Current status

The current root implementation is in:

- `sumotrack/projected_activation.py`
- `sumotrack/optimizer.py`
- `experiments/llm_synth_smoke.py`
- `tests/test_projected_activation.py`
- `tests/test_llm_harness.py`

What is now implemented:

- explicit SumoTrack projected-gradient queue via `SumoTrack.queue_projected_grad(param, projected_grad)`;
- custom projected-activation `Linear` autograd path;
- fused projected LFM/SwiGLU MLP autograd path;
- opt-in harness backend `--projected-activation-backend lfm`;
- LFM wrapper coverage for:
  - MLP `w1`, `w3`, `w2`;
  - standard-attention projection Linears `q_proj`, `k_proj`, `v_proj`, `out_proj`;
  - short-conv operator projection Linears `in_proj`, `out_proj`;
- layer-staggered refresh schedule to avoid burst refresh dominating measured peak;
- diagnostic `--projection-side-policy right` to separate activation-facing geometry from projected-backward implementation.
- harness-local LFM2 decoder-layer checkpoint repair, because the Transformers LFM2 model advertises gradient checkpointing but does not call `_gradient_checkpointing_func` in `Lfm2Model.forward()`.

Default training remains no activation projection unless explicitly requested.

## Core coordinate fact

For PyTorch `Linear.weight == [out_features, in_features]` and `out = input @ weight.T`:

```text
projected_input = input @ Q.T        # [tokens, rank]
projected_dW    = grad_out.T @ projected_input
```

This is storage-right / activation-facing projection. It matches residual-facing for MLP up/gate and attention q/k/v. It does not match residual-facing for MLP down or attention output, so activation projection changes side geometry for those tensors.

The corrected-LR all-right control showed this geometry change is real but not a basic backward-math bug: full `lfm` activation projection and ordinary no-projection all-right SumoTrack were nearly identical at the 100-step quality sensor.

## Proven gates

### Parameter hooks are too late

PyTorch materializes the full weight gradient before parameter hooks run. Hook-time projection removed retained `.grad` buffers but barely moved LFM peak memory at faithful shapes. This path was removed.

### Custom Linear works

The projected `Linear` path:

- runs the same full-rank forward;
- saves projected input for weight-gradient formation;
- emits `grad_out.T @ (input @ Q.T)` through a side channel;
- returns exact `grad_input` and optional bias grad;
- leaves captured `weight.grad` as `None` on projected steps.

Tiny tests prove equality to ordinary full `weight.grad @ Q.T`.

### Optimizer ingress works

`SumoTrack.queue_projected_grad()` accepts already-projected matrix gradients for initialized, non-refresh params. It is transient optimizer-side state, cleared by `zero_grad()`, not serialized, and rejected before basis init or on refresh steps.

Queued projected-gradient steps match ordinary full-gradient SumoTrack steps after warm basis init.

### Fused MLP math works

The fused LFM/SwiGLU path computes exact forward and input gradients and projected gate/up/down gradients matching ordinary full gradients projected by each stored basis.

Early versions saved full `gate_pre` and `up` to keep the proof clean. That was an experiment hack, not a design endpoint.

### Fused MLP no longer saves full gate/up

The current fused MLP backward saves projected activations plus the MLP input, then recomputes `gate_pre` and `up` in backward. The recompute is scheduled so `up` is not materialized until the gate path needs it.

This removes the two largest obvious full MLP saved intermediates: `[tokens, intermediate]` `gate_pre` and `up`.

Tests now include saved-tensor coverage proving the fused MLP no longer saves `[batch, intermediate]` gate/up tensors, while equality tests still prove projected grads and input grads match reference.

## Memory milestones

### Isolated Linear smoke

bf16, `tokens=8192`, `source_features=1024`, `in_features=4096`, `out_features=4096`, `rank=64`:

- ordinary peak: `872,547,328`
- projected peak: `815,531,008`
- peak delta: `57,016,320`
- ordinary target `weight.grad`: `33,554,432`
- projected replacement: `524,288`

The saving scales with tokens because the removed object is saved activation state, not only final full `.grad`.

### Fused MLP before recompute

bf16, `hidden=4096`, `intermediate=8192`, rank 64:

- `tokens=4096`: projected peak was worse than ordinary because saved `gate_pre/up` plus projected buffers outweighed the removed activation saves.
- `tokens=8192`: projected peak saved `112,066,560` bytes.
- `tokens=16384`: projected peak saved `445,513,728` bytes.

This proved the thesis but exposed `gate_pre/up` as the next storage hog.

### Fused MLP after scheduled recompute

Synthetic gated MLP smoke after recompute scheduling:

- `tokens=4096`: projected peak still slightly worse than ordinary by `21,626,880` bytes, but much better than naive recompute.
- `tokens=8192`: projected peak saved `178,126,848` bytes; after-forward allocation saved `313,917,440` bytes; retained after-backward memory saved `194,510,848` bytes.

Naive recompute was worse because it materialized too many full intermediates at once. Scheduling mattered.

### Real LFM no-checkpoint smoke after MLP recompute

Full `lfm`, no activation checkpointing, `bs4 × seq1024`, rank 64, layer-staggered refresh:

- 10 measured steps: peak `2,754,171,392`, step `0.205771s`, tokens/s `19,905.6`.
- 200 measured steps crossing refresh: peak `2,931,879,424`, step `0.206220s`, tokens/s `19,862.3`, mean update norm `0.057830`.

Before MLP recompute, the comparable full `lfm` 200-step no-checkpoint peak was `~3.91 GB`. The recompute cut saved roughly another `~0.98 GB` on the real path.

## Quality and geometry controls

The old faithful rank-ablation lane used `--sumotrack-lr 2e-4`. A stale parser default `0.0025` caused false catastrophic quality runs by overdriving update norms. The harness default is now corrected to `2e-4`.

Corrected-LR checkpointed 100-step controls:

- old residual-facing baseline step 100: target `1.941913`, source `2.966022`;
- current corrected residual-facing baseline step 100: target `1.932040`, source `2.969326`;
- full `lfm` activation projection step 100: target `1.916492`, source `2.984066`, mean update norm `0.061237`;
- no-projection all-right control step 100: target `1.916419`, source `2.985453`, mean update norm `0.060246`.

Interpretation: activation projection is faithfully realizing all-right/activation-facing geometry. The remaining quality tradeoff is side geometry, not an obvious projected-backward arithmetic bug.

The 200-step corrected-LR checkpointed full `lfm` run was a milestone sanity test, not a product lane:

- W&B run `je0dc9xc`, `lfm-activation-projection-lr2e4-r64-bs8-200`;
- step 100: target `1.917199`, source `2.984890`;
- step 200: target `1.868035`, source `2.997959`;
- mean update norm `0.054502`.

Checkpointing is not forbidden. It is now a practical substrate again after the LFM2 repair: layer checkpointing stores boundary activations and recomputes internals, while projected activation still prevents full projection-weight `dW` materialization and keeps queued projected gradients small.

## LFM2 checkpointing repair

Transformers' LFM2 implementation sets `supports_gradient_checkpointing = True` and `Lfm2Model.gradient_checkpointing = False`, and `gradient_checkpointing_enable()` sets the flag plus `_gradient_checkpointing_func`. But `Lfm2Model.forward()` loops through decoder layers and calls them directly. The flag was lit; the engine was not connected.

The harness now narrowly repairs this when `--activation-checkpointing` is requested:

- only `Lfm2Model` is patched;
- each decoder layer forward is wrapped with non-reentrant `torch.utils.checkpoint.checkpoint()`;
- grad-enabled execution checkpoints the layer;
- `torch.no_grad()` / eval-style execution calls the original layer directly;
- no LFM model fork is maintained.

One-forward/backward monitor, full `lfm`, `bs8 × seq1024`, one warm basis step then measured step:

| mode | after forward+loss | final peak | queued projected grads | retained full grads |
| --- | ---: | ---: | ---: | ---: |
| no checkpoint | `4,370,416,640` | `4,731,528,192` | `28,573,696` | `128,512` |
| stock HF checkpoint flag | `4,420,781,056` | `4,781,892,608` | `28,573,696` | `128,512` |
| repaired decoder-layer checkpoint | `1,094,175,744` | `1,692,342,272` | `28,573,696` | `128,512` |

Corrected-LR 200-step quality sensor after the repair:

- W&B run `51nsznhn`, `lfm-projected-repaired-checkpoint-r64-bs8-200`;
- step 100 target/source: `1.916834 / 2.984027`;
- step 200 target/source: `1.861770 / 3.004450`;
- last train loss `1.849765`, mean update norm `0.059228`;
- peak allocated `1,797,460,480`, peak reserved `2,472,542,208`;
- step time `0.570883s`, tokens/s `14,349.7`.

Interpretation: real checkpointing cuts memory hard and preserves the corrected-LR quality shape, but it costs throughput. That is the actual tradeoff to optimize, not the old fake checkpoint curve.

Artifacts:

- `/tmp/opencode/projected_activation_monitor_bs8_checkpoint_repaired_timeline/repaired_checkpoint_timeline_table.md`
- `/tmp/opencode/projected_activation_monitor_bs8_checkpoint_repaired_timeline/repaired_checkpoint_memory_timeline.svg`

Saved-tensor owner hooks are useful for no-checkpoint forensics, but they are not the authoritative checkpoint monitor because non-reentrant checkpointing itself relies on saved-tensor machinery. Use CUDA timeline/peak events for checkpoint memory claims.

## Attention and short-conv status

Current attention/short-conv support is a Linear-boundary cut:

- save projected inputs for projection Linear weight gradients;
- avoid full projected Linear `.grad` materialization;
- preserve standard attention / SDPA / FA behavior and short-conv kernels.

It does not make the whole attention block projected-activation clean.

Standard attention still has nonlinear/kernel state around q/k norms, rotary, attention probabilities/logsumexp, and SDPA backward. Short-conv still has branch products and conv state. A projected input activation is enough for a projection Linear's `dW`; it is not enough by itself to reconstruct exact attention/conv gradients.

If exact backward for a block requires recomputing full q/k/v/attention or conv internals, that approaches standard checkpointing. The question is not whether checkpointing is morally allowed; it is whether projected saves replace a real full activation storage hog rather than becoming redundant with recomputed full activations.

## Latest no-checkpoint monitor

One-off monitor after commit `4597b6c`, full `lfm`, no checkpointing, `bs8 × seq1024`, one warm basis step then one measured forward/backward:

- CSV: `/tmp/opencode/lfm_no_checkpoint_recompute_bs8_monitor.csv`
- installed projected modules: `60`
- side counts: `left=0`, `right=92`, `auto=0`
- final allocated: `803,525,120`
- peak allocated: `4,731,528,192`
- queued projected grad bytes after backward: `28,573,696`
- retained full grad bytes after backward: `128,512`

Top events show the peak occurs around the last layer's MLP backward and immediately adjacent short-conv/attention module boundaries:

- `bwd_post:mlp:model.layers.15.feed_forward` hit peak `4,731,528,192`;
- adjacent layer 15 short-conv and layer 14 MLP/attention events kept the same peak while allocation fell.

Interpretation still needs care. The monitor shows where the peak occurs in module time, not yet which internal tensor owns it. Do not optimize by analogy until the owner is identified.

## Open design questions

1. **Remaining peak owner.** Is the next dominant object MLP input save, attention/short-conv kernel state, CCE/loss state, refresh fallback, optimizer update temps, or allocator timing?
2. **Attention/short-conv economics.** Do projected Linear boundary saves still buy enough to matter once MLP is fixed, or are kernel internals now the wall?
3. **Checkpointing as substrate.** Repaired decoder-layer checkpointing is now empirically best for the current `bs8` memory shape. Next decide what projected activation buys on top of real checkpointing and whether any remaining full-activation saves are worth cutting without kernel work.
4. **Quality without checkpointing.** Now that the no-checkpoint full `lfm` path fits much better, run corrected-LR target/source sensors on the product-shaped path.

## Next coherent cut

Read the monitor before optimizing:

1. validate repaired checkpointing in ordinary harness runs, not only monitor scripts;
2. rerun corrected-LR quality/throughput sensors on the repaired checkpoint substrate;
3. measure the incremental value of projected activation versus ordinary right-side/no-projection under real checkpointing;
4. avoid kernel-level/Triton work unless a measured bottleneck leaves no higher-level route.
