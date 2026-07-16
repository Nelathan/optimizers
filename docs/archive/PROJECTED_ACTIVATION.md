# Projected activation backward

This is the design and experiment record for UsuiTrack's projected-activation backward path.

The goal is not merely to save some VRAM. The goal is to replace the dominant full activation storage needed for matrix weight-gradient formation with small projected activations, then fold loss gradients onto those small tensors during backward while feeding UsuiTrack's projected-moment/Aurora update path.

Checkpointing remains an empirical tool and may still be part of a practical implementation for some blocks, but it is not the product story by itself. A checkpointed run can validate projected-gradient geometry and, when actually wired, can be the right substrate for avoiding depth-wise activation buildup. It does not by itself prove that every useful full activation has been replaced by a projected save.

## Design intent from the original pitch

The original idea lived in `Activation_projection.md`. Its useful content is folded here; that duplicate is removed so this file is the single first-class record.

The durable values from the pitch are:

- **Single-GPU full-parameter adaptation.** The target is full-parameter training under consumer-GPU memory pressure, not LoRA/adapters that permanently constrain forward capacity.
- **Activation storage is the wall.** Optimizer state matters, but after UsuiTrack's projected moments the dominant obstacle is full activation storage and backward temporaries scaling with `batch × sequence × depth × feature`.
- **Full-rank forward stays sacred.** The forward model should not be low-rank bottlenecked. Projection is for optimizer/update state and weight-gradient formation, not for reducing model expressivity.
- **Store small when it replaces large.** Projected activations are valuable when they replace a full activation that would otherwise need to survive from forward to backward for matrix weight-gradient formation.
- **Checkpointing is a tool, not the thesis.** Normal checkpointing may still be best for some blocks if it stores a single block/layer boundary activation and recomputes internals while projected saves reduce `dW` materialization. It becomes a dead end only when projected saves are redundant with full recompute and do not buy meaningful memory.
- **Avoid kernel heroics until forced.** Chunking, slicing, Triton, or custom attention kernels are possible later, but not the goal. First remove the biggest activation-storage hogs with clean PyTorch/custom-autograd boundaries and measure where the peak lands.
- **Clean progress beats clever increments.** Each phase should answer a design question. Do not keep scratch lanes, duplicate docs, option gardens, or milestone hacks as product surfaces after their evidence has been harvested.

Important parts of the original pitch are now superseded by implementation evidence:

- Basis init is stable side-Gram `eigh`, not random init or SVD.
- Refresh smoothing is layer-staggered phase offsets for peak control, not general round-robin optimizer activity.
- Projected gradients are queued into UsuiTrack's optimizer side channel, not expanded or injected into full `.grad` slots.
- The generic projected `Linear` primitive is now side-aware: storage-right projection saves projected activations; storage-left projection forms projected loss-gradient contractions in custom backward. Residual-facing remains the optimizer-quality geometry. All-right/activation-facing was a measured tradeoff, not a default.

## Current status

The current root implementation is in:

- `usuitrack/projected_activation.py`
- `usuitrack/optimizer.py`
- `experiments/llm_synth_smoke.py`
- `tests/test_projected_activation.py`
- `tests/test_llm_harness.py`

What is now implemented:

- explicit UsuiTrack projected-gradient queue via `UsuiTrack.queue_projected_grad(param, projected_grad)`;
- generic side-aware projected `Linear` autograd path;
- fused projected LFM/SwiGLU MLP autograd path for all-right/activation-facing bases;
- opt-in harness backend `--projected-activation-backend lfm`;
- opt-in compiled projected-activation backward tensor helpers when `--torch-compile` is used with the `lfm` backend;
- LFM wrapper coverage for:
  - MLP `w1`, `w3`, `w2`;
  - standard-attention projection Linears `q_proj`, `k_proj`, `v_proj`, `out_proj`;
  - short-conv operator projection Linears `in_proj`, `out_proj`;
- layer-staggered refresh schedule to avoid burst refresh dominating measured peak;
- diagnostic `--projection-side-policy right` to separate activation-facing geometry from projected-backward implementation;
- true CCE hot-path compile targeting: the harness compiles the inner base model that CCE actually calls, not just the outer CausalLM wrapper.
- harness-local LFM2 decoder-layer checkpoint repair, because the Transformers LFM2 model advertises gradient checkpointing but does not call `_gradient_checkpointing_func` in `Lfm2Model.forward()`.

Default training remains no activation projection unless explicitly requested. As of the repaired-checkpoint runs, projected activation is best understood as a performance/library-design sidequest rather than the main optimizer-quality lane. The first full `lfm` path was mathematically faithful to one-sided activation-facing projection and saved memory at rank64 pressure points, but activation-facing/storage-right geometry is not the residual-facing quality default. The current cleanup restores residual-facing side policy for wrapped params and makes the generic `Linear` projected-gradient path side-aware. Right side remains the actual activation-save path; left side preserves residual-facing geometry by projecting `grad_out` in custom backward and avoiding full `weight.grad`, but it does not have the same activation-storage economics. Fused LFM MLP remains right-only until a side-aware fused primitive is written. The current mainline remains repaired checkpointing plus residual-facing UsuiTrack; projected activation is opt-in.

## Core coordinate fact

For PyTorch `Linear.weight == [out_features, in_features]` and `out = input @ weight.T`, the generic projected `Linear` has two one-sided forms.

Storage-right / activation-facing projection stores `Q ∈ R^{rank×in}`:

```text
projected_input = input @ Q.T        # [tokens, rank]
projected_dW    = grad_out.T @ projected_input
```

This matches `full_dW @ Q.T` without returning a full `weight.grad`, and can replace a saved full activation with a saved projected activation.

Storage-left / loss-gradient-facing projection stores `P ∈ R^{out×rank}`:

```text
projected_grad_out = grad_out @ P    # [tokens, rank]
projected_dW       = projected_grad_out.T @ input
```

This matches `P.T @ full_dW` without returning a full `weight.grad`. It preserves residual-facing geometry for MLP down and attention output tensors, but the projectable tensor exists only in backward, so it is not the same activation-save win as the right-side path.

Right side matches residual-facing for MLP up/gate and attention q/k/v. Left side matches residual-facing for MLP down and attention output. The old all-right `lfm` activation projection changed side geometry for down/out tensors.

The corrected-LR all-right control showed this geometry change is real but not a basic backward-math bug: full `lfm` activation projection and ordinary no-projection all-right UsuiTrack were nearly identical at the 100-step quality sensor.

## Proven gates

### Parameter hooks are too late

PyTorch materializes the full weight gradient before parameter hooks run. Hook-time projection removed retained `.grad` buffers but barely moved LFM peak memory at faithful shapes. This path was removed.

### Generic projected Linear works

The projected `Linear` path:

- runs the same full-rank forward;
- saves projected input for right-side weight-gradient formation, or saves the ordinary input for left-side projected loss-gradient formation;
- emits `grad_out.T @ (input @ Q.T)` for right-side bases or `(grad_out @ P).T @ input` for left-side bases through a side channel;
- returns exact `grad_input` and optional bias grad;
- leaves captured `weight.grad` as `None` on projected steps.

Tiny tests prove equality to ordinary full-gradient projection on both storage sides: `full_grad @ Q.T` for right bases and `P.T @ full_grad` for left bases. Optimizer-ingress tests prove both sides drive the same UsuiTrack update as ordinary full-gradient projection after basis initialization.

### Optimizer ingress works

`UsuiTrack.queue_projected_grad()` accepts already-projected matrix gradients for initialized, non-refresh params. It is transient optimizer-side state, cleared by `zero_grad()`, not serialized, and rejected before basis init or on refresh steps.

Queued projected-gradient steps match ordinary full-gradient UsuiTrack steps after warm basis init.

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

The old faithful rank-ablation lane used `--usuitrack-lr 2e-4`. A stale parser default `0.0025` caused false catastrophic quality runs by overdriving update norms. The harness default is now corrected to `2e-4`.

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

Matched repaired-checkpoint controls clarify the incremental value of projected activation under real layer checkpointing:

| run | target/source @ 200 | peak allocated | step sec | interpretation |
| --- | --- | ---: | ---: | --- |
| residual-facing `off` | `1.877504 / 2.991532` | `2,072,345,088` | `0.524733` | baseline geometry preserves source slightly better |
| all-right `off` | `1.861207 / 3.005637` | `2,072,345,088` | `0.524142` | same geometry as activation projection, no custom projected backward |
| full `lfm` projected activation | `1.861770 / 3.004450` | `1,797,460,480` | `0.570883` | same quality shape as all-right, `~275 MB` lower peak, slower eager wrappers |

This is the current honest boundary: projected activation is faithful and still saves memory on top of real checkpointing, but the giant win was fixing checkpointing itself. The next decision is whether the extra `~275 MB` at `bs8×seq1024` and possible larger-shape headroom justify optimizing wrapper overhead, or whether the product path should lean on repaired checkpointing plus UsuiTrack state savings first.

Current answer: lean on repaired checkpointing plus UsuiTrack state savings first. Keep projected activation as an opt-in branch for memory pressure and future performance work, not as the default quality lane. If revisited, the next work should be speed/perf cleanup and larger-shape memory evidence, not more proof of gradient arithmetic.

### Compile spike after repaired checkpointing

The first speed cleanup moved model compilation after projected-activation backend installation and added `set_projected_activation_compile(enabled)`, which compiles only the pure tensor work inside the projected Linear and fused gated-MLP backward paths. The custom autograd Functions still own the Python side-channel that queues projected gradients into UsuiTrack; optimizer bookkeeping stays outside Dynamo. Later audit found that compiling only the outer CausalLM wrapper misses the CCE hot path, because `cce_causal_lm_loss()` calls the inner base model plus `lm_head` directly. The harness now compiles that inner training base model for CCE-style models and uses PyTorch functional AdamW for fallback parameters; fused functional AdamW is used only for fp32 CUDA fallback tensors because PyTorch rejects mixed bf16-param/fp32-state fused AdamW.

Short validation-skipped smokes, all with repaired activation checkpointing, rank 64, `seq1024`, two warmup steps, and `--torch-compile` where noted:

| run | backend | compile | batch | peak allocated | peak reserved | step sec | tokens/s |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| pre-change `off` control | `off` | yes | 4 | `1,697,642,496` | n/a | `0.273423` | `14,980` |
| pre-change eager PA | `lfm` | no | 4 | `1,237,795,840` | `2,392,850,432` | `0.279863` | `14,636` |
| compile-after-install only | `lfm` | yes | 4 | `1,237,795,840` | n/a | `0.279618` | `14,649` |
| compiled PA backward tensors | `lfm` | yes | 4 | `1,197,753,344` | `2,392,850,432` | `0.256436` | `15,973` |
| sequential `off` control | `off` | yes | 8 | `2,056,244,736` | `2,220,883,968` | `0.536572` | `15,267` |
| sequential compiled PA | `lfm` | yes | 8 | `1,616,779,776` | `2,824,863,744` | `0.518161` | `15,810` |

The `bs8` rows were rerun sequentially after an invalid parallel same-GPU attempt; only the sequential numbers are evidence. The result was positive but preliminary: projected activation can be smaller and modestly faster under compile at rank64, while reserved memory may rise because of allocator/compile behavior.

Rank256 current-lane follow-up: `--projected-activation-backend lfm --torch-compile`, repaired checkpointing, `bs16 × seq1024`, rank256, LR `3e-4`, LR warmup `50`, burst100, dual rails `2/6`, source retention enabled, W&B `anqz70ln`, completed 1k measured steps. Final target/source was `1.677808 / 3.074101`, train loss `1.663006`, and mean update norm `0.093577`. This is slightly worse than the residual-facing rank256 dual-rail quality anchor, consistent with activation-facing geometry costing some quality/source retention. Do **not** use the run's reported `3,248,483,328` peak, `13,610` tokens/s, or `1.203802s` step as clean performance evidence: the harness then included periodic eval in measured throughput, read peak memory after final eval/sample generation, and did not compile the inner CCE transformer base. Keep this path opt-in until rerun with fixed measurement and true hot-path compile.

The side-aware generic `Linear` variant now preserves residual-facing geometry without forcing all wrapped weights to activation-facing bases. It is correct and portable, but the per-Linear custom-autograd scheduling did not earn more performance work. Fair rank256 `bs64 × seq1024` smokes with true CCE hot-path compile and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` favored the baseline: `off` ran `20,089 tok/s`, `3.262229s/step`, peak `8,422,493,184`; side-aware `lfm` ran `14,564 tok/s`, `4.499743s/step`, peak `9,557,816,320`. A default-allocator `off` attempt OOMed while `lfm` fit, but the expandable-segments retry showed that was allocator fragmentation, not a durable projected-gradient capacity win.

End this try: generic projected `Linear` stays as a tested correctness primitive, not as the performance route.

Boundary smokes with repaired checkpointing and full `lfm` projected activation at `seq1024`, validation skipped:

| batch | tokens/update | peak allocated | peak reserved | step sec | tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | `16,384` | `2,609,973,760` | `2,990,538,752` | `1.1632` | `14,085` |
| 32 | `32,768` | `4,439,957,504` | `5,087,690,752` | `2.3478` | `13,957` |
| 64 | `65,536` | `8,099,395,072` | `9,386,852,352` | `4.7135` | `13,904` |
| 80 | `81,920` | `9,928,586,752` | `10,710,155,264` | `5.9103` | `13,861` |

This is a boundary map, not a default recommendation. It shows repaired checkpointing plus projected activation has moved the local LFM-350M memory regime dramatically; useful token mass and throughput/quality now need measurement rather than assumption.

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

1. **Side-aware fused MLP.** Can one fused custom autograd primitive handle gate/up right-side projected activations and down left-side projected loss gradients with fewer launches and better scheduling than per-Linear wrappers?
2. **Fused MLP microbench gate.** At token counts matching `bs16` and `bs64`, and ranks `64/256`, does the fused primitive beat ordinary baseline backward on step time and peak? If not, do not integrate it.
3. **Remaining peak owner after fused MLP.** If fused MLP wins locally, is the next dominant object MLP input save, attention/short-conv kernel state, CCE/loss state, refresh fallback, optimizer update temps, or allocator timing?
4. **Checkpointing as substrate.** Repaired decoder-layer checkpointing is now empirically best for the current memory shape. Any projected-gradient primitive must prove incremental value on top of real checkpointing, not fake stock-LFM checkpointing.

## Side-aware fused MLP: gate run, gate failed

The side-aware fused MLP primitive was built and gated on 2026-07-05 (`_ProjectedActivationGatedMlpSideAware`, `experiments/fused_mlp_side_aware_microbench.py`, RESULTS.md entry). Math is fp64-exact on all three sides and no `[T, intermediate]` tensor is saved. The synthetic microbench at LFM-350M MLP dims tied the compiled ordinary baseline on step time and lost to it on peak at every (T, rank) tested.

The failure is structural. The theoretical dW-FLOP saving is consumed by the gate/up recompute, and the down-left contraction needs `hidden` alive alongside the other `[T, f]` transients — a perfect schedule still holds ~3–4 simultaneously, which is parity with Inductor's save/recompute on the plain baseline, not a win. At `hidden=1024` the full weight grads avoided are ~13 MB each; the real memory object in this block is `[T, f]` activation liveness, and `torch.compile` already manages that on ordinary backward.

Verdict: do not integrate into LFM. The projected-backward performance lane is closed at 350M shapes. Compiled ordinary backward plus repaired decoder-layer checkpointing is the performance path; the generic side-aware `Linear` and both fused MLP Functions remain tested correctness primitives for possible future shapes (much larger `hidden`, where avoided full weight grads stop being negligible), not active leads.
