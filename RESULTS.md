# SumoTrack Results

Short empirical notes from local runs. Treat these as terrain markers, not claims of optimizer quality.

This file is chronological experiment history. Older entries may describe defaults, flags, model choices, or harness behavior that have since been superseded. The current contract and durable facts distilled from these runs are in `PLAN.md`. When an old run used SVD init, packed SYNTH formatting, random init, HF loss, or a now-stale LR prior, read it as dated evidence for that specific setup, not as current guidance.

## 2026-06-30: Faithful SYNTH rank sweep with qualitative samples

Question: after rank 64 became the boring default, does increasing or decreasing uniform projected rank change the target/source tradeoff and qualitative target-format adoption on the faithful SYNTH lane?

Harness change: final validation can now print one qualitative target-eval sample. Sampling defaults are `temperature=0.6`, `top_k=20`, `top_p=0.95`, no `min_p`, and max prompt+generated length `4096`. This path is guarded for dirty HF target formats; non-SYNTH HF target samples require an explicit override and were not printed for Sugar Quill.

Shared setup: `LiquidAI/LFM2.5-350M-Base`, broad no embeddings, residual-facing side, stable `eigh`, Aurora `pp=2/ns=5`, CCE, SDPA, faithful right-padded no-mask SYNTH batches, `batch_size=8`, `seq_len=1024`, activation checkpointing, `sumotrack_lr=2e-4`, `basis_refresh_interval=100`, `measure_steps=1000`, `eval_every=100`, source sensor `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled:data/train-00000-of-00100.parquet`. Training prints target/source eval every 100 measured steps, so each run exposes whether progress is smooth or stalled.

Runs:

| rank | wandb | target val loss | source val loss | state bytes | peak allocated CUDA | step sec | tok/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | https://wandb.ai/pink-marker/sumotrack/runs/ddshpeov | `2.300731 → 1.828967` | `2.899438 → 3.027462` | `12,507,136` | `9,840,612,864` | `0.418382` | `19,580` |
| 32 | https://wandb.ai/pink-marker/sumotrack/runs/n68iwyi2 | `2.300731 → 1.804122` | `2.899438 → 3.019680` | `24,500,224` | `9,852,605,952` | `0.417277` | `19,632` |
| 64 | https://wandb.ai/pink-marker/sumotrack/runs/kkmk57jz | `2.300731 → 1.778049` | `2.899438 → 3.014893` | `48,486,400` | `9,876,592,128` | `0.424949` | `19,278` |
| 128 | https://wandb.ai/pink-marker/sumotrack/runs/milti2cv | `2.300731 → 1.750586` | `2.899438 → 3.010585` | `96,458,752` | `9,924,564,480` | `0.435693` | `18,802` |
| 256 | https://wandb.ai/pink-marker/sumotrack/runs/cxy82wsx | `2.300731 → 1.722463` | `2.899438 → 3.012350` | `192,403,456` | `10,020,509,184` | `0.467022` | `17,541` |

Interpretation:

- Higher rank improves target loss monotonically across this sweep. The improvement is not subtle: rank 256 beats rank 64 by `~0.056` target loss at 1k steps.
- Source loss is worst at low rank and roughly flat/better from rank 64 upward. Honorable mention: higher rank did not buy target movement by obviously paying more source loss on this source sensor.
- Optimizer state scales linearly with rank as expected, but peak CUDA is dominated by model activations/temporaries; rank 16→256 changed peak allocated by only `~180 MB` while optimizer state changed by `~180 MB`.
- Step time is nearly flat through rank 64, modestly worse at rank 128, and visibly worse at rank 256. Compute was not the deciding question here, but the cost curve is now visible.
- Qualitative samples show target-format adoption even at rank 16 on a tiny model. Lower ranks adopt the reasoning/answer shape but are more repetitive and less grounded; higher ranks look somewhat more fluent, though all samples still contain technical confabulation. Treat samples as texture, not evidence stronger than eval curves.
- Current read: uniform rank 64 remains a good boring default for byte budget, but rank is a real quality lever. If the product budget allows it, rank 128/256 deserve attention before inventing more geometry knobs.

## 2026-06-30: First-basis accumulation and Sugar Quill head-to-head

Question: does initializing the first `eigh` bases from a `4×` accumulated first gradient estimate improve the known faithful-lane curve, and does the current default behave similarly on `Nelathan/synthetic-sugar-quill`?

Safety/format contract for Sugar Quill: row contents were not read, printed, decoded, or logged. A schema-only parquet check found one shard, 9,619 rows, columns `id`, `text`, `profile`, `model`. The harness formats target rows as `profile\n\n---\n\ntext`, masks `profile + divider`, and trains only on `text`, analogous to masking the SYNTH question/divider.

Harness changes used for this run:

- Added `--target-hf-dataset`, `--target-format profile_text`, and `--target-val-offset` for HF target datasets.
- Temporarily added `--first-step-grad-accum-steps` to affect only the first optimizer step / basis-forming gradient estimate, not normal training-time gradient accumulation. After these near-null results, that flag was removed rather than kept as an undead knob.
- Cleared CUDA cache after initial validation before training. The first `bs8`/`4×` attempt OOMed before useful training when desktop/browser VRAM pressure was present; retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` completed.

Shared setup: `LiquidAI/LFM2.5-350M-Base`, broad no embeddings, rank 64, residual-facing side, stable `eigh`, Aurora `pp=2/ns=5`, CCE, SDPA, faithful right-padded no-mask batches, `batch_size=8`, `seq_len=1024`, activation checkpointing, `sumotrack_lr=2e-4`, `measure_steps=1000`, `eval_every=100`, source sensor `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled:data/train-00000-of-00100.parquet`.

Runs:

| target | first-step accum | refresh | wandb | supervised tokens/update | target val loss | source val loss | mean basis rotation | mean update/param |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| SYNTH | 1 | 100 | previous baseline `8yn89vj9` | 6399 | `2.300731 → 1.776949` | `2.899437 → 3.010900` | n/a in old table | `0.000227` |
| SYNTH | 4 | 100 | https://wandb.ai/pink-marker/sumotrack/runs/goiyjzg9 | 6399 | `2.300731 → 1.775587` | `2.899438 → 3.012776` | `0.149512` | `0.000213` |
| SYNTH | 4 | 20 | https://wandb.ai/pink-marker/sumotrack/runs/5w6v7125 | 6399 | `2.300731 → 1.774600` | `2.899438 → 3.011622` | `0.123417` | `0.000225` |
| Sugar Quill | 1 | 100 | https://wandb.ai/pink-marker/sumotrack/runs/74ymlx78 | 5656 | `3.277249 → 3.183513` | `2.899438 → 2.940904` | `0.148967` | `0.000224` |
| Sugar Quill | 4 | 100 | https://wandb.ai/pink-marker/sumotrack/runs/onduo46m | 5656 | `3.277249 → 3.182505` | `2.899438 → 2.942369` | `0.151005` | `0.000221` |

Interpretation:

- `4×` first-basis accumulation is a near-null on the known SYNTH lane at 1k steps: target/source are essentially baseline. It did not reveal an obvious first-basis quality win.
- Refresh interval `20` with `4×` first-basis slightly lowered logged mean chordal rotation and landed a hair better on target/source, but the effect is too small to reopen the refresh axis.
- Sugar Quill is not SYNTH-like under this setup: target loss moved much less (`~0.094–0.095` vs `~0.525`) and source loss rose much less (`~0.041–0.043` vs `~0.111–0.113`) at matched update scale. Treat this as dataset/contact evidence, not as a clean optimizer win/loss.
- `4×` first-basis accumulation is again a near-null on Sugar Quill.
- Current read: the basis-initialization/refesh knob is not the next high-value lever for the 350M faithful lane. The interesting question is why Sugar Quill has much weaker target movement at the same optimizer scale: data difficulty/format/contact, not optimizer geometry by default.

## 2026-06-24: Compile and basis-rotation telemetry cleanup

Question: can we use `torch.compile` without compiling Python optimizer bookkeeping, and is basis refresh telemetry worth keeping in routine wandb runs?

Compile result: compiling the whole `optimizer.step()` failed before measured training because Dynamo specialized on `Parameter` object identity through the fallback path and HeavyBall fused Adam internals. The viable cut is targeted compile: compile the model forward/backward and SumoTrack's pure tensor orthogonalization kernel, while leaving Python optimizer state/bookkeeping eager. A 2-step smoke and a 100-step smoke both passed with `--torch-compile`, SDPA, faithful SYNTH batches, activation checkpointing, and LR `2e-4`; no quality claim is attached to those smokes.

Basis telemetry result: the 250-step and accidental uncompiled 1k telemetry runs both showed the same shape. Basis-capture before/after refresh was essentially flat (`~+3e-4`), so capture did not explain the run and is not useful routine telemetry. Chordal basis rotation is the remaining useful sensor: it is cheap enough to keep because it runs only at cadence/refresh and computes an SVD on the rank×rank basis overlap, not on model-sized matrices. In the 1k run, mean chordal rotation was lower than the 250-step probe (`~0.148` vs `~0.176`), so there is no current evidence that basis motion is exploding or that more frequent refresh is needed.

Wandb cleanup decision: routine runs now log only `target_val_loss`, `source_val_loss`, `train_loss`, `grad_norm`, `update_norm`, `update_to_param_ratio`, and `basis_rotation_chordal`. Basis capture, top-1 rotation, leverage, param norm, memory summaries, and all `final_*` wandb summaries were dropped. Memory/state/final-loss numbers remain in terminal output and result records, not graph clutter.

Refresh compile decision: do not compile the whole refresh path. It includes state lookup, basis replacement, moment transport, `eigh`/QR/Grassmann logic, shape/side branches, and diagnostics, and it only runs every refresh interval. If profiling later shows refresh is material, split a pure tensor refresh primitive and compile that; do not point Dynamo at the full refresh machinery.

## 2026-06-24: Faithful SYNTH grad accumulation check at 2e-4

Question: at the same total supervised-token budget, does `grad_accum=2` improve quality by reducing update noise, or does fewer optimizer updates lose target/source quality?

Setup matched the `2e-4`, `bs8 + activation_checkpointing`, `pp=2/ns=5`, SDPA, faithful SYNTH baseline except `grad_accum_steps=2`, `measure_steps=500`, and `eval_every=50`. The `eval_every=50` choice aligned validation by token budget but made the wandb step-axis visually non-comparable to the 1000-step baseline; future same-token comparisons should keep `eval_every=100` for wandb consistency and compare token budget explicitly.

Run: https://wandb.ai/pink-marker/sumotrack/runs/eahca2ey

| run | optimizer steps | supervised tokens/update | target val loss | source val loss | last train loss | mean grad norm | mean update norm | update/param |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `grad_accum=1` baseline | 1000 | 6399 | `2.300731 → 1.776949` | `2.899437 → 3.010900` | `1.816579` | `9.482308` | `0.056273` | `0.000227` |
| `grad_accum=2` | 500 | 12798 | `2.300731 → 1.795405` | `2.899438 → 3.018159` | `1.725927` | `6.868297` | `0.057202` | `0.000231` |

Interpretation: ignore throughput here. At fixed total token budget and LR, `grad_accum=2` lost on quality: worse target and slightly worse source, with essentially matched update scale. It did reduce raw grad norm, but that did not translate into a better target/source curve. Do not promote gradient accumulation as a quality fix from this result.

## 2026-06-24: Faithful SYNTH Aurora/NS cycle sweep at 2e-4

Question: at the useful LR knee, how much Aurora / Newton-Schulz work is enough? Low tokens/step makes optimizer compute worth watching, but the target is not perfect polar math; it is loss quality per memory-constrained step.

Shared setup:

- Model: `LiquidAI/LFM2.5-350M-Base`.
- Source sensor: first parquet shard from `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` (`data/train-00000-of-00100.parquet`).
- Faithful SYNTH right-padded no-mask batches, CCE, broad no embeddings, rank 64, residual-facing side policy, stable `eigh` basis init, `basis_refresh_interval=100`, `batch_size=8`, `seq_len=1024`, activation checkpointing, `warmup_steps=1`, `measure_steps=1000`, `eval_every=100`, `sumotrack_lr=2e-4`.
- Local attention contract: `attn_implementation=sdpa`. An earlier attempted run without pinning the local SDPA contract OOMed before training and is not optimizer evidence.
- Actual supervised tokens/update: `6399`; sequence tokens/update: `8192`; optimizer state bytes: `48,486,400`; peak allocated CUDA: `9,878,730,240`.

Runs:

| Aurora pp | NS steps | wandb | target val loss | source val loss | last train loss | mean update norm | update/param | step sec |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5 | https://wandb.ai/pink-marker/sumotrack/runs/8yn89vj9 | `2.300731 → 1.776949` (`-0.523782`) | `2.899437 → 3.010900` (`+0.111462`) | `1.816579` | `0.056273` | `0.000227` | `0.663540` |
| 1 | 1 | https://wandb.ai/pink-marker/sumotrack/runs/6vlwt7ef | `2.300731 → 1.826631` (`-0.474100`) | `2.899438 → 3.018623` (`+0.119185`) | `1.868589` | `0.052900` | `0.000214` | `0.651163` |
| 1 | 2 | https://wandb.ai/pink-marker/sumotrack/runs/tldgmkm9 | `2.300731 → 1.789151` (`-0.511580`) | `2.899438 → 3.000318` (`+0.100880`) | `1.829834` | `0.056128` | `0.000227` | `0.618963` |
| 1 | 3 | https://wandb.ai/pink-marker/sumotrack/runs/lkptzi0s | `2.300731 → 1.802733` (`-0.497998`) | `2.899438 → 3.015043` (`+0.115605`) | `1.843152` | `0.054630` | `0.000221` | `0.475700` |
| 1 | 5 | https://wandb.ai/pink-marker/sumotrack/runs/24m68537 | `2.300731 → 1.779830` (`-0.520901`) | `2.899438 → 3.009695` (`+0.110257`) | `1.821850` | `0.056691` | `0.000229` | `0.431666` |

Interpretation:

- `2e-4` itself lands in the useful LR knee: better target than `1e-4`, slightly less source cost than `3e-4`, and much less source cost than `1e-3`.
- `pp=1, ns=1` is visibly too cheap: worse target and no source benefit.
- `pp=1, ns=2` is the interesting cheap candidate: target is close to default, source is slightly better, update scale is matched.
- `pp=1, ns=5` essentially matches the default `pp=2, ns=5` curve in this run, suggesting the second Aurora precondition pass is not obviously earning its keep at this scale.
- Step-time ordering in this single sequential sweep is noisy enough that it should not be overfit, but all useful `pp=1` candidates avoided a quality collapse. The next default-candidate comparison should focus on `pp=1, ns=2` versus `pp=1, ns=5` and current `pp=2, ns=5` under the same LR/source contract, not expand the grid.

## 2026-06-24: Faithful SYNTH LR knee sweep

Question: under the faithful `bs8 + activation_checkpointing` diagnostic lane, where is the useful SumoTrack LR/update-scale knee? SumoTrack already applies Muon-style scaling, so do not assume it needs Muon's larger raw LR priors.

Shared setup:

- Model: `LiquidAI/LFM2.5-350M-Base`.
- Source sensor: first parquet shard from `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` (`data/train-00000-of-00100.parquet`). This is a source-behavior sensor, not proof of broad retention.
- Harness defaults unless named: faithful SYNTH right-padded no-mask batches, CCE, broad no embeddings, rank 64, residual-facing side policy, stable `eigh` basis init, Aurora, `basis_refresh_interval=100`, `batch_size=8`, `seq_len=1024`, activation checkpointing, `warmup_steps=1`, `measure_steps=1000`, `eval_every=100`.
- Actual supervised tokens/update: `6399`; sequence tokens/update: `8192`; optimizer state bytes: `48,486,400`; peak allocated CUDA: `9,878,730,240`; peak reserved CUDA: `10,368,319,488`.

Command shape:

```bash
uv run python experiments/llm_synth_smoke.py \
  --batch-size 8 \
  --measure-steps 1000 \
  --eval-every 100 \
  --activation-checkpointing \
  --sumotrack-lr <lr> \
  --wandb-run sumotrack-350m-faithful-bs8-checkpoint-lr-<lr>-1k \
  --retention-hf-dataset HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
```

Runs:

| LR | wandb | target val loss | source val loss | last train loss | mean grad norm | mean update norm | update/param |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-5` | https://wandb.ai/pink-marker/sumotrack/runs/ppo5piwj | `2.300632 → 2.099905` (`-0.200727`) | `2.899438 → 2.911166` (`+0.011728`) | `2.179358` | `11.277402` | `0.015174` | `0.000061` |
| `3e-5` | https://wandb.ai/pink-marker/sumotrack/runs/wuvt5f7h | `2.300728 → 1.977568` (`-0.323160`) | `2.899438 → 2.943315` (`+0.043877`) | `2.041459` | `10.295789` | `0.023435` | `0.000095` |
| `1e-4` | https://wandb.ai/pink-marker/sumotrack/runs/58s0amuo | `2.300731 → 1.844126` (`-0.456605`) | `2.899406 → 2.999864` (`+0.100457`) | `1.891413` | `9.731564` | `0.039362` | `0.000159` |
| `3e-4` | https://wandb.ai/pink-marker/sumotrack/runs/2m9eq5r4 | `2.300731 → 1.749390` (`-0.551341`) | `2.899438 → 3.020600` (`+0.121162`) | `1.786439` | `9.280159` | `0.070321` | `0.000284` |
| `1e-3` | https://wandb.ai/pink-marker/sumotrack/runs/k9olzza0 | `2.300731 → 1.779225` (`-0.521506`) | `2.899438 → 3.167889` (`+0.268451`) | `1.787664` | `8.282332` | `0.165576` | `0.000669` |

Interpretation:

- `1e-5` is conservative and keeps the source sensor almost flat, but underfits the target at 1k steps.
- `1e-4` is the clean balance point in this sweep: strong target movement with moderate source rise.
- `3e-4` gives the best target validation loss and train loss, with a slightly higher source cost. This is the aggressive useful side of the knee.
- `1e-3` is past the knee: target is worse than `3e-4` while source degradation is much larger.
- The prior default LR `0.0025` / update norm around `0.35` was not proven intrinsically bad by its absolute norm, but this sweep shows the faithful lane gets a better target/source tradeoff at much lower update scales (`~0.04-0.07` update norm, `~1.6e-4-2.8e-4` update/param).

Next useful cut: refine around `1e-4` to `3e-4` depending on source tolerance, before adding tracking knobs or changing refresh cadence. Do not read the source shard as broad retention solved or failed; it is one accessible sensor.

## 2026-06-24: Faithful SYNTH 1k tracking comparison

Question: after making faithful SYNTH right-padded no-mask batching the default diagnostic lane, does target learning improve more from frequent basis refresh, larger token mass via checkpointed `batch_size=8`, or both?

Shared setup:

- Model: `LiquidAI/LFM2.5-350M-Base`.
- Source sensor: first parquet shard from `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled` (`data/train-00000-of-00100.parquet`).
- Harness defaults unless named: broad no embeddings, rank 64, residual-facing side policy, stable `eigh` basis init, Aurora, CCE, faithful SYNTH right-padded no-mask batches, `seq_len=1024`, `warmup_steps=1`, `measure_steps=1000`, `eval_every=100`, wandb log cadence 20.
- Historical note: this section predates the later telemetry cleanup. Current routine wandb sensors are only target/source validation loss, train loss, raw grad norm, update norm, update/param, and basis rotation chordal. Leverage and basis-capture graphs are not routine harness telemetry.
- `peak_cuda_bytes` is PyTorch peak allocated tensor memory, not driver-visible reserved VRAM. The harness now also records `peak_cuda_reserved_bytes` for future runs because checkpointed runs can still leave the card nearly full even when allocated peak is below total VRAM.

Runs:

| run | wandb | supervised tokens/update | target val loss | source val loss | last train loss | mean grad norm | mean update norm | update/param | step sec | peak allocated |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bs4`, refresh 20 | https://wandb.ai/pink-marker/sumotrack/runs/5ol91981 | 3340 | `2.226388 → 2.196474` | `2.875980 → 3.593040` | `2.172782` | `10.574589` | `0.354263` | `0.001426` | `0.223883` | `5,311,750,144` |
| `bs8`, checkpoint, refresh 100 | https://wandb.ai/pink-marker/sumotrack/runs/se1ggkj0 | 6399 | `2.300725 → 2.024565` | `2.899438 → 3.508711` | `2.015278` | `7.305237` | `0.354651` | `0.001428` | `0.448283` | `9,878,730,240` |
| `bs8`, checkpoint, refresh 20 | https://wandb.ai/pink-marker/sumotrack/runs/38z9bjca | 6399 | `2.300731 → 2.022594` | `2.899438 → 3.519228` | `2.003851` | `7.168745` | `0.355365` | `0.001431` | `0.464087` | `9,878,730,240` |

Interpretation:

- `bs4 + refresh20` did not rescue the faithful lane. It made only small target progress and the source sensor rose hard. Frequent basis refresh at low token mass is not the obvious main fix.
- `bs8 + activation checkpointing` is the useful improvement. More supervised tokens/update gave better train loss and much better target validation movement, with slightly less source damage than `bs4 + refresh20`.
- Adding frequent refresh on top of `bs8 + checkpointing` was essentially indifferent: target was almost identical, source was slightly worse, and step time rose. Leave `basis_refresh_interval=100` as the boring default until a more targeted tracking metric proves otherwise.
- Update magnitude was strikingly stable across runs: update norm around `0.354-0.355`, update/param around `0.00143`. The better `bs8` curve therefore looks like better signal going into the same update scale, not a quieter optimizer. That update scale is not obviously wrong in isolation and may be in the normal LoRA-training ballpark; the question is empirical target/source tradeoff. Because SumoTrack already applies Muon-style scaling, do not assume it needs Muon's larger raw LR priors. An earlier short faithful `2e-4` probe had update norm around `0.075` and source nearly flat, so a 1k `bs8 + checkpointing` LR/update-scale sweep across roughly `1e-5` to `1e-3` is the next clean cut, not because `0.35` is proven bad, but because the source curve needs a scale response.
- Checkpointing allowed `batch_size=8`, and the larger token mass clearly improved the signal, but it did not create enough headroom for another clean batch-size rung. PyTorch allocated peak was about `9.88 GB` and driver-visible reserved VRAM was observed near the card limit, so `batch_size=16` is not a free next move on this card. If `bs16`-equivalent signal is needed, the honest routes are grad accumulation or deeper memory cuts, not simply more checkpointing. Future memory reports need both allocated and reserved peaks.

## 2026-06-24: Stable side-Gram EIGH basis init

Question: is full SVD necessary for SumoTrack's one-sided basis initialization, or can side-Gram `eigh` be the default without cargo-culting SVD as a correctness rail?

Answer: full SVD is unnecessary for the current projector contract. SumoTrack only needs one dominant side subspace: right basis from top eigenvectors of `GᵀG`, left basis from top eigenvectors of `GGᵀ`. The implemented default now uses stable `eigh`: fp32 finite check, Frobenius normalization, symmetric Gram, and trace-scaled jitter retry only if the eigensolve backend fails. Random init remains an explicit ablation/stress path; `svd` is no longer an accepted basis-init method.

Empirical basis probe on a faithful-format LFM-350M first-step gradient, all 92 matrix tensors:

- `eigh` was faster than exact SVD on every measured shape: about 2.1x on `[512,1024]` qkv and about 5.4-7.3x on larger/square groups.
- Projection-energy deltas versus SVD were about `1e-3`; subspaces agreed within fp noise for the rank-64 basis contract.
- `eigh` bases had better measured orthonormality (`~1e-6`) than SVD bases in this probe (SVD up to `~6e-4`).
- Symmetrization and Frobenius normalization are mathematically neutral for the subspace and protect the eigensolve contract. Always-on jitter did not improve clean gradients, so it is a retry/degeneracy fallback, not sauce.
- MoleGrad's tiny projected `YᵀY` trick is a different optimizer design. The portable lesson here is stable Gram eigensolve hygiene, not Mbar/scout/erf damping.

Implementation smoke after switching defaults:

```bash
uv run python experiments/llm_synth_smoke.py \
  --batch-size 4 \
  --warmup-steps 0 \
  --measure-steps 1 \
  --batching synth_right_padded_no_mask \
  --retention-hf-dataset HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
```

- Model: `LiquidAI/LFM2.5-350M-Base`.
- `basis_init=eigh` through the harness default; no SVD path involved.
- Actual supervised tokens/update: `3340` (`4096` sequence tokens).
- Initial/final held-out SYNTH loss: `2.226388 → 2.207341`.
- Initial/final source loss: `2.875980 → 2.931102`; one-step movement only, not retention evidence.
- Mean grad norm: `17.132582`; update/param: `0.002921`; peak CUDA: `5.18 GB`.

Validation: `uv run python -m unittest discover -s tests` passed 50 tests.

## 2026-06-24: Source-mix completion sanity eval

Status after later input-format audit: **not interpretable as retention evidence**. Keep this entry as a harness/plumbing record for periodic target/source eval and wandb logging, not as evidence that source behavior was preserved.

Question: after the 1k Aurora target-only win, does current SumoTrack make target SYNTH progress while avoiding obvious base-model sandblasting on a broad text-completion source sensor?

Source sensor:

- Dataset: first parquet shard only from `HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled`, resolved as `data/train-00000-of-00100.parquet`.
- This is not the original LFM pretraining distribution and not a proof of broad retention. It is an accessible, diverse pretraining-mixture completion set, but the later audit found the training stream formatting was not faithful enough to interpret retention/source movement.
- Metrics were intentionally minimal: target SYNTH validation loss, source-mix validation loss, ordinary global grad norm at wandb log cadence. No KL, prompt suite, or extra projector diagnostics.

Harness changes for this run:

- `experiments/llm_synth_smoke.py` can load source/retention texts from the first parquet of a Hugging Face dataset via `--retention-hf-dataset`.
- Periodic target/source validation logging is controlled by `--eval-every`.
- Wandb logging is explicit via `--wandb-run`, default entity `pink-marker`; empty run name keeps local/dummy behavior.

350M run:

```bash
uv run python experiments/llm_synth_smoke.py \
  --batch-size 8 \
  --measure-steps 200 \
  --eval-every 25 \
  --log-grad-norm \
  --wandb-run sumotrack-350m-source-mix-200 \
  --retention-hf-dataset HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
```

- Model: `LiquidAI/LFM2.5-350M-Base`.
- Tokens/update: `8192` (`batch_size=8`, `seq_len=1024`, no grad accumulation).
- Defaults at the time: broad no embeddings, rank 64, residual-facing side policy, SVD basis init, Aurora, CCE, packed no-mask batches. Current code now defaults basis init to stable `eigh`.
- Wandb: `https://wandb.ai/pink-marker/sumotrack/runs/0spqj89v`.

| metric | initial | final | delta |
| --- | ---: | ---: | ---: |
| target SYNTH val loss | 3.983130 | 2.235465 | -1.747665 |
| source-mix val loss | 4.328842 | 3.409084 | -0.919758 |

Other signals: last train loss `2.139143`, mean logged grad norm `7.256254`, state bytes `48,486,400` (`47,972,352` matrix / `514,048` fallback), peak CUDA `9,722,962,432`, measured step seconds `0.486636`.

1.2B run:

- `batch_size=4` fit initial validation but OOMed on the first backward, so it was not an optimizer result.
- Successful rerun used `batch_size=2`, keeping the same model, source shard, loss path, and optimizer defaults.

```bash
uv run python experiments/llm_synth_smoke.py \
  --model LiquidAI/LFM2.5-1.2B-Base \
  --batch-size 2 \
  --measure-steps 200 \
  --eval-every 25 \
  --log-grad-norm \
  --wandb-run sumotrack-1p2b-source-mix-200-b2 \
  --retention-hf-dataset HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled
```

- Tokens/update: `2048` (`batch_size=2`, `seq_len=1024`, no grad accumulation).
- Wandb: `https://wandb.ai/pink-marker/sumotrack/runs/z35dj5dg`.

| metric | initial | final | delta |
| --- | ---: | ---: | ---: |
| target SYNTH val loss | 3.124986 | 1.787799 | -1.337187 |
| source-mix val loss | 3.642093 | 2.700852 | -0.941241 |

Other signals: last train loss `1.756662`, mean logged grad norm `10.424753`, state bytes `89,888,768` (`88,866,816` matrix / `1,021,952` fallback), peak CUDA `6,591,101,440`, measured step seconds `0.376855`.

Interpretation:

- Both models reduced loss on the current SYNTH stream and the source-mix completion loss also improved, but this should not be read as useful target progress or retention signal after the input-format audit. The safest interpretation is adaptation to a mismatched training stream.
- Later inspection of decoded packed blocks showed the current SYNTH stream can begin with raw query/reasoning/answer text and no visible BOS, EOS, or divider inside the first 1024-token block. The harness appends EOS after rows, but long rows can push that boundary outside the block. Basic document separators and conscious BOS/EOS/label-masking policy are missing.
- Therefore the honest claim is only: periodic target/source eval plumbing works and wandb logging works. The run does not answer whether SumoTrack preserves source behavior under faithful continued-pretraining formatting.
- The next algorithmic comparison should wait until formatting and gradient-scale probes are corrected. Relevant later comparisons remain SUMO/Muon-style and SubTrack/GaLore-style baselines, not full AdamW as the main opponent. AdamW destruction at medium/high LR is unsurprising and not the sharp question.

## 2026-06-13: LFM/SYNTH matrix-path smoke

Setup:

- Model: `LiquidAI/LFM2.5-1.2B-Base`, local cache, `HF_HUB_OFFLINE=1`.
- Data: local SYNTH parquet shards from `/home/djg/.cache/nanochat/base_data_synth`.
- Scope: `--param-scope matrices-no-embeddings`.
- Trainable: 92 non-embedding 2D tensors, 1,035,993,088 params.
- Excluded: 134,219,776 embedding/lm-head params and 61,440 tiny 3D conv params.
- Sequence/batch: `--seq-len 192 --batch-size 1`.
- Timing: `--warmup-steps 1 --measure-steps 3`; reported step time excludes warmup and first basis initialization.
- Optimizer path: `SumoTrack`, Grassmann refresh, no torch compile.

Random-init rank sweep:

| rank | state bytes | measured step seconds | peak CUDA bytes | initial val loss | final val loss |
| ---: | ----------: | --------------------: | --------------: | ---------------: | -------------: |
| 64 | 88,866,816 | 0.181071 | 4,864,393,216 | 1.943316 | 1.939324 |
| 128 | 177,733,632 | 0.321723 | 4,961,828,864 | 1.943316 | 1.942642 |
| 256 | 355,467,264 | 0.721046 | 5,152,342,528 | 1.943316 | 1.944172 |

Rank-64 SVD-init comparison:

| init | rank | state bytes | measured step seconds | peak CUDA bytes | initial val loss | final val loss |
| --- | ---: | ----------: | --------------------: | --------------: | ---------------: | -------------: |
| random | 64 | 88,866,816 | 0.181071 | 4,864,393,216 | 1.943316 | 1.939324 |
| svd | 64 | 88,866,816 | 0.165038 | 4,864,393,216 | 1.943316 | 1.944222 |

Notes:

- State size scales linearly with rank, as expected.
- All tested ranks fit comfortably on the local CUDA device in this short sequence-length setting.
- The three measured steps are too few to interpret loss movement. The random rank-64 final-val improvement is a smoke signal only.
- The SVD-init run used warmup to absorb cold basis initialization. Its measured step time is therefore not SVD cold-start timing.
- Rank 256 exposes the eager projected-space SVD tax: step time grows faster than the state-memory story alone would suggest.
- Next meaningful comparison needs more measured steps, AdamW/Muon baselines, and eventually HeavyBall-style orthogonalization instead of eager SVD inside projected space.

## 2026-06-13: SUMO orthogonalization ablation

Question: does SUMO/Muon-style orthogonalization of the projected first moment help, or is SubTrack-style projected momentum enough?

Setup:

- Model/data/scope: same LFM/SYNTH `matrices-no-embeddings` setup as above.
- Trainable: 92 non-embedding 2D tensors, 1,035,993,088 params.
- Rank: 64.
- Init: SVD basis initialization.
- Steps: 20 optimizer steps.
- Tokens/update: 768 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=4`).
- Validation texts: 16.
- No torch compile.

Results:

| optimizer | lr | state bytes | peak CUDA bytes | initial val | final val | mean train | step seconds |
| --- | ---: | ----------: | --------------: | ----------: | --------: | ---------: | -----------: |
| SumoTrack, SVD ortho | 0.001 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.800248 | 2.192415 | 1.981366 |
| SumoTrack, SVD ortho | 0.0025 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.701340 | 2.046949 | 1.976006 |
| SumoTrack, SVD ortho | 0.01 | 88,866,816 | 5,035,031,040 | 2.345853 | 2.625968 | 2.397669 | 1.980586 |
| Projected momentum, no ortho | 0.0025 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.975293 | 2.309294 | 1.865864 |
| Projected momentum, no ortho | 0.01 | 88,866,816 | 5,035,031,040 | 2.345853 | 1.815476 | 2.182461 | 1.866572 |
| torch AdamW quality anchor | 0.00002 | 4,143,972,720 | 9,097,523,712 | 2.345853 | 1.655455 | 1.962064 | 0.200987 |

Notes:

- The SUMO-style orthogonalized projected moment beat the no-orthogonalization projected-momentum ablation at the tested LRs.
- No-ortho improved when moved from `0.0025` to `0.01`, but still did not catch orthogonalized `0.0025`.
- Orthogonalized `0.01` overshot badly, so LR tuning matters; HeavyBall's `0.0025` default landed near the useful region for this setup.
- AdamW remains a quality anchor, not the target budget. Torch fused AdamW fit this matrix-only scope but used about 4.14 GB of optimizer state and 9.1 GB peak CUDA before embeddings/full fallback params. HeavyBall AdamW OOMed on the same scope during a one-step plumbing check.
- At the time of this ablation, local compiled HeavyBall paths were blocked by missing `Python.h`, so HeavyBall polar/NS was treated as a separate integration task rather than judged from uncompiled one-step smoke timing. Later notes below record the machine fix and compiled Newton-Schulz results.

## 2026-06-13: Norm-logged longer SUMO vs projected momentum

Question: after adding grad/param/update norm telemetry, does no-orthogonalization projected momentum catch SUMO when given a wider LR search and more tokens per optimizer update?

Setup:

- Same LFM/SYNTH `matrices-no-embeddings` scope as above: 92 2D tensors, 1,035,993,088 trainable params.
- Rank: 64.
- Init: SVD basis initialization.
- Steps: 1 warmup + 40 measured optimizer steps.
- Tokens/update: 1536 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=8`).
- Validation texts: 16.
- Norm logging: enabled. `update_norm` is the actual `SumoTrack` update vector norm after LR scaling, excluding decoupled weight decay; weight decay was `0` for these runs. AdamW-style generic update norms are intentionally not computed because cloning/streaming 1B params would distort the memory/perf story.

Representative command shape at the time:

```bash
HF_HUB_OFFLINE=1 uv run python experiments/llm_synth_smoke.py \
  --optimizers subspace --param-scope matrices-no-embeddings \
  --warmup-steps 1 --measure-steps 40 \
  --seq-len 192 --batch-size 1 --grad-accum-steps 8 --val-texts 16 \
  --rank 64 --basis-init svd --log-norms
```

Results:

| orthogonalization | lr | final val | mean train | mean grad norm | mean update norm | update/param | step seconds |
| --- | ---: | --------: | ---------: | -------------: | ---------------: | -----------: | -----------: |
| SVD SUMO | 0.0025 | 1.593341 | 1.861647 | 10.945538 | 0.191900 | 0.000491 | 0.559974 |
| SVD SUMO | 0.003 | 1.616218 | 1.863523 | 10.740152 | 0.230280 | 0.000590 | 0.555029 |
| SVD SUMO | 0.004 | 1.663988 | 1.893716 | 10.412884 | 0.307040 | 0.000786 | 0.553619 |
| no ortho | 0.01 | 1.699018 | 2.031259 | 11.752147 | 0.023460 | 0.000060 | 0.449519 |
| no ortho | 0.02 | 1.631776 | 1.948476 | 11.283048 | 0.036870 | 0.000094 | 0.437304 |
| no ortho | 0.04 | 1.612252 | 1.911619 | 10.968435 | 0.060896 | 0.000156 | 0.437575 |
| no ortho | 0.08 | 1.639280 | 1.943709 | 10.999331 | 0.108202 | 0.000277 | 0.437675 |

Notes:

- No-ortho was under-tuned in the earlier ablation; it improves substantially when moved above `0.01`, with the best tested point at `0.04`.
- SUMO still has the best tested validation loss (`1.593341` at LR `0.0025`) at the same 88.9 MB optimizer-state budget.
- The norm telemetry argues this is not just “take a bigger raw projected-momentum step.” No-ortho degrades by LR `0.08` while its mean update norm (`0.108`) remains well below SUMO LR `0.0025` (`0.192`). SUMO is changing the update geometry and scale together.
- Eager projected-space SVD costs about 25-30% step time in this norm-logged run (`~0.56s` vs `~0.44s`). This strengthens the case for integrating HeavyBall polar/Newton-Schulz as the next performance cut, but not at the cost of losing the observed SUMO geometry.

## 2026-06-13: Compiled HeavyBall Newton-Schulz orthogonalization

Question: can HeavyBall's compiled Newton-Schulz orthogonalization replace exact projected-space SVD without drifting from the SVD correctness baseline?

Machine fix:

- Torch/Inductor initially failed because `/usr/include/python3.14/Python.h` was missing.
- Fedora package fix: install `python3-devel` matching Python `3.14.5`.
- Downstream signal after install: HeavyBall compiled `inplace_orthogonal_` ran on CUDA for `newtonschulz`, `thinky_polar_express`, and `svd` modes.

Scale semantics:

- HeavyBall Muon's `orthogonalize_update` defaults to `scale_mode="scale"`, which flattens the full parameter to `[rows, cols]` and multiplies by `sqrt(max(1, rows / cols))`.
- Applying that same `"scale"` mode directly to a SUMO projected tensor can multiply by `sqrt(full_dim / rank)`, which is a different and often much larger scale.
- Added `orthogonalization_scale_mode="muon"`: orthogonalize in projected space, but scale according to the original full matrix shape. This preserves HeavyBall Muon aspect scaling without letting rank become an accidental LR multiplier.
- HeavyBall's `thinky_polar_express` path is only selected for square matrices; HeavyBall falls back to Newton-Schulz for rectangular tensors. SUMO projected updates are generally rectangular, so the practical HeavyBall hot path here is Newton-Schulz.

Short scale smoke:

| mode | scale mode | final val | mean update norm | note |
| --- | --- | ---: | ---: | --- |
| HeavyBall NS | projected `scale` | 2.898771 | 1.513056 | unstable at LR `0.0025`; projected-rank scaling too large |
| HeavyBall NS | `none` | 2.058131 | 0.095534 | descends, but drops HeavyBall Muon aspect scaling |
| HeavyBall NS | `muon` | 1.984581 | 0.143622 | descends and preserves full-matrix Muon scaling |

Rank-64 LFM/SYNTH comparison:

- Same matrix-only LFM/SYNTH setup as above.
- Rank: 64.
- Init: SVD basis initialization.
- Steps: 1 warmup + 40 measured optimizer steps.
- Tokens/update: 1536.
- LR: `0.0025`.
- Scale mode: `muon`.

| orthogonalization | mode | final val | mean train | mean update norm | update/param | step seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| exact SVD | baseline | 1.632313 | 1.860567 | 0.288551 | 0.000739 | 0.560159 |
| HeavyBall | newtonschulz | 1.629435 | 1.871363 | 0.287822 | 0.000737 | 0.479736 |

Notes:

- HeavyBall Newton-Schulz closely matches exact SVD quality and update scale in this short run, with lower measured step time.
- Exact SVD remains valuable as a correctness/debug baseline, but it should no longer be the main performance path.
- The previous best unscaled SVD run (`1.593341`) is not directly comparable to the Muon-scaled runs; scale and LR are coupled. Muon-scale mode likely needs its own LR sweep.

## 2026-06-14: Broad no-embedding LFM/SYNTH quality smoke

Question: does rank-64 broad-scope `SumoTrack` preserve the state-memory story and keep useful movement when real model topology adds fallback tensors, or was the positive matrix-only signal too narrow?

Setup:

- Model/data: `LiquidAI/LFM2.5-1.2B-Base` from local cache, local SYNTH shards, `HF_HUB_OFFLINE=1`.
- Scope: `--param-scope broad-no-embeddings`; embeddings/lm-head frozen.
- Trainable: 1,036,120,832 params across 146 tensors: 92 matrix tensors / 1,035,993,088 params and 54 fallback tensors / 127,744 params.
- Rank/init: rank 64, random basis initialization.
- Steps: 1 warmup + 20 measured optimizer steps.
- Tokens/update: 768 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=4`).
- Validation texts: 8.
- Norm logging: enabled for `SumoTrack` runs.

Results:

| optimizer | ortho | lr | state bytes | matrix/fallback state bytes | peak CUDA bytes | final val | mean train | mean update/param | step seconds |
| --- | --- | ---: | ----------: | ---: | --------------: | --------: | ---------: | ----------------: | -----------: |
| SumoTrack | HeavyBall NS + Muon scale | 0.0025 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 1.734250 | 2.072461 | 0.001181 | 0.323579 |
| SumoTrack | HeavyBall NS + Muon scale | 0.005 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 1.793099 | 2.168231 | 0.002193 | 0.326059 |
| SumoTrack | none | 0.04 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 7.085179 | 7.481242 | 0.010982 | 0.285293 |
| SumoTrack | none | 0.005 | 89,888,768 | 88,866,816 / 1,021,952 | 5,067,364,864 | 1.811141 | 2.219735 | 0.001681 | 0.284458 |
| torch AdamW | n/a | 0.00002 | 4,144,483,912 | n/a / 4,144,483,912 | 9,129,377,280 | 1.512150 | 1.927465 | n/a | 0.203278 |

Notes:

- Broad topology preserved the SumoTrack state-memory story: fallback state was ~1.0 MB and did not dominate the ~89.9 MB total.
- Orthogonalized SumoTrack stayed numerically stable at LR `0.0025` and `0.005`, but `0.005` looked too hot on this short run.
- The no-ortho matrix-only LR prior did not transfer. LR `0.04` was an obvious broad-topology failure, not a noisy loss comparison. LR `0.005` was stable but still worse than orthogonalized LR `0.0025`.
- AdamW remains materially better in this short quality anchor, but it used about 46x SumoTrack optimizer state and much higher peak CUDA. That is the intended product tension, not a contradiction.
- Next useful gate is a longer broad run centered on orthogonalized LR `0.0025`, with a lower no-ortho bracket and possibly SVD init as a quality-initialization comparison. Continuing should be falsified if the AdamW quality gap fails to narrow with more steps/tokens or if no-ortho catches up after a fair broad LR tune.

## 2026-06-14: Aurora and two-sided projected geometry check

Question: should SumoTrack's projected orthogonalization remain one-sided rectangular HeavyBall NS, use Aurora-style leverage-uniform rectangular polar, or move to a two-sided square-core projection?

Setup:

- Model/data: `LiquidAI/LFM2.5-1.2B-Base`, local SYNTH, `HF_HUB_OFFLINE=1`.
- Scope: `--param-scope broad-no-embeddings`.
- Rank/init: rank 64, fixed bases unless noted.
- Steps: 1 warmup + 10 measured optimizer steps.
- Tokens/update: 768 (`seq_len=192`, `batch_size=1`, `grad_accum_steps=4`).
- Validation texts: 8.
- Norm/leverage logging: enabled.

Results:

| projection | init | ortho | state bytes | final val | mean train | update/param | leverage CV | leverage min/max vs mean | step seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one-sided | random | HeavyBall NS | 89,888,768 | 1.735645 | 2.123392 | 0.001307 | 1.277685 | 0.109 / 36.609 | 0.347413 |
| one-sided | random | Aurora | 89,888,768 | 1.751235 | 2.145004 | 0.001313 | 0.025380 | 0.829 / 1.066 | 0.418227 |
| two-sided | random | HeavyBall NS | 90,642,432 | 1.801806 | 2.180583 | 0.001318 | 0.023224 | 0.917 / 1.043 | 0.341462 |
| two-sided | SVD | HeavyBall NS | 90,642,432 | 1.813795 | 2.162647 | 0.001303 | 0.019022 | 0.946 / 1.042 | 0.335513 |

Notes:

- One-sided HeavyBall NS has extreme large-axis leverage concentration in this diagnostic: average projected row-energy max was ~36x mean.
- Aurora fixes that mechanically, reducing leverage CV to ~0.025 and max/mean to ~1.07, with similar update/param. It slightly lost immediate target movement and cost ~20% step time over this short run.
- Two-sided square-core projection is stable and nearly state-neutral, but underperformed one-sided rectangular HeavyBall on immediate target movement. SVD init did not rescue it in this short fixed-basis comparison.
- Interpretation: one-sided rectangular updates remain the mainline for target adaptation. Aurora is still worth a retention-aware run because its value may be smoother/less forgetting-prone movement, not faster 10-step target loss. Two-sided should stay experimental until it shows retention or rank-budget advantage.

## 2026-06-17: Packed SDPA throughput sanity check

Question: can the LFM broad-no-embeddings harness run a realistic no-grad-accum throughput shape without accidentally disabling fast attention?

Findings:

- Padded `padding=max_length` throughput batches were the wrong test shape. The resulting non-null `attention_mask` made PyTorch SDPA reject flash attention (`Flash Attention does not support non-null attn_mask`) and made HF expand GQA key/value heads instead of using native SDPA GQA.
- Added packed full-sequence batches (`--pack-sequences`) that omit `attention_mask`; this preserves causal SDPA flash eligibility for throughput tests.
- `causal-conv1d` was temporarily built and verified during measurement, making LFM report its short-conv fast path available. It is not kept as a repo dependency because proper installation needs a coherent CUDA extension toolchain, not ad-hoc local build scaffolding. `flash-attn` source build was abandoned after memory pressure; do not source-build it casually in this environment.
- With `LiquidAI/LFM2.5-1.2B-Base`, broad-no-embeddings, rank 64, residual-facing side, CCE loss, packed `seq_len=1024`, and no grad accumulation, the local 12GB card fits `batch_size=4` but not `5`, `6`, `8`, or `16`.

Measured fitting point:

| batch × seq | tokens/step | checkpointing | compile | Aurora PP / NS | step seconds | tokens/sec | peak CUDA | state bytes |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 × 1024 | 4096 | off | off | 2 / 5 | 0.623791 | 6566 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | on | off | 2 / 5 | 0.625118 | 6552 | 10.75 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | on | 2 / 5 | 0.622031 | 6585 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | off | 1 / 5 | 0.607037 | 6748 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | off | 2 / 3 | 0.612473 | 6688 | 10.70 GB | 89.9 MB |
| 4 × 1024 | 4096 | off | off | 1 / 3 | 0.601221 | 6813 | 10.70 GB | 89.9 MB |

Notes:

- Checkpointing and `torch.compile` were noise at this shape.
- Reducing Aurora/polar cycles gave a modest optimizer-side speedup; `1` Aurora preconditioning pass and `3` NS steps was ~3.6% faster than the default timing point. This is throughput-only evidence, not quality evidence.
- The remaining memory wall is LFM layer temporary/activation memory under dense broad training, not optimizer state and not SDPA masking.

## 2026-06-18: Packed SDPA, CCE path, and Aurora-cycle smoke

Question: do packed batches keep SDPA on the flash path, does the CCE harness path run, and do Aurora cycle knobs buy real throughput on the local 12GB card?

Setup:

- Model/data: `LiquidAI/LFM2.5-1.2B-Base`, local SYNTH, `HF_HUB_OFFLINE=1`.
- Scope: `--param-scope broad-no-embeddings`, rank 64, residual-facing side.
- Shape: packed `seq_len=256`, `batch_size=1`, `--skip-validation`.
- Init: `basis_init=random` for perf-path smoke; exact SVD init OOMed at this shape on the local card.
- Loss paths: `hf` and `cce` (`--chunked-lm-loss-tokens 128`).

Results:

| loss / Aurora cycles | tokens/step | step seconds | tokens/sec | peak CUDA |
| --- | ---: | ---: | ---: | ---: |
| HF / default `2/5` | 256 | 0.220295 | 1162.079 | 4.53 GB |
| HF / reduced `1/3` | 256 | 0.086944 | 2944.421 | 4.53 GB |
| CCE / default `2/5` | 256 | 0.187094 | 1368.299 | 4.53 GB |

Notes:

- Packed batches omit `attention_mask`, so the harness keeps SDPA flash eligibility instead of falling back to padded-mask behavior.
- The CCE path works in this harness shape with random basis init.
- On this tiny-token smoke, lowering Aurora preconditioning / Newton-Schulz cycles was a big throughput win with no visible memory change. That is optimizer-side overhead amortization, not a quality claim.
- Historical note: exact SVD basis initialization was the cold-start tax at the time of this smoke and OOMed at the same packed shape on the local card. Current SumoTrack defaults to stable side-Gram `eigh`, so rerun throughput-path fit/perf claims before carrying this old random-init workaround forward.
- Follow-up packed smoke at `seq_len=512`, `batch_size=2` (1024 tokens/step, random init, HF loss) sharpened the cycle story: default `2/5` cycles OOMed on this card, while reduced `1/3` cycles fit and ran at `5701 tok/s` with `5.27 GB` peak CUDA. Cycle count is therefore not just a micro-optimization at this shape; it can be the difference between fit and no fit.
- These throughput numbers were taken without `causal-conv1d`; the environment fell back to LFM2's Torch `Conv1d` short-conv path because no Triton/`causal-conv1d` package was installed.

## 2026-06-18: Qwen3.5-2B-Base + FLA pull

Question: does pulling Qwen3.5-2B-Base plus `flash-linear-attention` lift the local 12GB throughput ceiling?

Setup:

- Model: `Qwen/Qwen3.5-2B-Base`, now fully cached locally.
- Extra dependency: `flash-linear-attention` installed successfully; `causal-conv1d` build was attempted but failed because the environment lacks `nvcc` and the package does not cleanly build under the current Python/CUDA toolchain.
- Fast-path status: transformers still reports Qwen3.5 fast path unavailable because one required library is missing, so the model falls back to Torch for the short-conv path.

Results:

| shape | outcome | tokens/sec | peak CUDA |
| --- | --- | ---: | ---: |
| `batch=1, seq=128` | fit | 661 | 6.66 GB |
| `batch=2, seq=1024` | OOM in MLP | n/a | ~10.30 GB |
| `batch=4, seq=1024` | OOM | n/a | ~10.30 GB |

Notes:

- Qwen3.5 does load and train offline, but without `causal-conv1d` it does not unlock the intended fast path.
- The model switch alone does not buy the bs4×1k target on this 12GB card; it still blows up around the MLP/activation side before the optimizer matters.
- Net: `flash-linear-attention` is useful, but it is not the missing piece by itself.

## 2026-06-18: causal-conv1d build blocker

Question: can `causal-conv1d` be made uv-native on this Fedora box without hand-built hacks?

Findings:

- The project floor was dropped to Python `3.13`, and `uv sync --python 3.13` works.
- `uv add causal-conv1d --no-build-isolation-package causal-conv1d --no-binary-package causal-conv1d` still fails when the build uses the local CUDA 13.3 toolkit.
- The actual blocker is the host compiler: Fedora ships GCC `16.1`, while CUDA 13.3 rejects compilers newer than `15`.

Required next step:

- Install GCC 15 (`gcc15`, `gcc15-c++`) and point `CC`/`CXX` at it for the build shell, then retry the `uv add` command under `--python 3.13`.

## 2026-06-18: causal-conv1d built and Qwen fast path enabled

Question: did the Fedora box now satisfy the real build contract for `causal-conv1d`?

Findings:

- `.python-version` now pins the repo to Python `3.13`, so `uv run` / `uv sync` stop drifting back to system 3.14.
- Building `causal-conv1d` succeeded when the shell used `/usr/bin/gcc-15` and `/usr/bin/g++-15` explicitly together with CUDA 13.3 and Python 3.13.
- Verification signal: `transformers.utils.import_utils.is_causal_conv1d_available()` returns `True`, and `transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn` / `causal_conv1d_update` are both bound.

Meaning:

- Qwen3.5’s short-conv fast path is now live in this repo environment; the remaining work is throughput measurement, not kernel archaeology.

## 2026-06-18: Qwen3.5 throughput try

Question: does Qwen3.5-2B-Base with `causal-conv1d` and FLA actually clear the earlier bs4×1k ceiling?

Result:

- No. `batch=4, seq=1024` still OOMed during the Qwen3.5 linear-attention / MLP path.
- `batch=1, seq=1024` also OOMed, and activation checkpointing did not rescue it.

Meaning:

- Enabling the fast short-conv path is necessary plumbing, but it does not by itself make Qwen3.5 the 12GB bs4×1k vehicle. The remaining wall is still model activations / temporary tensors, not optimizer state.

## 2026-06-18: Qwen3.5 memory math check

Follow-up smoke runs with `basis_init=random` separated the loss path from the basis-init path:

- `batch=1, seq=1024, loss_impl=cce` fit at `7.28 GB` peak CUDA.
- `batch=1, seq=1024, loss_impl=hf` fit at `10.04 GB` peak CUDA.
- `batch=4, seq=1024, loss_impl=cce` OOMed in `causal_conv1d_fn` while allocating a `48.00 MiB` buffer (`empty_like(x)`), with only `31.88 MiB` free on the GPU.

Math:

- Qwen3.5-2B-Base has `2,274,069,824` parameters total.
- The text-side language model is `1,881,825,088` params; the non-embedding text-side trainable set is `1,373,265,728` params.
- One packed `bf16` activation at `batch=4, seq=1024, hidden=6144` is `4*1024*6144*2 = 50,331,648` bytes, i.e. the exact `48 MiB` buffer that failed.

Takeaway:

- The OOM is not a mystery leak. `batch=4, seq=1024` is already at the edge of the 11.59 GiB card once weights + grads + optimizer state + per-layer temporaries are resident, and the causal-conv output buffer is the shove that finally tips it over.

## 2026-06-21: Clean LFM 350M packed CCE throughput ceiling

Question: after cleaning the harness to one faithful route — EOS-packed no-mask batches plus CCE loss, no model fallback, no HF/chunked loss switch — what token step fits on the local 11.59 GiB 4070 SUPER with the default `LiquidAI/LFM2.5-350M-Base`?

Setup:

- Model/data: `LiquidAI/LFM2.5-350M-Base`, local SYNTH shards.
- Scope: `--param-scope broad-no-embeddings`, rank 64, residual-facing side.
- Batching/loss: EOS-packed fixed-length blocks, no `attention_mask`, CCE loss.
- Init: `--basis-init random`, because this was a throughput/fit check rather than an SVD cold-start quality claim.
- Shape: `seq_len=1024`, no grad accumulation, `--skip-validation`, one warmup + three measured steps.

Results:

| batch × seq | tokens/step | checkpointing | outcome | tokens/sec | step seconds | peak CUDA | state bytes |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| 4 × 1024 | 4096 | off | fit | 18,873 | 0.217 | 5.24 GB | 48.5 MB |
| 8 × 1024 | 8192 | off | fit | 18,859 | 0.434 | 9.70 GB | 48.5 MB |
| 9 × 1024 | 9216 | off | OOM in forward MLP | n/a | n/a | ~10.2 GB allocated at failure | n/a |
| 10 × 1024 | 10240 | off | OOM in forward MLP | n/a | n/a | ~10.2 GB allocated at failure | n/a |
| 16 × 1024 | 16384 | off/on | OOM in forward MLP | n/a | n/a | ~10.2 GB allocated at failure | n/a |

Notes:

- The clean no-mask CCE route works and reports `attn_implementation=sdpa`.
- The practical no-grad-accum ceiling at `seq_len=1024` is `batch_size=8` / `8192` tokens per optimizer step on the current card state.
- OOMs happen during the model forward MLP path before optimizer state matters. Activation checkpointing did not rescue `batch_size=16`, consistent with a forward temporary/activation wall rather than a backward-only saved-activation wall.
- SumoTrack state is ~48.5 MB total at this 350M broad-no-embeddings topology: ~48.0 MB matrix state and ~0.5 MB fallback state.
