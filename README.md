# Optimizers Lab

This repo is currently the **UsuiTrack** lab: a place to design and test a memory-efficient optimizer for high-capacity continued pretraining on consumer GPUs.

UsuiTrack's product target is distribution adaptation under memory pressure: move a pretrained model across a real data shift without full AdamW state, slow gradient accumulation, or adapter-only capacity limits.

## Documentation

- `docs/SPEC.md` — canonical mathematical specification of the current update.
- `docs/PLAN.md` — live product direction, evidence, research map, and release gates.
- `docs/SUBSPACE_TRACKING.md` — live geometry ledger and open questions.
- `docs/QUANTIZATION.md` — scoped rank-state quantization sidequest and evidence gates.
- `docs/compile-memory-research.md` — external research on compiled gradient release and regional compilation.
- `docs/LEGEND.md` — the design's anthropomorphic story; evocative, not normative.
- `AGENTS.md` — operating contract for agents working in this repo.
- `docs/archive/` — complete superseded run and design histories. Use for provenance,
  never to infer current defaults.
- `whiteflow/README.md` and `muonfactor/README.md` — older optimizer sketches.

## Current mainline

The active optimizer class is `UsuiTrack`.

Current quality/diagnostic harness defaults are broad no-embedding training,
uniform rank `128`, stable side-Gram `eigh` basis initialization, residual-facing
one-sided projection, faithful SYNTH right-padded no-mask batches, CCE loss,
one-state basis tracking using Oja covariance tangents, and fixed Aurora/Muon
update geometry.

Read in this order: `docs/SPEC.md`, `docs/PLAN.md`, then the specialized live
ledger. Enter `docs/archive/` only when the provenance of a decision matters.
