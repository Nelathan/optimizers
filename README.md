# Optimizers Lab

This repo is currently the **SumoTrack** lab: a place to design and test a memory-efficient optimizer for high-capacity continued pretraining on consumer GPUs.

SumoTrack's product target is distribution adaptation under memory pressure: move a pretrained model across a real data shift without full AdamW state, slow gradient accumulation, or adapter-only capacity limits.

## Documentation

- `SPEC.md` — canonical mathematical specification of the current update.
- `PLAN.md` — live product direction, evidence, research map, and release gates.
- `SUBSPACE_TRACKING.md` — live geometry ledger and open questions.
- `LEGEND.md` — the design's anthropomorphic story; evocative, not normative.
- `AGENTS.md` — operating contract for agents working in this repo.
- `archive/` — complete superseded run and design histories. Use for provenance,
  never to infer current defaults.
- `whiteflow/README.md` and `muonfactor/README.md` — older optimizer sketches.

## Current mainline

The active optimizer class is `SumoTrack`.

Current quality/diagnostic harness defaults are broad no-embedding training,
uniform rank `64`, stable side-Gram `eigh` basis initialization, residual-facing
one-sided projection, faithful SYNTH right-padded no-mask batches, CCE loss,
position-controlled Grassmann burst refresh, and Aurora/Muon update geometry.

Read in this order: `SPEC.md`, `PLAN.md`, then the specialized live ledger. Enter
`archive/` only when the provenance of a decision matters.
