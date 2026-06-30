# Optimizers Lab

This repo is currently the **SumoTrack** lab: a place to design and test a memory-efficient optimizer for high-capacity continued pretraining on consumer GPUs.

SumoTrack's product target is distribution adaptation under memory pressure: move a pretrained model across a real data shift without full AdamW state, slow gradient accumulation, or adapter-only capacity limits.

## Documentation

- `PLAN.md` — current direction, active invariants, and next cuts.
- `INSIGHTS.md` — durable empirical facts and local terrain.
- `RESULTS.md` — brief experiment records: question, setup, run links, metrics, interpretation.
- `AGENTS.md` — operating contract for agents working in this repo.
- `whiteflow/README.md` and `muonfactor/README.md` — historical optimizer sketches, not active specs.

## Current mainline

The active optimizer class is `SumoTrack`.

Current quality/diagnostic harness defaults are broad no-embedding training, uniform rank `64`, stable side-Gram `eigh` basis initialization, residual-facing one-sided projection, faithful SYNTH right-padded no-mask batches, CCE loss, Transformers `sdpa` attention, Grassmann burst basis refresh, and Aurora/Muon projected update geometry.

For the current working contract and stop conditions, read `PLAN.md` first.
