# StellaStiefel / MuonFactor

Status: historical design sketch.

This document is preserved for reference. It is not the active optimizer direction, not a current benchmark contract, and not a specification for `UsuiTrack`.

The active optimizer line is **UsuiTrack**. Current direction and durable facts live in `../PLAN.md`; superseded experiment records live in `../archive/RESULTS.md`.

## Historical idea

StellaStiefel explored a Muon + AdaFactor hybrid for memory-efficient full finetuning at batch size 1. The sketch combined:

- AdaFactor-style factored second-moment state for 2D matrices,
- Muon/Newton-Schulz orthogonalized update geometry,
- AdamW fallback for embeddings and 1D tensors,
- token-half-life scheduling for second-moment responsiveness,
- bf16 parameter updates with stochastic rounding.

The intended regime was TRL/Transformers-compatible full finetuning on a single 24 GB-class GPU with packed long contexts and no gradient accumulation.

## Why it is not the active spec

- It uses factored second-moment state as a central design feature. UsuiTrack's active matrix path currently avoids full-size first moments and full-size second moments, and instead keeps projected first moments.
- It is framed around batch-size-1 full finetuning, while UsuiTrack's current product question is high-capacity distribution adaptation with as many tokens per step as practical.
- It predates the current UsuiTrack harness contract: faithful SYNTH right-padded diagnostics, CCE loss, residual-facing one-sided projection, stable `eigh` basis init, Grassmann tracking, and Aurora projected direction geometry.
- Its hyperparameter grids and VRAM estimates are not current local UsuiTrack evidence.

## If revisiting

The useful surviving questions are narrow:

- whether factored second-moment information can improve UsuiTrack's projected direction without breaking the state budget,
- whether stochastic rounding helps bf16 projected-state or update writeback,
- whether token-half-life scheduling is useful for fallback state.

Treat any such work as a new UsuiTrack experiment with explicit invariants, not as resurrection of this whole optimizer design.
