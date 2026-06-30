# WhiteFlow

Status: historical design sketch.

This document is preserved as optimizer-idea archaeology. It is not the active direction of this repo, and it should not be read as a current implementation contract, benchmark plan, or product thesis.

The active optimizer line is **SumoTrack**. Current direction and durable facts live in `../PLAN.md`, and experiment records live in `../RESULTS.md`.

## Historical idea

WhiteFlow explored a whitening-inspired optimizer for memory-constrained continued pretraining. The sketch combined:

- SPlus-style whitening/preconditioning intuition for fast early convergence,
- Apollo-style low-rank/channel-wise adaptation to avoid full second-moment matrices,
- DolphinFlow-style gradient orthogonalization for stable directions,
- bf16 optimizer state to reduce memory pressure.

The motivating hypothesis was that orthogonalized gradients plus cheap adaptive scaling could avoid Apollo's slow projection-settling phase while using much less memory than SPlus.

## Why it is not the active spec

- It stores full-size momentum and second-moment buffers in the sketch, which conflicts with SumoTrack's active matrix-state invariant: no full-size first or second moments on the main matrix path.
- Its memory claims were prospective design estimates, not validated local benchmark results.
- Its target hardware/model framing predates the current SumoTrack product target and harness defaults.
- The current project direction has moved toward one-sided projected moments, Grassmann basis tracking, and Aurora/Muon projected update geometry.

## If revisiting

Only reopen this line if there is a concrete question that SumoTrack does not answer, such as a validated need for adaptive channel scaling in the projected path. Any revival should start as a new design note that compares directly against the current SumoTrack invariants rather than mutating this historical sketch into a second competing truth.
