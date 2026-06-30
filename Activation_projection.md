# Technical Architecture Design: Low-Rank, Subspace-Isolated Optimization for Memory-Constrained Deep Learning

This document outlines the engineering specification and architectural design for a high-efficiency deep learning training pipeline. The goal is to perform full parameter supervised fine-tuning (SFT) of deep Transformer models (64+ layers) on hardware constrained to a **single GPU**, without relying on low-rank parameter adapters (LoRA) that permanently constrain forward model capacity.

---

## 1. Motivation & Context

Full-parameter fine-tuning of large language models yields superior downstream capability and cross-dialect alignment compared to parameter-efficient methods like LoRA. However, scaling full fine-tuning on a single workstation hits a hard physical wall: **VRAM exhaustion (Out of Memory)**.

The standard VRAM profile of an optimization step is comprised of three primary vectors:

1. **Model Weights:** Persistent memory proportional to parameter count.
2. **Optimizer States:** Spatially dominant allocations (e.g., Adam tracking two moving averages per parameter in FP32).
3. **Activation Memory:** Spatially dominant allocations proportional to batch size ($B$), sequence length ($T$), and depth ($L$). For a $[B \cdot T, 16\text{k}]$ intermediate MLP representation, storing these activations across 64+ layers completely nukes the VRAM ceiling.

Standard engineering mitigations present critical trade-offs:

* **Gradient Checkpointing:** Erases the persistent activation memory across layers but introduces a transient $[B \cdot T, 16\text{k}]$ memory spike during the layer-wise backward pass, which still dictates the maximum peak VRAM bottleneck.
* **Low-Rank Gradients (e.g., GaLoRE):** Compresses optimizer states via low-rank subspaces ($Q$) calculated by Singular Value Decomposition (SVD). However, online SVD updates introduce heavy computational overhead, and tracking these across all modules simultaneously scales poorly.

Furthermore, empirical testing reveals a fundamental geometric asymmetry in the MLP block: **compressing the wide internal feature space ($M \approx 16\text{k}$) introduces severe representation degradation, whereas compressing along the backbone-facing hidden dimension ($H \approx 4\text{k}$) preserves loss dynamics.**

---

## 2. Goals & Scope

### Strategic Goals

* **VRAM Minimization:** Compress both the long-term activation tracking and optimizer states so that full-parameter updates can scale to large sequence contexts on a single consumer GPU.
* **Loss Preservation:** Avoid mapping the forward pass or the internal MLP hidden dimensions into destructive bottlenecks, preserving the model's full-rank expressive power.
* **Low Computational Overhead:** Eradicate expensive online SVD calls for tracking subspace drift.

### Architectural Scope

This design specifies the implementation details for the **Feed-Forward Network (MLP)** blocks of the network—specifically targeting the Up-Projection ($W_{\text{up}}$) and Down-Projection ($W_{\text{down}}$) matrices—as they represent the largest memory sinks outside of attention blocks.

---

## 3. Mathematical Foundations

Let $X \in \mathbb{R}^{(B \cdot T) \times H}$ be the layer input activations, where $H$ is the hidden dimension ($4\text{k}$). Let $M$ be the intermediate MLP dimension ($16\text{k}$). Let $R$ be the low optimization rank ($128 \ll H < M$).

### 3.1 Up-Projection Layer ($W_{\text{up}} \in \mathbb{R}^{H \times M}$)

The full-rank forward transformation is defined as:


$$Y = X \cdot W_{\text{up}} \quad \in \mathbb{R}^{(B \cdot T) \times M}$$

The true weight gradient matrix is defined by the outer product:


$$G_{\text{up}} = X^T \cdot \nabla_Y L \quad \in \mathbb{R}^{H \times M}$$

To enforce backbone-aligned compression, we project the gradient from the left using a subspace matrix $Q_{\text{up}} \in \mathbb{R}^{H \times R}$:


$$\hat{G}_{\text{up\_small}} = Q_{\text{up}}^T \cdot G_{\text{up}} = (X \cdot Q_{\text{up}})^T \cdot \nabla_Y L = Q_X^T \cdot \nabla_Y L \quad \in \mathbb{R}^{R \times M}$$

Where $Q_X = X \cdot Q_{\text{up}} \in \mathbb{R}^{(B \cdot T) \times R}$ acts as the crunched forward activation tracker.

### 3.2 Down-Projection Layer ($W_{\text{down}} \in \mathbb{R}^{M \times H}$)

The input to the down-projection is the unconstrained intermediate activation $Y \in \mathbb{R}^{(B \cdot T) \times M}$. The forward operation is:


$$Z = Y \cdot W_{\text{down}} \quad \in \mathbb{R}^{(B \cdot T) \times H}$$

The true weight gradient matrix is:


$$G_{\text{down}} = Y^T \cdot \nabla_Z L \quad \in \mathbb{R}^{M \times H}$$

Because empirical observations warn against compressing the $M$ dimension, we maintain backbone-alignment by projecting $G_{\text{down}}$ from the *right-hand side* using a separate subspace matrix $Q_{\text{down}} \in \mathbb{R}^{H \times R}$:


$$\hat{G}_{\text{down\_small}} = G_{\text{down}} \cdot Q_{\text{down}} = Y^T \cdot (\nabla_Z L \cdot Q_{\text{down}}) \quad \in \mathbb{R}^{M \times R}$$

---

## 4. System Architecture Blueprint

The system coordinates three structural pillars to manage memory execution: **Subspace Tracking**, **Module Round-Robin Execution**, and a **Custom Autograd Graph Pass**.

### 4.1 Subspace Tracking via SubTrack (Grassmannian Manifold)

To bypass the computational penalty of repeated Singular Value Decompositions, subspace drift is managed via **SubTrack**. The matrices $Q_{\text{up}}$ and $Q_{\text{down}}$ are initialized randomly as orthogonal bases and updated continuously using cheap, rank-1 geodesic steps directly along a Grassmannian manifold.

### 4.2 Module-Level Round-Robin Coordination

Instead of materializing optimizer states and tracking subspace gradients for all layers simultaneously, the training loop coordinates execution on a rotating timeline.

During any given step $N$, only a designated subset of modules are "Active." Inactive modules still pass gradients backward linearly to ensure full deep network connectivity but freeze their moving optimizer moments and skip tracking updates for that step.

```
Step N:
├── Module 01 (MLP Up Layer 1)   ──► Active: Compute SubTrack + Update Moments
├── Module 02 (MLP Down Layer 1) ──► Inactive: Pass Gradients Backwards, Skip Update
└── Module 03 (MLP Up Layer 2)   ──► Inactive: Pass Gradients Backwards, Skip Update

Step N+1:
├── Module 01 (MLP Up Layer 1)   ──► Inactive: Pass Gradients Backwards, Skip Update
├── Module 02 (MLP Down Layer 1) ──► Active: Compute SubTrack + Update Moments
└── ...

```

### 4.3 Fused Autograd and Activation Discard Pipeline

To achieve the optimization, the standard PyTorch execution graph must be broken. We deploy a custom `torch.autograd.Function` coupled with a localized activation checkpoint wrapper over the MLP block.

1. **Forward Pass Execution:**
* $Y = X \cdot W_{\text{up}}$ is evaluated fully, preserving model capacity.
* $Q_X = X \cdot Q_{\text{up}}$ is computed and cached.
* **Graph Severance:** The massive input matrix $X$ is explicitly excluded from `ctx.save_for_backward()`, forcing its immediate eviction from GPU VRAM. Only $Q_X$, $W_{\text{up}}$, $W_{\text{down}}$, and the $Q$ operators are retained.


2. **Backward Pass Execution & Sequential Memory Asymmetry:**
* **Stage 1 (Down-Projection):** The backward pass hits $W_{\text{down}}$ first. The framework triggers a localized activation checkpoint to transiently materialize $Y \in \mathbb{R}^{(B \cdot T) \times 16\text{k}}$.
* The low-rank gradient is calculated: $\hat{G}_{\text{down\_small}} = Y^T \cdot (\nabla_Z L \cdot Q_{\text{down}})$.
* **Immediate Eviction:** The massive intermediate $Y$ tensor is immediately freed from VRAM before moving deeper.
* **Stage 2 (Up-Projection):** The backward pass enters $W_{\text{up}}$. The system accesses the small saved $Q_X \in \mathbb{R}^{(B \cdot T) \times R}$ tensor to directly calculate $\hat{G}_{\text{up\_small}} = Q_X^T \cdot \nabla_Y L$. No $16\text{k}$ dimensions are re-materialized during this phase.



---

## 5. Implementation Specification

The implementation handles gradient calculation via custom backward pathways, while side-loading the low-rank updates directly into the standard `.grad` attribute slots to maintain compatibility with standard downstream deep learning optimizers.

```python
import torch

class BackboneAlignedLowRankMLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W_up, W_down, Q_up, Q_down):
        """
        Executes a full-rank forward pass while aggressively purging 
        raw activation matrices from the persistent memory graph.
        """
        # Full-rank matrix operations
        Y = X @ W_up
        Z = Y @ W_down
        
        # Capture the crunched representation for W_up backward pass
        Q_X = X @ Q_up  # Shape: [B*T, R]
        
        # Save only low-rank representations and weight dimensions
        # Full dimension X is dropped from memory entirely here
        ctx.save_for_backward(Q_X, W_up, W_down, Q_up, Q_down)
        
        return Z

    @staticmethod
    def backward(ctx, grad_Z):
        """
        Calculates backwards gradients layer-by-layer, utilizing sequential 
        asymmetry to keep peak activation memory bounded.
        """
        Q_X, W_up, W_down, Q_up, Q_down = ctx.saved_tensors
        
        #---------------------------------------------------------------------
        # 1. DOWN-PROJECTION LAYER (W_down) GRADIENT
        # Note: In production execution, full Y is provided here via a 
        # localized, single-layer activation checkpoint recomputation wrapper.
        #---------------------------------------------------------------------
        # Map back to previous activation layer dimension
        grad_Y = grad_Z @ W_down.t()
        
        # Compute right-side projected gradient: Shape [M, R]
        grad_Z_compressed = grad_Z @ Q_down
        
        # Placeholder for checkpoint-recomputed intermediate activation Y
        # grad_W_down_small = Y.t() @ grad_Z_compressed 
        
        #---------------------------------------------------------------------
        # 2. UP-PROJECTION LAYER (W_up) GRADIENT
        #---------------------------------------------------------------------
        # Propagate structural gradient to the previous layer block
        grad_X = grad_Y @ W_up.t()
        
        # Compute left-side projected gradient using crunched Q_X tracker
        grad_W_up_small = Q_X.t() @ grad_Y  # Shape: [R, M]
        
        # Project up to full dimensions and inject into standard parameter grad slots
        # to ensure out-of-the-box compatibility with Cautious/Adam optimizers.
        if W_up.grad is None:
            W_up.grad = torch.zeros_like(W_up)
        
        # Fused in-place GEMM to avoid intermediate allocation memory spikes
        torch.mm(Q_up, grad_W_up_small, out=W_up.grad)
        
        # Return structural gradients matching forward inputs
        return grad_X, None, None, None, None

```

---

## 6. Review Hotspots & Critical Risks

The following elements require deep scrutiny during technical review:

1. **The $W_{\text{down}}$ Activation Spike:** Although checkpointing isolates the $Y \in \mathbb{R}^{(B \cdot T) \times 16\text{k}}$ memory footprint to a single layer, for massive sequence contexts ($T \ge 64\text{k}$), this isolated allocation may still exceed single-GPU capacity. If verified under testing, intra-layer micro-batch tensor slicing must be implemented within the custom backward loop.
2. **SubTrack Geodesic Overhead:** The computational cost of updating $Q$ via Grassmannian steps must be profiled against standard SVD to ensure module round-robin pacing yields a net training throughput speedup.
3. **Autograd Interoperability:** Overriding standard gradient routing by injecting manually projected gradients into `W.grad` while returning `None` from the autograd method may disrupt mixed-precision gradient scaling frameworks (e.g., `torch.cuda.amp.GradScaler`). Robust verification in $bf16$ optimization pipelines is mandatory.
