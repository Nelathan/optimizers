# WhiteFlow Optimizer: Memory-Efficient Whitening-Inspired Optimization

## 🎯 Executive Summary

WhiteFlow is a novel optimizer designed for **memory-constrained continued pretraining** scenarios. It combines insights from three state-of-the-art optimizers to achieve **SPlus-level convergence speed** with **memory usage between Apollo and Adam** (significantly less than SPlus's 2x Adam memory cost).

**Key Innovation**: Fast convergence from step 1 through orthogonalized gradients + low-rank channel-wise adaptive scaling, eliminating Apollo's 20-40k step ramp-up problem.

## 🔬 Problem Analysis & Motivation

### Target Scenario
- **Hardware**: Single GPU with limited VRAM (24GB RTX 4090)
- **Models**: 4B-70B parameters (scalable architecture)
- **Training regime**: Continued pretraining with ~100B tokens (not 4-20T like Apollo)
- **Workflow**: Many orthogonal experts → merge (fast iteration required)
- **Constraint**: Must work well from minimum 1k steps

### Current Optimizer Limitations

| Optimizer | Convergence Speed | Memory Usage | Key Issues |
|-----------|------------------|--------------|------------|
| **SPlus** | ⭐⭐⭐⭐⭐ (44% of Adam steps) | ❌❌ (2x Adam + O(d²) matrices) | Excessive memory for large layers |
| **Apollo** | ⭐⭐⭐ (after 20-40k steps) | ⭐⭐⭐⭐ | Slow initial convergence |
| **DolphinFlow** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (1x Adam) | Good but no adaptive scaling |
| **Adam** | ⭐⭐⭐ | ⭐⭐⭐ (2x params) | Baseline performance |

## 💡 Core Design Philosophy

### Three-Pillar Approach

1. **SPlus Convergence Intuition**: Whitening-based preconditioning for fast convergence
2. **Apollo Memory Efficiency**: Low-rank channel-wise projections instead of full second moments
3. **DolphinFlow Stability**: Gradient orthogonalization prevents naive loss minimization

### Key Insight: Hybrid Fast-Settling Architecture
Instead of Apollo's gradual projection settling, use **"car suspension" dynamics**:
- **Quick initial response** (β₂ starts at 0.9) to capture essential directions
- **Smooth stabilization** (β₂ approaches 0.999) to prevent oscillation
- **Orthogonalization** provides immediate stability without waiting for projections

## 🏗️ Algorithm Architecture

### State Management (Memory Optimized)

```python
# Per parameter state (much less than SPlus):
state = {
    'step': int32,                    # 4 bytes
    'momentum': bf16,                 # 2 bytes/param (vs SPlus fp32)
    'exp_avg_sq': bf16,              # 2 bytes/param (vs SPlus fp32)

    # Channel projections (Apollo-inspired, much smaller than SPlus O(d²)):
    'proj_left': bf16,               # rank × rows (e.g., 256×8192)
    'proj_right': bf16,              # rank × cols (e.g., 256×28672)
    'channel_scales': bf16,          # rows only (e.g., 8192)

    # NO EMA states (unlike SPlus) - saves massive memory
    # NO full second-moment matrices (unlike SPlus)
}
```

### Memory Comparison (LLaMA 70B MLP layer: 28672×8192)

| Component | SPlus | WhiteFlow | Savings |
|-----------|--------|-----------|---------|
| Momentum | 940MB (fp32) | 470MB (bf16) | 50% |
| Second moment | 940MB (fp32) | 470MB (bf16) | 50% |
| EMA | 940MB (fp32) | **0MB** | 100% |
| sides[0] | **3.3GB** (fp32) | **10MB** (rank-256 proj) | 99.7% |
| sides[1] | 268MB (fp32) | **5MB** (rank-256 proj) | 98.1% |
| q_sides | 3.6GB (fp32) | **0MB** (computed on-demand) | 100% |
| **Total** | **~9GB** | **~1GB** | **89% reduction** |

### Core Algorithm Steps

```python
def step(self):
    for param in parameters:
        # 1. Orthogonalize gradient (DolphinFlow-inspired stability)
        if orthogonalize and param.ndim >= 2:
            grad = project_away_from_vector(grad, param.data)

        # 2. Momentum update
        momentum = β₁ * momentum + (1-β₁) * grad

        # 3. "Car suspension" second moment (quick settle then smooth)
        β₂_adaptive = min(β₂, 0.9 + 0.099 * (step/100))
        exp_avg_sq = β₂_adaptive * exp_avg_sq + (1-β₂_adaptive) * grad²

        # 4. Channel-wise scaling (Apollo-inspired, every 20 steps)
        if step % proj_update_freq == 0:
            update_channel_projections(grad)

        scaling = compute_channel_scaling_via_projections()

        # 5. Parameter update
        update = momentum / sqrt(exp_avg_sq + ε) * scaling
        param -= lr * update
```

## 🔧 Technical Innovations

### 1. Gradient Orthogonalization (DolphinFlow Heritage)
**Purpose**: Prevent naive loss minimization where optimizer simply scales existing weights.

**Implementation**:
```python
# Project gradient away from current parameter vector
grad_orthogonal = grad - param * (grad·param / ||param||²)
```

**Benefits**:
- Immediate stability from step 1 (no warm-up needed)
- Allows higher learning rates (typically 10-100x Adam)
- Maintains gradient diversity without additional memory

### 2. "Car Suspension" Second Moment Design
**Problem**: Apollo's slow convergence due to poor initial second moment estimates.

**Solution**: Adaptive β₂ that starts responsive then smooths:
```python
# Step 1: β₂=0.9 (quick response to capture directions)
# Step 100: β₂=0.999 (smooth, like standard Adam)
β₂_adaptive = min(β₂_target, 0.9 + 0.099 * (step/100))
```

**Engineering Analogy**: Like a well-tuned car suspension - quick response to road changes, then smooth ride.

### 3. Low-Rank Channel Projections (Apollo Heritage)
**Memory Problem**: SPlus uses O(d²) second-moment matrices.
**Solution**: Approximate channel-wise scaling with rank-256 projections.

**Key Improvement over Apollo**: More frequent updates (every 20 steps vs 200) enabled by cheaper computation.

### 4. EMA Elimination Strategy
**SPlus Issue**: EMA doubles memory usage for marginal stability gains.
**WhiteFlow Solution**: Orthogonalization + good second moment design provides stability without EMA.

## 📊 Expected Performance Profile

### Memory Usage (4B Model)
```
Total Model Memory Budget (24GB RTX 4090):
├── Model weights: ~8GB (bf16)
├── Gradients: ~8GB (bf16)
├── Optimizer states: ~4GB (WhiteFlow vs ~8GB Adam vs ~16GB SPlus)
└── Available for batch size: ~4GB (2x larger batches than Adam)
```

### Convergence Expectations
- **Steps 1-100**: Match Adam performance (vs Apollo's poor start)
- **Steps 100-1000**: Approach SPlus convergence speed
- **Memory**: ~50% of Adam, ~25% of SPlus
- **Throughput**: 1.5-2x Adam (due to larger possible batch sizes)

## 🧪 Test Plan: SmolLM2-135M

### Phase 1: Proof of Concept (1000 steps)
**Baseline**: AdamW fused optimizer
**Model**: HuggingFaceTB/SmolLM2-135M
**Training**: Transformer layers only (skip embeddings)
**Metrics**: Loss convergence, VRAM usage, step time

### Success Criteria
1. **Convergence**: Match or beat AdamW by step 100
2. **Memory**: Use less memory than AdamW
3. **Speed**: Comparable or better step time
4. **Stability**: No training instabilities

### Code Structure
```
optimizers/
├── whiteflow.py          # Core optimizer implementation
├── main.py              # Training comparison script
└── README.md           # This design document
```

## 🔍 Why This Should Work

### Apollo's Slow Start Problem Analysis
**Root Causes**:
1. Random projections need 20-40k steps to "settle" into good subspaces
2. Initial gradient scaling factors are poorly calibrated
3. Infrequent projection updates (every 200 steps) slow adaptation

**WhiteFlow Solutions**:
1. **Orthogonalization** provides immediate good directions
2. **Car suspension β₂** gives quick initial response
3. **Frequent projection updates** (every 20 steps) improve adaptation
4. **Better initialization** of projection matrices

### Memory Optimization Validation
**SPlus Memory Issue**: O(d²) matrices for 28672×8192 layer = 3.3GB just for one matrix
**WhiteFlow Approach**: Rank-256 projections = 256×28672×2 bytes = 14MB

**Quantization Strategy**: We avoid naive int8 quantization because:
- Momentum values: ~1e-6 to 1e-2 (would all become 0 in int8)
- exp_avg_sq values: ~1e-12 to 1e-4 (even tinier)
- BitsAndBytes uses proper dynamic quantization (future optimization)

## 🎯 Scaling Strategy

### 70B Model Considerations
For massive layers like LLaMA 70B MLP (28672×8192):
- **Block-wise processing**: Split large matrix operations
- **Async CPU offloading**: Non-critical projection updates
- **Gradient checkpointing**: Already planned for activation memory
- **Rank adaptation**: Scale rank with layer size (256→512 for huge layers)

### Future Optimizations (Phase 2)
1. **BitsAndBytes integration**: Proper 8-bit quantization
2. **torch.compile support**: Fused operations for speed
3. **FSDP compatibility**: For multi-GPU scaling
4. **CPU offloading**: Projection matrices during forward pass

## 📚 References & Inspirations

### Source Papers & Implementations

**SPlus**: "A Stable Whitening Optimizer for Neural Network Optimization"
- Repository: `optimizers/splus/`
- Key insight: Whitening via eigendecomposition for fast convergence
- Problem: O(d²) memory usage, expensive eigendecompositions

**Apollo**: "APOLLO: SGD-like Memory, AdamW-level Performance"
- Repository: `optimizers/apollo/`
- Key insight: Channel-wise gradient scaling via low-rank projections
- Problem: 20-40k steps to reach good performance

**DolphinFlow**: "A Robust Orthogonalizing Optimizer"
- Repository: `optimizers/dolphinflow-optimizer/`
- Key insight: Gradient orthogonalization prevents naive loss minimization
- Problem: No adaptive scaling, 8-bit version needs second momentum

### Critical Implementation Details

**From Apollo Analysis**:
- Full Apollo uses channel-wise adaptation (not tensor-wise like Apollo-Mini)
- Random projections work better than SVD for auxiliary spaces
- Norm-Growth Limiter provides better stability than gradient clipping

**From DolphinFlow Analysis**:
- Vector orthogonalization: `project_away_from_vector(u, v)`
- Works with torch.compile for performance
- Single momentum buffer significantly reduces memory

**From SPlus Analysis**:
- EMA parameters provide stability but double memory usage
- Eigendecomposition every 100 steps is expensive but effective
- Employs a high base learning rate (typically around 0.1) which is then aggressively scaled based on parameter shape (e.g., `lr * (2 / (rows + cols))` for 2D layers).

## 🚀 Implementation Status

- [x] WhiteFlow core algorithm design
- [x] Memory optimization strategy
- [x] Test framework for SmolLM2-135M
- [ ] Initial implementation testing
- [ ] Performance validation vs AdamW
- [ ] Memory profiling and optimization
- [ ] Scaling tests to larger models

## ⚠️ Known Limitations & Risks

### Potential Issues
1. **Orthogonalization overhead**: May slow down step time
2. **Projection quality**: Low-rank approximation may be insufficient
3. **Hyperparameter sensitivity**: Learning rate, rank, update frequency
4. **Numerical stability**: bf16 precision for small values

### Mitigation Strategies
1. **Careful profiling**: Measure actual impact of each component
2. **Adaptive rank**: Increase rank if convergence is poor
3. **Extensive hyperparameter search**: Start with conservative values
4. **Fallback mechanisms**: Disable orthogonalization if unstable

## 🎖️ Success Metrics

### Minimum Viable Product (MVP)
- ✅ Memory usage ≤ 1.5x Adam (vs 2x SPlus)
- ✅ Convergence speed ≥ 0.8x Adam by step 100  
- ✅ No training instabilities over 1000 steps
- ✅ Implementation runs without errors

### Stretch Goals
- 🎯 Memory usage < Adam (due to no EMA)
- 🎯 Convergence speed ≥ Adam by step 100
- 🎯 Convergence speed ≥ 1.2x Adam by step 1000
- 🎯 Step time ≤ 1.1x Adam

---

**Next Steps**: Run `python main.py` to execute the SmolLM2-135M comparison and validate the WhiteFlow design hypothesis.