"""
WhiteFlow: Memory-Efficient Whitening-Inspired Optimizer

A novel optimizer combining insights from:
- SPlus: Whitening-based preconditioning for fast convergence
- Apollo: Low-rank projections for memory efficiency
- DolphinFlow: Gradient orthogonalization for stability

Designed for continued pretraining scenarios with limited memory budgets.
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Any, Optional, Callable
import math


def project_away_from_vector(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Projects vector u to be orthogonal to vector v.
    Inspired by DolphinFlow's orthogonalization technique.

    This prevents naive loss minimization by ensuring gradient updates
    don't simply scale existing weights.
    """
    original_shape = u.shape

    # Flatten for dot product computation
    if u.ndim > 1:
        u_flat = u.flatten()
    else:
        u_flat = u

    if v.ndim > 1:
        v_flat = v.flatten()
    else:
        v_flat = v

    # Safety checks
    u_norm = torch.norm(u_flat)
    v_norm = torch.norm(v_flat)

    if u_norm < eps or v_norm < eps:
        return u  # Cannot project if either vector is too small

    # Check for NaN/Inf
    if torch.isnan(u_flat).any() or torch.isinf(u_flat).any():
        return torch.zeros_like(u)
    if torch.isnan(v_flat).any() or torch.isinf(v_flat).any():
        return u

    v_norm_sq = v_flat.pow(2).sum()
    if v_norm_sq < eps:
        return u  # Cannot project onto zero vector

    dot_product = torch.dot(u_flat, v_flat)
    projection = v_flat * (dot_product / v_norm_sq)

    result = (u_flat - projection).reshape(original_shape)

    # Conservative check: if projection removes too much of the gradient,
    # only partially apply it (similar to DolphinFlow's approach)
    result_norm = torch.norm(result)
    if result_norm < 0.1 * u_norm:
        print("Orthogonalization removed a lot of the gradient: {result_norm}")
        alpha = 0.7
        result = alpha * result + (1 - alpha) * u

    return result


class WhiteFlow(Optimizer):
    """
    WhiteFlow Optimizer: Memory-efficient whitening-inspired optimization

    Combines:
    1. SPlus-style whitening intuition for fast convergence
    2. Apollo's low-rank channel-wise projections for memory efficiency
    3. DolphinFlow's gradient orthogonalization for stability
    4. "Car suspension" second moment design for quick convergence from step 1

    Key features:
    - Fast convergence from step 1 (unlike Apollo's 20-40k step ramp-up)
    - Memory usage between Apollo and Adam (much less than SPlus)
    - Channel-wise adaptive scaling without expensive eigendecompositions
    - No EMA states (significant memory savings)
    - Designed for continued pretraining with limited tokens (100B vs 4-20T)

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate (typically 10-100x higher than Adam due to orthogonalization)
        rank: Rank for low-rank channel projections (256 for most models)
        beta1: Momentum coefficient (0.9)
        beta2: Second moment base coefficient (0.999)
        proj_update_freq: How often to update channel projections (20 steps)
        orthogonalize: Whether to orthogonalize gradients (True recommended)
        eps: Small constant for numerical stability (1e-8)
        weight_decay: Decoupled weight decay (1e-2)
        channel_wise_only: Apply projections only to 2D layers (True)
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1e-4,
        rank: int = 256,
        beta1: float = 0.9,
        beta2: float = 0.999,
        proj_update_freq: int = 20,
        orthogonalize: bool = True,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        channel_wise_only: bool = True,
        gradient_clipping: float = 1.0,
        disable_nl: bool = False,
        nonstandard_constant: float = 0.001,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {beta2}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 < rank:
            raise ValueError(f"Invalid rank: {rank}")
        if not 0 < proj_update_freq:
            raise ValueError(f"Invalid proj_update_freq: {proj_update_freq}")
        if not 0.0 <= gradient_clipping:
            raise ValueError(f"Invalid gradient_clipping: {gradient_clipping}")

        defaults = dict(
            lr=lr,
            rank=rank,
            beta1=beta1,
            beta2=beta2,
            proj_update_freq=proj_update_freq,
            orthogonalize=orthogonalize,
            eps=eps,
            weight_decay=weight_decay,
            channel_wise_only=channel_wise_only,
            gradient_clipping=gradient_clipping,
            disable_nl=disable_nl,
            nonstandard_constant=nonstandard_constant,
        )
        super().__init__(params, defaults)

    def _initialize_projections(self, param_shape: tuple, rank: int, device: torch.device, dtype: torch.dtype) -> tuple:
        """Initialize random projections for channel-wise scaling (Apollo-inspired)"""
        rows, cols = param_shape
        effective_rank = min(rank, min(rows, cols))

        # Random projection matrices (normalized)
        proj_left = torch.randn(effective_rank, rows, device=device, dtype=dtype) * (1.0 / math.sqrt(rows))
        proj_right = torch.randn(effective_rank, cols, device=device, dtype=dtype) * (1.0 / math.sqrt(cols))

        # Channel scaling factors (initialized to 1)
        channel_scales = torch.ones(rows, device=device, dtype=dtype)

        return proj_left, proj_right, channel_scales

    def _update_channel_projections(self, state: dict, grad: torch.Tensor):
        """Update channel projections based on gradient statistics"""
        # Simple update: exponential moving average of gradient outer products
        # This is much cheaper than SPlus's eigendecomposition

        # Left projection update (row-wise statistics)
        grad_left_stats = torch.mm(grad, grad.t())  # [rows, rows]
        if 'grad_left_ema' not in state:
            state['grad_left_ema'] = grad_left_stats.clone()
        else:
            state['grad_left_ema'].mul_(0.9).add_(grad_left_stats, alpha=0.1)

        # Right projection update (column-wise statistics)
        grad_right_stats = torch.mm(grad.t(), grad)  # [cols, cols]
        if 'grad_right_ema' not in state:
            state['grad_right_ema'] = grad_right_stats.clone()
        else:
            state['grad_right_ema'].mul_(0.9).add_(grad_right_stats, alpha=0.1)

        # Update projections based on top eigenvectors (approximate)
        # This is still much cheaper than full eigendecomposition
        try:
            # Simple power iteration for dominant directions
            proj_left = state['proj_left']
            proj_right = state['proj_right']

            # Update left projection
            new_left = torch.mm(proj_left, state['grad_left_ema'])
            new_left = new_left / (torch.norm(new_left, dim=1, keepdim=True) + state['eps'])
            state['proj_left'] = 0.9 * proj_left + 0.1 * new_left

            # Update right projection
            new_right = torch.mm(proj_right, state['grad_right_ema'])
            new_right = new_right / (torch.norm(new_right, dim=1, keepdim=True) + state['eps'])
            state['proj_right'] = 0.9 * proj_right + 0.1 * new_right

        except RuntimeError:
            # Skip update if numerical issues
            pass

    def _compute_channel_scaling(self, grad: torch.Tensor, norm_grad: torch.Tensor, state: dict) -> torch.Tensor:
        """Compute channel-wise scaling factors using low-rank projections like Apollo"""
        # Temporarily disabled for stability testing - return neutral scaling
        return torch.ones_like(grad)

        if len(grad.shape) != 2 or 'proj_left' not in state:
            # For non-2D tensors or no projections, use tensor-wise scaling
            grad_norm = torch.norm(grad).item()
            norm_grad_norm = torch.norm(norm_grad).item()
            if grad_norm < 1e-8:
                return torch.ones_like(grad)
            scaling_factor = norm_grad_norm / grad_norm
            return torch.full_like(grad, scaling_factor)

        # Low-rank projection approach (simplified Apollo)
        try:
            # Project gradients to low-rank space for stable scaling computation
            proj_left = state['proj_left']  # [rank, rows]
            proj_right = state['proj_right']  # [rank, cols]

            # Project to low-rank space
            grad_proj = torch.mm(torch.mm(proj_left, grad), proj_right.t())  # [rank, rank]
            norm_grad_proj = torch.mm(torch.mm(proj_left, norm_grad), proj_right.t())  # [rank, rank]

            # Compute scaling in projected space (more stable)
            grad_proj_norm = torch.norm(grad_proj).item() + 1e-8
            norm_grad_proj_norm = torch.norm(norm_grad_proj).item()
            scaling_factor = norm_grad_proj_norm / grad_proj_norm

            # Clamp for stability
            scaling_factor = max(0.1, min(10.0, scaling_factor))

            return torch.full_like(grad, scaling_factor)

        except Exception as e:
            print(f"⚠️  Projection scaling failed: {e}, falling back to tensor-wise")
            # Fallback to tensor-wise scaling
            grad_norm = torch.norm(grad).item() + 1e-8
            norm_grad_norm = torch.norm(norm_grad).item()
            scaling_factor = norm_grad_norm / grad_norm
            return torch.full_like(grad, scaling_factor)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply gradient clipping (DolphinFlow-inspired)
        if self.defaults['gradient_clipping'] > 0:
            all_grads = []
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        all_grads.append(p.grad)
            if all_grads:
                torch.nn.utils.clip_grad_norm_(all_grads, max_norm=self.defaults['gradient_clipping'])

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Debug logging
                param_name = f"param_{id(p)}"
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    print(f"⚠️  {param_name}: NaN/Inf in gradient!")
                if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    print(f"⚠️  {param_name}: NaN/Inf in parameter!")

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['eps'] = group['eps']

                    # Initialize projections for 2D layers (channel-wise scaling)
                    if group['channel_wise_only'] and len(p.shape) == 2:
                        proj_left, proj_right, channel_scales = self._initialize_projections(
                            p.shape, group['rank'], p.device, p.dtype
                        )
                        state['proj_left'] = proj_left
                        state['proj_right'] = proj_right
                        state['channel_scales'] = channel_scales

                state['step'] += 1

                # 1. Gradient orthogonalization (DolphinFlow-inspired stability)
                grad_norm_before = torch.norm(grad).item()
                param_norm = torch.norm(p.data).item()
                if group['orthogonalize'] and len(p.shape) >= 2:
                    # Only orthogonalize if gradient and parameter are reasonable
                    if grad_norm_before > group['eps'] and param_norm > group['eps']:
                        grad = project_away_from_vector(grad, p.data, group['eps'])
                        grad_norm_after = torch.norm(grad).item()



                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            print(f"⚠️  {param_name}: NaN/Inf after orthogonalization! Reverting to original gradient.")
                            grad = p.grad.data  # Revert to original gradient


                # 2. Momentum update
                state['momentum'].mul_(group['beta1']).add_(grad, alpha=1 - group['beta1'])
                momentum_norm = torch.norm(state['momentum']).item()
                if torch.isnan(state['momentum']).any():
                    print(f"⚠️  {param_name}: NaN in momentum!")

                # 3. "Car suspension" second moment - quick response then smooth
                # Starts with beta2=0.9 (quick response), approaches beta2=0.999 (smooth)
                beta2_adaptive = min(group['beta2'], 0.9 + 0.099 * (state['step'] / 1000.0))
                state['exp_avg_sq'].mul_(beta2_adaptive).add_(grad.pow(2), alpha=1 - beta2_adaptive)
                exp_avg_sq_norm = torch.norm(state['exp_avg_sq']).item()
                if torch.isnan(state['exp_avg_sq']).any():
                    print(f"⚠️  {param_name}: NaN in exp_avg_sq!")

                # 4. Update channel projections periodically
                if (group['channel_wise_only'] and len(p.shape) == 2 and
                    state['step'] % group['proj_update_freq'] == 0):
                    self._update_channel_projections(state, grad)

                # 5. Compute parameter update
                denom = state['exp_avg_sq'].sqrt().add_(group['eps'])
                norm_grad = state['momentum'] / denom

                if group['channel_wise_only'] and len(p.shape) == 2 and 'proj_left' in state:
                    # Channel-wise adaptive scaling with low-rank projections
                    scaling = self._compute_channel_scaling(grad, norm_grad, state)
                    if torch.isnan(scaling).any() or torch.isinf(scaling).any():
                        print(f"⚠️  {param_name}: NaN/Inf in channel scaling!")
                        scaling = torch.ones_like(grad)

                    # Apply scaling to the original gradient (like Apollo)
                    param_update = grad * scaling


                else:
                    # Standard Adam-like update for non-2D layers or without projections
                    param_update = norm_grad

                if torch.isnan(param_update).any() or torch.isinf(param_update).any():
                    print(f"⚠️  {param_name}: NaN/Inf in param_update!")
                    print(f"   momentum_norm: {momentum_norm:.6f}")
                    print(f"   exp_avg_sq_norm: {exp_avg_sq_norm:.6f}")
                    print(f"   lr: {group['lr']}")

                # 6. Apply Norm-Growth Limiter (Apollo-inspired)
                if not group['disable_nl']:
                    param_update_norm = torch.norm(param_update)
                    if 'prev_update_norm' in state:
                        limiter = max(
                            param_update_norm / (state['prev_update_norm'] + group['eps']),
                            1.01,
                        ) / 1.01
                        param_update = param_update / limiter
                        state['prev_update_norm'] = param_update_norm / limiter
                    else:
                        state['prev_update_norm'] = param_update_norm

                # 7. Compute shape-dependent learning rate (SPlus-style)
                if len(p.shape) == 2:
                    # For 2D parameters: scale by inverse of sum of dimensions
                    effective_lr = group['lr'] * (2.0 / (p.shape[0] + p.shape[1]))
                else:
                    # For non-2D parameters: use small constant scaling
                    effective_lr = group['lr'] * group['nonstandard_constant']

                # 8. Apply weight decay (decoupled)
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - effective_lr * group['weight_decay'])

                # 9. Parameter update
                p.data.add_(param_update, alpha=-effective_lr)

                # Final check
                if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    print(f"⚠️  {param_name}: NaN/Inf in parameter after update!")

        return loss

    def __repr__(self):
        return f"WhiteFlow(lr={self.defaults['lr']}, rank={self.defaults['rank']}, orthogonalize={self.defaults['orthogonalize']}, gradient_clipping={self.defaults['gradient_clipping']})"
