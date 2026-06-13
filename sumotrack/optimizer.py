from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .projector import ProjectionSide, SubspaceProjector


class SubspaceMuon(Optimizer):
    """Eager SUMOTrack baseline optimizer.

    Matrix parameters keep optimizer state in projected space: an orthonormal
    basis plus a projected first moment. Non-matrix parameters use a small local
    AdamW fallback so biases/norms still train without contaminating the matrix
    state invariant.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        beta: float = 0.9,
        fallback_betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 32,
        side: ProjectionSide | str = ProjectionSide.AUTO,
        recovery_scale: float = 0.0,
        orthogonalization: str = "svd",
        subspace_refresh_budget: int = 1,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if len(fallback_betas) != 2 or not all(0 <= b < 1 for b in fallback_betas):
            raise ValueError(f"fallback_betas must contain two values in [0, 1), got {fallback_betas}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if recovery_scale < 0:
            raise ValueError(f"recovery_scale must be non-negative, got {recovery_scale}")
        if orthogonalization != "svd":
            raise ValueError("only orthogonalization='svd' is implemented in the eager baseline")
        if subspace_refresh_budget <= 0:
            raise ValueError(f"subspace_refresh_budget must be positive, got {subspace_refresh_budget}")

        defaults = dict(
            lr=lr,
            beta=beta,
            fallback_betas=fallback_betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            side=ProjectionSide(side).value,
            recovery_scale=recovery_scale,
            orthogonalization=orthogonalization,
            subspace_refresh_budget=subspace_refresh_budget,
            subspace_refresh_cursor=0,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            matrix_params = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
            refresh_ids = self._refresh_param_ids(group, matrix_params)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SubspaceMuon does not support sparse gradients")
                if p.ndim == 2:
                    self._step_matrix_param(p, grad, group, id(p) in refresh_ids)
                else:
                    self._step_fallback_param(p, grad, group)

        return loss

    def _refresh_param_ids(self, group: dict, matrix_params: list[Tensor]) -> set[int]:
        if not matrix_params:
            group["subspace_refresh_cursor"] = 0
            return set()
        budget = min(group["subspace_refresh_budget"], len(matrix_params))
        cursor = group["subspace_refresh_cursor"] % len(matrix_params)
        refresh = [(cursor + offset) % len(matrix_params) for offset in range(budget)]
        group["subspace_refresh_cursor"] = (cursor + budget) % len(matrix_params)
        return {id(matrix_params[index]) for index in refresh}

    def _step_matrix_param(self, p: Tensor, grad: Tensor, group: dict, refresh_basis: bool) -> None:
        state = self.state[p]
        state["step"] = state.get("step", 0) + 1

        projector = self._projector_from_state(p, group, state)
        if not projector.is_initialized or refresh_basis:
            old_shape = tuple(state["projected_exp_avg"].shape) if "projected_exp_avg" in state else None
            projector.fit_svd(grad)
            state["basis"] = projector.basis
            resolved_side = projector.resolved_side if projector.resolved_side is not None else projector.side
            state["projection_side_is_right"] = resolved_side is ProjectionSide.RIGHT
            projected_shape = tuple(projector.project(grad).shape)
            if old_shape is not None and old_shape != projected_shape:
                state.pop("projected_exp_avg", None)

        projected_grad = projector.project(grad)
        projected_exp_avg = state.get("projected_exp_avg")
        if projected_exp_avg is None:
            projected_exp_avg = torch.zeros_like(projected_grad)
        projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        state["projected_exp_avg"] = projected_exp_avg

        update_hat = self._orthogonalize_svd(projected_exp_avg)
        update = projector.project_back(update_hat).to(dtype=p.dtype)

        recovery_scale = group["recovery_scale"]
        if recovery_scale:
            projected_back_grad = projector.project_back(projected_grad).to(dtype=grad.dtype)
            update = update + recovery_scale * (grad - projected_back_grad)

        if group["weight_decay"]:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
        p.add_(update, alpha=-group["lr"])

    def _step_fallback_param(self, p: Tensor, grad: Tensor, group: dict) -> None:
        state = self.state[p]
        state["step"] = state.get("step", 0) + 1
        beta1, beta2 = group["fallback_betas"]
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(p)
            exp_avg_sq = torch.zeros_like(p)
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq

        if group["weight_decay"]:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])

        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        denom = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(group["eps"])
        p.addcdiv_(exp_avg / bias_correction1, denom, value=-group["lr"])

    @staticmethod
    def _orthogonalize_svd(update: Tensor) -> Tensor:
        svd_input = update.float() if update.dtype in (torch.float16, torch.bfloat16) else update
        u, _s, vh = torch.linalg.svd(svd_input, full_matrices=False)
        return (u @ vh).to(dtype=update.dtype, device=update.device)

    @staticmethod
    def _projector_from_state(p: Tensor, group: dict, state: dict) -> SubspaceProjector:
        projector = SubspaceProjector(rank=group["rank"], side=ProjectionSide(group["side"]))
        basis = state.get("basis")
        if basis is not None:
            projector.basis = basis
            is_right = state.get("projection_side_is_right")
            if is_right is None:
                projector.resolved_side = projector.effective_side(p)
            else:
                projector.resolved_side = ProjectionSide.RIGHT if is_right else ProjectionSide.LEFT
        return projector
