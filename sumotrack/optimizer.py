from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from heavyball import utils as heavyball_utils

from .projector import ProjectionSide, ProjectorInitMethod, SubspaceProjector


AURORA_PP_ITERATIONS = 2
AURORA_PP_BETA = 0.5
ORTHOGONALIZATION_SCALE_MODE = "muon"
NEWTON_SCHULZ_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)


@dataclass
class MatrixUpdate:
    param: Tensor
    projector: SubspaceProjector
    projected_grad: Tensor
    projected_exp_avg: Tensor
    original_shape: tuple[int, ...]


class SumoTrack(Optimizer):
    """Eager SumoTrack baseline optimizer.

    Matrix parameters keep optimizer state in projected space: an orthonormal
    basis plus a projected first moment. Non-matrix parameters use HeavyBall's
    fused AdamW update path so biases/norms still train without contaminating
    the matrix state invariant.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        beta: float = 0.9,
        fallback_betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 32,
        side: ProjectionSide | str = ProjectionSide.AUTO,
        basis_init: str = "svd",
        grassmann_step_size: float = 0.01,
        basis_refresh_interval: int = 100,
        aurora_pp_iterations: int = AURORA_PP_ITERATIONS,
        polar_ns_steps: int = len(NEWTON_SCHULZ_COEFFICIENTS),
        consume_grad: bool = True,
        ecc: str | None = None,
        param_ecc: str | None = None,
    ) -> None:
        if ecc is not None or param_ecc is not None:
            raise NotImplementedError(
                "SumoTrack does not yet support HeavyBall ECC/param-ECC. "
                "Fallback updates use HeavyBall fused AdamW math, but ECC requires HeavyBall's ChainOpt state hooks."
            )
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
        basis_init = ProjectorInitMethod(basis_init).value
        if grassmann_step_size <= 0:
            raise ValueError(f"grassmann_step_size must be positive, got {grassmann_step_size}")
        if basis_refresh_interval <= 0:
            raise ValueError(f"basis_refresh_interval must be positive, got {basis_refresh_interval}")
        if aurora_pp_iterations <= 0:
            raise ValueError(f"aurora_pp_iterations must be positive, got {aurora_pp_iterations}")
        if not 1 <= polar_ns_steps <= len(NEWTON_SCHULZ_COEFFICIENTS):
            raise ValueError(f"polar_ns_steps must be in [1, {len(NEWTON_SCHULZ_COEFFICIENTS)}], got {polar_ns_steps}")

        defaults = dict(
            lr=lr,
            beta=beta,
            fallback_betas=fallback_betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            side=ProjectionSide(side).value,
            basis_init=basis_init,
            grassmann_step_size=grassmann_step_size,
            basis_refresh_interval=basis_refresh_interval,
            aurora_pp_iterations=aurora_pp_iterations,
            polar_ns_steps=polar_ns_steps,
            consume_grad=consume_grad,
            basis_refresh_step=0,
        )
        super().__init__(params, defaults)
        self.diagnostics_enabled = False
        self.last_step_diagnostics: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        diagnostics = self._new_diagnostics()

        for group in self.param_groups:
            matrix_params = [p for p in group["params"] if p.grad is not None and p.ndim == 2]
            refresh_ids = self._refresh_param_ids(group, matrix_params)
            matrix_updates = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SumoTrack does not support sparse gradients")
                if p.ndim == 2:
                    matrix_updates.append(self._prepare_matrix_update(p, grad, group, id(p) in refresh_ids))
                    if group["consume_grad"]:
                        p.grad = None
                else:
                    self._step_fallback_param(p, grad, group, diagnostics)
                    if group["consume_grad"]:
                        p.grad = None
            self._apply_matrix_update_buckets(matrix_updates, group, diagnostics)

        self.last_step_diagnostics = self._finalize_diagnostics(diagnostics)

        return loss

    def _new_diagnostics(self) -> dict[str, float] | None:
        if not self.diagnostics_enabled:
            return None
        return {
            "matrix_update_norm_sq": 0.0,
            "fallback_update_norm_sq": 0.0,
            "matrix_params": 0,
            "fallback_params": 0,
            "projected_leverage_cv_sum": 0.0,
            "projected_leverage_min_ratio_sum": 0.0,
            "projected_leverage_max_ratio_sum": 0.0,
            "projected_leverage_tensors": 0,
        }

    def _finalize_diagnostics(self, diagnostics: dict[str, float] | None) -> dict[str, float]:
        if diagnostics is None:
            return {}
        diagnostics["matrix_update_norm"] = diagnostics["matrix_update_norm_sq"] ** 0.5
        diagnostics["fallback_update_norm"] = diagnostics["fallback_update_norm_sq"] ** 0.5
        diagnostics["update_norm"] = (diagnostics["matrix_update_norm_sq"] + diagnostics["fallback_update_norm_sq"]) ** 0.5
        count = diagnostics["projected_leverage_tensors"]
        diagnostics["mean_projected_leverage_cv"] = diagnostics["projected_leverage_cv_sum"] / count if count else float("nan")
        diagnostics["mean_projected_leverage_min_ratio"] = diagnostics["projected_leverage_min_ratio_sum"] / count if count else float("nan")
        diagnostics["mean_projected_leverage_max_ratio"] = diagnostics["projected_leverage_max_ratio_sum"] / count if count else float("nan")
        return diagnostics

    def _refresh_param_ids(self, group: dict, matrix_params: list[Tensor]) -> set[int]:
        if not matrix_params:
            return set()
        step = group["basis_refresh_step"]
        group["basis_refresh_step"] = step + 1
        if step > 0 and step % group["basis_refresh_interval"] == 0:
            return {id(param) for param in matrix_params}
        return set()

    def _prepare_matrix_update(self, p: Tensor, grad: Tensor, group: dict, refresh_basis: bool) -> MatrixUpdate:
        state = self.state[p]
        state["step"] = state.get("step", 0) + 1

        projector = self._projector_from_state(p, group, state)
        if not projector.is_initialized or refresh_basis:
            self._refresh_projector(projector, grad, group, state)

        projected_grad = projector.project(grad)
        projected_exp_avg = state.get("projected_exp_avg")
        if projected_exp_avg is None:
            projected_exp_avg = torch.zeros_like(projected_grad)
        projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        state["projected_exp_avg"] = projected_exp_avg

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_grad=projected_grad,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
        )

    def _apply_matrix_update_buckets(self, entries: list[MatrixUpdate], group: dict, diagnostics: dict | None) -> None:
        if not entries:
            return

        buckets: dict[tuple, list[MatrixUpdate]] = {}
        for entry in entries:
            projected_exp_avg = entry.projected_exp_avg
            key = (tuple(projected_exp_avg.shape), entry.original_shape)
            buckets.setdefault(key, []).append(entry)

        for bucket_entries in buckets.values():
            if len(bucket_entries) == 1:
                update_hats = [
                    self._orthogonalize_update(
                        bucket_entries[0].projected_exp_avg,
                        group,
                        bucket_entries[0].original_shape,
                    )
                ]
            else:
                stacked = torch.stack([entry.projected_exp_avg for entry in bucket_entries])
                stacked_update_hats = self._orthogonalize_update(stacked, group, bucket_entries[0].original_shape)
                update_hats = list(stacked_update_hats.unbind(0))
            for entry, update_hat in zip(bucket_entries, update_hats, strict=True):
                self._apply_matrix_update(entry, update_hat, group, diagnostics)

    def _apply_matrix_update(self, entry: MatrixUpdate, update_hat: Tensor, group: dict, diagnostics: dict | None) -> None:
        if diagnostics is not None:
            leverage_cv, min_ratio, max_ratio = self._large_axis_leverage_stats(update_hat)
            diagnostics["projected_leverage_cv_sum"] += leverage_cv
            diagnostics["projected_leverage_min_ratio_sum"] += min_ratio
            diagnostics["projected_leverage_max_ratio_sum"] += max_ratio
            diagnostics["projected_leverage_tensors"] += 1
        update = entry.projector.project_back(update_hat).to(dtype=entry.param.dtype)

        if group["weight_decay"]:
            entry.param.mul_(1.0 - group["lr"] * group["weight_decay"])
        if diagnostics is not None:
            diagnostics["matrix_update_norm_sq"] += float((update.float().norm() * group["lr"]).square().detach().cpu())
            diagnostics["matrix_params"] += entry.param.numel()
        entry.param.add_(update, alpha=-group["lr"])

    def _step_fallback_param(self, p: Tensor, grad: Tensor, group: dict, diagnostics: dict | None) -> None:
        state = self.state[p]
        state["step"] = state.get("step", 0) + 1
        beta1, beta2 = group["fallback_betas"]
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(p, dtype=torch.float32)
            exp_avg_sq = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        before = p.detach().clone() if diagnostics is not None else None
        update = grad.detach().clone()
        heavyball_utils.fused_adam_(
            [p],
            [exp_avg],
            [exp_avg_sq],
            [update],
            [grad],
            beta1,
            beta2,
            state["step"],
            group["lr"],
            group["eps"],
            group["weight_decay"],
            False,
        )
        if diagnostics is not None:
            delta = p.detach().float() - before.float()
            diagnostics["fallback_update_norm_sq"] += float(delta.norm().square().detach().cpu())
            diagnostics["fallback_params"] += p.numel()

    def _refresh_projector(self, projector: SubspaceProjector, grad: Tensor, group: dict, state: dict) -> None:
        old_projected_exp_avg = state.get("projected_exp_avg")
        old_projector = None
        if projector.is_initialized and old_projected_exp_avg is not None:
            old_projector = SubspaceProjector(rank=projector.rank, side=projector.side, init_method=projector.init_method)
            old_projector.basis = projector.basis
            old_projector.resolved_side = projector.resolved_side

        if not projector.is_initialized:
            projector.fit(grad)
        else:
            projector.update_grassmann(grad, step_size=group["grassmann_step_size"])

        state["basis"] = projector.basis
        resolved_side = projector.resolved_side if projector.resolved_side is not None else projector.side
        state["projection_side_is_right"] = resolved_side is ProjectionSide.RIGHT

        if old_projector is not None:
            lifted_moment = old_projector.project_back(old_projected_exp_avg)
            state["projected_exp_avg"] = projector.project(lifted_moment)
        elif old_projected_exp_avg is not None:
            projected_shape = tuple(projector.project(grad).shape)
            if tuple(old_projected_exp_avg.shape) != projected_shape:
                state.pop("projected_exp_avg", None)

    @staticmethod
    def _orthogonalize_update(update: Tensor, group: dict, original_shape: tuple[int, ...] | None = None) -> Tensor:
        return SumoTrack._orthogonalize_aurora(update, group, original_shape)

    @staticmethod
    def _orthogonalize_aurora(update: Tensor, _group: dict, original_shape: tuple[int, ...] | None) -> Tensor:
        aurora_update = SumoTrack._aurora_leverage_uniform_polar(
            update,
            pp_iterations=_group.get("aurora_pp_iterations", AURORA_PP_ITERATIONS),
            pp_beta=AURORA_PP_BETA,
            polar_ns_steps=_group.get("polar_ns_steps", len(NEWTON_SCHULZ_COEFFICIENTS)),
        )
        return SumoTrack._scale_orthogonalized_update(
            update,
            aurora_update,
            ORTHOGONALIZATION_SCALE_MODE,
            original_shape,
        )

    @staticmethod
    def _aurora_leverage_uniform_polar(
        update: Tensor,
        pp_iterations: int = 2,
        pp_beta: float = 0.5,
        eps: float = 1e-7,
        polar_ns_steps: int = len(NEWTON_SCHULZ_COEFFICIENTS),
    ) -> Tensor:
        """Aurora-style leverage-uniform polar direction for rectangular projected moments.

        SumoTrack owns momentum, LR, weight decay, and full-matrix Muon scaling. This
        helper extracts only Aurora's rectangular direction map: diagonally
        precondition a non-square matrix before polar/NS so the large-side row
        leverage approaches the Stiefel target. For wide matrices, transpose to
        tall form, balance, then transpose back, matching Aurora's convention.
        """

        if update.ndim < 2:
            raise ValueError(f"Aurora orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        if pp_iterations < 1:
            raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
        if pp_beta <= 0:
            raise ValueError(f"pp_beta must be positive, got {pp_beta}")
        if not 1 <= polar_ns_steps <= len(NEWTON_SCHULZ_COEFFICIENTS):
            raise ValueError(f"polar_ns_steps must be in [1, {len(NEWTON_SCHULZ_COEFFICIENTS)}], got {polar_ns_steps}")
        if update.shape[-2] == update.shape[-1]:
            return SumoTrack._heavyball_polar(update, steps=polar_ns_steps)

        transposed = update.shape[-2] < update.shape[-1]
        work = update.mT if transposed else update
        work32 = work.float()
        rows, cols = work32.shape[-2:]
        target_row_sq = cols / rows
        diagonal = work32.norm(dim=-1, keepdim=True).clamp_min(eps).reciprocal()
        balanced = None
        for iteration in range(pp_iterations):
            balanced = SumoTrack._heavyball_polar(diagonal * work32, steps=polar_ns_steps).float()
            if iteration < pp_iterations - 1:
                row_sq = balanced.square().sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                diagonal = diagonal * (target_row_sq / row_sq).pow(pp_beta)
        assert balanced is not None
        result = balanced.mT if transposed else balanced
        return result.to(device=update.device, dtype=update.dtype)

    @staticmethod
    def _heavyball_polar(update: Tensor, steps: int = len(NEWTON_SCHULZ_COEFFICIENTS)) -> Tensor:
        return SumoTrack._batched_newton_schulz(update, steps=steps)

    @staticmethod
    def _batched_newton_schulz(update: Tensor, steps: int = len(NEWTON_SCHULZ_COEFFICIENTS), eps: float = 1e-7) -> Tensor:
        if update.ndim < 2:
            raise ValueError(f"Newton-Schulz orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        if not 1 <= steps <= len(NEWTON_SCHULZ_COEFFICIENTS):
            raise ValueError(f"steps must be in [1, {len(NEWTON_SCHULZ_COEFFICIENTS)}], got {steps}")
        work = update.float()
        work = work / work.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
        transposed = work.shape[-2] > work.shape[-1]
        x = work.mT if transposed else work

        for a, b, c in NEWTON_SCHULZ_COEFFICIENTS[:steps]:
            gram = x @ x.mT
            y = c * gram
            y.diagonal(dim1=-2, dim2=-1).add_(b)
            y = y @ gram
            y.diagonal(dim1=-2, dim2=-1).add_(a)
            x = y @ x

        result = x.mT if transposed else x
        return result.to(device=update.device, dtype=update.dtype)

    @staticmethod
    def _scale_orthogonalized_update(
        original_update: Tensor,
        orthogonalized_update: Tensor,
        scale_mode: str,
        original_shape: tuple[int, ...] | None,
    ) -> Tensor:
        if scale_mode == "none":
            return orthogonalized_update
        if scale_mode == "scale":
            return orthogonalized_update * math.sqrt(max(1.0, original_update.shape[-2] / original_update.shape[-1]))
        if scale_mode == "graft":
            if original_update.ndim > 2:
                original_norm = original_update.norm(dim=(-2, -1), keepdim=True)
                ortho_norm = orthogonalized_update.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                return orthogonalized_update * (original_norm / ortho_norm)
            return orthogonalized_update * (original_update.norm() / orthogonalized_update.norm().clamp(min=1e-6))
        if scale_mode == "muon":
            if original_shape is None or len(original_shape) < 2:
                raise ValueError("muon scale mode requires the original parameter shape")
            rows = original_shape[0]
            cols = math.prod(original_shape[1:])
            return orthogonalized_update * math.sqrt(max(1.0, rows / cols))
        raise AssertionError(f"unexpected orthogonalization scale mode: {scale_mode}")

    @staticmethod
    def _large_axis_leverage_stats(update: Tensor, eps: float = 1e-12) -> tuple[float, float, float]:
        if update.ndim != 2 or min(update.shape) == 0:
            return float("nan"), float("nan"), float("nan")
        tall = update if update.shape[-2] >= update.shape[-1] else update.mT
        row_sq = tall.float().square().sum(dim=-1)
        mean = row_sq.mean().clamp_min(eps)
        cv = row_sq.std(unbiased=False) / mean
        return (
            float(cv.detach().cpu()),
            float((row_sq.min() / mean).detach().cpu()),
            float((row_sq.max() / mean).detach().cpu()),
        )

    @staticmethod
    def _projector_from_state(p: Tensor, group: dict, state: dict) -> SubspaceProjector:
        projector = SubspaceProjector(
            rank=group["rank"],
            side=ProjectionSide(group["side"]),
            init_method=ProjectorInitMethod(group["basis_init"]),
        )
        basis = state.get("basis")
        if basis is not None:
            projector.basis = basis
            is_right = state.get("projection_side_is_right")
            if is_right is None:
                projector.resolved_side = projector.effective_side(p)
            else:
                projector.resolved_side = ProjectionSide.RIGHT if is_right else ProjectionSide.LEFT
        return projector
