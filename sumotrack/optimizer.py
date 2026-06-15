from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from heavyball import utils as heavyball_utils

from .projector import ProjectionSide, ProjectorInitMethod, SubspaceProjector


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
        projection_mode: str = "one_sided",
        side: ProjectionSide | str = ProjectionSide.AUTO,
        recovery_scale: float = 0.0,
        orthogonalization: str = "aurora",
        orthogonalization_scale_mode: str = "muon",
        heavyball_orthogonalization_mode: str | None = None,
        aurora_pp_iterations: int = 2,
        aurora_pp_beta: float = 0.5,
        subspace_init: str = "svd",
        subspace_update_method: str = "svd_refresh",
        grassmann_step_size: float = 0.01,
        subspace_refresh_budget: int = 1,
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
        if projection_mode not in {"one_sided", "two_sided"}:
            raise ValueError("projection_mode must be 'one_sided' or 'two_sided'")
        if recovery_scale < 0:
            raise ValueError(f"recovery_scale must be non-negative, got {recovery_scale}")
        if orthogonalization not in {"none", "svd", "heavyball", "aurora"}:
            raise ValueError("orthogonalization must be 'none', 'svd', 'heavyball', or 'aurora'")
        if orthogonalization_scale_mode not in {"none", "scale", "graft", "muon"}:
            raise ValueError("orthogonalization_scale_mode must be 'none', 'scale', 'graft', or 'muon'")
        if aurora_pp_iterations < 1:
            raise ValueError(f"aurora_pp_iterations must be >= 1, got {aurora_pp_iterations}")
        if aurora_pp_beta <= 0:
            raise ValueError(f"aurora_pp_beta must be positive, got {aurora_pp_beta}")
        if subspace_update_method not in {"none", "svd_refresh", "grassmann"}:
            raise ValueError("subspace_update_method must be 'none', 'svd_refresh', or 'grassmann'")
        if projection_mode == "two_sided" and subspace_update_method == "grassmann":
            raise ValueError("two-sided projection does not yet support Grassmann tracking; use subspace_update_method='none' or 'svd_refresh'")
        subspace_init = ProjectorInitMethod(subspace_init).value
        if grassmann_step_size <= 0:
            raise ValueError(f"grassmann_step_size must be positive, got {grassmann_step_size}")
        if subspace_refresh_budget <= 0:
            raise ValueError(f"subspace_refresh_budget must be positive, got {subspace_refresh_budget}")

        defaults = dict(
            lr=lr,
            beta=beta,
            fallback_betas=fallback_betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            projection_mode=projection_mode,
            side=ProjectionSide(side).value,
            recovery_scale=recovery_scale,
            orthogonalization=orthogonalization,
            orthogonalization_scale_mode=orthogonalization_scale_mode,
            heavyball_orthogonalization_mode=heavyball_orthogonalization_mode,
            aurora_pp_iterations=aurora_pp_iterations,
            aurora_pp_beta=aurora_pp_beta,
            subspace_init=subspace_init,
            subspace_update_method=subspace_update_method,
            grassmann_step_size=grassmann_step_size,
            subspace_refresh_budget=subspace_refresh_budget,
            subspace_refresh_cursor=0,
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

        diagnostics = (
            {
                "matrix_update_norm_sq": 0.0,
                "fallback_update_norm_sq": 0.0,
                "matrix_params": 0,
                "fallback_params": 0,
                "projected_leverage_cv_sum": 0.0,
                "projected_leverage_min_ratio_sum": 0.0,
                "projected_leverage_max_ratio_sum": 0.0,
                "projected_leverage_tensors": 0,
            }
            if self.diagnostics_enabled
            else None
        )

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
                    if group["projection_mode"] == "one_sided":
                        matrix_updates.append(self._prepare_one_sided_matrix_update(p, grad, group, id(p) in refresh_ids))
                    else:
                        self._step_matrix_param(p, grad, group, id(p) in refresh_ids, diagnostics)
                else:
                    self._step_fallback_param(p, grad, group, diagnostics)
            self._apply_matrix_update_buckets(matrix_updates, group, diagnostics)

        if diagnostics is not None:
            matrix_update_norm = diagnostics["matrix_update_norm_sq"] ** 0.5
            fallback_update_norm = diagnostics["fallback_update_norm_sq"] ** 0.5
            diagnostics["matrix_update_norm"] = matrix_update_norm
            diagnostics["fallback_update_norm"] = fallback_update_norm
            diagnostics["update_norm"] = (diagnostics["matrix_update_norm_sq"] + diagnostics["fallback_update_norm_sq"]) ** 0.5
            count = diagnostics["projected_leverage_tensors"]
            if count:
                diagnostics["mean_projected_leverage_cv"] = diagnostics["projected_leverage_cv_sum"] / count
                diagnostics["mean_projected_leverage_min_ratio"] = diagnostics["projected_leverage_min_ratio_sum"] / count
                diagnostics["mean_projected_leverage_max_ratio"] = diagnostics["projected_leverage_max_ratio_sum"] / count
            else:
                diagnostics["mean_projected_leverage_cv"] = float("nan")
                diagnostics["mean_projected_leverage_min_ratio"] = float("nan")
                diagnostics["mean_projected_leverage_max_ratio"] = float("nan")
            self.last_step_diagnostics = diagnostics
        else:
            self.last_step_diagnostics = {}

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

    def _step_matrix_param(self, p: Tensor, grad: Tensor, group: dict, refresh_basis: bool, diagnostics: dict | None) -> None:
        state = self.state[p]
        state["step"] = state.get("step", 0) + 1

        if group["projection_mode"] == "two_sided":
            self._step_two_sided_matrix_param(p, grad, group, refresh_basis, diagnostics)
            return

        projector = self._projector_from_state(p, group, state)
        if not projector.is_initialized or refresh_basis:
            self._refresh_projector(projector, grad, group, state)

        projected_grad = projector.project(grad)
        projected_exp_avg = state.get("projected_exp_avg")
        if projected_exp_avg is None:
            projected_exp_avg = torch.zeros_like(projected_grad)
        projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        state["projected_exp_avg"] = projected_exp_avg

        update_hat = self._orthogonalize_update(projected_exp_avg, group, tuple(p.shape))
        if diagnostics is not None and group["orthogonalization"] in {"svd", "heavyball", "aurora"}:
            leverage_cv, min_ratio, max_ratio = self._large_axis_leverage_stats(update_hat)
            diagnostics["projected_leverage_cv_sum"] += leverage_cv
            diagnostics["projected_leverage_min_ratio_sum"] += min_ratio
            diagnostics["projected_leverage_max_ratio_sum"] += max_ratio
            diagnostics["projected_leverage_tensors"] += 1
        update = projector.project_back(update_hat).to(dtype=p.dtype)

        recovery_scale = group["recovery_scale"]
        if recovery_scale:
            projected_back_grad = projector.project_back(projected_grad).to(dtype=grad.dtype)
            update = update + recovery_scale * (grad - projected_back_grad)

        if group["weight_decay"]:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
        if diagnostics is not None:
            diagnostics["matrix_update_norm_sq"] += float((update.float().norm() * group["lr"]).square().detach().cpu())
            diagnostics["matrix_params"] += p.numel()
        p.add_(update, alpha=-group["lr"])

    def _prepare_one_sided_matrix_update(self, p: Tensor, grad: Tensor, group: dict, refresh_basis: bool) -> dict:
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

        return {
            "param": p,
            "grad": grad,
            "projector": projector,
            "projected_grad": projected_grad,
            "projected_exp_avg": projected_exp_avg,
            "original_shape": tuple(p.shape),
        }

    def _apply_matrix_update_buckets(self, entries: list[dict], group: dict, diagnostics: dict | None) -> None:
        if not entries:
            return

        buckets: dict[tuple, list[dict]] = {}
        for entry in entries:
            projected_exp_avg = entry["projected_exp_avg"]
            key = (
                group["orthogonalization"],
                group["orthogonalization_scale_mode"],
                tuple(projected_exp_avg.shape),
                entry["original_shape"],
                projected_exp_avg.dtype,
                projected_exp_avg.device,
                group["heavyball_orthogonalization_mode"],
                group["aurora_pp_iterations"],
                group["aurora_pp_beta"],
            )
            buckets.setdefault(key, []).append(entry)

        for bucket_entries in buckets.values():
            if len(bucket_entries) == 1:
                update_hats = [
                    self._orthogonalize_update(
                        bucket_entries[0]["projected_exp_avg"],
                        group,
                        bucket_entries[0]["original_shape"],
                    )
                ]
            else:
                stacked = torch.stack([entry["projected_exp_avg"] for entry in bucket_entries])
                stacked_update_hats = self._orthogonalize_update(stacked, group, bucket_entries[0]["original_shape"])
                update_hats = list(stacked_update_hats.unbind(0))
            for entry, update_hat in zip(bucket_entries, update_hats, strict=True):
                self._apply_one_sided_matrix_update(entry, update_hat, group, diagnostics)

    def _apply_one_sided_matrix_update(self, entry: dict, update_hat: Tensor, group: dict, diagnostics: dict | None) -> None:
        p = entry["param"]
        grad = entry["grad"]
        projector = entry["projector"]
        projected_grad = entry["projected_grad"]

        if diagnostics is not None and group["orthogonalization"] in {"svd", "heavyball", "aurora"}:
            leverage_cv, min_ratio, max_ratio = self._large_axis_leverage_stats(update_hat)
            diagnostics["projected_leverage_cv_sum"] += leverage_cv
            diagnostics["projected_leverage_min_ratio_sum"] += min_ratio
            diagnostics["projected_leverage_max_ratio_sum"] += max_ratio
            diagnostics["projected_leverage_tensors"] += 1
        update = projector.project_back(update_hat).to(dtype=p.dtype)

        recovery_scale = group["recovery_scale"]
        if recovery_scale:
            projected_back_grad = projector.project_back(projected_grad).to(dtype=grad.dtype)
            update = update + recovery_scale * (grad - projected_back_grad)

        if group["weight_decay"]:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
        if diagnostics is not None:
            diagnostics["matrix_update_norm_sq"] += float((update.float().norm() * group["lr"]).square().detach().cpu())
            diagnostics["matrix_params"] += p.numel()
        p.add_(update, alpha=-group["lr"])

    def _step_two_sided_matrix_param(self, p: Tensor, grad: Tensor, group: dict, refresh_basis: bool, diagnostics: dict | None) -> None:
        state = self.state[p]
        if self._two_sided_needs_fit(state) or refresh_basis:
            self._refresh_two_sided_projector(grad, group, state)

        projected_grad = self._two_sided_project(grad, state)
        projected_exp_avg = state.get("projected_exp_avg")
        if projected_exp_avg is None:
            projected_exp_avg = torch.zeros_like(projected_grad)
        projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        state["projected_exp_avg"] = projected_exp_avg

        update_hat = self._orthogonalize_update(projected_exp_avg, group, tuple(p.shape))
        if diagnostics is not None and group["orthogonalization"] in {"svd", "heavyball", "aurora"}:
            leverage_cv, min_ratio, max_ratio = self._large_axis_leverage_stats(update_hat)
            diagnostics["projected_leverage_cv_sum"] += leverage_cv
            diagnostics["projected_leverage_min_ratio_sum"] += min_ratio
            diagnostics["projected_leverage_max_ratio_sum"] += max_ratio
            diagnostics["projected_leverage_tensors"] += 1
        update = self._two_sided_project_back(update_hat, state).to(dtype=p.dtype)

        recovery_scale = group["recovery_scale"]
        if recovery_scale:
            projected_back_grad = self._two_sided_project_back(projected_grad, state).to(dtype=grad.dtype)
            update = update + recovery_scale * (grad - projected_back_grad)

        if group["weight_decay"]:
            p.mul_(1.0 - group["lr"] * group["weight_decay"])
        if diagnostics is not None:
            diagnostics["matrix_update_norm_sq"] += float((update.float().norm() * group["lr"]).square().detach().cpu())
            diagnostics["matrix_params"] += p.numel()
        p.add_(update, alpha=-group["lr"])

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
        elif group["subspace_update_method"] == "none":
            return
        elif group["subspace_update_method"] == "svd_refresh":
            projector.fit_svd(grad)
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
    def _two_sided_rank(matrix: Tensor, rank: int) -> int:
        if matrix.ndim != 2:
            raise ValueError(f"two-sided projection only supports 2D tensors, got shape {tuple(matrix.shape)}")
        return min(rank, matrix.shape[0], matrix.shape[1])

    @staticmethod
    def _two_sided_needs_fit(state: dict) -> bool:
        return state.get("basis_left") is None or state.get("basis_right") is None

    @staticmethod
    def _fit_two_sided_projector(matrix: Tensor, group: dict, init_method: str | None = None) -> tuple[Tensor, Tensor]:
        rank = SumoTrack._two_sided_rank(matrix, group["rank"])
        work_dtype = torch.float32 if matrix.dtype in (torch.float16, torch.bfloat16) else matrix.dtype
        init_method = init_method or group["subspace_init"]
        if init_method == "svd":
            work = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
            u, _s, vh = torch.linalg.svd(work, full_matrices=False)
            left = u[:, :rank]
            right = vh[:rank, :]
        else:
            left_random = torch.randn(matrix.shape[0], rank, device=matrix.device, dtype=work_dtype)
            left, _left_r = torch.linalg.qr(left_random, mode="reduced")
            right_random = torch.randn(matrix.shape[1], rank, device=matrix.device, dtype=work_dtype)
            right_q, _right_r = torch.linalg.qr(right_random, mode="reduced")
            right = right_q.mT
        return (
            left.to(device=matrix.device, dtype=matrix.dtype).contiguous(),
            right.to(device=matrix.device, dtype=matrix.dtype).contiguous(),
        )

    @staticmethod
    def _refresh_two_sided_projector(grad: Tensor, group: dict, state: dict) -> None:
        old_projected_exp_avg = state.get("projected_exp_avg")
        old_left = state.get("basis_left")
        old_right = state.get("basis_right")

        if old_left is not None and old_right is not None and group["subspace_update_method"] == "none":
            return

        init_method = "svd" if old_left is not None and group["subspace_update_method"] == "svd_refresh" else group["subspace_init"]
        new_left, new_right = SumoTrack._fit_two_sided_projector(grad, group, init_method=init_method)
        state["basis_left"] = new_left
        state["basis_right"] = new_right
        state["projection_mode"] = "two_sided"

        if old_projected_exp_avg is not None and old_left is not None and old_right is not None:
            lifted_moment = old_left @ old_projected_exp_avg @ old_right
            state["projected_exp_avg"] = new_left.mT @ lifted_moment @ new_right.mT
        elif old_projected_exp_avg is not None:
            projected_shape = tuple(SumoTrack._two_sided_project(grad, state).shape)
            if tuple(old_projected_exp_avg.shape) != projected_shape:
                state.pop("projected_exp_avg", None)

    @staticmethod
    def _two_sided_project(matrix: Tensor, state: dict) -> Tensor:
        left = state.get("basis_left")
        right = state.get("basis_right")
        if left is None or right is None:
            raise RuntimeError("two-sided projector has not been fitted")
        return left.mT @ matrix @ right.mT

    @staticmethod
    def _two_sided_project_back(projected: Tensor, state: dict) -> Tensor:
        left = state.get("basis_left")
        right = state.get("basis_right")
        if left is None or right is None:
            raise RuntimeError("two-sided projector has not been fitted")
        expected = (left.shape[1], right.shape[0])
        if tuple(projected.shape) != expected:
            raise ValueError(f"two-sided projected tensor must have shape {expected}, got {tuple(projected.shape)}")
        return left @ projected @ right

    @staticmethod
    def _orthogonalize_update(update: Tensor, group: dict, original_shape: tuple[int, ...] | None = None) -> Tensor:
        if group["orthogonalization"] == "none":
            return update
        if group["orthogonalization"] == "svd":
            return SumoTrack._scale_orthogonalized_update(
                update,
                SumoTrack._orthogonalize_svd(update),
                group["orthogonalization_scale_mode"],
                original_shape,
            )
        if group["orthogonalization"] == "heavyball":
            return SumoTrack._orthogonalize_heavyball(update, group, original_shape)
        if group["orthogonalization"] == "aurora":
            return SumoTrack._orthogonalize_aurora(update, group, original_shape)
        raise AssertionError(f"unexpected orthogonalization mode: {group['orthogonalization']}")

    @staticmethod
    def _orthogonalize_svd(update: Tensor) -> Tensor:
        svd_input = update.float() if update.dtype in (torch.float16, torch.bfloat16) else update
        u, _s, vh = torch.linalg.svd(svd_input, full_matrices=False)
        return (u @ vh).to(dtype=update.dtype, device=update.device)

    @staticmethod
    def _orthogonalize_heavyball(update: Tensor, group: dict, original_shape: tuple[int, ...] | None) -> Tensor:
        if update.ndim > 2:
            work = SumoTrack._batched_newton_schulz5(update)
            if group["orthogonalization_scale_mode"] == "muon":
                work = SumoTrack._scale_orthogonalized_update(update, work, "muon", original_shape)
            return work
        work = update.clone()
        scale_mode = group["orthogonalization_scale_mode"]
        heavyball_utils.inplace_orthogonal_(
            work,
            mode=group["heavyball_orthogonalization_mode"],
            out=work,
            scale_mode="none" if scale_mode == "muon" else scale_mode,
        )
        if scale_mode == "muon":
            work = SumoTrack._scale_orthogonalized_update(update, work, scale_mode, original_shape)
        return work

    @staticmethod
    def _orthogonalize_aurora(update: Tensor, group: dict, original_shape: tuple[int, ...] | None) -> Tensor:
        aurora_update = SumoTrack._aurora_leverage_uniform_polar(
            update,
            pp_iterations=group["aurora_pp_iterations"],
            pp_beta=group["aurora_pp_beta"],
        )
        return SumoTrack._scale_orthogonalized_update(
            update,
            aurora_update,
            group["orthogonalization_scale_mode"],
            original_shape,
        )

    @staticmethod
    def _aurora_leverage_uniform_polar(update: Tensor, pp_iterations: int = 2, pp_beta: float = 0.5, eps: float = 1e-7) -> Tensor:
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
        if update.shape[-2] == update.shape[-1]:
            return SumoTrack._heavyball_polar(update)

        transposed = update.shape[-2] < update.shape[-1]
        work = update.mT if transposed else update
        work32 = work.float()
        rows, cols = work32.shape[-2:]
        target_row_sq = cols / rows
        diagonal = work32.norm(dim=-1, keepdim=True).clamp_min(eps).reciprocal()
        balanced = None
        for iteration in range(pp_iterations):
            balanced = SumoTrack._heavyball_polar(diagonal * work32).float()
            if iteration < pp_iterations - 1:
                row_sq = balanced.square().sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                diagonal = diagonal * (target_row_sq / row_sq).pow(pp_beta)
        assert balanced is not None
        result = balanced.mT if transposed else balanced
        return result.to(device=update.device, dtype=update.dtype)

    @staticmethod
    def _heavyball_polar(update: Tensor) -> Tensor:
        if update.ndim > 2:
            return SumoTrack._batched_newton_schulz5(update)
        work = update.clone()
        heavyball_utils.inplace_orthogonal_(work, mode="newtonschulz", out=work, scale_mode="none")
        return work

    @staticmethod
    def _batched_newton_schulz5(update: Tensor, eps: float = 1e-7) -> Tensor:
        if update.ndim < 2:
            raise ValueError(f"Newton-Schulz orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        work = update.float()
        work = work / work.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
        transposed = work.shape[-2] > work.shape[-1]
        x = work.mT if transposed else work

        for a, b, c in [
            (4.0848, -6.8946, 2.9270),
            (3.9505, -6.3029, 2.6377),
            (3.7418, -5.5913, 2.3037),
            (2.8769, -3.1427, 1.2046),
            (2.8366, -3.0525, 1.2012),
        ]:
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
            init_method=ProjectorInitMethod(group["subspace_init"]),
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
