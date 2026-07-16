from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

from .projector import ProjectionSide, SubspaceProjector
AURORA_PP_ITERATIONS = 1
AURORA_PP_BETA = 0.5
ORTHOGONALIZATION_SCALE_MODE = "muon"
NEWTON_SCHULZ_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
MIN_BASIS_UPDATE_STEP = 0.01


@dataclass
class MatrixUpdate:
    param: Tensor
    projector: SubspaceProjector
    projected_exp_avg: Tensor
    original_shape: tuple[int, ...]
    oja_tangent: Tensor | None = None
    raw_grad_norm: Tensor | None = None


class UsuiTrack(Optimizer):
    """Eager UsuiTrack baseline optimizer.

    Matrix parameters keep optimizer state in projected space: an orthonormal
    basis plus a projected first moment. UsuiTrack accepts only 2D parameters;
    callers own any bias, norm, or other fallback optimizer separately. The
    default one-state basis tracker conditions every full matrix gradient and
    updates its live basis on the configured cadence.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 32,
        side: ProjectionSide | str = ProjectionSide.AUTO,
        adafactor_beta2: float = 0.99,
        adafactor_eps: float = 1e-30,
        grad_clip_norm: float | None = 1.0,
        basis_update_interval: int = 1,
        consume_grad: bool = True,
        release_matrix_grads: bool = False,
        compile_tensor_kernels: bool = False,
        ecc: str | None = None,
        param_ecc: str | None = None,
    ) -> None:
        if ecc is not None or param_ecc is not None:
            raise NotImplementedError(
                "UsuiTrack does not yet support HeavyBall ECC/param-ECC. "
                "ECC requires HeavyBall's ChainOpt state hooks."
            )
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 <= beta < 1:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0 <= adafactor_beta2 < 1:
            raise ValueError(f"adafactor_beta2 must be in [0, 1), got {adafactor_beta2}")
        if adafactor_eps <= 0:
            raise ValueError(f"adafactor_eps must be positive, got {adafactor_eps}")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive when set, got {grad_clip_norm}")
        if basis_update_interval <= 0:
            raise ValueError(f"basis_update_interval must be positive, got {basis_update_interval}")
        if release_matrix_grads and not consume_grad:
            raise ValueError("release_matrix_grads requires consume_grad=True")

        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            side=ProjectionSide(side).value,
            adafactor_beta2=adafactor_beta2,
            adafactor_eps=adafactor_eps,
            grad_clip_norm=grad_clip_norm,
            consume_grad=consume_grad,
            compile_tensor_kernels=compile_tensor_kernels,
            basis_update_interval=basis_update_interval,
            matrix_step=0,
            basis_update_step=0,
        )
        super().__init__(params, defaults)
        self.diagnostics_enabled = False
        self.diagnostics_leverage_enabled = False
        self.diagnostics_basis_enabled = False
        self.diagnostics_aurora_health_enabled = False
        self.last_step_diagnostics: dict[str, float] = {}
        self._compiled_orthogonalize_update = torch.compile(UsuiTrack._orthogonalize_aurora_muon_tensor) if compile_tensor_kernels else None
        self._compiled_prepare_tracker_adafactor_right = (
            torch.compile(UsuiTrack._prepare_tracker_adafactor_right_tensors, dynamic=True)
            if compile_tensor_kernels
            else None
        )
        self._compiled_prepare_tracker_adafactor_left = (
            torch.compile(UsuiTrack._prepare_tracker_adafactor_left_tensors, dynamic=True)
            if compile_tensor_kernels
            else None
        )
        self._pending_matrix_updates: dict[Tensor, MatrixUpdate] = {}
        self._pending_diagnostics: dict[str, Any] | None = None
        self._matrix_grad_hook_handles = []
        self._matrix_param_groups: dict[Tensor, dict] = {}
        self.release_matrix_grads = release_matrix_grads
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim == 2:
                    self._matrix_param_groups[param] = group
        if release_matrix_grads:
            optimizer_ref = weakref.ref(self)

            def release_grad(param: Tensor) -> None:
                optimizer = optimizer_ref()
                if optimizer is not None:
                    optimizer.prepare(param)

            for group in self.param_groups:
                if not group["consume_grad"] and any(param.ndim == 2 for param in group["params"]):
                    raise ValueError("release_matrix_grads requires consume_grad=True for every matrix parameter group")
                for param in group["params"]:
                    if param.ndim == 2 and param.requires_grad:
                        self._matrix_grad_hook_handles.append(param.register_post_accumulate_grad_hook(release_grad))

    def add_param_group(self, param_group: dict) -> None:
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        try:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        "UsuiTrack only supports 2D matrix parameters; "
                        f"got shape {tuple(param.shape)}"
                    )
                if group["rank"] > min(param.shape):
                    raise ValueError(
                        f"rank {group['rank']} exceeds the smaller dimension "
                        f"of matrix parameter shape {tuple(param.shape)}"
                    )
        except Exception:
            self.param_groups.pop()
            raise

        matrix_param_groups = getattr(self, "_matrix_param_groups", None)
        if matrix_param_groups is not None:
            for param in group["params"]:
                matrix_param_groups[param] = group

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._pending_matrix_updates:
            raise RuntimeError(
                "cannot discard released matrix updates or explicitly prepared matrix updates with zero_grad(); preparation has already mutated "
                "optimizer state, so the pending updates must be consumed by step()"
        )
        super().zero_grad(set_to_none=set_to_none)
        if self.release_matrix_grads:
            self._pending_diagnostics = self._new_diagnostics()

    @torch.no_grad()
    def prepare(self, param: Tensor) -> None:
        """Consume and prepare one owned full matrix gradient exactly once."""

        group = self._matrix_group(param)
        if group is None:
            if not self._owns_param(param):
                raise ValueError("cannot prepare a parameter not owned by this optimizer")
            raise ValueError(f"prepare() only supports 2D matrix parameters, got shape {tuple(param.shape)}")
        if not group["consume_grad"]:
            raise RuntimeError("prepare() requires consume_grad=True for the parameter group")
        self._prepare_matrix_param(param, require_full_grad=True)

    @torch.no_grad()
    def _prepare_matrix_param(
        self,
        param: Tensor,
        require_full_grad: bool = False,
    ) -> MatrixUpdate:
        if param in self._pending_matrix_updates:
            raise RuntimeError(
                "matrix parameter is already prepared; prepare/release does not support gradient accumulation before step()"
            )
        grad = param.grad
        if require_full_grad and grad is None:
            raise RuntimeError("prepare() requires a live full matrix gradient")
        if grad is None:
            raise RuntimeError("matrix update requires a full grad")
        if grad is not None and grad.is_sparse:
            raise RuntimeError("UsuiTrack does not support sparse gradients")
        group = self._matrix_group(param)
        if group is None:
            raise ValueError("cannot prepare a matrix parameter not owned by this optimizer")
        if self._pending_diagnostics is None:
            self._pending_diagnostics = self._new_diagnostics()
        update = self._prepare_matrix_update(param, grad, group, self._pending_diagnostics)
        self._pending_matrix_updates[param] = update
        if group["consume_grad"]:
            param.grad = None
        return update

    def released_matrix_grad_norms(self) -> tuple[Tensor, ...]:
        """Raw full-gradient norms retained for telemetry after matrix grads are released."""

        return tuple(
            self._pending_matrix_updates[param].raw_grad_norm
            for group in self.param_groups
            for param in group["params"]
            if param in self._pending_matrix_updates
            and self._pending_matrix_updates[param].raw_grad_norm is not None
        )

    def _matrix_group(self, param: Tensor) -> dict | None:
        group = self._matrix_param_groups.get(param)
        if group is not None:
            return group
        if param.ndim != 2:
            return None
        for candidate_group in self.param_groups:
            if any(param is candidate for candidate in candidate_group["params"]):
                self._matrix_param_groups[param] = candidate_group
                return candidate_group
        return None

    def _owns_param(self, param: Tensor) -> bool:
        return any(param is candidate for group in self.param_groups for candidate in group["params"])

    def _validate_step_inputs(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                pending = param in self._pending_matrix_updates
                grad = param.grad
                if pending and grad is not None:
                    raise RuntimeError("a prepared matrix parameter cannot also have a new live gradient")
                if self.release_matrix_grads and grad is not None:
                    raise RuntimeError("release_matrix_grads requires matrix gradients to be produced by backward hooks")
                if grad is not None and grad.is_sparse:
                    raise RuntimeError("UsuiTrack does not support sparse gradients")

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None and (self.release_matrix_grads or self._pending_matrix_updates):
            raise RuntimeError("optimizer closures cannot run while matrix updates are pending")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._validate_step_inputs()
        diagnostics = self._pending_diagnostics if self._pending_diagnostics is not None else self._new_diagnostics()
        self._pending_diagnostics = diagnostics

        for group in self.param_groups:
            matrix_params = [p for p in group["params"] if p in self._pending_matrix_updates or p.grad is not None]
            matrix_updates = []

            for p in group["params"]:
                if p in self._pending_matrix_updates:
                    matrix_updates.append(self._pending_matrix_updates[p])
                    continue
                if p.grad is None:
                    continue
                matrix_updates.append(self._prepare_matrix_param(p))
            if matrix_updates:
                basis_update_due = self._basis_update_due(group)
                group["matrix_step"] += 1
                if basis_update_due:
                    group["basis_update_step"] += 1
            self._apply_basis_updates(matrix_updates, group, diagnostics)
            self._apply_matrix_update_buckets(matrix_updates, group, diagnostics)

        self.last_step_diagnostics = self._finalize_diagnostics(diagnostics)
        self._pending_matrix_updates.clear()
        self._pending_diagnostics = None
        return loss

    def _new_diagnostics(self) -> dict[str, Any] | None:
        if not self.diagnostics_enabled:
            return None
        # Logging contract: collect device tensors during the optimizer step and
        # move/reduce them to Python scalars only in `_finalize_diagnostics()`.
        # Per-parameter `.cpu()`/`.item()` calls are silent synchronization traps.
        return {
            "matrix_update_norm_sq": None,
            "nonfinite_grad_tensors": None,
            "projected_grad_norm_sum": None,
            "projected_grad_norm_tensors": 0,
            "projected_grad_to_moment_ratio_sum": None,
            "projected_grad_to_moment_ratio_tensors": 0,
            "matrix_params": 0,
            "projected_leverage_cv_sum": 0.0,
            "projected_leverage_min_ratio_sum": 0.0,
            "projected_leverage_max_ratio_sum": 0.0,
            "projected_leverage_tensors": 0,
            "rotation_angle_sum": 0.0,
            "basis_update_tensors": 0,
            "basis_capture_sum": None,
            "basis_capture_tensors": 0,
            "aurora_alignment_sum": 0.0,
            "aurora_erank_sum": 0.0,
            "aurora_erank_pct_sum": 0.0,
            "aurora_health_tensors": 0,
        }

    def _finalize_diagnostics(self, diagnostics: dict[str, Any] | None) -> dict[str, float]:
        if diagnostics is None:
            return {}
        matrix_norm_sq = diagnostics["matrix_update_norm_sq"]
        if matrix_norm_sq is None:
            matrix_norm_sq = torch.tensor(0.0)
        diagnostics["matrix_update_norm"] = float(matrix_norm_sq.sqrt().detach().cpu())
        diagnostics["update_norm"] = float(matrix_norm_sq.sqrt().detach().cpu())
        projected_grad_norm_sum = diagnostics.pop("projected_grad_norm_sum")
        projected_grad_norm_count = diagnostics.pop("projected_grad_norm_tensors")
        diagnostics["mean_projected_grad_norm"] = float((projected_grad_norm_sum / projected_grad_norm_count).detach().cpu()) if projected_grad_norm_count else float("nan")
        nonfinite = diagnostics["nonfinite_grad_tensors"]
        diagnostics["nonfinite_grad_tensors"] = float(nonfinite.detach().cpu()) if nonfinite is not None else 0.0
        count = diagnostics["projected_leverage_tensors"]
        diagnostics["mean_projected_leverage_cv"] = diagnostics["projected_leverage_cv_sum"] / count if count else float("nan")
        diagnostics["mean_projected_leverage_min_ratio"] = diagnostics["projected_leverage_min_ratio_sum"] / count if count else float("nan")
        diagnostics["mean_projected_leverage_max_ratio"] = diagnostics["projected_leverage_max_ratio_sum"] / count if count else float("nan")
        ratio_sum = diagnostics.pop("projected_grad_to_moment_ratio_sum")
        ratio_count = diagnostics.pop("projected_grad_to_moment_ratio_tensors")
        diagnostics["mean_projected_grad_to_moment_ratio"] = float((ratio_sum / ratio_count).detach().cpu()) if ratio_count else float("nan")
        capture_sum = diagnostics.pop("basis_capture_sum")
        capture_count = diagnostics.pop("basis_capture_tensors")
        diagnostics["mean_basis_capture"] = float((capture_sum / capture_count).detach().cpu()) if capture_count else float("nan")
        basis_count = diagnostics["basis_update_tensors"]
        diagnostics["mean_rotation_angle"] = diagnostics["rotation_angle_sum"] / basis_count if basis_count else float("nan")
        diagnostics["basis_update_tensors"] = float(basis_count)
        aurora_count = diagnostics["aurora_health_tensors"]
        diagnostics["mean_aurora_alignment"] = diagnostics["aurora_alignment_sum"] / aurora_count if aurora_count else float("nan")
        diagnostics["mean_aurora_erank"] = diagnostics["aurora_erank_sum"] / aurora_count if aurora_count else float("nan")
        diagnostics["mean_aurora_erank_pct"] = diagnostics["aurora_erank_pct_sum"] / aurora_count if aurora_count else float("nan")
        diagnostics["aurora_health_tensors"] = float(aurora_count)
        return diagnostics

    def _prepare_matrix_update(self, p: Tensor, grad: Tensor, group: dict, diagnostics: dict | None) -> MatrixUpdate:
        state = self.state[p]
        projector = self._projector_from_state(p, group, state)
        oja_tangent = None
        raw_grad_norm = None
        if self._can_use_initialized_tracker_adafactor_prepare(projector, state, group):
            return self._prepare_initialized_tracker_adafactor_update(
                p,
                grad,
                projector,
                state,
                group,
                diagnostics,
            )
        else:
            # Sync-free non-finite guard. A NaN/inf batch otherwise poisons every
            # downstream consumer at once: the clip scale (NaN norm -> NaN scale ->
            # whole grad NaN), adafactor's row/col vars, the tangent buffer, and
            # eventually the refresh SVD -- and an `isfinite().all()` branch would be
            # a device->host sync in the hot path. Zeroing non-finite elements drops
            # their contribution for one step instead. Visibility lives in the
            # diagnostics path (which already syncs at finalize), not here.
            if diagnostics is not None:
                nonfinite = (~torch.isfinite(grad)).any().detach()
                current = diagnostics["nonfinite_grad_tensors"]
                diagnostics["nonfinite_grad_tensors"] = nonfinite if current is None else current + nonfinite
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip the RAW gradient before it reaches adafactor. A blip batch (grad
            # norm spiking ~180x) otherwise poisons the adafactor row/col second
            # moment: grad_sq of the blip is ~30000x normal, and with beta2=0.99 that
            # spike decays over ~100 steps, over-dampening whole gradient directions
            # for the entire window -- the basis then tracks a *starved* residual and
            # each subsequent refresh adapts to the corruption (observed: a single
            # step-54 blip collapsed alignment for the whole back half of a 100-step
            # run). Clipping here protects adafactor's state, the basis refresh, and
            # the projection in one place -- upstream of everything, which is why the
            # projected-grad clip (downstream, moment-only) could not stop it.
            raw_grad_norm = grad.float().norm().detach()
            grad_clip_norm = group.get("grad_clip_norm")
            if grad_clip_norm is not None:
                clip_scale = (grad.new_tensor(float(grad_clip_norm)) / raw_grad_norm.clamp_min(1e-12)).clamp(max=1.0)
                grad = grad.mul(clip_scale)
            grad = self._adafactor_dampen_full_grad(grad, group, state)
            held_projected_grad = projector.project(grad) if projector.is_initialized else None
            if diagnostics is not None and held_projected_grad is not None:
                capture = held_projected_grad.float().norm() / grad.float().norm().clamp_min(1e-12)
                current_sum = diagnostics["basis_capture_sum"]
                diagnostics["basis_capture_sum"] = capture if current_sum is None else current_sum + capture
                diagnostics["basis_capture_tensors"] += 1
            if held_projected_grad is None:
                self._initialize_projector(projector, grad, state)
                projected_grad = projector.project(grad)
            else:
                # The held-frame projection supplies both the projected moment
                # update and Oja's covariance action. Basis motion is deferred
                # until every tangent can be batched in step().
                if self._basis_update_due(group):
                    oja_tangent = projector.oja_tangent(grad, projected=held_projected_grad)
                projected_grad = held_projected_grad

        projected_grad_norm = projected_grad.float().norm().detach()
        if diagnostics is not None:
            current_sum = diagnostics["projected_grad_norm_sum"]
            diagnostics["projected_grad_norm_sum"] = projected_grad_norm if current_sum is None else current_sum + projected_grad_norm
            diagnostics["projected_grad_norm_tensors"] += 1
        projected_exp_avg = state.get("projected_exp_avg")
        moment_norm = projected_exp_avg.float().norm().detach() if projected_exp_avg is not None else None
        if moment_norm is not None and diagnostics is not None:
            ratio = projected_grad_norm / moment_norm.clamp_min(1e-12)
            current_sum = diagnostics["projected_grad_to_moment_ratio_sum"]
            diagnostics["projected_grad_to_moment_ratio_sum"] = ratio.detach() if current_sum is None else current_sum + ratio.detach()
            diagnostics["projected_grad_to_moment_ratio_tensors"] += 1
        state["step"] = state.get("step", 0) + 1
        if projected_exp_avg is None:
            projected_exp_avg = torch.zeros_like(projected_grad)
        projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        state["projected_exp_avg"] = projected_exp_avg

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
            oja_tangent=oja_tangent,
            raw_grad_norm=raw_grad_norm,
        )

    @staticmethod
    def _can_use_initialized_tracker_adafactor_prepare(
        projector: SubspaceProjector,
        state: dict,
        _group: dict,
    ) -> bool:
        return (
            projector.is_initialized
            and UsuiTrack._basis_update_due(_group)
            and _group.get("grad_clip_norm") is not None
            and state.get("adafactor_row_var") is not None
            and state.get("adafactor_col_var") is not None
            and state.get("projected_exp_avg") is not None
        )

    def _prepare_initialized_tracker_adafactor_update(
        self,
        p: Tensor,
        grad: Tensor,
        projector: SubspaceProjector,
        state: dict,
        group: dict,
        diagnostics: dict | None,
    ) -> MatrixUpdate:
        if diagnostics is not None:
            nonfinite = (~torch.isfinite(grad)).any().detach()
            current = diagnostics["nonfinite_grad_tensors"]
            diagnostics["nonfinite_grad_tensors"] = nonfinite if current is None else current + nonfinite

        adafactor_step = state.get("adafactor_step", 0) + 1
        state["adafactor_step"] = adafactor_step
        state["step"] = state.get("step", 0) + 1
        basis = projector.basis
        assert basis is not None
        side = projector._basis_side()
        prepare = (
            self._compiled_prepare_tracker_adafactor_right
            if side is ProjectionSide.RIGHT
            else self._compiled_prepare_tracker_adafactor_left
        )
        if prepare is None:
            prepare = (
                self._prepare_tracker_adafactor_right_tensors
                if side is ProjectionSide.RIGHT
                else self._prepare_tracker_adafactor_left_tensors
            )
        conditioned_grad, oja_tangent, raw_grad_norm, projected_grad_norm, moment_norm = prepare(
            grad,
            basis,
            state["adafactor_row_var"],
            state["adafactor_col_var"],
            state["projected_exp_avg"],
            adafactor_step,
            float(group["grad_clip_norm"]),
            float(group["adafactor_beta2"]),
            float(group["adafactor_eps"]),
            float(group["beta"]),
        )
        projected_exp_avg = state["projected_exp_avg"]

        if diagnostics is not None:
            current_sum = diagnostics["projected_grad_norm_sum"]
            diagnostics["projected_grad_norm_sum"] = (
                projected_grad_norm if current_sum is None else current_sum + projected_grad_norm
            )
            diagnostics["projected_grad_norm_tensors"] += 1
            ratio = projected_grad_norm / moment_norm.clamp_min(1e-12)
            current_sum = diagnostics["projected_grad_to_moment_ratio_sum"]
            diagnostics["projected_grad_to_moment_ratio_sum"] = ratio if current_sum is None else current_sum + ratio
            diagnostics["projected_grad_to_moment_ratio_tensors"] += 1
            capture = projected_grad_norm / conditioned_grad.float().norm().clamp_min(1e-12)
            current_sum = diagnostics["basis_capture_sum"]
            diagnostics["basis_capture_sum"] = capture if current_sum is None else current_sum + capture
            diagnostics["basis_capture_tensors"] += 1

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
            oja_tangent=oja_tangent,
            raw_grad_norm=raw_grad_norm,
        )

    @staticmethod
    def _prepare_tracker_adafactor_right_tensors(
        grad: Tensor,
        basis: Tensor,
        row_var: Tensor,
        col_var: Tensor,
        projected_exp_avg: Tensor,
        adafactor_step: int,
        grad_clip_norm: float,
        adafactor_beta2: float,
        adafactor_eps: float,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        grad, raw_grad_norm = UsuiTrack._sanitize_and_clip_grad_tensors(grad, grad_clip_norm)
        conditioned_grad = UsuiTrack._adafactor_dampen_tensors(
            grad,
            row_var,
            col_var,
            adafactor_step,
            adafactor_beta2,
            adafactor_eps,
        )
        projected_grad = conditioned_grad @ basis.mT
        work = conditioned_grad.float()
        frame = basis.float().mT
        low = projected_grad.float()
        action = work.mT @ low
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        projected_grad_norm = low.norm().detach()
        moment_norm = projected_exp_avg.float().norm().detach()
        projected_exp_avg.mul_(beta).add_(projected_grad, alpha=1.0 - beta)
        return conditioned_grad, tangent, raw_grad_norm, projected_grad_norm, moment_norm

    @staticmethod
    def _prepare_tracker_adafactor_left_tensors(
        grad: Tensor,
        basis: Tensor,
        row_var: Tensor,
        col_var: Tensor,
        projected_exp_avg: Tensor,
        adafactor_step: int,
        grad_clip_norm: float,
        adafactor_beta2: float,
        adafactor_eps: float,
        beta: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        grad, raw_grad_norm = UsuiTrack._sanitize_and_clip_grad_tensors(grad, grad_clip_norm)
        conditioned_grad = UsuiTrack._adafactor_dampen_tensors(
            grad,
            row_var,
            col_var,
            adafactor_step,
            adafactor_beta2,
            adafactor_eps,
        )
        projected_grad = basis.mT @ conditioned_grad
        work = conditioned_grad.float()
        frame = basis.float()
        low = projected_grad.float()
        action = work @ low.mT
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        tangent = action - frame @ rayleigh
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        projected_grad_norm = low.norm().detach()
        moment_norm = projected_exp_avg.float().norm().detach()
        projected_exp_avg.mul_(beta).add_(projected_grad, alpha=1.0 - beta)
        return conditioned_grad, tangent, raw_grad_norm, projected_grad_norm, moment_norm

    @staticmethod
    def _sanitize_and_clip_grad_tensors(grad: Tensor, grad_clip_norm: float) -> tuple[Tensor, Tensor]:
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        raw_grad_norm = grad.float().norm().detach()
        clip_scale = (grad.new_tensor(grad_clip_norm) / raw_grad_norm.clamp_min(1e-12)).clamp(max=1.0)
        return grad.mul(clip_scale), raw_grad_norm

    @staticmethod
    def _adafactor_dampen_tensors(
        grad: Tensor,
        row_var: Tensor,
        col_var: Tensor,
        step: int,
        beta2: float,
        eps: float,
    ) -> Tensor:
        grad32 = grad.float()
        grad_sq = grad32.square() + eps
        row_var.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1.0 - beta2)
        col_var.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1.0 - beta2)
        bias_correction = 1.0 - beta2**step
        row_hat = row_var / bias_correction
        col_hat = col_var / bias_correction
        # Reconstruct the inverse factored RMS as two broadcast vectors rather
        # than a second matrix-sized tensor, then restore the raw gradient RMS.
        # The latter keeps Oja's quadratic covariance scale comparable to the
        # unconditioned gradient while retaining Adafactor's SNR reweighting.
        mean_row = row_hat.mean().clamp_min(eps)
        row_scale = (mean_row / row_hat.clamp_min(eps)).sqrt().unsqueeze(1)
        col_scale = col_hat.clamp_min(eps).rsqrt().unsqueeze(0)
        dampened = grad32 * row_scale * col_scale
        grad_rms = grad32.square().mean().sqrt().clamp_min(eps)
        return (dampened * grad_rms).to(dtype=grad.dtype)

    @staticmethod
    def _adafactor_dampen_full_grad(grad: Tensor, group: dict, state: dict) -> Tensor:
        """Adafactor-style row/col factored second moment on the full gradient,
        applied before basis tracking and before projection so the same dampened
        gradient feeds both consumers, then still passed through the same
        first-moment EMA as ema mode.
        """

        beta2 = group["adafactor_beta2"]
        eps = group["adafactor_eps"]
        step = state.get("adafactor_step", 0) + 1
        state["adafactor_step"] = step

        row_var = state.get("adafactor_row_var")
        col_var = state.get("adafactor_col_var")
        if row_var is None:
            row_var = torch.zeros(grad.shape[0], device=grad.device, dtype=torch.float32)
            col_var = torch.zeros(grad.shape[1], device=grad.device, dtype=torch.float32)
        state["adafactor_row_var"] = row_var
        state["adafactor_col_var"] = col_var
        return UsuiTrack._adafactor_dampen_tensors(grad, row_var, col_var, step, beta2, eps)

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
                    self._orthogonalize_update_runtime(
                        bucket_entries[0].projected_exp_avg,
                        group,
                        bucket_entries[0].original_shape,
                    )
                ]
            else:
                stacked = torch.stack([entry.projected_exp_avg for entry in bucket_entries])
                stacked_update_hats = self._orthogonalize_update_runtime(stacked, group, bucket_entries[0].original_shape)
                update_hats = list(stacked_update_hats.unbind(0))
            if diagnostics is not None and self.diagnostics_aurora_health_enabled:
                self._accumulate_aurora_health(diagnostics, bucket_entries, update_hats)
            for entry, update_hat in zip(bucket_entries, update_hats, strict=True):
                self._apply_matrix_update(entry, update_hat, group, diagnostics)

    @staticmethod
    def _accumulate_aurora_health(
        diagnostics: dict,
        entries: list[MatrixUpdate],
        update_hats: list[Tensor],
        eps: float = 1e-12,
    ) -> None:
        moments = torch.stack([entry.projected_exp_avg for entry in entries]).float()
        updates = torch.stack(update_hats).float()
        inner = (moments * updates).sum(dim=(-2, -1))
        alignment = inner / (moments.norm(dim=(-2, -1)) * updates.norm(dim=(-2, -1))).clamp_min(eps)

        gram = moments @ moments.mT if moments.shape[-2] <= moments.shape[-1] else moments.mT @ moments
        singular_values = torch.linalg.eigvalsh(gram).clamp_min(0.0).sqrt()
        probabilities = singular_values / singular_values.sum(dim=-1, keepdim=True).clamp_min(eps)
        entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=-1)
        effective_rank = entropy.exp()

        diagnostics["aurora_alignment_sum"] += alignment.sum()
        diagnostics["aurora_erank_sum"] += effective_rank.sum()
        diagnostics["aurora_erank_pct_sum"] += (effective_rank / min(moments.shape[-2:])).sum()
        diagnostics["aurora_health_tensors"] += len(entries)

    def _apply_basis_updates(self, entries: list[MatrixUpdate], group: dict, diagnostics: dict | None) -> None:
        pending = [entry for entry in entries if entry.oja_tangent is not None]
        if not pending:
            return

        buckets: dict[tuple, list[MatrixUpdate]] = {}
        for entry in pending:
            tangent = entry.oja_tangent
            assert tangent is not None
            key = (tangent.device, tangent.dtype, tangent.shape[1])
            buckets.setdefault(key, []).append(entry)

        step_size = self._basis_update_step_size(group)
        record_rotation = diagnostics is not None and self.diagnostics_basis_enabled
        for bucket_entries in buckets.values():
            tangents = [entry.oja_tangent for entry in bucket_entries]
            assert all(tangent is not None for tangent in tangents)
            grams = torch.stack([tangent.mT @ tangent for tangent in tangents if tangent is not None])
            grams = 0.5 * (grams + grams.mT)
            eigenvalues, eigenvectors = torch.linalg.eigh(grams)
            geometry_buckets: dict[tuple, list[int]] = {}
            for index, entry in enumerate(bucket_entries):
                tangent = entry.oja_tangent
                assert tangent is not None
                geometry_buckets.setdefault((tuple(tangent.shape), entry.projector._basis_side()), []).append(index)
            for (_shape, side), indices in geometry_buckets.items():
                selected_entries = [bucket_entries[index] for index in indices]
                frames = torch.stack([entry.projector.canonical_basis() for entry in selected_entries])
                selected_tangents = torch.stack([entry.oja_tangent for entry in selected_entries if entry.oja_tangent is not None])
                selected_values = eigenvalues[indices]
                selected_vectors = eigenvectors[indices]
                new_frames = SubspaceProjector.oja_geodesic_from_eigh(
                    frames,
                    selected_tangents,
                    selected_values,
                    selected_vectors,
                    step_size,
                )
                rotation_angles = None
                if record_rotation:
                    rotation_angles = (step_size * selected_values.clamp_min(0.0).sqrt()).abs().sum(dim=-1).detach().cpu()
                for local_index, (entry, new_frame) in enumerate(zip(selected_entries, new_frames, strict=True)):
                    basis = new_frame.mT if side is ProjectionSide.RIGHT else new_frame
                    entry.projector.basis = basis.to(
                        device=entry.projector.basis.device,
                        dtype=entry.projector.basis.dtype,
                    ).contiguous()
                    entry.projector.resolved_side = side
                    state = self.state[entry.param]
                    if rotation_angles is not None and diagnostics is not None:
                        entry.projector.last_rotation_angle = float(rotation_angles[local_index])
                        self._record_basis_motion(diagnostics, entry.projector)
                    state["basis"] = entry.projector.basis
                    state["projection_side_is_right"] = side is ProjectionSide.RIGHT

    @staticmethod
    def _basis_update_step_size(group: dict) -> float:
        return max(MIN_BASIS_UPDATE_STEP, 1.0 / group["basis_update_step"])

    @staticmethod
    def _basis_update_due(group: dict) -> bool:
        return (group["matrix_step"] + 1) % group["basis_update_interval"] == 0

    def _apply_matrix_update(self, entry: MatrixUpdate, update_hat: Tensor, group: dict, diagnostics: dict | None) -> None:
        if diagnostics is not None:
            if self.diagnostics_leverage_enabled:
                leverage_cv, min_ratio, max_ratio = self._large_axis_leverage_stats(update_hat)
                diagnostics["projected_leverage_cv_sum"] += leverage_cv
                diagnostics["projected_leverage_min_ratio_sum"] += min_ratio
                diagnostics["projected_leverage_max_ratio_sum"] += max_ratio
                diagnostics["projected_leverage_tensors"] += 1
        update = entry.projector.project_back(update_hat).to(dtype=entry.param.dtype)

        if group["weight_decay"]:
            entry.param.mul_(1.0 - group["lr"] * group["weight_decay"])
        if diagnostics is not None:
            update_norm_sq = (update.float().norm() * group["lr"]).square().detach()
            diagnostics["matrix_update_norm_sq"] = update_norm_sq if diagnostics["matrix_update_norm_sq"] is None else diagnostics["matrix_update_norm_sq"] + update_norm_sq
            diagnostics["matrix_params"] += entry.param.numel()
        entry.param.add_(update, alpha=-group["lr"])

    def _initialize_projector(
        self,
        projector: SubspaceProjector,
        grad: Tensor,
        state: dict,
    ) -> None:
        projector.fit(grad)
        state["basis"] = projector.basis
        resolved_side = projector.resolved_side if projector.resolved_side is not None else projector.side
        state["projection_side_is_right"] = resolved_side is ProjectionSide.RIGHT
        old_projected_exp_avg = state.get("projected_exp_avg")
        if old_projected_exp_avg is not None:
            projected_shape = tuple(projector.project(grad).shape)
            if tuple(old_projected_exp_avg.shape) != projected_shape:
                state.pop("projected_exp_avg", None)

    @staticmethod
    def _expected_projected_grad_shape(p: Tensor, projector: SubspaceProjector) -> tuple[int, int]:
        if projector.basis is None:
            raise RuntimeError("basis is not initialized")
        side = projector.resolved_side if projector.resolved_side is not None else projector.side
        if side is ProjectionSide.AUTO:
            side = projector.effective_side(p)
        if side is ProjectionSide.RIGHT:
            return (p.shape[0], projector.basis.shape[0])
        return (projector.basis.shape[1], p.shape[1])

    @staticmethod
    def _accumulate_basis_diagnostics(diagnostics: dict, rotation_angle: float) -> None:
        diagnostics["rotation_angle_sum"] += rotation_angle
        diagnostics["basis_update_tensors"] += 1

    def _record_basis_motion(self, diagnostics: dict, projector: SubspaceProjector) -> None:
        self._accumulate_basis_diagnostics(diagnostics, projector.last_rotation_angle)

    @staticmethod
    def _orthogonalize_update(update: Tensor, _group: dict, original_shape: tuple[int, ...] | None = None) -> Tensor:
        return UsuiTrack._orthogonalize_aurora(update, original_shape)

    def _orthogonalize_update_runtime(self, update: Tensor, group: dict, original_shape: tuple[int, ...] | None = None) -> Tensor:
        if self._compiled_orthogonalize_update is None:
            return self._orthogonalize_update(update, group, original_shape)
        if original_shape is None or len(original_shape) < 2:
            raise ValueError("compiled UsuiTrack orthogonalization requires the original parameter shape")
        rows = int(original_shape[0])
        cols = int(math.prod(original_shape[1:]))
        return self._compiled_orthogonalize_update(
            update,
            rows,
            cols,
        )

    @staticmethod
    def _orthogonalize_aurora_muon_tensor(
        update: Tensor,
        original_rows: int,
        original_cols: int,
    ) -> Tensor:
        aurora_update = UsuiTrack._aurora_leverage_uniform_polar(update)
        return aurora_update * math.sqrt(max(1.0, original_rows / original_cols))

    @staticmethod
    def _orthogonalize_aurora(update: Tensor, original_shape: tuple[int, ...] | None) -> Tensor:
        aurora_update = UsuiTrack._aurora_leverage_uniform_polar(update)
        return UsuiTrack._scale_orthogonalized_update(
            update,
            aurora_update,
            ORTHOGONALIZATION_SCALE_MODE,
            original_shape,
        )

    @staticmethod
    def _aurora_leverage_uniform_polar(
        update: Tensor,
        eps: float = 1e-7,
    ) -> Tensor:
        """Aurora-style leverage-uniform polar direction for rectangular projected moments.

        UsuiTrack owns momentum, LR, weight decay, and full-matrix Muon scaling. This
        helper extracts only Aurora's rectangular direction map: diagonally
        precondition a non-square matrix before polar/NS so the large-side row
        leverage approaches the Stiefel target. For wide matrices, transpose to
        tall form, balance, then transpose back, matching Aurora's convention.
        """

        if update.ndim < 2:
            raise ValueError(f"Aurora orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        if update.shape[-2] == update.shape[-1]:
            return UsuiTrack._heavyball_polar(update)

        transposed = update.shape[-2] < update.shape[-1]
        work = update.mT if transposed else update
        work32 = work.float()
        rows, cols = work32.shape[-2:]
        target_row_sq = cols / rows
        diagonal = work32.norm(dim=-1, keepdim=True).clamp_min(eps).reciprocal()
        balanced = None
        for iteration in range(AURORA_PP_ITERATIONS):
            balanced = UsuiTrack._heavyball_polar(diagonal * work32).float()
            if iteration < AURORA_PP_ITERATIONS - 1:
                row_sq = balanced.square().sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                diagonal = diagonal * (target_row_sq / row_sq).pow(AURORA_PP_BETA)
        assert balanced is not None
        result = balanced.mT if transposed else balanced
        return result.to(device=update.device, dtype=update.dtype)

    @staticmethod
    def _heavyball_polar(update: Tensor) -> Tensor:
        return UsuiTrack._batched_newton_schulz(update)

    @staticmethod
    def _batched_newton_schulz(update: Tensor, eps: float = 1e-7) -> Tensor:
        if update.ndim < 2:
            raise ValueError(f"Newton-Schulz orthogonalization expects at least 2D input, got shape {tuple(update.shape)}")
        work = update.float()
        work = work / work.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
        transposed = work.shape[-2] > work.shape[-1]
        x = work.mT if transposed else work

        for a, b, c in NEWTON_SCHULZ_COEFFICIENTS:
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
    def _aurora_alignment(pre_aurora: Tensor, update_hat: Tensor, eps: float = 1e-12) -> float:
        """Cosine similarity between the pre-Aurora tensor and Aurora's orthogonalized
        output, in Frobenius inner-product terms: ``<M, O>_F / (||M||_F ||O||_F)``.

        Newton-Schulz/polar orthogonalization always lands on a semi-orthogonal
        matrix regardless of input quality, so the output's own properties cannot
        tell you whether the input was worth orthogonalizing. This measures how
        much Aurora had to distort the input direction to reach the manifold: high
        alignment means the pre-Aurora tensor was already well-shaped: low
        alignment means Aurora is fighting a noisy or ill-conditioned input.
        """

        m = pre_aurora.float()
        o = update_hat.float()
        inner = (m * o).sum()
        denom = (m.norm() * o.norm()).clamp_min(eps)
        return float((inner / denom).detach().cpu())

    @staticmethod
    def _effective_rank(matrix: Tensor, eps: float = 1e-12) -> float:
        """Spectral effective rank (Roy & Vetterli 2007): ``exp(-sum(p_i log p_i))``
        where ``p_i`` are the matrix's singular values normalized to sum to 1.

        A single dominant singular value (rank-1-ish, e.g. one noisy gradient
        direction) gives erank near 1. A flat spectrum across all r singular
        values (using the full available rank) gives erank near r. Computed via
        eigh on the smaller-side Gram matrix (eigenvalues are squared singular
        values) rather than a full SVD, matching the cost profile Newton-Schulz
        already uses internally for this same rank-sized tensor.
        """

        if matrix.ndim != 2 or min(matrix.shape) == 0:
            return float("nan")
        work = matrix.float()
        gram = work @ work.mT if work.shape[-2] <= work.shape[-1] else work.mT @ work
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        singular_values = eigenvalues.sqrt()
        total = singular_values.sum().clamp_min(eps)
        p = singular_values / total
        # p_i log p_i -> 0 as p_i -> 0; clamp inside log to avoid log(0) on exact zeros.
        entropy = -(p * p.clamp_min(eps).log()).sum()
        return float(entropy.exp().detach().cpu())

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
