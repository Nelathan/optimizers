from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim import _functional as torch_optim_functional

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


class UsuiTrack(Optimizer):
    """Eager UsuiTrack baseline optimizer.

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
        basis_init: str = "eigh",
        moment_mode: str = "adafactor_ema",
        adafactor_beta2: float = 0.99,
        adafactor_eps: float = 1e-30,
        grad_clip_norm: float | None = 2.5,
        grassmann_step_size: float = 0.25,
        grassmann_rotate_rank: int | None = None,
        grassmann_aim: str = "eigh",
        basis_refresh_interval: int = 100,
        aurora_pp_iterations: int = AURORA_PP_ITERATIONS,
        polar_ns_steps: int = len(NEWTON_SCHULZ_COEFFICIENTS),
        projected_grad_clip_norm: float | None = None,
        projected_grad_clip_ratio: float | None = None,
        consume_grad: bool = True,
        compile_tensor_kernels: bool = False,
        ecc: str | None = None,
        param_ecc: str | None = None,
    ) -> None:
        if ecc is not None or param_ecc is not None:
            raise NotImplementedError(
                "UsuiTrack does not yet support HeavyBall ECC/param-ECC. "
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
        if moment_mode not in ("ema", "adafactor_ema"):
            raise ValueError(f"moment_mode must be one of 'ema', 'adafactor_ema', got {moment_mode!r}")
        if not 0 <= adafactor_beta2 < 1:
            raise ValueError(f"adafactor_beta2 must be in [0, 1), got {adafactor_beta2}")
        if adafactor_eps <= 0:
            raise ValueError(f"adafactor_eps must be positive, got {adafactor_eps}")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(f"grad_clip_norm must be positive when set, got {grad_clip_norm}")
        if grassmann_step_size <= 0:
            raise ValueError(f"grassmann_step_size must be positive, got {grassmann_step_size}")
        if grassmann_rotate_rank is not None and grassmann_rotate_rank < 1:
            raise ValueError(f"grassmann_rotate_rank must be None (all planes) or >= 1, got {grassmann_rotate_rank}")
        if grassmann_aim not in ("tangent", "eigh"):
            raise ValueError(f"grassmann_aim must be one of 'tangent', 'eigh', got {grassmann_aim!r}")
        if basis_refresh_interval <= 0:
            raise ValueError(f"basis_refresh_interval must be positive, got {basis_refresh_interval}")
        if aurora_pp_iterations <= 0:
            raise ValueError(f"aurora_pp_iterations must be positive, got {aurora_pp_iterations}")
        if not 1 <= polar_ns_steps <= len(NEWTON_SCHULZ_COEFFICIENTS):
            raise ValueError(f"polar_ns_steps must be in [1, {len(NEWTON_SCHULZ_COEFFICIENTS)}], got {polar_ns_steps}")
        if projected_grad_clip_norm is not None and projected_grad_clip_norm <= 0:
            raise ValueError(f"projected_grad_clip_norm must be positive when set, got {projected_grad_clip_norm}")
        if projected_grad_clip_ratio is not None and projected_grad_clip_ratio <= 0:
            raise ValueError(f"projected_grad_clip_ratio must be positive when set, got {projected_grad_clip_ratio}")

        defaults = dict(
            lr=lr,
            beta=beta,
            fallback_betas=fallback_betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            side=ProjectionSide(side).value,
            basis_init=basis_init,
            moment_mode=moment_mode,
            adafactor_beta2=adafactor_beta2,
            adafactor_eps=adafactor_eps,
            grad_clip_norm=grad_clip_norm,
            grassmann_step_size=grassmann_step_size,
            grassmann_rotate_rank=grassmann_rotate_rank,
            grassmann_aim=grassmann_aim,
            basis_refresh_interval=basis_refresh_interval,
            aurora_pp_iterations=aurora_pp_iterations,
            polar_ns_steps=polar_ns_steps,
            projected_grad_clip_norm=projected_grad_clip_norm,
            projected_grad_clip_ratio=projected_grad_clip_ratio,
            consume_grad=consume_grad,
            compile_tensor_kernels=compile_tensor_kernels,
            basis_refresh_step=0,
        )
        super().__init__(params, defaults)
        self.diagnostics_enabled = False
        self.diagnostics_leverage_enabled = False
        self.diagnostics_basis_enabled = False
        self.diagnostics_aurora_health_enabled = False
        self.last_step_diagnostics: dict[str, float] = {}
        self._compiled_orthogonalize_update = torch.compile(UsuiTrack._orthogonalize_aurora_muon_tensor) if compile_tensor_kernels else None
        self._queued_projected_grads: dict[Tensor, Tensor] = {}

    @torch.no_grad()
    def queue_projected_grad(self, param: Tensor, projected_grad: Tensor) -> None:
        """Queue an already-projected matrix gradient for the next ``step``.

        This is the explicit ingress for custom projected-activation backward
        paths. The queued tensor must already live in UsuiTrack's current basis;
        basis initialization and refresh still require a full matrix gradient.
        """

        if not self._owns_param(param):
            raise ValueError("cannot queue a projected gradient for a parameter not owned by this optimizer")
        if param.ndim != 2:
            raise ValueError(f"projected gradients are only supported for 2D matrix parameters, got shape {tuple(param.shape)}")
        if projected_grad.ndim != 2:
            raise ValueError(f"projected gradient must be 2D, got shape {tuple(projected_grad.shape)}")
        if projected_grad.is_sparse:
            raise RuntimeError("UsuiTrack does not support sparse projected gradients")

        projected_grad = projected_grad.detach()
        existing = self._queued_projected_grads.get(param)
        if existing is None:
            self._queued_projected_grads[param] = projected_grad
        else:
            existing.add_(projected_grad)

    def clear_projected_grads(self) -> None:
        """Clear queued projected gradients without touching ``Parameter.grad``."""

        self._queued_projected_grads.clear()

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.clear_projected_grads()

    def _owns_param(self, param: Tensor) -> bool:
        return any(param is candidate for group in self.param_groups for candidate in group["params"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        diagnostics = self._new_diagnostics()

        for group in self.param_groups:
            matrix_params = [p for p in group["params"] if p.ndim == 2 and (p.grad is not None or p in self._queued_projected_grads)]
            refresh_ids = self._refresh_param_ids(group, matrix_params)
            matrix_updates = []

            for p in group["params"]:
                has_queued_projected_grad = p in self._queued_projected_grads
                if p.grad is None and not has_queued_projected_grad:
                    continue
                grad = p.grad
                if grad is not None and grad.is_sparse:
                    raise RuntimeError("UsuiTrack does not support sparse gradients")
                if p.ndim == 2:
                    matrix_updates.append(self._prepare_matrix_update(p, grad, group, id(p) in refresh_ids, diagnostics))
                    if group["consume_grad"]:
                        p.grad = None
                else:
                    if has_queued_projected_grad:
                        raise RuntimeError("queued projected gradients are only supported for 2D matrix parameters")
                    assert grad is not None
                    self._step_fallback_param(p, grad, group, diagnostics)
                    if group["consume_grad"]:
                        p.grad = None
            self._apply_matrix_update_buckets(matrix_updates, group, diagnostics)

        self.last_step_diagnostics = self._finalize_diagnostics(diagnostics)

        return loss

    def _new_diagnostics(self) -> dict[str, Any] | None:
        if not self.diagnostics_enabled:
            return None
        # Logging contract: collect device tensors during the optimizer step and
        # move/reduce them to Python scalars only in `_finalize_diagnostics()`.
        # Per-parameter `.cpu()`/`.item()` calls are silent synchronization traps.
        return {
            "matrix_update_norm_sq": None,
            "fallback_update_norm_sq": None,
            "nonfinite_grad_tensors": None,
            "projected_grad_max_norm": None,
            "projected_grad_ratio_values": [],
            "matrix_params": 0,
            "fallback_params": 0,
            "projected_leverage_cv_sum": 0.0,
            "projected_leverage_min_ratio_sum": 0.0,
            "projected_leverage_max_ratio_sum": 0.0,
            "projected_leverage_tensors": 0,
            "rotation_angle_sum": 0.0,
            "tangent_sigma_max_sum": 0.0,
            "basis_refresh_tensors": 0,
            "eigh_target_self_angle_sum": 0.0,
            "eigh_target_probe_tensors": 0,
            "eigh_target_cutoff_ratio_sum": 0.0,
            "eigh_target_cutoff_tensors": 0,
            "basis_lag_mean_angle_sum": 0.0,
            "basis_lag_top_angle_sum": 0.0,
            "basis_lag_tensors": 0,
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
        fallback_norm_sq = diagnostics["fallback_update_norm_sq"]
        if matrix_norm_sq is None and fallback_norm_sq is None:
            matrix_norm_sq = fallback_norm_sq = torch.tensor(0.0)
        elif matrix_norm_sq is None:
            matrix_norm_sq = fallback_norm_sq.new_zeros(())
        elif fallback_norm_sq is None:
            fallback_norm_sq = matrix_norm_sq.new_zeros(())
        diagnostics["matrix_update_norm"] = float(matrix_norm_sq.sqrt().detach().cpu())
        diagnostics["fallback_update_norm"] = float(fallback_norm_sq.sqrt().detach().cpu())
        diagnostics["update_norm"] = float((matrix_norm_sq + fallback_norm_sq).sqrt().detach().cpu())
        projected_grad_max_norm = diagnostics["projected_grad_max_norm"]
        diagnostics["projected_grad_max_norm"] = float(projected_grad_max_norm.detach().cpu()) if projected_grad_max_norm is not None else float("nan")
        nonfinite = diagnostics["nonfinite_grad_tensors"]
        diagnostics["nonfinite_grad_tensors"] = float(nonfinite.detach().cpu()) if nonfinite is not None else 0.0
        count = diagnostics["projected_leverage_tensors"]
        diagnostics["mean_projected_leverage_cv"] = diagnostics["projected_leverage_cv_sum"] / count if count else float("nan")
        diagnostics["mean_projected_leverage_min_ratio"] = diagnostics["projected_leverage_min_ratio_sum"] / count if count else float("nan")
        diagnostics["mean_projected_leverage_max_ratio"] = diagnostics["projected_leverage_max_ratio_sum"] / count if count else float("nan")
        ratio_values = diagnostics.pop("projected_grad_ratio_values")
        if ratio_values:
            ratio_tensor = torch.stack(ratio_values)
            diagnostics["projected_grad_p90_to_moment_ratio"] = float(torch.quantile(ratio_tensor.float(), 0.9).detach().cpu())
        else:
            diagnostics["projected_grad_p90_to_moment_ratio"] = float("nan")
        capture_sum = diagnostics.pop("basis_capture_sum")
        capture_count = diagnostics.pop("basis_capture_tensors")
        diagnostics["mean_basis_capture"] = float((capture_sum / capture_count).detach().cpu()) if capture_count else float("nan")
        basis_count = diagnostics["basis_refresh_tensors"]
        diagnostics["mean_rotation_angle"] = diagnostics["rotation_angle_sum"] / basis_count if basis_count else float("nan")
        diagnostics["mean_tangent_sigma_max"] = diagnostics["tangent_sigma_max_sum"] / basis_count if basis_count else float("nan")
        diagnostics["basis_refresh_tensors"] = float(basis_count)
        probe_count = diagnostics.pop("eigh_target_probe_tensors")
        diagnostics["mean_eigh_target_self_angle"] = diagnostics.pop("eigh_target_self_angle_sum") / probe_count if probe_count else float("nan")
        cutoff_count = diagnostics.pop("eigh_target_cutoff_tensors")
        diagnostics["mean_eigh_target_cutoff_ratio"] = diagnostics.pop("eigh_target_cutoff_ratio_sum") / cutoff_count if cutoff_count else float("nan")
        lag_count = diagnostics.pop("basis_lag_tensors")
        diagnostics["mean_basis_lag_angle"] = diagnostics.pop("basis_lag_mean_angle_sum") / lag_count if lag_count else float("nan")
        diagnostics["mean_basis_lag_top_angle"] = diagnostics.pop("basis_lag_top_angle_sum") / lag_count if lag_count else float("nan")
        aurora_count = diagnostics["aurora_health_tensors"]
        diagnostics["mean_aurora_alignment"] = diagnostics["aurora_alignment_sum"] / aurora_count if aurora_count else float("nan")
        diagnostics["mean_aurora_erank"] = diagnostics["aurora_erank_sum"] / aurora_count if aurora_count else float("nan")
        diagnostics["mean_aurora_erank_pct"] = diagnostics["aurora_erank_pct_sum"] / aurora_count if aurora_count else float("nan")
        diagnostics["aurora_health_tensors"] = float(aurora_count)
        return diagnostics

    def _refresh_param_ids(self, group: dict, matrix_params: list[Tensor]) -> set[int]:
        if not matrix_params:
            return set()
        step = group["basis_refresh_step"]
        group["basis_refresh_step"] = step + 1
        interval = group["basis_refresh_interval"]
        refresh_offsets = group.get("basis_refresh_offsets")
        if step > 0:
            if refresh_offsets is None:
                if step % interval != 0:
                    return set()
                return {id(param) for param in matrix_params}
            if step < interval:
                return set()
            return {
                id(param)
                for param in matrix_params
                if (step - refresh_offsets.get(id(param), 0)) % interval == 0
            }
        return set()

    def _prepare_matrix_update(self, p: Tensor, grad: Tensor | None, group: dict, refresh_basis: bool, diagnostics: dict | None) -> MatrixUpdate:
        queued_projected_grad = self._queued_projected_grads.pop(p, None)
        if grad is not None and queued_projected_grad is not None:
            raise RuntimeError("a matrix parameter cannot have both a full grad and a queued projected grad")

        state = self.state[p]
        projector = self._projector_from_state(p, group, state)
        if queued_projected_grad is not None:
            if group["moment_mode"] == "adafactor_ema":
                raise RuntimeError("adafactor_ema moment_mode dampens the full gradient before projection and is incompatible with queued projected gradients")
            if not projector.is_initialized:
                raise RuntimeError("queued projected gradients require an initialized UsuiTrack basis; run a full-gradient step first")
            if refresh_basis:
                raise RuntimeError("queued projected gradients cannot refresh a UsuiTrack basis; provide a full matrix gradient on refresh steps")
            expected_shape = self._expected_projected_grad_shape(p, projector)
            if tuple(queued_projected_grad.shape) != expected_shape:
                raise ValueError(
                    f"queued projected gradient shape {tuple(queued_projected_grad.shape)} does not match expected {expected_shape}"
                )
            if queued_projected_grad.device != p.device:
                raise ValueError(f"queued projected gradient device {queued_projected_grad.device} does not match parameter device {p.device}")
            projected_grad = queued_projected_grad
        else:
            if grad is None:
                raise RuntimeError("matrix update requires either a full grad or a queued projected grad")
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
            grad_clip_norm = group.get("grad_clip_norm")
            if grad_clip_norm is not None:
                raw_norm = grad.float().norm()
                clip_scale = (grad.new_tensor(float(grad_clip_norm)) / raw_norm.clamp_min(1e-12)).clamp(max=1.0)
                grad = grad.mul(clip_scale)
            moment_mode = group["moment_mode"]
            if moment_mode == "adafactor_ema":
                grad = self._adafactor_dampen_full_grad(grad, group, state)
            if not projector.is_initialized or refresh_basis:
                self._refresh_projector(projector, grad, group, state, diagnostics)
            projected_grad = projector.project(grad)

        projected_grad_norm = projected_grad.float().norm().detach()
        if diagnostics is not None:
            current_max = diagnostics["projected_grad_max_norm"]
            diagnostics["projected_grad_max_norm"] = projected_grad_norm if current_max is None else torch.maximum(current_max, projected_grad_norm)
            # Fit/capture: fraction of the (dampened) gradient the basis captures,
            # ||Q^T g|| / ||g||. This is the basis-QUALITY readout; sigma is only
            # contact/steepness and is non-monotone in quality (sigma=0 both for a
            # perfect basis and one orthogonal to the signal).
            if grad is not None:
                capture = projected_grad_norm / grad.float().norm().clamp_min(1e-12)
                current_sum = diagnostics["basis_capture_sum"]
                diagnostics["basis_capture_sum"] = capture if current_sum is None else current_sum + capture
                diagnostics["basis_capture_tensors"] += 1
        projected_exp_avg = state.get("projected_exp_avg")
        moment_norm = projected_exp_avg.float().norm().detach() if projected_exp_avg is not None else None
        if moment_norm is not None and diagnostics is not None:
            ratio = projected_grad_norm / moment_norm.clamp_min(1e-12)
            diagnostics["projected_grad_ratio_values"].append(ratio.detach())
        clip_norm = group.get("projected_grad_clip_norm")
        clip_ratio = group.get("projected_grad_clip_ratio")
        allowed_norm = None
        if clip_norm is not None:
            allowed_norm = projected_grad_norm.new_tensor(float(clip_norm))
        if clip_ratio is not None and moment_norm is not None:
            ratio_allowed = moment_norm * float(clip_ratio)
            allowed_norm = ratio_allowed if allowed_norm is None else torch.minimum(allowed_norm, ratio_allowed)
        if allowed_norm is not None:
            clip_scale = (allowed_norm / projected_grad_norm.clamp_min(1e-12)).clamp(max=1.0)
            projected_grad = projected_grad.mul(clip_scale.to(device=projected_grad.device, dtype=projected_grad.dtype))

        state["step"] = state.get("step", 0) + 1
        moment_mode = group["moment_mode"]
        if moment_mode in ("ema", "adafactor_ema"):
            # adafactor_ema applies the same full-gradient Adafactor dampening
            # above (before basis refresh and projection); the dampened,
            # projected gradient still feeds this same first-moment EMA rather
            # than replacing the moment slot outright.
            if projected_exp_avg is None:
                projected_exp_avg = torch.zeros_like(projected_grad)
            projected_exp_avg.mul_(group["beta"]).add_(projected_grad, alpha=1.0 - group["beta"])
        else:  # pragma: no cover - validated in __init__
            raise AssertionError(f"unexpected moment_mode: {moment_mode}")
        state["projected_exp_avg"] = projected_exp_avg

        return MatrixUpdate(
            param=p,
            projector=projector,
            projected_grad=projected_grad,
            projected_exp_avg=projected_exp_avg,
            original_shape=tuple(p.shape),
        )

    @staticmethod
    def _adafactor_dampen_full_grad(grad: Tensor, group: dict, state: dict) -> Tensor:
        """Adafactor-style row/col factored second moment on the full gradient,
        applied before basis refresh and before projection so the same dampened
        gradient feeds both consumers, then still passed through the same
        first-moment EMA as ema mode.
        """

        beta2 = group["adafactor_beta2"]
        eps = group["adafactor_eps"]
        step = state.get("adafactor_step", 0) + 1
        state["adafactor_step"] = step

        grad32 = grad.float()
        grad_sq = grad32.square() + eps
        row_var = state.get("adafactor_row_var")
        col_var = state.get("adafactor_col_var")
        if row_var is None:
            row_var = torch.zeros(grad32.shape[0], device=grad32.device, dtype=torch.float32)
            col_var = torch.zeros(grad32.shape[1], device=grad32.device, dtype=torch.float32)
        row_var.mul_(beta2).add_(grad_sq.mean(dim=1), alpha=1.0 - beta2)
        col_var.mul_(beta2).add_(grad_sq.mean(dim=0), alpha=1.0 - beta2)
        state["adafactor_row_var"] = row_var
        state["adafactor_col_var"] = col_var

        bias_correction = 1.0 - beta2**step
        row_hat = row_var / bias_correction
        col_hat = col_var / bias_correction
        # Shazeer & Stern (2018) factored second-moment reconstruction: R C^T / sum(R).
        # row_hat/col_hat are stored as per-row/per-column means (not sums), so the
        # sum-based normalizer is row_hat.mean() (mean of means == sum/n cancelling n).
        factor = (row_hat.unsqueeze(1) @ col_hat.unsqueeze(0)) / row_hat.mean().clamp_min(eps)
        dampened = grad32 / factor.sqrt().clamp_min(eps)
        # Restore the raw gradient's scale. The factored second moment normalizes each
        # element to RMS~1, which is Adafactor's *direction* convention but blows the
        # magnitude up: an RMS-1 [m,n] matrix has Frobenius norm sqrt(m*n) (e.g. ~2610
        # for a 6656x1024 MLP grad, ~261x a norm-10 raw grad). Real Adafactor hides
        # this behind the learning rate; we feed the dampened grad into the basis
        # tracker's tangent/sigma path, where sigma scales *quadratically* with
        # magnitude and so explodes (~1e5). Rescaling to the raw grad's RMS keeps
        # Adafactor's SNR reweighting (the point) while restoring scale: dampened is
        # RMS-1, so multiplying by grad's RMS makes it RMS-match the raw grad
        # per-element. This returns projected-grad norm to the pre-Adafactor regime
        # (~1.77 at rank 32), which is why the clip rail belongs back at ~2, not 2000.
        grad_rms = grad32.square().mean().sqrt().clamp_min(eps)
        dampened = dampened * grad_rms
        return dampened.to(dtype=grad.dtype)

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
            for entry, update_hat in zip(bucket_entries, update_hats, strict=True):
                self._apply_matrix_update(entry, update_hat, group, diagnostics)

    def _apply_matrix_update(self, entry: MatrixUpdate, update_hat: Tensor, group: dict, diagnostics: dict | None) -> None:
        if diagnostics is not None:
            if self.diagnostics_leverage_enabled:
                leverage_cv, min_ratio, max_ratio = self._large_axis_leverage_stats(update_hat)
                diagnostics["projected_leverage_cv_sum"] += leverage_cv
                diagnostics["projected_leverage_min_ratio_sum"] += min_ratio
                diagnostics["projected_leverage_max_ratio_sum"] += max_ratio
                diagnostics["projected_leverage_tensors"] += 1
            if self.diagnostics_aurora_health_enabled:
                alignment = self._aurora_alignment(entry.projected_exp_avg, update_hat)
                erank = self._effective_rank(entry.projected_exp_avg)
                diagnostics["aurora_alignment_sum"] += alignment
                diagnostics["aurora_erank_sum"] += erank
                diagnostics["aurora_erank_pct_sum"] += erank / min(entry.projected_exp_avg.shape)
                diagnostics["aurora_health_tensors"] += 1
        update = entry.projector.project_back(update_hat).to(dtype=entry.param.dtype)

        if group["weight_decay"]:
            entry.param.mul_(1.0 - group["lr"] * group["weight_decay"])
        if diagnostics is not None:
            update_norm_sq = (update.float().norm() * group["lr"]).square().detach()
            diagnostics["matrix_update_norm_sq"] = update_norm_sq if diagnostics["matrix_update_norm_sq"] is None else diagnostics["matrix_update_norm_sq"] + update_norm_sq
            diagnostics["matrix_params"] += entry.param.numel()
        entry.param.add_(update, alpha=-group["lr"])

    def _step_fallback_param(self, p: Tensor, grad: Tensor, group: dict, diagnostics: dict | None) -> None:
        state = self.state[p]
        beta1, beta2 = group["fallback_betas"]
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(p, dtype=torch.float32)
            exp_avg_sq = torch.zeros_like(p, dtype=torch.float32)
            state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
        state["exp_avg"] = exp_avg
        state["exp_avg_sq"] = exp_avg_sq
        step = state["step"]
        before = p.detach().clone() if diagnostics is not None else None
        torch_optim_functional.adamw(
            [p],
            [grad.detach().float()],
            [exp_avg],
            [exp_avg_sq],
            [],
            [step],
            foreach=False,
            capturable=False,
            differentiable=False,
            fused=p.is_cuda and p.dtype == torch.float32,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=group["lr"],
            weight_decay=group["weight_decay"],
            eps=group["eps"],
            maximize=False,
        )
        if diagnostics is not None:
            assert before is not None
            delta = p.detach().float() - before.float()
            update_norm_sq = delta.norm().square().detach()
            diagnostics["fallback_update_norm_sq"] = update_norm_sq if diagnostics["fallback_update_norm_sq"] is None else diagnostics["fallback_update_norm_sq"] + update_norm_sq
            diagnostics["fallback_params"] += p.numel()

    # Lag between basis snapshots for the convergence metric, in refresh events
    # (5 refreshes = 50 steps at the default interval 10). The instantaneous
    # angles (rotation_angle, target self-angle) are floored by target noise and
    # structurally cannot show convergence; the angle of the basis against its
    # own past can: decaying lag-angle = settling, plateau = stable orbit radius.
    BASIS_LAG_REFRESHES = 5

    def _refresh_projector(self, projector: SubspaceProjector, grad: Tensor, group: dict, state: dict, diagnostics: dict | None) -> None:
        was_initialized = projector.is_initialized
        aim = group["grassmann_aim"]
        if not was_initialized:
            projector.fit(grad)
        elif aim == "eigh":
            # Position control (the default path): eigh target frame from the
            # dampened boundary grad names WHERE the signal subspace is; the
            # geodesic contracts a fraction step_size of every principal angle
            # toward it. Target noise decays geometrically instead of integrating
            # -- the basis is a streaming Karcher mean of the target stream, its
            # own accumulator, zero persistent state.
            rotate_rank = self._resolve_rotate_rank(group, projector, grad)
            target_frame, gram_eigenvalues = projector.eigh_target_frame(grad)
            tangent = projector.tangent_toward(target_frame, top_k=rotate_rank)
            projector.update_grassmann_from_tangent(
                tangent,
                step_size=group["grassmann_step_size"],
                rotate_rank=rotate_rank,
            )
            if diagnostics is not None and self.diagnostics_basis_enabled:
                # Q10 probe (demoted to background): target self-consistency and the
                # Gram spectrum ratio at the rank cutoff. Both measure the target
                # stream, not the basis -- they are rank-starvation thermometers,
                # not convergence reads (that is the basis lag angle below).
                prev_target = state.get("prev_eigh_target")
                if prev_target is not None:
                    self_angle = SubspaceProjector.top_principal_angle(prev_target, target_frame)
                    diagnostics["eigh_target_self_angle_sum"] += float(self_angle.detach().cpu())
                    diagnostics["eigh_target_probe_tensors"] += 1
                state["prev_eigh_target"] = target_frame
                rank = target_frame.shape[1]
                if gram_eigenvalues.shape[0] > rank:
                    kept_min = gram_eigenvalues[-rank]
                    dropped_max = gram_eigenvalues[-rank - 1]
                    ratio = (dropped_max / kept_min.clamp_min(1e-30)).clamp(0.0, 1.0)
                    diagnostics["eigh_target_cutoff_ratio_sum"] += float(ratio.detach().cpu())
                    diagnostics["eigh_target_cutoff_tensors"] += 1
        else:
            # Velocity control (SubTrack-faithful single-grad tangent step): kept as
            # the reference/ablation arm. The C1 window accumulator was deleted when
            # position control beat it with zero state -- noise in a velocity command
            # integrates as a random walk with no restoring force, which is why this
            # path needed milliradian steps and lost to drift (see ledger).
            projector.update_grassmann(
                grad,
                step_size=group["grassmann_step_size"],
                rotate_rank=self._resolve_rotate_rank(group, projector, grad),
            )

        if was_initialized and diagnostics is not None and self.diagnostics_basis_enabled:
            self._accumulate_basis_diagnostics(diagnostics, projector.last_rotation_angle, projector.last_tangent_sigma_max)
            # Convergence metric: principal angles between the basis and its own
            # snapshot from BASIS_LAG_REFRESHES boundaries ago. Mean angle is the
            # settling read (-> 0 iff every plane stops moving); top angle is the
            # orbit-radius read (churn planes keep it elevated). Snapshot buffer is
            # diagnostic-only state, gated like the Q10 probe buffer.
            snapshot = state.get("basis_lag_snapshot")
            lag_count = state.get("basis_lag_refreshes", 0) + 1
            if snapshot is None:
                state["basis_lag_snapshot"] = projector.canonical_basis().detach().clone()
                state["basis_lag_refreshes"] = 0
            elif lag_count >= self.BASIS_LAG_REFRESHES:
                current = projector.canonical_basis()
                lag_angles = SubspaceProjector.principal_angles_sine(snapshot, current)
                diagnostics["basis_lag_mean_angle_sum"] += float(lag_angles.mean().detach().cpu())
                diagnostics["basis_lag_top_angle_sum"] += float(lag_angles.max().detach().cpu())
                diagnostics["basis_lag_tensors"] += 1
                state["basis_lag_snapshot"] = current.detach().clone()
                state["basis_lag_refreshes"] = 0
            else:
                state["basis_lag_refreshes"] = lag_count

        state["basis"] = projector.basis
        resolved_side = projector.resolved_side if projector.resolved_side is not None else projector.side
        state["projection_side_is_right"] = resolved_side is ProjectionSide.RIGHT

        # Moment across a geodesic refresh: parallel transport, which is the
        # IDENTITY in projected coordinates. The retraction is a rigid frame
        # rotation (Q_new = R @ Q_old with R rotating the [Q@V, U] planes; pinned
        # by test), so rotating the lifted moment with the frame and re-reading its
        # coordinates returns them unchanged -- the honest transfer is a no-op, and
        # nothing is dropped. The previous project-back/re-project transfer was
        # orthogonal projection instead: it lost sin(angle) of every rotated
        # plane's moment, and Aurora's polar map then re-amplified those shrunken,
        # noise-dominated directions back to full strength (the alignment
        # degradation measured at hot step sizes). Only a fresh fit (no geodesic)
        # has no transport; there the stale coordinates are dropped on shape
        # mismatch and otherwise kept as the least-wrong option.
        old_projected_exp_avg = state.get("projected_exp_avg")
        if not was_initialized and old_projected_exp_avg is not None:
            projected_shape = tuple(projector.project(grad).shape)
            if tuple(old_projected_exp_avg.shape) != projected_shape:
                state.pop("projected_exp_avg", None)

    @staticmethod
    def _resolve_rotate_rank(group: dict, projector: SubspaceProjector, grad: Tensor) -> int:
        configured = group["grassmann_rotate_rank"]
        return configured if configured is not None else projector.effective_rank(grad)

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
    def _accumulate_basis_diagnostics(diagnostics: dict, rotation_angle: float, tangent_sigma_max: float) -> None:
        diagnostics["rotation_angle_sum"] += rotation_angle
        diagnostics["tangent_sigma_max_sum"] += tangent_sigma_max
        diagnostics["basis_refresh_tensors"] += 1

    @staticmethod
    def _orthogonalize_update(update: Tensor, group: dict, original_shape: tuple[int, ...] | None = None) -> Tensor:
        return UsuiTrack._orthogonalize_aurora(update, group, original_shape)

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
            int(group.get("aurora_pp_iterations", AURORA_PP_ITERATIONS)),
            int(group.get("polar_ns_steps", len(NEWTON_SCHULZ_COEFFICIENTS))),
        )

    @staticmethod
    def _orthogonalize_aurora_muon_tensor(
        update: Tensor,
        original_rows: int,
        original_cols: int,
        aurora_pp_iterations: int,
        polar_ns_steps: int,
    ) -> Tensor:
        aurora_update = UsuiTrack._aurora_leverage_uniform_polar(
            update,
            pp_iterations=aurora_pp_iterations,
            pp_beta=AURORA_PP_BETA,
            polar_ns_steps=polar_ns_steps,
        )
        return aurora_update * math.sqrt(max(1.0, original_rows / original_cols))

    @staticmethod
    def _orthogonalize_aurora(update: Tensor, _group: dict, original_shape: tuple[int, ...] | None) -> Tensor:
        aurora_update = UsuiTrack._aurora_leverage_uniform_polar(
            update,
            pp_iterations=_group.get("aurora_pp_iterations", AURORA_PP_ITERATIONS),
            pp_beta=AURORA_PP_BETA,
            polar_ns_steps=_group.get("polar_ns_steps", len(NEWTON_SCHULZ_COEFFICIENTS)),
        )
        return UsuiTrack._scale_orthogonalized_update(
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

        UsuiTrack owns momentum, LR, weight decay, and full-matrix Muon scaling. This
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
            return UsuiTrack._heavyball_polar(update, steps=polar_ns_steps)

        transposed = update.shape[-2] < update.shape[-1]
        work = update.mT if transposed else update
        work32 = work.float()
        rows, cols = work32.shape[-2:]
        target_row_sq = cols / rows
        diagonal = work32.norm(dim=-1, keepdim=True).clamp_min(eps).reciprocal()
        balanced = None
        for iteration in range(pp_iterations):
            balanced = UsuiTrack._heavyball_polar(diagonal * work32, steps=polar_ns_steps).float()
            if iteration < pp_iterations - 1:
                row_sq = balanced.square().sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                diagonal = diagonal * (target_row_sq / row_sq).pow(pp_beta)
        assert balanced is not None
        result = balanced.mT if transposed else balanced
        return result.to(device=update.device, dtype=update.dtype)

    @staticmethod
    def _heavyball_polar(update: Tensor, steps: int = len(NEWTON_SCHULZ_COEFFICIENTS)) -> Tensor:
        return UsuiTrack._batched_newton_schulz(update, steps=steps)

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
