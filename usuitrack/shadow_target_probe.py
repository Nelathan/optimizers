from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor

from .projector import ProjectionSide, SubspaceProjector


SEPARATE_OJA_STEPS = (0.03125, 0.0625)
DIRECT_OJA_STEPS = (0.02, 0.03, 0.04)


def _step_label(prefix: str, step: float) -> str:
    return f"{prefix}_{str(step).replace('.', '')}"


SEPARATE_OJA_LABELS = tuple(_step_label("oja", step) for step in SEPARATE_OJA_STEPS)
DIRECT_OJA_LABELS = tuple(_step_label("direct", step) for step in DIRECT_OJA_STEPS)
SHADOW_LABELS = ("current_q", "eigh", "warm", *SEPARATE_OJA_LABELS, *DIRECT_OJA_LABELS)


@dataclass
class ShadowParamState:
    side: ProjectionSide
    warm_sum: Tensor
    separate_frames: dict[str, Tensor]
    direct_frames: dict[str, Tensor]
    snapshots: dict[str, Tensor]
    capture_sums: dict[str, Tensor] = field(default_factory=dict)
    capture_count: int = 0


class ShadowTargetProbe:
    """Compare target estimators without allowing them to affect optimization."""

    def __init__(self, params: Iterable[Tensor]) -> None:
        self.param_ids = {id(param) for param in params}
        self.states: dict[Tensor, ShadowParamState] = {}
        self.last_step_diagnostics: dict[str, float] = {}
        self._step_metric_sums: dict[str, float] = {}
        self._step_metric_count = 0

    def includes(self, param: Tensor) -> bool:
        return id(param) in self.param_ids

    @staticmethod
    def _canonical_frame(projector: SubspaceProjector) -> Tensor:
        return projector.canonical_basis().detach().float()

    @staticmethod
    def _project(gradient: Tensor, frame: Tensor, side: ProjectionSide) -> Tensor:
        return gradient @ frame if side is ProjectionSide.RIGHT else frame.mT @ gradient

    @classmethod
    def _covariance_action(cls, gradient: Tensor, frame: Tensor, side: ProjectionSide, norm_sq: Tensor) -> tuple[Tensor, Tensor]:
        projected = cls._project(gradient, frame, side)
        action = gradient.mT @ projected if side is ProjectionSide.RIGHT else gradient @ projected.mT
        return action / norm_sq, projected

    @staticmethod
    def _orthonormalize(frame: Tensor) -> Tensor:
        return torch.linalg.qr(frame, mode="reduced").Q

    @classmethod
    def _separate_oja_step(cls, frame: Tensor, action: Tensor, step_size: float) -> Tensor:
        rayleigh = frame.mT @ action
        tangent = action - frame @ (0.5 * (rayleigh + rayleigh.mT))
        mean_energy = rayleigh.diagonal().mean().clamp_min(1e-12)
        return cls._orthonormalize(frame + (step_size / mean_energy) * tangent)

    @classmethod
    def _direct_oja_step(cls, frame: Tensor, action: Tensor, step_size: float) -> Tensor:
        rayleigh = frame.mT @ action
        tangent = action - frame @ (0.5 * (rayleigh + rayleigh.mT))
        tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
        normal, singular_values, right_h = torch.linalg.svd(tangent, full_matrices=False)
        right = right_h.mT
        moved = (frame @ right) * torch.cos(step_size * singular_values)
        moved = moved + normal * torch.sin(step_size * singular_values)
        return cls._orthonormalize(moved @ right_h)

    def initialize(self, param: Tensor, projector: SubspaceProjector) -> None:
        frame = self._canonical_frame(projector)
        side = projector.resolved_side
        if side is None:
            raise RuntimeError("shadow target probe requires a resolved projection side")
        self.states[param] = ShadowParamState(
            side=side,
            warm_sum=torch.zeros_like(frame),
            separate_frames={label: frame.clone() for label in SEPARATE_OJA_LABELS},
            direct_frames={label: frame.clone() for label in DIRECT_OJA_LABELS},
            snapshots={label: frame.clone() for label in SHADOW_LABELS},
            capture_sums={label: frame.new_zeros(()) for label in SHADOW_LABELS},
        )

    def observe(self, param: Tensor, gradient: Tensor, projector: SubspaceProjector) -> None:
        state = self.states[param]
        work = gradient.detach().float()
        norm = work.norm().clamp_min(1e-12)
        norm_sq = norm.square()

        current_frame = self._canonical_frame(projector)
        current_projected = self._project(work, current_frame, state.side)
        state.capture_sums["current_q"] += current_projected.norm() / norm
        for label, snapshot in state.snapshots.items():
            if label == "current_q":
                continue
            state.capture_sums[label] += self._project(work, snapshot, state.side).norm() / norm
        state.capture_count += 1

        warm_action, _ = self._covariance_action(work, current_frame, state.side, norm_sq)
        state.warm_sum.add_(warm_action)

        for label, step_size in zip(SEPARATE_OJA_LABELS, SEPARATE_OJA_STEPS, strict=True):
            frame = state.separate_frames[label]
            action, _ = self._covariance_action(work, frame, state.side, norm_sq)
            state.separate_frames[label] = self._separate_oja_step(frame, action, step_size)

        for label, step_size in zip(DIRECT_OJA_LABELS, DIRECT_OJA_STEPS, strict=True):
            frame = state.direct_frames[label]
            action, _ = self._covariance_action(work, frame, state.side, norm_sq)
            state.direct_frames[label] = self._direct_oja_step(frame, action, step_size)

    def finish_boundary(
        self,
        param: Tensor,
        held_frame: Tensor,
        updated_frame: Tensor,
        eigh_target: Tensor,
    ) -> None:
        state = self.states[param]
        warm_target = self._orthonormalize(state.warm_sum)
        new_snapshots = {
            "current_q": updated_frame.detach().float(),
            "eigh": eigh_target.detach().float(),
            "warm": warm_target,
            **{label: frame.detach().clone() for label, frame in state.separate_frames.items()},
            **{label: frame.detach().clone() for label, frame in state.direct_frames.items()},
        }

        if state.capture_count:
            for label in SHADOW_LABELS:
                self._add_step_metric(f"predictive_capture/{label}", float((state.capture_sums[label] / state.capture_count).detach().cpu()))
                demand = SubspaceProjector.principal_angles_sine(held_frame, new_snapshots[label]).sum()
                churn = SubspaceProjector.principal_angles_sine(state.snapshots[label], new_snapshots[label]).sum()
                self._add_step_metric(f"target_angle_mass/{label}", float(demand.detach().cpu()))
                self._add_step_metric(f"target_churn_mass/{label}", float(churn.detach().cpu()))
            self._step_metric_count += 1

        state.snapshots = new_snapshots
        state.warm_sum.zero_()
        for label in SHADOW_LABELS:
            state.capture_sums[label].zero_()
        state.capture_count = 0

    def _add_step_metric(self, name: str, value: float) -> None:
        self._step_metric_sums[name] = self._step_metric_sums.get(name, 0.0) + value

    def finalize_step(self) -> None:
        if self._step_metric_count:
            self.last_step_diagnostics = {
                name: value / self._step_metric_count for name, value in self._step_metric_sums.items()
            }
            self.last_step_diagnostics["tensors"] = float(self._step_metric_count)
        else:
            self.last_step_diagnostics = {}
        self._step_metric_sums.clear()
        self._step_metric_count = 0

    def tensor_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for state in self.states.values()
            for value in vars(state).values()
            for tensor in self._tensors(value)
        )

    @classmethod
    def _tensors(cls, value) -> list[Tensor]:
        if isinstance(value, Tensor):
            return [value]
        if isinstance(value, dict):
            tensors = []
            for nested in value.values():
                tensors.extend(cls._tensors(nested))
            return tensors
        return []
