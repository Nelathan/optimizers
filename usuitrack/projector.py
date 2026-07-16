from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch
from heavyball.utils import ABC_LIST_STABLE
from torch import Tensor


class ProjectionSide(StrEnum):
    AUTO = "auto"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class SubspaceProjector:
    """One-sided rank-r projector with stable side-Gram EIGH initialization."""

    rank: int = 32
    side: ProjectionSide | str = ProjectionSide.AUTO
    basis: Tensor | None = None
    resolved_side: ProjectionSide | None = field(default=None, init=False)
    last_rotation_angle: float = field(default=float("nan"), init=False)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        self.side = ProjectionSide(self.side)

    @property
    def is_initialized(self) -> bool:
        return self.basis is not None

    def effective_side(self, matrix: Tensor) -> ProjectionSide:
        self._check_matrix(matrix)
        if self.side is ProjectionSide.AUTO:
            return ProjectionSide.RIGHT if matrix.shape[0] >= matrix.shape[1] else ProjectionSide.LEFT
        return ProjectionSide(self.side)

    def effective_rank(self, matrix: Tensor) -> int:
        self._check_matrix(matrix)
        return min(self.rank, matrix.shape[0], matrix.shape[1])

    @torch.no_grad()
    def fit(self, matrix: Tensor) -> Tensor:
        self._check_matrix(matrix)
        side = self.effective_side(matrix)
        rank = self.effective_rank(matrix)
        work = self._spectral_input(matrix)
        work = work / work.norm().clamp_min(1e-12)
        _values, vectors = self._side_gram_eigh(work, side)
        basis = vectors[:, -rank:].mT if side is ProjectionSide.RIGHT else vectors[:, -rank:]
        self.basis = basis.to(device=matrix.device, dtype=matrix.dtype).contiguous()
        self.resolved_side = side
        return self.basis

    @staticmethod
    def _side_gram_eigh(work: Tensor, side: ProjectionSide) -> tuple[Tensor, Tensor]:
        gram = work.mT @ work if side is ProjectionSide.RIGHT else work @ work.mT
        gram = 0.5 * (gram + gram.mT)
        try:
            return torch.linalg.eigh(gram)
        except RuntimeError:
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            jitter = 1e-6 * (gram.diagonal().sum() / max(1, gram.shape[0])).clamp_min(1e-12)
            return torch.linalg.eigh(gram + jitter * eye)

    @torch.no_grad()
    def canonical_basis(self) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot read a canonical basis before fitting")
        work = self.basis.float() if self.basis.dtype in (torch.float16, torch.bfloat16) else self.basis
        return work.mT if self._basis_side() is ProjectionSide.RIGHT else work

    @torch.no_grad()
    def project(self, matrix: Tensor) -> Tensor:
        basis = self._basis_for(matrix)
        return matrix @ basis.mT if self.effective_side(matrix) is ProjectionSide.RIGHT else basis.mT @ matrix

    @torch.no_grad()
    def project_back(self, projected: Tensor) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot project back before fitting a basis")
        if self._basis_side() is ProjectionSide.RIGHT:
            if projected.ndim != 2 or projected.shape[1] != self.basis.shape[0]:
                raise ValueError(f"right-basis projected tensor does not match basis {tuple(self.basis.shape)}")
            return projected @ self.basis
        if projected.ndim != 2 or projected.shape[0] != self.basis.shape[1]:
            raise ValueError(f"left-basis projected tensor does not match basis {tuple(self.basis.shape)}")
        return self.basis @ projected

    @torch.no_grad()
    def project_and_back(self, matrix: Tensor) -> Tensor:
        return self.project_back(self.project(matrix))

    @torch.no_grad()
    def oja_tangent(self, matrix: Tensor, projected: Tensor | None = None) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot compute an Oja tangent before fitting a basis")
        self._check_basis_matches(matrix)
        side = self._basis_side()
        work = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
        frame = self.canonical_basis()
        low = projected.float() if projected is not None else (work @ frame if side is ProjectionSide.RIGHT else frame.mT @ work)
        action = work.mT @ low if side is ProjectionSide.RIGHT else work @ low.mT
        rayleigh = frame.mT @ action
        rayleigh = 0.5 * (rayleigh + rayleigh.mT)
        return (action - frame @ rayleigh) / rayleigh.diagonal().mean().clamp_min(1e-12)

    @staticmethod
    def oja_geodesic_from_eigh(frame: Tensor, tangent: Tensor, eigenvalues: Tensor, eigenvectors: Tensor, step_size: float) -> Tensor:
        sigma = eigenvalues.clamp_min(0.0).sqrt()
        rotation = step_size * sigma
        sin_over_sigma = torch.where(sigma > 1e-7, torch.sin(rotation) / sigma.clamp_min(1e-12), torch.full_like(sigma, step_size))
        moved = ((frame @ eigenvectors) * torch.cos(rotation).unsqueeze(-2) + (tangent @ eigenvectors) * sin_over_sigma.unsqueeze(-2)) @ eigenvectors.mT
        return SubspaceProjector._polar_express_stiefel_correction(moved)

    @staticmethod
    def _polar_express_stiefel_correction(frame: Tensor) -> Tensor:
        gram = frame.mT @ frame
        a, b, c = ABC_LIST_STABLE[-1]
        correction = c * gram
        correction.diagonal(dim1=-2, dim2=-1).add_(b)
        correction = correction @ gram
        correction.diagonal(dim1=-2, dim2=-1).add_(a)
        return frame @ correction

    @torch.no_grad()
    def orthonormality_error(self) -> Tensor:
        if self.basis is None:
            raise RuntimeError("cannot measure orthonormality before fitting a basis")
        basis = self.basis.float()
        gram = basis @ basis.mT if self._basis_side() is ProjectionSide.RIGHT else basis.mT @ basis
        return (gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)).abs().max()

    def _basis_for(self, matrix: Tensor) -> Tensor:
        if self.basis is None:
            return self.fit(matrix)
        self._check_basis_matches(matrix)
        return self.basis

    def _basis_side(self) -> ProjectionSide:
        if self.basis is None:
            raise RuntimeError("basis has not been fitted")
        if self.side is not ProjectionSide.AUTO:
            return ProjectionSide(self.side)
        if self.resolved_side is None:
            raise RuntimeError("basis side has not been resolved")
        return self.resolved_side

    def _check_basis_matches(self, matrix: Tensor) -> None:
        self._check_matrix(matrix)
        assert self.basis is not None
        side = self.effective_side(matrix)
        rank = self.effective_rank(matrix)
        expected = (rank, matrix.shape[1]) if side is ProjectionSide.RIGHT else (matrix.shape[0], rank)
        if tuple(self.basis.shape) != expected:
            raise ValueError(f"basis shape {tuple(self.basis.shape)} does not match expected {expected}")
        if self.basis.device != matrix.device:
            raise ValueError(f"basis device {self.basis.device} does not match matrix device {matrix.device}")

    @staticmethod
    def _check_matrix(matrix: Tensor) -> None:
        if matrix.ndim != 2 or min(matrix.shape) == 0:
            raise ValueError(f"SubspaceProjector requires a non-empty 2D tensor, got shape {tuple(matrix.shape)}")

    @staticmethod
    def _spectral_input(matrix: Tensor) -> Tensor:
        matrix = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
        if not torch.isfinite(matrix).all():
            raise RuntimeError("cannot fit a projection basis from non-finite matrix values")
        return matrix
