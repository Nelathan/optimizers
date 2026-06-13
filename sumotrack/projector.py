from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum

import torch
from torch import Tensor


class ProjectionSide(StrEnum):
    """Which side of a matrix gradient is represented by the tracked basis."""

    AUTO = "auto"
    LEFT = "left"
    RIGHT = "right"


@dataclass
class SubspaceProjector:
    """Project 2D gradients into a low-dimensional orthonormal subspace.

    For a gradient ``G`` with shape ``[m, n]`` and rank ``r``:

    - right projection stores ``Q`` as ``[r, n]`` and computes ``G @ Q.T``;
    - left projection stores ``Q`` as ``[m, r]`` and computes ``Q.T @ G``.

    ``AUTO`` chooses the smaller ambient side, matching the GaLore/SubTrack/SUMO
    convention: tall matrices use a right basis, wide matrices use a left basis.
    """

    rank: int = 32
    side: ProjectionSide | str = ProjectionSide.AUTO
    basis: Tensor | None = None
    resolved_side: ProjectionSide | None = field(default=None, init=False)

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
        return self.side

    def effective_rank(self, matrix: Tensor) -> int:
        self._check_matrix(matrix)
        return min(self.rank, matrix.shape[0], matrix.shape[1])

    @torch.no_grad()
    def fit_svd(self, matrix: Tensor) -> Tensor:
        """Initialize or refresh the basis from an exact SVD of ``matrix``."""

        self._check_matrix(matrix)
        side = self.effective_side(matrix)
        rank = self.effective_rank(matrix)

        svd_input = self._svd_input(matrix)
        u, _s, vh = torch.linalg.svd(svd_input, full_matrices=False)

        if side is ProjectionSide.RIGHT:
            basis = vh[:rank, :]
        elif side is ProjectionSide.LEFT:
            basis = u[:, :rank]
        else:  # pragma: no cover - effective_side never returns AUTO
            raise AssertionError(f"unexpected effective side: {side}")

        self.basis = basis.to(device=matrix.device, dtype=matrix.dtype).contiguous()
        self.resolved_side = side
        return self.basis

    @torch.no_grad()
    def project(self, matrix: Tensor) -> Tensor:
        """Project ``matrix`` into the current basis, fitting by SVD if needed."""

        basis = self._basis_for(matrix)
        if self.effective_side(matrix) is ProjectionSide.RIGHT:
            return matrix @ basis.mT
        return basis.mT @ matrix

    @torch.no_grad()
    def project_back(self, projected: Tensor) -> Tensor:
        """Lift a projected matrix back into the original matrix shape."""

        if self.basis is None:
            raise RuntimeError("cannot project back before fitting a basis")
        if self._basis_side() is ProjectionSide.RIGHT:
            if projected.ndim != 2 or projected.shape[1] != self.basis.shape[0]:
                raise ValueError(
                    "right-basis projected tensor must have shape [m, rank]; "
                    f"got {tuple(projected.shape)} for basis {tuple(self.basis.shape)}"
                )
            return projected @ self.basis
        if projected.ndim != 2 or projected.shape[0] != self.basis.shape[1]:
            raise ValueError(
                "left-basis projected tensor must have shape [rank, n]; "
                f"got {tuple(projected.shape)} for basis {tuple(self.basis.shape)}"
            )
        return self.basis @ projected

    @torch.no_grad()
    def project_and_back(self, matrix: Tensor) -> Tensor:
        return self.project_back(self.project(matrix))

    @torch.no_grad()
    def orthonormality_error(self) -> Tensor:
        """Return max absolute deviation from basis orthonormality."""

        if self.basis is None:
            raise RuntimeError("cannot measure orthonormality before fitting a basis")
        basis = self.basis.float()
        if self._basis_side() is ProjectionSide.RIGHT:
            gram = basis @ basis.mT
        else:
            gram = basis.mT @ basis
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return (gram - eye).abs().max()

    def _basis_for(self, matrix: Tensor) -> Tensor:
        if self.basis is None:
            return self.fit_svd(matrix)
        self._check_basis_matches(matrix)
        return self.basis

    def _basis_side(self) -> ProjectionSide:
        if self.basis is None:
            raise RuntimeError("basis has not been fitted")
        if self.side is not ProjectionSide.AUTO:
            return self.side
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
        if matrix.ndim != 2:
            raise ValueError(f"SubspaceProjector only supports 2D tensors, got shape {tuple(matrix.shape)}")
        if min(matrix.shape) == 0:
            raise ValueError(f"matrix dimensions must be non-empty, got shape {tuple(matrix.shape)}")

    @staticmethod
    def _svd_input(matrix: Tensor) -> Tensor:
        if matrix.dtype in (torch.float16, torch.bfloat16):
            return matrix.float()
        return matrix
