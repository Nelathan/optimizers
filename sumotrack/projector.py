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


class ProjectorInitMethod(StrEnum):
    """How an unfitted projector initializes its basis."""

    EIGH = "eigh"
    RANDOM = "random"


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
    init_method: ProjectorInitMethod | str = ProjectorInitMethod.EIGH
    basis: Tensor | None = None
    resolved_side: ProjectionSide | None = field(default=None, init=False)
    last_tangent_sigma_max: float = field(default=float("nan"), init=False)
    last_rotation_angle: float = field(default=float("nan"), init=False)

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {self.rank}")
        self.side = ProjectionSide(self.side)
        self.init_method = ProjectorInitMethod(self.init_method)

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
    def fit_eigh(self, matrix: Tensor) -> Tensor:
        """Initialize or refresh the basis from a stable side-Gram eigendecomposition."""

        self._check_matrix(matrix)
        side = self.effective_side(matrix)
        rank = self.effective_rank(matrix)

        work = self._spectral_input(matrix)
        norm = work.norm()
        if norm == 0:
            return self.fit_random(matrix)
        work = work / norm.clamp_min(1e-12)

        _eigenvalues, eigenvectors = self._side_gram_eigh(work, side)

        if side is ProjectionSide.RIGHT:
            basis = eigenvectors[:, -rank:].mT
        elif side is ProjectionSide.LEFT:
            basis = eigenvectors[:, -rank:]
        else:  # pragma: no cover - effective_side never returns AUTO
            raise AssertionError(f"unexpected effective side: {side}")

        self.basis = basis.to(device=matrix.device, dtype=matrix.dtype).contiguous()
        self.resolved_side = side
        return self.basis

    @staticmethod
    def _side_gram_eigh(work: Tensor, side: ProjectionSide) -> tuple[Tensor, Tensor]:
        """Ascending eigendecomposition of the side Gram, with the jitter retry."""

        gram = work.mT @ work if side is ProjectionSide.RIGHT else work @ work.mT
        gram = 0.5 * (gram + gram.mT)
        try:
            return torch.linalg.eigh(gram)
        except RuntimeError:
            eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
            trace = gram.diagonal().sum()
            jitter = 1e-6 * (trace / max(1, gram.shape[0])).clamp_min(1e-12)
            return torch.linalg.eigh(gram + jitter * eye)

    @torch.no_grad()
    def eigh_target_frame(self, matrix: Tensor) -> tuple[Tensor, Tensor]:
        """Transient eigh target frame from ``matrix``: no basis mutation, no state.

        Same side-Gram decomposition as ``fit_eigh``, returned in the canon
        column-orthonormal ``[dim, rank]`` layout (the side-Gram eigenvectors ARE
        the canon frame on both sides). Also returns the full ascending Gram
        eigenvalue vector so callers can probe the spectrum at the rank cutoff
        (near-degenerate eigenvalues there make target membership swap
        batch-to-batch -- the churn hazard the A' probe measures). Skips the
        finite check: refresh-path gradients are already nan_to_num'd upstream.
        """

        if self.basis is None:
            raise RuntimeError("cannot compute an eigh target frame before fitting a basis")
        self._check_basis_matches(matrix)
        side = self._basis_side()
        rank = self.effective_rank(matrix)
        work = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
        work = work / work.norm().clamp_min(1e-12)
        eigenvalues, eigenvectors = self._side_gram_eigh(work, side)
        return eigenvectors[:, -rank:].contiguous(), eigenvalues

    @torch.no_grad()
    def tangent_toward(self, target_frame: Tensor, top_k: int) -> Tensor:
        """Grassmann log-map tangent from the current basis toward ``target_frame``,
        restricted to the ``top_k`` LARGEST principal angles. Canon ``[dim, rank]``.

        ``SVD(Q^T Q_t) = U cos(theta) V^T`` pairs each of Q's directions
        ``Q @ u_i`` with a target direction; the log map is ``sum_i theta_i w_i
        u_i^T`` with ``w_i = (I - QQ^T) Q_t v_i / sin(theta_i)``, computed as
        ``(I - QQ^T) Q_t V diag(theta/sin(theta)) U^T`` so the theta -> 0 limit is
        finite. Returned NEGATED: ``update_grassmann_from_tangent`` treats the
        tangent as a cost gradient and steps along ``-U``, so retracting the
        returned tangent with ``step_size=1, rotate_rank >= top_k`` lands exactly
        on the target's ``top_k`` principal planes (pinned by test). The tangent's
        singular values are the principal angles in radians -- sigma is a true
        angle here, scale-free by construction.
        """

        if self.basis is None:
            raise RuntimeError("cannot aim at a target frame before fitting a basis")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        side = self._basis_side()
        work_basis = self.basis.float() if self.basis.dtype in (torch.float16, torch.bfloat16) else self.basis
        canon_basis = work_basis.mT if side is ProjectionSide.RIGHT else work_basis
        target = target_frame.to(dtype=canon_basis.dtype)

        overlap = canon_basis.mT @ target
        u, s, vh = torch.linalg.svd(overlap)
        # SVD orders s = cos(theta) descending, so principal angles are ascending:
        # the top_k largest angles are the LAST top_k triples.
        keep = slice(max(0, s.shape[0] - top_k), s.shape[0])
        u_k, s_k, v_k = u[:, keep], s[keep], vh.mT[:, keep]

        # (I - QQ^T) Q_t v_i = Q_t v_i - Q u_i s_i, columns of norm sin(theta_i).
        # theta from atan2(sin, cos), NOT arccos(s): arccos is ill-conditioned near
        # s=1 (error ~sqrt(fp32 eps) ~ 3e-4 rad on small angles); atan2 with the
        # exactly-known sine is well-conditioned everywhere. Scale each column by
        # theta/sin(theta) (-> 1 as theta -> 0) to land at length theta_i.
        residual = target @ v_k - canon_basis @ (u_k * s_k)
        sin_theta = residual.norm(dim=0)
        theta_k = torch.atan2(sin_theta, s_k.clamp_min(0.0))
        scale = torch.where(sin_theta > 1e-7, theta_k / sin_theta.clamp_min(1e-12), torch.ones_like(theta_k))
        return -(residual * scale) @ u_k.mT

    @staticmethod
    @torch.no_grad()
    def top_principal_angle(frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """Largest principal angle (radians) between two column-orthonormal frames."""

        s = torch.linalg.svdvals(frame_a.mT @ frame_b)
        return torch.arccos(s.min().clamp(-1.0, 1.0))

    @torch.no_grad()
    def fit_random(self, matrix: Tensor) -> Tensor:
        """Initialize the basis with QR-orthonormalized random vectors."""

        self._check_matrix(matrix)
        side = self.effective_side(matrix)
        rank = self.effective_rank(matrix)
        work_dtype = torch.float32 if matrix.dtype in (torch.float16, torch.bfloat16) else matrix.dtype

        if side is ProjectionSide.RIGHT:
            random_matrix = torch.randn(matrix.shape[1], rank, device=matrix.device, dtype=work_dtype)
            q, _r = torch.linalg.qr(random_matrix, mode="reduced")
            basis = q.mT
        elif side is ProjectionSide.LEFT:
            random_matrix = torch.randn(matrix.shape[0], rank, device=matrix.device, dtype=work_dtype)
            basis, _r = torch.linalg.qr(random_matrix, mode="reduced")
        else:  # pragma: no cover - effective_side never returns AUTO
            raise AssertionError(f"unexpected effective side: {side}")

        self.basis = basis.to(device=matrix.device, dtype=matrix.dtype).contiguous()
        self.resolved_side = side
        return self.basis

    @torch.no_grad()
    def fit(self, matrix: Tensor) -> Tensor:
        if self.init_method is ProjectorInitMethod.EIGH:
            return self.fit_eigh(matrix)
        if self.init_method is ProjectorInitMethod.RANDOM:
            return self.fit_random(matrix)
        raise AssertionError(f"unexpected init method: {self.init_method}")

    @torch.no_grad()
    def project(self, matrix: Tensor) -> Tensor:
        """Project ``matrix`` into the current basis, fitting if needed."""

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
    def compute_tangent(self, matrix: Tensor) -> Tensor:
        """Canonical-layout Grassmann tangent of ``matrix`` at the current basis.

        Returns the tangent in SubTrack's column-orthonormal canonical frame,
        shape ``[dim, rank]`` (RIGHT bases are transposed into this frame; LEFT
        bases already are it -- see the layout note in ``update_grassmann``).

        Called every optimizer step under the accumulation path, so this must
        stay free of device->host syncs: no ``.item()``, ``.cpu()``, or the
        ``torch.isfinite(...).all()`` check from ``_spectral_input`` (that check
        is itself a hidden sync). Finite validation stays confined to ``fit_*``.
        """

        if self.basis is None:
            raise RuntimeError("cannot compute a Grassmann tangent before fitting a basis")
        self._check_basis_matches(matrix)
        basis = self.basis
        side = self._basis_side()
        work_matrix = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
        # NO Frobenius normalization. We previously divided the gradient by its global
        # norm here for gradient-scale invariance, but that made the tangent sigma a
        # scale-free ratio -- structurally blind to residual *magnitude*, so it could
        # not self-anneal and reported the same ~0.07 for a well-fit (eigh) and a
        # garbage (random) basis alike (measured: both ~0.066 at step 100). Self-
        # annealing *needs* sigma to carry residual size (big residual -> big step ->
        # settle). Adafactor upstream already conditions the gradient scale by SNR, so
        # the invariance was both redundant and poisonous. sigma now carries magnitude.
        work_basis = basis.float() if basis.dtype in (torch.float16, torch.bfloat16) else basis

        # SubTrack's track_the_subspace is defined for a COLUMN-orthonormal basis
        # Q:[dim, rank] with the gradient G:[dim, cols] laid out so Q projects its
        # rows. Its tangent is (I - Q Q.T) @ partial -- the partial with its
        # Q-column-space component removed from the LEFT, which is exactly what makes
        # the left-singular vectors U orthogonal to Q so that [Q@V, U] is a valid
        # orthonormal 2k-frame. Our RIGHT basis is stored row-orthonormal [rank, n];
        # transpose it (and G) into that canonical [dim, rank] form, run the one
        # shared geodesic, then transpose the result back. LEFT is already canonical.
        if side is ProjectionSide.RIGHT:
            canon_basis = work_basis.mT  # [n, rank], column-orthonormal
            canon_grad = work_matrix.mT  # [n, rows]
        else:
            canon_basis = work_basis  # [m, rank], column-orthonormal
            canon_grad = work_matrix  # [m, cols]

        # Least-squares residual of projecting the gradient into the current basis,
        # then the Euclidean gradient of ||residual||^2 w.r.t. the basis, projected
        # onto the tangent space at Q (component orthogonal to Q's columns).
        estimated_w = canon_basis.mT @ canon_grad
        residual = canon_grad - canon_basis @ estimated_w
        partial = -2.0 * (residual @ estimated_w.mT)
        return partial - canon_basis @ (canon_basis.mT @ partial)

    @torch.no_grad()
    def update_grassmann_from_tangent(self, tangent: Tensor, step_size: float, rotate_rank: int = 1) -> Tensor:
        """Retract along the exact Grassmann geodesic from a precomputed canon tangent.

        Port of SubTrack's ``track_the_subspace`` retraction half: take the
        tangent's top-``rank`` singular triple and retract along the exact
        geodesic (``cos``/``sin`` rotation of the ``[Q@V, U]`` principal-angle
        frame) rather than a first-order QR retraction. SubTrack calls this with
        ``k=1``; ``k=rank`` here is the direct spectral generalization to a
        rank-``r`` tracked subspace.

        The rotation is ``step_size * sigma`` with raw singular values (self-annealing:
        big residual -> big step, well-fit -> small step, so the tracker settles).
        The one storage-layout adaptation: RIGHT-side bases are stored row-orthonormal
        and are transposed into SubTrack's column-orthonormal convention before the
        shared geodesic and transposed back after -- ``tangent`` is expected already
        in that canonical ``[dim, rank]`` layout (see ``compute_tangent``).
        """

        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if rotate_rank < 1:
            raise ValueError(f"rotate_rank must be >= 1, got {rotate_rank}")
        if self.basis is None:
            raise RuntimeError("cannot retract a Grassmann geodesic before fitting a basis")
        side = self._basis_side()
        work_basis = self.basis.float() if self.basis.dtype in (torch.float16, torch.bfloat16) else self.basis
        canon_basis = work_basis.mT if side is ProjectionSide.RIGHT else work_basis

        # Default rank-1 rotation, faithful to SubTrack (``rank_k_matrix_estimation(..., k=1)``):
        # rotate the *single* dominant tangent direction and carry the rest of the
        # frame unrotated via the ``(I - V V.T)`` term below. This is deliberate --
        # a single-batch residual tail is near-isotropic noise, so rotating the whole
        # spectrum spins the basis on that noise and thrashes (measured; see the
        # drift-vs-spin record). ``rotate_rank > 1`` is an explicit ablation arm:
        # with a window-MEAN tangent the tail is denoised, which re-opens whether
        # multi-rank rotation helps -- that is a run flag, never a silent default.
        singular_u, singular_values, singular_v = self._rank_k_svd(tangent, rotate_rank)
        self.last_tangent_sigma_max = float(singular_values.max().detach().cpu()) if singular_values.numel() else float("nan")
        # No sigma clip. A clip existed as blip safety, but whenever sigma sat above
        # it the rotation became a CONSTANT angle (step*clip) -- self-annealing died
        # and the geodesic limit-cycled instead of converging (measured in vitro:
        # capture froze at 0.60 with rotation pinned at 0.5 rad; removing the clip
        # in the stable step regime converged to 0.87). Blip protection lives
        # upstream in the per-tensor raw-grad clip, which also guards adafactor.

        # Self-annealing geodesic retraction, faithful to SubTrack: rotate each
        # principal direction by ``step_size * sigma_i`` (raw singular values, not
        # normalized). This is deliberately NOT scale-free -- sigma carries the
        # residual magnitude, so a poorly-fit basis takes big steps and a well-fit
        # one takes small ones, letting the tracker settle to a fixed point instead
        # of being forced back to a constant top-angle every refresh. We briefly
        # normalized by sigma_max to fight a drifting sigma, but that drift was the
        # frame bug injecting garbage into the tangent; with the frame fixed sigma is
        # stable on its own, and normalizing actively re-inflated the residual tail
        # each refresh, preventing convergence (chordal rose instead of shrinking).
        # step_size is back in SubTrack's units (their default is 1e4), tuned against
        # our sigma scale rather than borrowed.
        rotation = step_size * singular_values
        # Honest rotation signal: the geodesic angle itself, in radians
        # (step_size * sigma_1 for the one rotated direction). Monotonic in how far
        # the basis turned -- 0 = static, grows without a ceiling. We deliberately do
        # NOT report |sin(angle)|: sin peaks at angle=pi/2 and *comes back down* past
        # 90 deg, so it aliases a big (wrapping) rotation as a small one. The raw
        # radian never lies about magnitude. (``.sum()`` is total radians over the
        # ``rotate_rank`` rotated directions -- length 1 on the default path.)
        self.last_rotation_angle = float(rotation.abs().sum().detach().cpu()) if rotation.numel() else float("nan")
        cos_block = torch.diag(torch.cos(rotation))
        sin_block = torch.diag(torch.sin(-rotation))
        basis_v = canon_basis @ singular_v
        rotated = torch.cat([basis_v, singular_u], dim=1) @ torch.cat([cos_block, sin_block], dim=0)
        eye_rank = torch.eye(singular_v.shape[0], device=canon_basis.device, dtype=canon_basis.dtype)
        canon_new = rotated @ singular_v.mT + canon_basis @ (eye_rank - singular_v @ singular_v.mT)

        # Back to storage layout: RIGHT is stored row-orthonormal, LEFT column-orthonormal.
        new_basis = canon_new.mT if side is ProjectionSide.RIGHT else canon_new

        self.basis = new_basis.to(device=self.basis.device, dtype=self.basis.dtype).contiguous()
        self.resolved_side = side
        return self.basis

    @torch.no_grad()
    def update_grassmann(self, matrix: Tensor, step_size: float, rotate_rank: int = 1) -> Tensor:
        """Refresh the basis with a Grassmann geodesic tangent step from ``matrix``.

        Composition of ``compute_tangent`` (tangent at the current basis) and
        ``update_grassmann_from_tangent`` (rank-1 SVD + geodesic retraction). Kept
        as a single entry point for callers that do not need windowed tangent
        accumulation (e.g. ``grassmann_accumulate=False``).
        """

        return self.update_grassmann_from_tangent(self.compute_tangent(matrix), step_size=step_size, rotate_rank=rotate_rank)

    @staticmethod
    def _rank_k_svd(matrix: Tensor, k: int) -> tuple[Tensor, Tensor, Tensor]:
        """Top-``k`` singular triple ``(U, sigma, V)`` with ``V`` as columns (not V^T)."""

        u, sigma, vh = torch.linalg.svd(matrix, full_matrices=False)
        return u[:, :k], sigma[:k], vh.mT[:, :k]

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
            return self.fit(matrix)
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
    def _spectral_input(matrix: Tensor) -> Tensor:
        if matrix.dtype in (torch.float16, torch.bfloat16):
            matrix = matrix.float()
        if not torch.isfinite(matrix).all():
            raise RuntimeError("cannot fit a projection basis from non-finite matrix values")
        return matrix

    @staticmethod
    def _orthonormalize_rows(matrix: Tensor) -> Tensor:
        q, _r = torch.linalg.qr(matrix.mT, mode="reduced")
        return q.mT

    @staticmethod
    def _orthonormalize_columns(matrix: Tensor) -> Tensor:
        q, _r = torch.linalg.qr(matrix, mode="reduced")
        return q
