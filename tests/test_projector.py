import unittest

import torch

from sumotrack import ProjectionSide, SubspaceProjector


class SubspaceProjectorTest(unittest.TestCase):
    def test_right_projection_shapes_and_orthonormality(self):
        grad = torch.randn(11, 7)
        projector = SubspaceProjector(rank=3)

        low = projector.project(grad)
        lifted = projector.project_back(low)

        self.assertEqual(tuple(projector.basis.shape), (3, 7))
        self.assertEqual(tuple(low.shape), (11, 3))
        self.assertEqual(tuple(lifted.shape), tuple(grad.shape))
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_left_projection_shapes_and_orthonormality(self):
        grad = torch.randn(5, 13)
        projector = SubspaceProjector(rank=4)

        low = projector.project(grad)
        lifted = projector.project_back(low)

        self.assertEqual(tuple(projector.basis.shape), (5, 4))
        self.assertEqual(tuple(low.shape), (4, 13))
        self.assertEqual(tuple(lifted.shape), tuple(grad.shape))
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_rank_clamps_to_matrix_dimension(self):
        grad = torch.randn(3, 9)
        projector = SubspaceProjector(rank=32)

        low = projector.project(grad)

        self.assertEqual(projector.effective_rank(grad), 3)
        self.assertEqual(tuple(projector.basis.shape), (3, 3))
        self.assertEqual(tuple(low.shape), (3, 9))

    def test_eigh_init_captures_known_right_subspace(self):
        right_modes, _ = torch.linalg.qr(torch.randn(7, 3), mode="reduced")
        coeffs = torch.randn(11, 3) * torch.tensor([4.0, 2.0, 1.0])
        grad = coeffs @ right_modes.mT
        projector = SubspaceProjector(rank=3, side=ProjectionSide.RIGHT)

        basis = projector.fit(grad)

        overlap = torch.linalg.svdvals(basis @ right_modes)
        self.assertGreater(float(overlap.min()), 1 - 1e-5)
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_eigh_init_captures_known_left_subspace(self):
        left_modes, _ = torch.linalg.qr(torch.randn(5, 4), mode="reduced")
        coeffs = torch.randn(4, 13) * torch.tensor([[4.0], [2.0], [1.0], [0.5]])
        grad = left_modes @ coeffs
        projector = SubspaceProjector(rank=4, side=ProjectionSide.LEFT)

        basis = projector.fit(grad)

        overlap = torch.linalg.svdvals(basis.mT @ left_modes)
        self.assertGreater(float(overlap.min()), 1 - 1e-5)
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_random_init_right_projection_shape_and_orthonormality(self):
        grad = torch.randn(11, 7)
        projector = SubspaceProjector(rank=3, init_method="random")

        basis = projector.fit(grad)

        self.assertEqual(tuple(basis.shape), (3, 7))
        self.assertIs(projector.resolved_side, ProjectionSide.RIGHT)
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_random_init_left_projection_shape_and_orthonormality(self):
        grad = torch.randn(5, 13)
        projector = SubspaceProjector(rank=4, init_method="random")

        basis = projector.fit(grad)

        self.assertEqual(tuple(basis.shape), (5, 4))
        self.assertIs(projector.resolved_side, ProjectionSide.LEFT)
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_random_init_preserves_dtype_device_and_clamps_rank(self):
        grad = torch.randn(3, 9, dtype=torch.bfloat16)
        projector = SubspaceProjector(rank=32, init_method="random")

        low = projector.project(grad)

        self.assertEqual(projector.effective_rank(grad), 3)
        self.assertEqual(tuple(projector.basis.shape), (3, 3))
        self.assertEqual(projector.basis.dtype, grad.dtype)
        self.assertEqual(projector.basis.device, grad.device)
        self.assertEqual(low.dtype, grad.dtype)
        self.assertEqual(low.device, grad.device)
        self.assertLess(float(projector.orthonormality_error()), 2e-2)

    def test_explicit_right_side(self):
        grad = torch.randn(4, 10)
        projector = SubspaceProjector(rank=2, side=ProjectionSide.RIGHT)

        low = projector.project(grad)
        lifted = projector.project_back(low)

        self.assertEqual(tuple(projector.basis.shape), (2, 10))
        self.assertEqual(tuple(low.shape), (4, 2))
        self.assertEqual(tuple(lifted.shape), tuple(grad.shape))

    def test_zero_matrix_uses_orthonormal_fallback_basis(self):
        grad = torch.zeros(4, 10)
        projector = SubspaceProjector(rank=2, side=ProjectionSide.RIGHT)

        basis = projector.fit(grad)

        self.assertEqual(tuple(basis.shape), (2, 10))
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_rejects_non_finite_matrix_for_spectral_fit(self):
        grad = torch.randn(4, 10)
        grad[0, 0] = float("nan")
        projector = SubspaceProjector(rank=2, side=ProjectionSide.RIGHT)

        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            projector.fit(grad)

    def test_rejects_removed_svd_init_method(self):
        with self.assertRaises(ValueError):
            SubspaceProjector(rank=2, init_method="svd")

    def test_rejects_non_matrix(self):
        projector = SubspaceProjector(rank=2)

        with self.assertRaises(ValueError):
            projector.project(torch.randn(2, 3, 4))

    def test_preserves_dtype_and_device_for_low_precision_input(self):
        grad = torch.randn(8, 5, dtype=torch.bfloat16)
        projector = SubspaceProjector(rank=3)

        low = projector.project(grad)
        lifted = projector.project_back(low)

        self.assertEqual(projector.basis.dtype, grad.dtype)
        self.assertEqual(projector.basis.device, grad.device)
        self.assertEqual(low.dtype, grad.dtype)
        self.assertEqual(low.device, grad.device)
        self.assertEqual(lifted.dtype, grad.dtype)
        self.assertEqual(lifted.device, grad.device)

    def test_grassmann_update_preserves_orthonormality_and_device(self):
        grad = torch.randn(9, 5)
        projector = SubspaceProjector(rank=3)
        old_basis = projector.fit_eigh(grad).clone()

        projector.update_grassmann(torch.randn_like(grad), step_size=0.01)

        self.assertEqual(projector.basis.device, grad.device)
        self.assertEqual(projector.basis.dtype, grad.dtype)
        self.assertEqual(tuple(projector.basis.shape), tuple(old_basis.shape))
        self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_grassmann_update_matches_geodesic_formula_rank_one(self):
        """Pin our rank-r retraction against the rank-1 geodesic formula (cos/sin
        rotation of the (Q@V, U) principal-angle frame). This guards against
        silently drifting back to a first-order QR retraction, which is not what
        SubTrack's paper/code do.

        We deliberately depart from SubTrack's *literal* ``step_size * sigma`` in
        one way: SubTrack tracks the raw gradient, so its sigma carries gradient
        magnitude and its ``st_step_size`` (1e4) is tuned to it. We normalize the
        spectral input (neutral basis, see fit_eigh) and run adafactor upstream,
        which makes raw sigma both tiny and drifting per refresh -- an unusable
        scale for a fixed step_size. So we rotate by ``step_size * sigma/sigma_max``:
        at rank=1, sigma/sigma_max == 1, so the single tracked direction rotates by
        exactly ``step_size`` radians. That is the invariant pinned here.
        """

        torch.manual_seed(0)
        grad = torch.randn(9, 5)
        refresh_grad = torch.randn_like(grad)
        step_size = 0.01

        projector = SubspaceProjector(rank=1, side=ProjectionSide.RIGHT)
        projector.fit_eigh(grad)
        basis = projector.basis.clone()  # [rank, n] row-orthonormal (storage layout)
        projector.update_grassmann(refresh_grad, step_size=step_size)

        # Reproduce the geodesic in SubTrack's canonical column-orthonormal frame:
        # transpose the RIGHT basis and gradient into [dim, rank] / [dim, cols] form,
        # project the residual gradient onto the tangent space with (I - Q Q.T), then
        # rotate the [Q@V, U] frame and transpose the result back to storage layout.
        canon_basis = basis.mT  # [n, rank], column-orthonormal
        canon_grad = (refresh_grad / refresh_grad.norm().clamp_min(1e-12)).mT
        estimated_w = canon_basis.mT @ canon_grad
        residual = canon_grad - canon_basis @ estimated_w
        partial = -2.0 * (residual @ estimated_w.mT)
        tangent = partial - canon_basis @ (canon_basis.mT @ partial)
        u, sigma, vh = torch.linalg.svd(tangent, full_matrices=False)
        u1, sigma1, v1 = u[:, :1], sigma[:1], vh.mT[:, :1]

        # Raw self-annealing angle: rotate by step_size * sigma (no normalization).
        rotation = step_size * sigma1
        cos_sigma = torch.cos(rotation)
        sin_sigma = torch.sin(-rotation)
        basis_v = canon_basis @ v1
        rotated = torch.cat([basis_v, u1], dim=1) @ torch.cat([torch.diag(cos_sigma), torch.diag(sin_sigma)], dim=0)
        expected_canon = rotated @ v1.mT + canon_basis @ (torch.eye(v1.shape[0]) - v1 @ v1.mT)
        expected_basis = expected_canon.mT

        self.assertTrue(torch.allclose(projector.basis, expected_basis, atol=1e-5))

    def test_grassmann_update_step_size_actually_moves_the_subspace(self):
        """Regression guard against the retraction collapsing to a no-op. The angle
        is ``step_size * sigma`` with raw singular values, so the step size needed to
        move the subspace scales inversely with sigma. This test does NOT assert a
        production step size -- it only pins that a small step_size barely moves the
        basis while a large one moves it measurably, i.e. that step_size stays
        load-bearing and the geodesic did not regress into a first-order no-op.
        """

        torch.manual_seed(0)
        grad = torch.randn(64, 256)
        base = SubspaceProjector(rank=32, side=ProjectionSide.RIGHT)
        base.fit_eigh(grad)
        original_basis = base.basis.clone()
        refresh_grad = grad + 0.1 * torch.randn_like(grad)

        small_step = SubspaceProjector(rank=32, side=ProjectionSide.RIGHT)
        small_step.basis = original_basis.clone()
        small_step.resolved_side = base.resolved_side
        small_step.update_grassmann(refresh_grad.clone(), step_size=0.1)

        large_step = SubspaceProjector(rank=32, side=ProjectionSide.RIGHT)
        large_step.basis = original_basis.clone()
        large_step.resolved_side = base.resolved_side
        large_step.update_grassmann(refresh_grad.clone(), step_size=1e4)

        # Sign-invariant subspace overlap: off-diagonal gram entries measure
        # real rotation, unlike raw basis diffs which are confounded by QR/SVD
        # sign-convention flips that occur regardless of step_size.
        small_gram = small_step.basis @ original_basis.mT
        large_gram = large_step.basis @ original_basis.mT
        small_off_diag = (small_gram - torch.diag(small_gram.diagonal())).abs().max()
        large_off_diag = (large_gram - torch.diag(large_gram.diagonal())).abs().max()

        self.assertLess(float(small_off_diag), 1e-4)
        self.assertGreater(float(large_off_diag), 1e-3)

    def test_grassmann_update_is_invariant_to_gradient_scale_right_side(self):
        torch.manual_seed(0)
        init_grad = torch.randn(9, 5)
        refresh_grad = torch.randn_like(init_grad)
        first = SubspaceProjector(rank=3, side=ProjectionSide.RIGHT)
        second = SubspaceProjector(rank=3, side=ProjectionSide.RIGHT)
        first.fit_eigh(init_grad)
        second.basis = first.basis.clone()
        second.resolved_side = first.resolved_side

        first.update_grassmann(refresh_grad, step_size=0.01)
        second.update_grassmann(refresh_grad * 1000.0, step_size=0.01)

        first_projector = first.basis.mT @ first.basis
        second_projector = second.basis.mT @ second.basis
        self.assertTrue(torch.allclose(first_projector, second_projector, atol=1e-5, rtol=1e-5))

    def test_grassmann_update_is_invariant_to_gradient_scale_left_side(self):
        torch.manual_seed(1)
        init_grad = torch.randn(5, 9)
        refresh_grad = torch.randn_like(init_grad)
        first = SubspaceProjector(rank=3, side=ProjectionSide.LEFT)
        second = SubspaceProjector(rank=3, side=ProjectionSide.LEFT)
        first.fit_eigh(init_grad)
        second.basis = first.basis.clone()
        second.resolved_side = first.resolved_side

        first.update_grassmann(refresh_grad, step_size=0.01)
        second.update_grassmann(refresh_grad * 1000.0, step_size=0.01)

        first_projector = first.basis @ first.basis.mT
        second_projector = second.basis @ second.basis.mT
        self.assertTrue(torch.allclose(first_projector, second_projector, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
