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


if __name__ == "__main__":
    unittest.main()
