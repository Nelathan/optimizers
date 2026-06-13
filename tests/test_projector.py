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
        old_basis = projector.fit_svd(grad).clone()

        projector.update_grassmann(torch.randn_like(grad), step_size=0.01)

        self.assertEqual(projector.basis.device, grad.device)
        self.assertEqual(projector.basis.dtype, grad.dtype)
        self.assertEqual(tuple(projector.basis.shape), tuple(old_basis.shape))
        self.assertLess(float(projector.orthonormality_error()), 1e-5)


if __name__ == "__main__":
    unittest.main()
