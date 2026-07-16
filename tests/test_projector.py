import unittest

import torch

from usuitrack import ProjectionSide, SubspaceProjector


class SubspaceProjectorTest(unittest.TestCase):
    def test_eigh_initialization_is_orthonormal_on_both_sides(self):
        for side, shape in ((ProjectionSide.RIGHT, (8, 4)), (ProjectionSide.LEFT, (4, 8))):
            projector = SubspaceProjector(rank=3, side=side)
            basis = projector.fit(torch.randn(*shape))
            self.assertEqual(tuple(basis.shape), (3, shape[1]) if side is ProjectionSide.RIGHT else (shape[0], 3))
            self.assertLess(float(projector.orthonormality_error()), 1e-5)

    def test_zero_input_uses_deterministic_eigh_frame(self):
        first = SubspaceProjector(rank=3, side="right")
        second = SubspaceProjector(rank=3, side="right")
        zero = torch.zeros(8, 4)
        torch.testing.assert_close(first.fit(zero), second.fit(zero), rtol=0, atol=0)

    def test_projection_and_lift_shapes(self):
        for side, shape in (("right", (8, 4)), ("left", (4, 8))):
            matrix = torch.randn(*shape)
            projector = SubspaceProjector(rank=3, side=side)
            projected = projector.project(matrix)
            lifted = projector.project_back(projected)
            self.assertEqual(tuple(lifted.shape), shape)

    def test_oja_tangent_is_tangent_and_geodesic_stays_orthonormal(self):
        matrix = torch.randn(8, 4)
        projector = SubspaceProjector(rank=3, side="right")
        projector.fit(matrix)
        projected = projector.project(matrix)
        tangent = projector.oja_tangent(matrix, projected=projected)
        frame = projector.canonical_basis()
        torch.testing.assert_close(frame.mT @ tangent, torch.zeros(3, 3), atol=1e-5, rtol=0)
        gram = tangent.mT @ tangent
        values, vectors = torch.linalg.eigh(0.5 * (gram + gram.mT))
        moved = SubspaceProjector.oja_geodesic_from_eigh(frame, tangent, values, vectors, 0.5)
        identity = torch.eye(3)
        torch.testing.assert_close(moved.mT @ moved, identity, atol=3e-3, rtol=0)


if __name__ == "__main__":
    unittest.main()
