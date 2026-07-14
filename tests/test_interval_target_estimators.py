import unittest

import torch

from experiments.interval_target_estimators import ProbeConfig
from experiments.interval_target_estimators import normalized_covariance
from experiments.interval_target_estimators import orthogonal_factor
from experiments.interval_target_estimators import principal_angle_mass
from experiments.interval_target_estimators import rayleigh_normalized_oja_step
from experiments.interval_target_estimators import rayleigh_normalized_geodesic_oja_step
from experiments.interval_target_estimators import top_frame


class IntervalTargetEstimatorTests(unittest.TestCase):
    def test_full_width_orthogonal_factor_preserves_covariance_target(self) -> None:
        generator = torch.Generator().manual_seed(3)
        gradient = torch.randn((12, 20), generator=generator, dtype=torch.float64)

        factor = orthogonal_factor(gradient, gradient.shape[0], generator)

        expected = top_frame(normalized_covariance(gradient), rank=5)
        actual = top_frame(factor.mT @ factor, rank=5)
        self.assertLess(principal_angle_mass(actual, expected), 1e-6)

    def test_oja_step_preserves_frame_shape_and_orthonormality(self) -> None:
        generator = torch.Generator().manual_seed(5)
        frame = torch.linalg.qr(torch.randn((24, 6), generator=generator, dtype=torch.float64), mode="reduced").Q
        gradient = torch.randn((16, 24), generator=generator, dtype=torch.float64)

        updated = rayleigh_normalized_oja_step(frame, normalized_covariance(gradient), step_size=0.25)

        self.assertEqual(updated.shape, frame.shape)
        torch.testing.assert_close(updated.mT @ updated, torch.eye(6, dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_geodesic_oja_step_preserves_frame_shape_and_orthonormality(self) -> None:
        generator = torch.Generator().manual_seed(7)
        frame = torch.linalg.qr(torch.randn((24, 6), generator=generator, dtype=torch.float64), mode="reduced").Q
        gradient = torch.randn((16, 24), generator=generator, dtype=torch.float64)

        updated = rayleigh_normalized_geodesic_oja_step(frame, normalized_covariance(gradient), step_size=0.03)

        self.assertEqual(updated.shape, frame.shape)
        torch.testing.assert_close(updated.mT @ updated, torch.eye(6, dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_default_factor_budgets_preserve_equal_interval_contact(self) -> None:
        ProbeConfig().validate()


if __name__ == "__main__":
    unittest.main()
