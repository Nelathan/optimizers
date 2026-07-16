import copy
import unittest

import torch

from usuitrack import UsuiTrack
from usuitrack.optimizer import AURORA_PP_ITERATIONS, MIN_BASIS_UPDATE_STEP, NEWTON_SCHULZ_COEFFICIENTS


class UsuiTrackTest(unittest.TestCase):
    def test_selected_default_contract(self):
        weight = torch.nn.Parameter(torch.randn(8, 4))
        optimizer = UsuiTrack([weight], rank=4)

        group = optimizer.param_groups[0]
        self.assertEqual(group["beta"], 0.95)
        self.assertEqual(group["adafactor_beta2"], 0.99)
        self.assertEqual(group["grad_clip_norm"], 1.0)
        self.assertEqual(group["basis_update_interval"], 1)
        self.assertEqual(AURORA_PP_ITERATIONS, 1)
        self.assertEqual(len(NEWTON_SCHULZ_COEFFICIENTS), 5)

    def test_rejects_non_matrix_and_excessive_rank(self):
        with self.assertRaisesRegex(ValueError, "only supports 2D"):
            UsuiTrack([torch.nn.Parameter(torch.randn(4))])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            UsuiTrack([torch.nn.Parameter(torch.randn(3, 4))], rank=4)

    def test_initializes_then_moves_every_gradient_by_default(self):
        weight = torch.nn.Parameter(torch.randn(8, 4))
        optimizer = UsuiTrack([weight], lr=0.01, rank=3, side="right")

        weight.grad = torch.randn_like(weight)
        optimizer.step()
        initial_basis = optimizer.state[weight]["basis"].clone()
        self.assertEqual(optimizer.param_groups[0]["matrix_step"], 1)
        self.assertEqual(optimizer.param_groups[0]["basis_update_step"], 1)

        weight.grad = torch.randn_like(weight)
        optimizer.step()
        self.assertFalse(torch.equal(initial_basis, optimizer.state[weight]["basis"]))
        self.assertEqual(optimizer.param_groups[0]["basis_update_step"], 2)

    def test_refresh_interval_controls_oja_moves_not_phase_one(self):
        weight = torch.nn.Parameter(torch.randn(8, 4))
        optimizer = UsuiTrack([weight], lr=0.01, rank=3, side="right", basis_update_interval=2)

        weight.grad = torch.randn_like(weight)
        optimizer.step()
        initial_basis = optimizer.state[weight]["basis"].clone()
        self.assertEqual(optimizer.param_groups[0]["basis_update_step"], 0)

        weight.grad = torch.randn_like(weight)
        optimizer.step()
        self.assertFalse(torch.equal(initial_basis, optimizer.state[weight]["basis"]))
        self.assertEqual(optimizer.param_groups[0]["basis_update_step"], 1)
        self.assertIn("projected_exp_avg", optimizer.state[weight])

    def test_harmonic_basis_schedule_uses_update_count(self):
        group = {"basis_update_step": 1}
        self.assertEqual(UsuiTrack._basis_update_step_size(group), 1.0)
        group["basis_update_step"] = 2
        self.assertEqual(UsuiTrack._basis_update_step_size(group), 0.5)
        group["basis_update_step"] = 1000
        self.assertEqual(UsuiTrack._basis_update_step_size(group), MIN_BASIS_UPDATE_STEP)

    def test_prepare_release_matches_ordinary_state(self):
        torch.manual_seed(0)
        ordinary_weight = torch.nn.Parameter(torch.randn(8, 4))
        released_weight = torch.nn.Parameter(ordinary_weight.detach().clone())
        ordinary = UsuiTrack([ordinary_weight], lr=0.01, rank=3, side="right")
        released = UsuiTrack([released_weight], lr=0.01, rank=3, side="right", release_matrix_grads=True)

        for _ in range(3):
            gradient = torch.randn_like(ordinary_weight)
            ordinary_weight.grad = gradient.clone()
            released_weight.grad = gradient.clone()
            released.prepare(released_weight)
            self.assertIsNone(released_weight.grad)
            ordinary.step()
            released.step()
            torch.testing.assert_close(ordinary_weight, released_weight, rtol=0, atol=0)
            self.assertEqual(ordinary.param_groups[0]["matrix_step"], released.param_groups[0]["matrix_step"])

    def test_prepare_is_exactly_once_and_zero_grad_cannot_discard(self):
        weight = torch.nn.Parameter(torch.randn(8, 4))
        optimizer = UsuiTrack([weight], rank=3)
        weight.grad = torch.randn_like(weight)
        optimizer.prepare(weight)
        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            optimizer.prepare(weight)
        with self.assertRaisesRegex(RuntimeError, "cannot discard"):
            optimizer.zero_grad()
        optimizer.step()

    def test_state_dict_continuation(self):
        torch.manual_seed(1)
        first = torch.nn.Parameter(torch.randn(8, 4))
        first_optimizer = UsuiTrack([first], lr=0.01, rank=3)
        for _ in range(2):
            first.grad = torch.randn_like(first)
            first_optimizer.step()

        second = torch.nn.Parameter(first.detach().clone())
        second_optimizer = UsuiTrack([second], lr=0.01, rank=3)
        second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))
        gradient = torch.randn_like(first)
        first.grad = gradient.clone()
        second.grad = gradient.clone()
        first_optimizer.step()
        second_optimizer.step()
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_compiled_prepare_handles_exist(self):
        weight = torch.nn.Parameter(torch.randn(8, 4, device="cuda" if torch.cuda.is_available() else "cpu"))
        optimizer = UsuiTrack([weight], rank=3, compile_tensor_kernels=True)
        self.assertIsNotNone(optimizer._compiled_orthogonalize_update)
        self.assertIsNotNone(optimizer._compiled_prepare_tracker_adafactor_left)
        self.assertIsNotNone(optimizer._compiled_prepare_tracker_adafactor_right)


if __name__ == "__main__":
    unittest.main()
