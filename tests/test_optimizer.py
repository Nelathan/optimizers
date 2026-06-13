import unittest

import torch

from sumotrack import SubspaceMuon


class SubspaceMuonTest(unittest.TestCase):
    def test_step_updates_matrix_and_fallback_params(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = SubspaceMuon([weight, bias], lr=0.01, rank=2)
        weight_before = weight.detach().clone()
        bias_before = bias.detach().clone()

        loss = (weight.square().mean() + bias.square().mean())
        loss.backward()
        opt.step()

        self.assertFalse(torch.equal(weight, weight_before))
        self.assertFalse(torch.equal(bias, bias_before))

    def test_matrix_state_keeps_projected_moment_only(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SubspaceMuon([weight], lr=0.01, rank=2)

        weight.square().mean().backward()
        opt.step()

        state = opt.state[weight]
        self.assertIn("basis", state)
        self.assertIn("projected_exp_avg", state)
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)
        self.assertEqual(tuple(state["basis"].shape), (2, 5))
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))
        self.assertNotEqual(tuple(state["projected_exp_avg"].shape), tuple(weight.shape))

    def test_fallback_state_uses_adamw_moments(self):
        bias = torch.nn.Parameter(torch.randn(5))
        opt = SubspaceMuon([bias], lr=0.01)

        bias.square().mean().backward()
        opt.step()

        state = opt.state[bias]
        self.assertEqual(tuple(state["exp_avg"].shape), tuple(bias.shape))
        self.assertEqual(tuple(state["exp_avg_sq"].shape), tuple(bias.shape))

    def test_state_dict_round_trip_preserves_state_shapes(self):
        weight = torch.nn.Parameter(torch.randn(7, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = SubspaceMuon([weight, bias], lr=0.01, rank=2)

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()
        saved = opt.state_dict()

        new_weight = torch.nn.Parameter(weight.detach().clone())
        new_bias = torch.nn.Parameter(bias.detach().clone())
        new_opt = SubspaceMuon([new_weight, new_bias], lr=0.01, rank=2)
        new_opt.load_state_dict(saved)

        new_matrix_state = new_opt.state[new_weight]
        new_fallback_state = new_opt.state[new_bias]
        self.assertEqual(tuple(new_matrix_state["basis"].shape), (2, 4))
        self.assertEqual(tuple(new_matrix_state["projected_exp_avg"].shape), (7, 2))
        self.assertEqual(tuple(new_fallback_state["exp_avg"].shape), (4,))

        new_opt.zero_grad()
        (new_weight.square().mean() + new_bias.square().mean()).backward()
        new_opt.step()
        self.assertEqual(tuple(new_opt.state[new_weight]["projected_exp_avg"].shape), (7, 2))


if __name__ == "__main__":
    unittest.main()
