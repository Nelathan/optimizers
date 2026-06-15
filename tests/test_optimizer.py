import unittest

import torch

import sumotrack
from sumotrack import SumoTrack, optimizer_state_bytes_by_category


class SumoTrackTest(unittest.TestCase):
    def test_public_optimizer_name_has_no_old_alias(self):
        self.assertIs(sumotrack.SumoTrack, SumoTrack)
        self.assertFalse(hasattr(sumotrack, "SubspaceMuon"))

    def test_default_direction_is_fixed_aurora_muon(self):
        weight = torch.nn.Parameter(torch.randn(4, 4))
        opt = SumoTrack([weight])

        self.assertNotIn("orthogonalization", opt.param_groups[0])
        self.assertEqual(opt.param_groups[0]["orthogonalization_scale_mode"], "muon")

    def test_step_updates_matrix_and_fallback_params(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = SumoTrack([weight, bias], lr=0.01, rank=2)
        opt.diagnostics_enabled = True
        weight_before = weight.detach().clone()
        bias_before = bias.detach().clone()

        loss = (weight.square().mean() + bias.square().mean())
        loss.backward()
        opt.step()

        self.assertFalse(torch.equal(weight, weight_before))
        self.assertFalse(torch.equal(bias, bias_before))
        self.assertGreater(opt.last_step_diagnostics["update_norm"], 0.0)
        self.assertGreater(opt.last_step_diagnostics["matrix_update_norm"], 0.0)
        self.assertGreater(opt.last_step_diagnostics["fallback_update_norm"], 0.0)

    def test_matrix_state_keeps_projected_moment_only(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2)

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
        opt = SumoTrack([bias], lr=0.01)

        bias.square().mean().backward()
        opt.step()

        state = opt.state[bias]
        self.assertEqual(tuple(state["exp_avg"].shape), tuple(bias.shape))
        self.assertEqual(tuple(state["exp_avg_sq"].shape), tuple(bias.shape))

    def test_bf16_fallback_uses_fp32_heavyball_moments(self):
        bias = torch.nn.Parameter(torch.randn(5, dtype=torch.bfloat16))
        opt = SumoTrack([bias], lr=0.01)

        bias.float().square().mean().backward()
        opt.step()

        state = opt.state[bias]
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
        self.assertEqual(bias.dtype, torch.bfloat16)

    def test_fallback_matches_heavyball_adamw_one_step(self):
        import heavyball

        torch.manual_seed(3)
        grad = torch.randn(5)
        base = torch.randn(5)
        sumo_bias = torch.nn.Parameter(base.clone())
        heavyball_bias = torch.nn.Parameter(base.clone())
        sumo_opt = SumoTrack([sumo_bias], lr=0.01, fallback_betas=(0.9, 0.99), weight_decay=0.01)
        heavyball_opt = heavyball.AdamW(
            [heavyball_bias],
            lr=0.01,
            betas=(0.9, 0.99),
            weight_decay=0.01,
            compile_step=False,
        )

        sumo_bias.grad = grad.clone()
        heavyball_bias.grad = grad.clone()
        sumo_opt.step()
        heavyball_opt.step()

        self.assertTrue(torch.allclose(sumo_bias, heavyball_bias))

    def test_state_dict_round_trip_preserves_state_shapes(self):
        weight = torch.nn.Parameter(torch.randn(7, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = SumoTrack([weight, bias], lr=0.01, rank=2)

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()
        saved = opt.state_dict()

        new_weight = torch.nn.Parameter(weight.detach().clone())
        new_bias = torch.nn.Parameter(bias.detach().clone())
        new_opt = SumoTrack([new_weight, new_bias], lr=0.01, rank=2)
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

    def test_mixed_path_state_dict_resume_changes_params_after_reload(self):
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(7, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = SumoTrack([weight, bias], lr=0.01, rank=2)

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()
        saved = opt.state_dict()

        new_weight = torch.nn.Parameter(weight.detach().clone())
        new_bias = torch.nn.Parameter(bias.detach().clone())
        new_opt = SumoTrack([new_weight, new_bias], lr=0.01, rank=2)
        new_opt.load_state_dict(saved)
        weight_before = new_weight.detach().clone()
        bias_before = new_bias.detach().clone()

        new_opt.zero_grad()
        (new_weight.square().mean() + new_bias.square().mean()).backward()
        new_opt.step()

        self.assertFalse(torch.equal(new_weight, weight_before))
        self.assertFalse(torch.equal(new_bias, bias_before))
        self.assertEqual(tuple(new_opt.state[new_weight]["projected_exp_avg"].shape), (7, 2))
        self.assertEqual(tuple(new_opt.state[new_bias]["exp_avg"].shape), (4,))
        state_bytes = optimizer_state_bytes_by_category(new_opt)
        self.assertGreater(state_bytes["matrix"], 0)
        self.assertGreater(state_bytes["fallback"], 0)

    def test_ecc_options_fail_loudly_until_heavyball_integration(self):
        weight = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.bfloat16))

        with self.assertRaises(NotImplementedError):
            SumoTrack([weight], ecc="bf16+8")
        with self.assertRaises(NotImplementedError):
            SumoTrack([weight], param_ecc="bf16+8")

    def test_random_subspace_init_wires_into_matrix_state(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2, subspace_init="random")

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        basis = state["basis"]
        gram = basis @ basis.mT
        self.assertEqual(tuple(basis.shape), (2, 5))
        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-5))
        self.assertEqual(opt.param_groups[0]["subspace_init"], "random")

    def test_muon_scale_uses_original_matrix_shape_not_projected_rank(self):
        update = torch.ones(1024, 64)
        ortho = torch.ones_like(update)

        projected_scaled = SumoTrack._scale_orthogonalized_update(update, ortho, "scale", (1024, 512))
        muon_scaled = SumoTrack._scale_orthogonalized_update(update, ortho, "muon", (1024, 512))

        self.assertAlmostEqual(float(projected_scaled[0, 0]), 4.0)
        self.assertAlmostEqual(float(muon_scaled[0, 0]), 2.0**0.5)

    def test_aurora_orthogonalization_keeps_projected_state_shape(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2)

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)

    def test_aurora_balances_rectangular_large_axis_leverage(self):
        torch.manual_seed(2)
        update = torch.randn(256, 16)

        heavyball_update = SumoTrack._heavyball_polar(update)
        aurora_update = SumoTrack._orthogonalize_aurora(
            update,
            {"orthogonalization_scale_mode": "none", "aurora_pp_iterations": 2, "aurora_pp_beta": 0.5},
            update.shape,
        )

        heavyball_cv, _heavyball_min, _heavyball_max = SumoTrack._large_axis_leverage_stats(heavyball_update)
        aurora_cv, aurora_min, aurora_max = SumoTrack._large_axis_leverage_stats(aurora_update)
        self.assertLess(aurora_cv, heavyball_cv)
        self.assertLess(aurora_cv, 0.05)
        self.assertGreater(aurora_min, 0.9)
        self.assertLess(aurora_max, 1.1)

    def test_batched_aurora_balances_rectangular_large_axis_leverage(self):
        torch.manual_seed(4)
        updates = torch.randn(3, 256, 16)

        aurora_updates = SumoTrack._orthogonalize_aurora(
            updates,
            {"orthogonalization_scale_mode": "none", "aurora_pp_iterations": 2, "aurora_pp_beta": 0.5},
            updates.shape[-2:],
        )

        self.assertEqual(tuple(aurora_updates.shape), tuple(updates.shape))
        for aurora_update in aurora_updates:
            aurora_cv, aurora_min, aurora_max = SumoTrack._large_axis_leverage_stats(aurora_update)
            self.assertLess(aurora_cv, 0.05)
            self.assertGreater(aurora_min, 0.9)
            self.assertLess(aurora_max, 1.1)

    def test_same_shape_one_sided_bucket_updates_multiple_params(self):
        torch.manual_seed(5)
        first = torch.nn.Parameter(torch.randn(8, 5))
        second = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([first, second], lr=0.01, rank=2)
        before_first = first.detach().clone()
        before_second = second.detach().clone()

        first.grad = torch.randn_like(first)
        second.grad = torch.randn_like(second)
        opt.step()

        self.assertFalse(torch.equal(first, before_first))
        self.assertFalse(torch.equal(second, before_second))
        self.assertEqual(tuple(opt.state[first]["projected_exp_avg"].shape), (8, 2))
        self.assertEqual(tuple(opt.state[second]["projected_exp_avg"].shape), (8, 2))

    def test_log_norm_diagnostics_include_projected_leverage(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2)
        opt.diagnostics_enabled = True

        weight.grad = torch.randn_like(weight)
        opt.step()

        self.assertIn("mean_projected_leverage_cv", opt.last_step_diagnostics)
        self.assertIn("mean_projected_leverage_min_ratio", opt.last_step_diagnostics)
        self.assertIn("mean_projected_leverage_max_ratio", opt.last_step_diagnostics)
        self.assertLess(opt.last_step_diagnostics["mean_projected_leverage_cv"], 0.1)

    def test_grassmann_refresh_transports_projected_moment(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack(
            [weight],
            lr=0.01,
            rank=2,
            grassmann_step_size=0.01,
            subspace_refresh_budget=1,
        )

        weight.grad = torch.randn_like(weight)
        opt.step()
        state = opt.state[weight]
        old_basis = state["basis"].clone()
        old_moment = state["projected_exp_avg"].clone()

        weight.grad = torch.randn_like(weight)
        old_lifted_moment = old_moment @ old_basis
        opt.step()

        new_basis = state["basis"]
        transported = old_lifted_moment @ new_basis.mT
        self.assertTrue(torch.allclose(state["projected_exp_avg"], 0.9 * transported + 0.1 * (weight.grad @ new_basis.mT), atol=1e-5))
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))


if __name__ == "__main__":
    unittest.main()
