import unittest

import torch

import sumotrack
from sumotrack import SumoTrack, optimizer_state_bytes_by_category


class SumoTrackTest(unittest.TestCase):
    def test_public_optimizer_name_has_no_old_alias(self):
        self.assertIs(sumotrack.SumoTrack, SumoTrack)
        self.assertFalse(hasattr(sumotrack, "SubspaceMuon"))

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

    def test_two_sided_matrix_state_keeps_square_projected_moment(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2, projection_mode="two_sided", subspace_update_method="none")

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        self.assertIn("basis_left", state)
        self.assertIn("basis_right", state)
        self.assertIn("projected_exp_avg", state)
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)
        self.assertEqual(tuple(state["basis_left"].shape), (8, 2))
        self.assertEqual(tuple(state["basis_right"].shape), (2, 5))
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (2, 2))

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

    def test_two_sided_state_dict_round_trip_preserves_state_shapes(self):
        weight = torch.nn.Parameter(torch.randn(7, 4))
        opt = SumoTrack([weight], lr=0.01, rank=2, projection_mode="two_sided", subspace_update_method="none")

        weight.square().mean().backward()
        opt.step()
        saved = opt.state_dict()

        new_weight = torch.nn.Parameter(weight.detach().clone())
        new_opt = SumoTrack([new_weight], lr=0.01, rank=2, projection_mode="two_sided", subspace_update_method="none")
        new_opt.load_state_dict(saved)

        new_state = new_opt.state[new_weight]
        self.assertEqual(tuple(new_state["basis_left"].shape), (7, 2))
        self.assertEqual(tuple(new_state["basis_right"].shape), (2, 4))
        self.assertEqual(tuple(new_state["projected_exp_avg"].shape), (2, 2))

        new_opt.zero_grad()
        new_weight.square().mean().backward()
        before = new_weight.detach().clone()
        new_opt.step()
        self.assertFalse(torch.equal(new_weight, before))
        self.assertEqual(tuple(new_opt.state[new_weight]["projected_exp_avg"].shape), (2, 2))

    def test_mixed_path_state_dict_resume_changes_params_after_reload(self):
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(7, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = SumoTrack([weight, bias], lr=0.01, rank=2, orthogonalization="none")

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()
        saved = opt.state_dict()

        new_weight = torch.nn.Parameter(weight.detach().clone())
        new_bias = torch.nn.Parameter(bias.detach().clone())
        new_opt = SumoTrack([new_weight, new_bias], lr=0.01, rank=2, orthogonalization="none")
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

    def test_two_sided_rejects_unsupported_grassmann_tracking(self):
        weight = torch.nn.Parameter(torch.randn(4, 4))

        with self.assertRaises(ValueError):
            SumoTrack([weight], projection_mode="two_sided", subspace_update_method="grassmann")

    def test_random_subspace_init_wires_into_matrix_state(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2, subspace_init="random", subspace_update_method="grassmann")

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        basis = state["basis"]
        gram = basis @ basis.mT
        self.assertEqual(tuple(basis.shape), (2, 5))
        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-5))
        self.assertEqual(opt.param_groups[0]["subspace_init"], "random")

    def test_no_orthogonalization_uses_projected_momentum_direction(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2, orthogonalization="none")

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        projector = opt._projector_from_state(weight, opt.param_groups[0], state)
        expected_update = projector.project_back(state["projected_exp_avg"])
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))
        self.assertTrue(torch.allclose(opt._orthogonalize_update(state["projected_exp_avg"], opt.param_groups[0]), state["projected_exp_avg"]))
        self.assertEqual(tuple(expected_update.shape), tuple(weight.shape))

    def test_orthogonalization_modes_produce_different_updates(self):
        torch.manual_seed(1)
        grad = torch.randn(8, 5)
        base = torch.randn(8, 5)
        plain = torch.nn.Parameter(base.clone())
        ortho = torch.nn.Parameter(base.clone())
        plain_opt = SumoTrack([plain], lr=0.01, rank=2, orthogonalization="none")
        ortho_opt = SumoTrack([ortho], lr=0.01, rank=2, orthogonalization="svd")

        plain.grad = grad.clone()
        ortho.grad = grad.clone()
        plain_opt.step()
        ortho_opt.step()

        self.assertFalse(torch.allclose(plain, ortho))

    def test_muon_scale_uses_original_matrix_shape_not_projected_rank(self):
        update = torch.ones(1024, 64)
        ortho = torch.ones_like(update)

        projected_scaled = SumoTrack._scale_orthogonalized_update(update, ortho, "scale", (1024, 512))
        muon_scaled = SumoTrack._scale_orthogonalized_update(update, ortho, "muon", (1024, 512))

        self.assertAlmostEqual(float(projected_scaled[0, 0]), 4.0)
        self.assertAlmostEqual(float(muon_scaled[0, 0]), 2.0**0.5)

    def test_heavyball_orthogonalization_keeps_projected_state_shape(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2, orthogonalization="heavyball")

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)

    def test_aurora_orthogonalization_keeps_projected_state_shape(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2, orthogonalization="aurora")

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)

    def test_aurora_balances_rectangular_large_axis_leverage(self):
        torch.manual_seed(2)
        update = torch.randn(256, 16)

        heavyball_update = SumoTrack._orthogonalize_heavyball(
            update,
            {"orthogonalization_scale_mode": "none", "heavyball_orthogonalization_mode": "newtonschulz"},
            update.shape,
        )
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

    def test_log_norm_diagnostics_include_projected_leverage(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack([weight], lr=0.01, rank=2, orthogonalization="aurora")
        opt.diagnostics_enabled = True

        weight.grad = torch.randn_like(weight)
        opt.step()

        self.assertIn("mean_projected_leverage_cv", opt.last_step_diagnostics)
        self.assertIn("mean_projected_leverage_min_ratio", opt.last_step_diagnostics)
        self.assertIn("mean_projected_leverage_max_ratio", opt.last_step_diagnostics)
        self.assertLess(opt.last_step_diagnostics["mean_projected_leverage_cv"], 0.1)

    def test_two_sided_orthogonalization_updates_params_and_logs_square_core(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack(
            [weight],
            lr=0.01,
            rank=2,
            projection_mode="two_sided",
            subspace_update_method="none",
            orthogonalization="heavyball",
        )
        opt.diagnostics_enabled = True
        before = weight.detach().clone()

        weight.grad = torch.randn_like(weight)
        opt.step()

        self.assertFalse(torch.equal(weight, before))
        self.assertEqual(tuple(opt.state[weight]["projected_exp_avg"].shape), (2, 2))
        self.assertGreater(opt.last_step_diagnostics["matrix_update_norm"], 0.0)
        self.assertEqual(opt.last_step_diagnostics["projected_leverage_tensors"], 1)

    def test_grassmann_refresh_transports_projected_moment(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = SumoTrack(
            [weight],
            lr=0.01,
            rank=2,
            subspace_update_method="grassmann",
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
