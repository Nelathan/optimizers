import copy
import math
import unittest
from unittest import mock

import torch

import usuitrack
from usuitrack import SubspaceProjector, UsuiTrack, optimizer_state_bytes_by_category
from usuitrack.optimizer import MatrixUpdate, OJA_STEP_SIZE, ORTHOGONALIZATION_SCALE_MODE


class UsuiTrackTest(unittest.TestCase):
    def test_public_optimizer_name_has_no_legacy_alias(self):
        self.assertIs(usuitrack.UsuiTrack, UsuiTrack)
        self.assertFalse(hasattr(usuitrack, "SumoTrack"))
        self.assertFalse(hasattr(usuitrack, "SubspaceMuon"))

    def test_default_direction_is_fixed_aurora_muon(self):
        weight = torch.nn.Parameter(torch.randn(4, 4))
        opt = UsuiTrack([weight])

        self.assertNotIn("orthogonalization", opt.param_groups[0])
        self.assertEqual(ORTHOGONALIZATION_SCALE_MODE, "muon")
        self.assertEqual(opt.param_groups[0]["aurora_pp_iterations"], 1)
        self.assertEqual(opt.param_groups[0]["polar_ns_steps"], 5)
        self.assertEqual(opt.param_groups[0]["grassmann_aim"], "oja")
        self.assertEqual(opt.param_groups[0]["oja_step_schedule"], "mature")

    def test_step_updates_matrix_and_fallback_params(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = UsuiTrack([weight, bias], lr=0.01, rank=2)
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
        self.assertGreater(opt.last_step_diagnostics["mean_projected_grad_norm"], 0.0)
        # Initialization has no held frame to measure. The next step's capture is
        # deliberately taken before any refresh.
        self.assertTrue(math.isnan(opt.last_step_diagnostics["mean_basis_capture"]))
        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()
        capture = opt.last_step_diagnostics["mean_basis_capture"]
        self.assertGreater(capture, 0.0)
        self.assertLessEqual(capture, 1.0 + 1e-5)

    def test_basis_capture_measures_the_held_frame_before_refresh(self):
        weight = torch.nn.Parameter(torch.zeros(2, 2))
        opt = UsuiTrack(
            [weight],
            lr=0.01,
            rank=1,
            side="left",
            moment_mode="ema",
            basis_refresh_interval=1,
            grassmann_step_size=0.25,
            grassmann_aim="eigh",
        )
        opt.diagnostics_enabled = True

        weight.grad = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        opt.step()  # initializes to e1; no held-frame capture exists yet
        weight.grad = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        opt.step()  # refreshes toward e2, but capture is measured against e1

        self.assertLess(opt.last_step_diagnostics["mean_basis_capture"], 1e-6)

    def test_eigh_aim_refresh_rotates_and_logs_the_q10_probe(self):
        torch.manual_seed(0)
        weight = torch.nn.Parameter(torch.randn(12, 6))
        opt = UsuiTrack([weight], lr=0.01, rank=3, basis_refresh_interval=2, grassmann_aim="eigh")
        opt.diagnostics_enabled = True
        opt.diagnostics_basis_enabled = True

        diagnostics = {}
        # Refresh-only metrics live on boundary steps, so keep the last boundary's
        # diagnostics rather than whatever step the loop happens to end on.
        for _ in range(6):
            opt.zero_grad()
            (weight @ torch.randn(6, 6)).square().mean().backward()
            opt.step()
            if opt.last_step_diagnostics["basis_refresh_tensors"] > 0.0:
                diagnostics = opt.last_step_diagnostics
        self.assertGreater(diagnostics["basis_refresh_tensors"], 0.0)
        self.assertGreater(diagnostics["mean_basis_target_angle_mass"], 0.0)
        self.assertGreater(diagnostics["mean_rotation_angle"], 0.0)
        self.assertLessEqual(diagnostics["mean_rotation_angle"], diagnostics["mean_basis_target_angle_mass"] + 1e-5)
        # No tangent-accumulation state or stale target stream under the eigh aim.
        state = opt.state[weight]
        self.assertIsNone(state.get("tangent_accum"))
        self.assertNotIn("prev_eigh_target", state)

    def test_oja_moves_live_basis_every_gradient_without_second_frame(self):
        torch.manual_seed(53)
        weight = torch.nn.Parameter(torch.randn(12, 8, dtype=torch.bfloat16))
        opt = UsuiTrack(
            [weight],
            lr=0.01,
            rank=4,
            side="right",
            moment_mode="ema",
            grad_clip_norm=None,
            basis_refresh_interval=10,
            grassmann_aim="oja",
        )

        weight.grad = torch.randn_like(weight)
        opt.step()
        initial_basis = opt.state[weight]["basis"].clone()
        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        self.assertFalse(torch.equal(state["basis"], initial_basis))
        self.assertNotIn("oja_target_basis", state)
        self.assertEqual(state["basis"].dtype, torch.bfloat16)
        self.assertLess(float((state["basis"].float() @ state["basis"].float().mT - torch.eye(4)).norm()), 2e-2)

    def test_mature_oja_step_schedule_is_harmonic_then_floored(self):
        group = {"oja_step_schedule": "mature", "basis_refresh_step": 2}
        self.assertEqual(UsuiTrack._oja_step_size(group), 0.5)
        group["basis_refresh_step"] = 3
        self.assertAlmostEqual(UsuiTrack._oja_step_size(group), 1.0 / 3.0)
        group["basis_refresh_step"] = 100
        self.assertEqual(UsuiTrack._oja_step_size(group), OJA_STEP_SIZE)
        group["basis_refresh_step"] = 1000
        self.assertEqual(UsuiTrack._oja_step_size(group), OJA_STEP_SIZE)

    def test_mature_oja_uses_scheduled_step_after_eigh_initialization(self):
        torch.manual_seed(55)
        weight = torch.nn.Parameter(torch.randn(10, 6))
        opt = UsuiTrack(
            [weight],
            lr=0.01,
            rank=3,
            side="right",
            moment_mode="ema",
            grad_clip_norm=None,
            grassmann_aim="oja",
            oja_step_schedule="mature",
        )
        opt.diagnostics_enabled = True
        opt.diagnostics_basis_enabled = True

        weight.grad = torch.randn_like(weight)
        opt.step()
        weight.grad = torch.randn_like(weight)
        with mock.patch(
            "usuitrack.optimizer.SubspaceProjector.oja_geodesic_from_eigh",
            wraps=SubspaceProjector.oja_geodesic_from_eigh,
        ) as geodesic:
            opt.step()

        self.assertEqual(geodesic.call_args.args[4], 0.5)

    def test_oja_reuses_held_projection_as_moving_frame_coordinates(self):
        torch.manual_seed(59)
        weight = torch.nn.Parameter(torch.randn(10, 6))
        opt = UsuiTrack(
            [weight],
            lr=0.01,
            beta=0.9,
            rank=3,
            side="right",
            moment_mode="ema",
            grad_clip_norm=None,
            grassmann_aim="oja",
        )

        weight.grad = torch.randn_like(weight)
        opt.step()
        state = opt.state[weight]
        old_basis = state["basis"].clone()
        old_moment = state["projected_exp_avg"].clone()
        gradient = torch.randn_like(weight)
        held_projection = gradient @ old_basis.mT
        weight.grad = gradient
        opt.step()

        self.assertFalse(torch.equal(state["basis"], old_basis))
        torch.testing.assert_close(state["projected_exp_avg"], 0.9 * old_moment + 0.1 * held_projection)

    def test_oja_batches_same_rank_eigendecompositions(self):
        torch.manual_seed(61)
        weights = [torch.nn.Parameter(torch.randn(10, 6)), torch.nn.Parameter(torch.randn(8, 7))]
        opt = UsuiTrack(
            weights,
            lr=0.01,
            rank=3,
            side="right",
            moment_mode="ema",
            grad_clip_norm=None,
            grassmann_aim="oja",
        )

        for weight in weights:
            weight.grad = torch.randn_like(weight)
        opt.step()

        for weight in weights:
            weight.grad = torch.randn_like(weight)
        with mock.patch("torch.linalg.eigh", wraps=torch.linalg.eigh) as eigh:
            opt.step()

        self.assertEqual(eigh.call_count, 1)
        self.assertEqual(tuple(eigh.call_args.args[0].shape), (2, 3, 3))

    def test_deferred_batched_oja_matches_sequential_parameter_groups(self):
        torch.manual_seed(63)
        initial = [torch.randn(10, 6), torch.randn(7, 11)]  # auto resolves right, then left
        batched_params = [torch.nn.Parameter(value.clone()) for value in initial]
        sequential_params = [torch.nn.Parameter(value.clone()) for value in initial]
        kwargs = dict(
            lr=0.01,
            beta=0.9,
            rank=3,
            side="auto",
            moment_mode="adafactor_ema",
            grad_clip_norm=None,
            grassmann_aim="oja",
        )
        batched = UsuiTrack(batched_params, **kwargs)
        sequential = UsuiTrack([{"params": [param]} for param in sequential_params], **kwargs)

        for _ in range(6):
            gradients = [torch.randn_like(value) for value in initial]
            for param, gradient in zip(batched_params, gradients, strict=True):
                param.grad = gradient.clone()
            for param, gradient in zip(sequential_params, gradients, strict=True):
                param.grad = gradient.clone()
            batched.step()
            sequential.step()

        tensor_state = (
            "basis",
            "projected_exp_avg",
            "adafactor_row_var",
            "adafactor_col_var",
        )
        for batched_param, sequential_param in zip(batched_params, sequential_params, strict=True):
            torch.testing.assert_close(batched_param, sequential_param, atol=3e-5, rtol=3e-5)
            batched_state = batched.state[batched_param]
            sequential_state = sequential.state[sequential_param]
            for key in tensor_state:
                torch.testing.assert_close(batched_state[key], sequential_state[key], atol=3e-5, rtol=3e-5)
            self.assertEqual(batched_state["step"], sequential_state["step"])
            self.assertEqual(batched_state["adafactor_step"], sequential_state["adafactor_step"])

    def test_oja_state_dict_continuation_is_deterministic(self):
        torch.manual_seed(67)
        first = torch.nn.Parameter(torch.randn(10, 6, dtype=torch.bfloat16))
        kwargs = dict(
            lr=0.01,
            rank=3,
            side="right",
            moment_mode="ema",
            grad_clip_norm=None,
            grassmann_aim="oja",
        )
        first_opt = UsuiTrack([first], **kwargs)
        for _ in range(4):
            first.grad = torch.randn_like(first)
            first_opt.step()

        second = torch.nn.Parameter(first.detach().clone())
        second_opt = UsuiTrack([second], **kwargs)
        second_opt.load_state_dict(copy.deepcopy(first_opt.state_dict()))
        gradient = torch.randn_like(first)
        first.grad = gradient.clone()
        second.grad = gradient.clone()
        first_opt.step()
        second_opt.step()

        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first_opt.state[first]["basis"], second_opt.state[second]["basis"])
        torch.testing.assert_close(
            first_opt.state[first]["projected_exp_avg"],
            second_opt.state[second]["projected_exp_avg"],
        )

    def test_oja_step_is_fixed_replay_contract(self):
        self.assertEqual(OJA_STEP_SIZE, 0.01)

    def test_projected_grad_clip_bounds_each_projected_matrix_input(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, beta=0.0, rank=2, side="right", projected_grad_clip_norm=1.0, basis_refresh_interval=100, moment_mode="ema", grassmann_aim="eigh")
        opt.diagnostics_enabled = True

        weight.grad = torch.randn_like(weight)
        opt.step()
        projector = opt._projector_from_state(weight, opt.param_groups[0], opt.state[weight])
        projected_grad = projector.project(torch.randn_like(weight))
        projected_grad.mul_(10.0 / projected_grad.float().norm())

        opt.queue_projected_grad(weight, projected_grad.clone())
        opt.step()

        self.assertAlmostEqual(opt.last_step_diagnostics["mean_projected_grad_norm"], 10.0, places=4)
        self.assertLessEqual(float(opt.state[weight]["projected_exp_avg"].float().norm()), 1.0001)

    def test_projected_grad_ratio_clip_bounds_gradient_relative_to_moment(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, beta=0.0, rank=2, side="right", projected_grad_clip_ratio=2.0, basis_refresh_interval=100, moment_mode="ema", grassmann_aim="eigh")
        opt.diagnostics_enabled = True

        weight.grad = torch.randn_like(weight)
        opt.step()
        projector = opt._projector_from_state(weight, opt.param_groups[0], opt.state[weight])
        moment = projector.project(torch.randn_like(weight))
        moment.mul_(0.5 / moment.float().norm())
        opt.state[weight]["projected_exp_avg"] = moment.clone()
        projected_grad = moment * 10.0

        opt.queue_projected_grad(weight, projected_grad.clone())
        opt.step()

        self.assertAlmostEqual(opt.last_step_diagnostics["mean_projected_grad_norm"], 5.0, places=4)
        self.assertAlmostEqual(opt.last_step_diagnostics["mean_projected_grad_to_moment_ratio"], 10.0, places=4)
        self.assertLessEqual(float(opt.state[weight]["projected_exp_avg"].float().norm()), 1.0001)

    def test_step_consumes_grads_after_projection_by_default(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = UsuiTrack([weight, bias], lr=0.01, rank=2)

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()

        self.assertIsNone(weight.grad)
        self.assertIsNone(bias.grad)

    def test_consume_grad_can_be_disabled_for_debugging(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2, consume_grad=False)

        weight.square().mean().backward()
        opt.step()

        self.assertIsNotNone(weight.grad)

    def test_hook_and_ordinary_step_share_matrix_preparation_seam(self):
        ordinary_weight = torch.nn.Parameter(torch.randn(6, 4))
        released_weight = torch.nn.Parameter(ordinary_weight.detach().clone())
        ordinary = UsuiTrack([ordinary_weight], rank=2)
        released = UsuiTrack([released_weight], rank=2, release_matrix_grads=True)

        with mock.patch.object(ordinary, "_prepare_matrix_param", wraps=ordinary._prepare_matrix_param) as ordinary_prepare:
            ordinary_weight.grad = torch.randn_like(ordinary_weight)
            ordinary.step()
        with mock.patch.object(released, "_prepare_matrix_param", wraps=released._prepare_matrix_param) as released_prepare:
            released_weight.square().sum().backward()

        self.assertEqual(ordinary_prepare.call_count, 1)
        self.assertEqual(released_prepare.call_count, 1)

    def test_explicit_prepare_is_reused_by_step_without_recomputation(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2)
        weight.grad = torch.randn_like(weight)

        with mock.patch.object(opt, "_prepare_matrix_update", wraps=opt._prepare_matrix_update) as compute:
            opt.prepare(weight)
            prepared = opt._pending_matrix_updates[weight]
            self.assertIsNone(weight.grad)
            opt.step()

        self.assertEqual(compute.call_count, 1)
        self.assertNotIn(weight, opt._pending_matrix_updates)
        self.assertIsNotNone(prepared.projected_exp_avg)

    def test_mixed_pending_and_fresh_matches_all_fresh_with_one_group_advance(self):
        torch.manual_seed(71)
        initial = [torch.randn(8, 5), torch.randn(8, 5)]
        mixed_params = [torch.nn.Parameter(value.clone()) for value in initial]
        fresh_params = [torch.nn.Parameter(value.clone()) for value in initial]
        kwargs = dict(lr=0.01, beta=0.9, rank=3, side="right", moment_mode="ema", grad_clip_norm=None)
        mixed = UsuiTrack(mixed_params, **kwargs)
        fresh = UsuiTrack(fresh_params, **kwargs)

        for _ in range(2):
            gradients = [torch.randn_like(value) for value in initial]
            for param, gradient in zip(mixed_params, gradients, strict=True):
                param.grad = gradient.clone()
            for param, gradient in zip(fresh_params, gradients, strict=True):
                param.grad = gradient.clone()
            if mixed.param_groups[0]["basis_refresh_step"]:
                mixed.prepare(mixed_params[0])
            mixed.step()
            fresh.step()

        self.assertEqual(mixed.param_groups[0]["basis_refresh_step"], 2)
        self.assertEqual(mixed.param_groups[0]["basis_refresh_step"], fresh.param_groups[0]["basis_refresh_step"])
        for mixed_param, fresh_param in zip(mixed_params, fresh_params, strict=True):
            torch.testing.assert_close(mixed_param, fresh_param, rtol=0, atol=0)
            for key, fresh_value in fresh.state[fresh_param].items():
                mixed_value = mixed.state[mixed_param][key]
                if isinstance(fresh_value, torch.Tensor):
                    torch.testing.assert_close(mixed_value, fresh_value, rtol=0, atol=0)
                else:
                    self.assertEqual(mixed_value, fresh_value)

    def test_duplicate_prepare_raises_before_mutating_state(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], rank=2)
        weight.grad = torch.randn_like(weight)
        opt.prepare(weight)
        state_before = copy.deepcopy(opt.state[weight])
        pending_before = opt._pending_matrix_updates[weight]
        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            opt.queue_projected_grad(weight, torch.randn(6, 2))
        closure = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "closures cannot run"):
            opt.step(closure)
        closure.assert_not_called()
        new_grad = torch.randn_like(weight)
        weight.grad = new_grad

        with self.assertRaisesRegex(RuntimeError, "already prepared"):
            opt.prepare(weight)

        self.assertIs(opt._pending_matrix_updates[weight], pending_before)
        self.assertIs(weight.grad, new_grad)
        self.assertEqual(opt.state[weight].keys(), state_before.keys())
        for key, before in state_before.items():
            after = opt.state[weight][key]
            if isinstance(before, torch.Tensor):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
            else:
                self.assertEqual(after, before)

    def test_step_validates_all_groups_before_applying_any_update(self):
        first = torch.nn.Parameter(torch.randn(6, 4))
        second = torch.nn.Parameter(torch.randn(5, 3))
        opt = UsuiTrack([{"params": [first]}, {"params": [second]}], lr=0.01, rank=2)
        first.grad = torch.randn_like(first)
        second.grad = torch.randn_like(second)
        opt.prepare(second)
        second.grad = torch.randn_like(second)
        first_before = first.detach().clone()
        first_step_before = opt.param_groups[0]["basis_refresh_step"]

        with self.assertRaisesRegex(RuntimeError, "already prepared|prepared matrix parameter"):
            opt.step()

        torch.testing.assert_close(first, first_before, rtol=0, atol=0)
        self.assertEqual(opt.param_groups[0]["basis_refresh_step"], first_step_before)
        self.assertNotIn("step", opt.state[first])

    def test_matrix_group_lookup_supports_params_added_after_construction(self):
        first = torch.nn.Parameter(torch.randn(6, 4))
        second = torch.nn.Parameter(torch.randn(5, 3))
        opt = UsuiTrack([first], lr=0.01, rank=2)
        opt.add_param_group({"params": [second]})
        second.grad = torch.randn_like(second)

        opt.prepare(second)
        self.assertIn(second, opt._pending_matrix_updates)
        opt.step()

        self.assertIn("step", opt.state[second])

    def test_prepare_rejects_foreign_non_matrix_and_non_consumable_params(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        frozen = torch.nn.Parameter(torch.randn(4, 3), requires_grad=False)
        bias = torch.nn.Parameter(torch.randn(4))
        foreign = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight, frozen, bias], rank=2)

        self.assertIs(opt._matrix_param_groups[weight], opt.param_groups[0])
        self.assertIs(opt._matrix_param_groups[frozen], opt.param_groups[0])

        with self.assertRaisesRegex(ValueError, "not owned"):
            opt.prepare(foreign)
        with self.assertRaisesRegex(ValueError, "only supports 2D"):
            opt.prepare(bias)

        retained = UsuiTrack([weight], rank=2, consume_grad=False)
        weight.grad = torch.randn_like(weight)
        with self.assertRaisesRegex(RuntimeError, "requires consume_grad=True"):
            retained.prepare(weight)

    def test_released_matrix_grads_match_ordinary_batched_step_and_state(self):
        torch.manual_seed(73)
        ordinary_model = torch.nn.Sequential(
            torch.nn.Linear(7, 9),
            torch.nn.SiLU(),
            torch.nn.Linear(9, 5),
        )
        released_model = copy.deepcopy(ordinary_model)
        kwargs = dict(
            lr=0.01,
            rank=3,
            side="auto",
            moment_mode="adafactor_ema",
            grad_clip_norm=1.0,
            grassmann_aim="oja",
        )
        ordinary = UsuiTrack(ordinary_model.parameters(), **kwargs)
        released = UsuiTrack(released_model.parameters(), release_matrix_grads=True, **kwargs)
        ordinary.diagnostics_enabled = True
        released.diagnostics_enabled = True

        for _ in range(4):
            inputs = torch.randn(6, 7)
            targets = torch.randn(6, 5)
            ordinary.zero_grad()
            released.zero_grad()
            torch.nn.functional.mse_loss(ordinary_model(inputs), targets).backward()
            torch.nn.functional.mse_loss(released_model(inputs), targets).backward()

            released_params = list(released_model.parameters())
            self.assertIsNone(released_params[0].grad)
            self.assertIsNotNone(released_params[1].grad)
            self.assertIsNone(released_params[2].grad)
            self.assertIsNotNone(released_params[3].grad)
            self.assertEqual(len(released._pending_matrix_updates), 2)

            ordinary.step()
            released.step()

            for ordinary_param, released_param in zip(ordinary_model.parameters(), released_model.parameters(), strict=True):
                torch.testing.assert_close(released_param, ordinary_param, rtol=0, atol=0)
                ordinary_state = ordinary.state[ordinary_param]
                released_state = released.state[released_param]
                self.assertEqual(ordinary_state.keys(), released_state.keys())
                for key, ordinary_value in ordinary_state.items():
                    released_value = released_state[key]
                    if isinstance(ordinary_value, torch.Tensor):
                        torch.testing.assert_close(released_value, ordinary_value, rtol=0, atol=0)
                    else:
                        self.assertEqual(released_value, ordinary_value)
            self.assertEqual(released.param_groups[0]["basis_refresh_step"], ordinary.param_groups[0]["basis_refresh_step"])
            self.assertEqual(released._pending_matrix_updates, {})

    def test_released_matrix_grads_reject_accumulation_and_discard(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2, release_matrix_grads=True)
        opt.zero_grad()
        weight.square().mean().backward()

        with self.assertRaisesRegex(RuntimeError, "cannot discard released matrix updates"):
            opt.zero_grad()
        with self.assertRaisesRegex(RuntimeError, "does not support gradient accumulation"):
            weight.square().mean().backward()

    def test_released_matrix_grads_require_consumed_backward_grads(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        with self.assertRaisesRegex(ValueError, "requires consume_grad=True"):
            UsuiTrack([weight], release_matrix_grads=True, consume_grad=False)

        opt = UsuiTrack([weight], release_matrix_grads=True)
        weight.grad = torch.randn_like(weight)
        with self.assertRaisesRegex(RuntimeError, "produced by backward hooks"):
            opt.step()

    def test_queued_projected_grad_matches_full_gradient_step(self):
        torch.manual_seed(24)
        base = torch.randn(7, 5, dtype=torch.float64)
        warm_grad = torch.randn_like(base)
        step_grad = torch.randn_like(base)
        full_weight = torch.nn.Parameter(base.clone())
        queued_weight = torch.nn.Parameter(base.clone())
        # grad_clip_norm=None: the raw-grad clip lives upstream of projection, so the
        # full-grad path clips while the queued-projected path structurally cannot --
        # this test asserts the projection equivalence, so keep the clip out of it.
        full_opt = UsuiTrack([full_weight], lr=0.01, beta=0.9, rank=3, side="right", basis_refresh_interval=100, moment_mode="ema", grad_clip_norm=None, grassmann_aim="eigh")
        queued_opt = UsuiTrack([queued_weight], lr=0.01, beta=0.9, rank=3, side="right", basis_refresh_interval=100, moment_mode="ema", grad_clip_norm=None, grassmann_aim="eigh")

        full_weight.grad = warm_grad.clone()
        queued_weight.grad = warm_grad.clone()
        full_opt.step()
        queued_opt.step()

        full_weight.grad = step_grad.clone()
        projector = full_opt._projector_from_state(full_weight, full_opt.param_groups[0], full_opt.state[full_weight])
        full_projected_grad = projector.project(step_grad)
        queued_opt.queue_projected_grad(queued_weight, full_projected_grad.clone())

        full_opt.step()
        queued_opt.step()

        self.assertIsNone(queued_weight.grad)
        self.assertTrue(torch.allclose(queued_weight, full_weight, atol=1e-12))
        self.assertTrue(torch.allclose(queued_opt.state[queued_weight]["projected_exp_avg"], full_opt.state[full_weight]["projected_exp_avg"], atol=1e-12))
        self.assertEqual(queued_opt._queued_projected_grads, {})

    def test_queued_projected_grad_requires_initialized_basis(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2, side="right", moment_mode="ema", grassmann_aim="eigh")

        opt.queue_projected_grad(weight, torch.randn(6, 2))

        with self.assertRaisesRegex(RuntimeError, "initialized"):
            opt.step()

    def test_queued_projected_grad_rejects_refresh_step_without_full_grad(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2, side="right", basis_refresh_interval=1, moment_mode="ema", grassmann_aim="eigh")
        weight.grad = torch.randn_like(weight)
        opt.step()
        projector = opt._projector_from_state(weight, opt.param_groups[0], opt.state[weight])

        opt.queue_projected_grad(weight, projector.project(torch.randn_like(weight)))

        with self.assertRaisesRegex(RuntimeError, "refresh"):
            opt.step()

    def test_default_oja_rejects_queued_projected_grad(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2, side="right", moment_mode="ema")
        weight.grad = torch.randn_like(weight)
        opt.step()
        projector = opt._projector_from_state(weight, opt.param_groups[0], opt.state[weight])
        opt.queue_projected_grad(weight, projector.project(torch.randn_like(weight)))

        with self.assertRaisesRegex(RuntimeError, "full matrix gradient on every step"):
            opt.step()

    def test_basis_refresh_offsets_stagger_due_params_after_first_interval(self):
        first = torch.nn.Parameter(torch.randn(6, 4))
        second = torch.nn.Parameter(torch.randn(6, 4))
        group = {"params": [first, second], "basis_refresh_offsets": {id(first): 0, id(second): 1}}
        opt = UsuiTrack([group], lr=0.01, rank=2, side="right", basis_refresh_interval=3, grassmann_aim="eigh")
        matrix_params = [first, second]

        self.assertEqual(opt._refresh_param_ids(opt.param_groups[0], matrix_params), set())
        self.assertEqual(opt._refresh_param_ids(opt.param_groups[0], matrix_params), set())
        self.assertEqual(opt._refresh_param_ids(opt.param_groups[0], matrix_params), set())
        self.assertEqual(opt._refresh_param_ids(opt.param_groups[0], matrix_params), {id(first)})
        self.assertEqual(opt._refresh_param_ids(opt.param_groups[0], matrix_params), {id(second)})

    def test_zero_grad_clears_queued_projected_grads(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2, side="right")
        opt.queue_projected_grad(weight, torch.randn(6, 2))

        opt.zero_grad()

        self.assertEqual(opt._queued_projected_grads, {})

    def test_aurora_cycle_counts_are_configurable(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = UsuiTrack([weight], lr=0.01, rank=2, aurora_pp_iterations=1, polar_ns_steps=3)

        weight.grad = torch.randn_like(weight)
        opt.step()

        self.assertEqual(opt.param_groups[0]["aurora_pp_iterations"], 1)
        self.assertEqual(opt.param_groups[0]["polar_ns_steps"], 3)
        self.assertEqual(tuple(opt.state[weight]["projected_exp_avg"].shape), (8, 2))

    def test_compile_tensor_kernels_targets_orthogonalization_and_default_prepare(self):
        compiled_calls = []

        def fake_compile(fn, **kwargs):
            compiled_calls.append((fn, kwargs))
            return fn

        with mock.patch("torch.compile", side_effect=fake_compile):
            weight = torch.nn.Parameter(torch.randn(8, 5))
            bias = torch.nn.Parameter(torch.randn(5))
            opt = UsuiTrack([weight, bias], lr=0.01, rank=2, compile_tensor_kernels=True)

        self.assertEqual(
            compiled_calls,
            [
                (UsuiTrack._orthogonalize_aurora_muon_tensor, {}),
                (UsuiTrack._prepare_oja_adafactor_right_tensors, {"dynamic": True}),
                (UsuiTrack._prepare_oja_adafactor_left_tensors, {"dynamic": True}),
            ],
        )
        self.assertIsNotNone(opt._compiled_orthogonalize_update)
        self.assertIsNotNone(opt._compiled_prepare_oja_adafactor_right)
        self.assertIsNotNone(opt._compiled_prepare_oja_adafactor_left)

        weight.grad = torch.randn_like(weight)
        bias.grad = torch.randn_like(bias)
        opt.step()

        self.assertEqual(tuple(opt.state[weight]["projected_exp_avg"].shape), (8, 2))
        self.assertIn("exp_avg", opt.state[bias])

    def test_matrix_state_keeps_projected_moment_only(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = UsuiTrack([weight], lr=0.01, rank=2)

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
        opt = UsuiTrack([bias], lr=0.01)

        bias.square().mean().backward()
        opt.step()

        state = opt.state[bias]
        self.assertEqual(tuple(state["exp_avg"].shape), tuple(bias.shape))
        self.assertEqual(tuple(state["exp_avg_sq"].shape), tuple(bias.shape))

    def test_bf16_fallback_uses_fp32_adamw_moments(self):
        bias = torch.nn.Parameter(torch.randn(5, dtype=torch.bfloat16))
        opt = UsuiTrack([bias], lr=0.01)

        bias.float().square().mean().backward()
        opt.step()

        state = opt.state[bias]
        self.assertEqual(state["exp_avg"].dtype, torch.float32)
        self.assertEqual(state["exp_avg_sq"].dtype, torch.float32)
        self.assertEqual(bias.dtype, torch.bfloat16)

    def test_fallback_matches_torch_adamw_one_step(self):
        torch.manual_seed(3)
        grad = torch.randn(5)
        base = torch.randn(5)
        usui_bias = torch.nn.Parameter(base.clone())
        torch_bias = torch.nn.Parameter(base.clone())
        usui_opt = UsuiTrack([usui_bias], lr=0.01, fallback_betas=(0.9, 0.99), weight_decay=0.01)
        torch_opt = torch.optim.AdamW(
            [torch_bias],
            lr=0.01,
            betas=(0.9, 0.99),
            weight_decay=0.01,
            foreach=False,
            fused=False,
        )

        usui_bias.grad = grad.clone()
        torch_bias.grad = grad.clone()
        usui_opt.step()
        torch_opt.step()

        self.assertTrue(torch.allclose(usui_bias, torch_bias))

    def test_state_dict_round_trip_preserves_state_shapes(self):
        weight = torch.nn.Parameter(torch.randn(7, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = UsuiTrack([weight, bias], lr=0.01, rank=2)

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()
        saved = opt.state_dict()

        new_weight = torch.nn.Parameter(weight.detach().clone())
        new_bias = torch.nn.Parameter(bias.detach().clone())
        new_opt = UsuiTrack([new_weight, new_bias], lr=0.01, rank=2)
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
        opt = UsuiTrack([weight, bias], lr=0.01, rank=2)

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()
        saved = opt.state_dict()

        new_weight = torch.nn.Parameter(weight.detach().clone())
        new_bias = torch.nn.Parameter(bias.detach().clone())
        new_opt = UsuiTrack([new_weight, new_bias], lr=0.01, rank=2)
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
            UsuiTrack([weight], ecc="bf16+8")
        with self.assertRaises(NotImplementedError):
            UsuiTrack([weight], param_ecc="bf16+8")

    def test_random_basis_init_wires_into_matrix_state(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = UsuiTrack([weight], lr=0.01, rank=2, basis_init="random")

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        basis = state["basis"]
        gram = basis @ basis.mT
        self.assertEqual(tuple(basis.shape), (2, 5))
        self.assertTrue(torch.allclose(gram, torch.eye(2), atol=1e-5))
        self.assertEqual(opt.param_groups[0]["basis_init"], "random")

    def test_muon_scale_uses_original_matrix_shape_not_projected_rank(self):
        update = torch.ones(1024, 64)
        ortho = torch.ones_like(update)

        projected_scaled = UsuiTrack._scale_orthogonalized_update(update, ortho, "scale", (1024, 512))
        muon_scaled = UsuiTrack._scale_orthogonalized_update(update, ortho, "muon", (1024, 512))

        self.assertAlmostEqual(float(projected_scaled[0, 0]), 4.0)
        self.assertAlmostEqual(float(muon_scaled[0, 0]), 2.0**0.5)

    def test_aurora_orthogonalization_keeps_projected_state_shape(self):
        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = UsuiTrack([weight], lr=0.01, rank=2)

        weight.grad = torch.randn_like(weight)
        opt.step()

        state = opt.state[weight]
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)

    def test_aurora_balances_rectangular_large_axis_leverage(self):
        torch.manual_seed(2)
        update = torch.randn(256, 16)

        heavyball_update = UsuiTrack._heavyball_polar(update)
        aurora_update = UsuiTrack._orthogonalize_aurora(
            update,
            {},
            update.shape,
        )

        heavyball_cv, _heavyball_min, _heavyball_max = UsuiTrack._large_axis_leverage_stats(heavyball_update)
        aurora_cv, aurora_min, aurora_max = UsuiTrack._large_axis_leverage_stats(aurora_update)
        self.assertLess(aurora_cv, heavyball_cv)
        self.assertLess(aurora_cv, 0.1)
        self.assertGreater(aurora_min, 0.75)
        self.assertLess(aurora_max, 1.3)

    def test_batched_aurora_balances_rectangular_large_axis_leverage(self):
        torch.manual_seed(4)
        updates = torch.randn(3, 256, 16)

        aurora_updates = UsuiTrack._orthogonalize_aurora(
            updates,
            {},
            updates.shape[-2:],
        )

        self.assertEqual(tuple(aurora_updates.shape), tuple(updates.shape))
        for update, aurora_update in zip(updates, aurora_updates):
            heavyball_cv, _heavyball_min, _heavyball_max = UsuiTrack._large_axis_leverage_stats(UsuiTrack._heavyball_polar(update))
            aurora_cv, aurora_min, aurora_max = UsuiTrack._large_axis_leverage_stats(aurora_update)
            self.assertLess(aurora_cv, heavyball_cv)
            self.assertLess(aurora_cv, 0.1)
            self.assertGreater(aurora_min, 0.75)
            self.assertLess(aurora_max, 1.3)

    def test_same_shape_one_sided_bucket_updates_multiple_params(self):
        torch.manual_seed(5)
        first = torch.nn.Parameter(torch.randn(8, 5))
        second = torch.nn.Parameter(torch.randn(8, 5))
        opt = UsuiTrack([first, second], lr=0.01, rank=2)
        before_first = first.detach().clone()
        before_second = second.detach().clone()

        first.grad = torch.randn_like(first)
        second.grad = torch.randn_like(second)
        opt.step()

        self.assertFalse(torch.equal(first, before_first))
        self.assertFalse(torch.equal(second, before_second))
        self.assertEqual(tuple(opt.state[first]["projected_exp_avg"].shape), (8, 2))
        self.assertEqual(tuple(opt.state[second]["projected_exp_avg"].shape), (8, 2))

    def test_refresh_interval_updates_all_bases_on_interval_step(self):
        params = [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(3)]
        opt = UsuiTrack(params, basis_refresh_interval=2, grassmann_aim="eigh")
        group = opt.param_groups[0]

        first = opt._refresh_param_ids(group, params)
        second = opt._refresh_param_ids(group, params)
        third = opt._refresh_param_ids(group, params)

        self.assertEqual(first, set())
        self.assertEqual(second, set())
        self.assertEqual(third, {id(param) for param in params})

    def test_log_norm_diagnostics_include_projected_leverage(self):
        weight = torch.nn.Parameter(torch.randn(256, 16))
        opt = UsuiTrack([weight], lr=0.01, rank=16)
        opt.diagnostics_enabled = True
        opt.diagnostics_leverage_enabled = True

        weight.grad = torch.randn_like(weight)
        opt.step()

        self.assertIn("mean_projected_leverage_cv", opt.last_step_diagnostics)
        self.assertIn("mean_projected_leverage_min_ratio", opt.last_step_diagnostics)
        self.assertIn("mean_projected_leverage_max_ratio", opt.last_step_diagnostics)
        self.assertLess(opt.last_step_diagnostics["mean_projected_leverage_cv"], 0.1)

    def test_basis_refresh_diagnostics_measure_rotation(self):
        torch.manual_seed(6)
        weight = torch.nn.Parameter(torch.randn(16, 8))
        opt = UsuiTrack([weight], lr=0.01, rank=4, basis_refresh_interval=1, grassmann_aim="eigh")

        weight.grad = torch.randn_like(weight)
        opt.step()

        opt.diagnostics_enabled = True
        opt.diagnostics_basis_enabled = True
        weight.grad = torch.randn_like(weight)
        opt.step()

        diagnostics = opt.last_step_diagnostics
        self.assertEqual(diagnostics["basis_refresh_tensors"], 1.0)
        self.assertGreaterEqual(diagnostics["mean_rotation_angle"], 0.0)
        self.assertNotIn("mean_basis_capture_before", diagnostics)
        self.assertNotIn("mean_basis_rotation_top1_sin", diagnostics)

    def test_grassmann_refresh_parallel_transports_projected_moment(self):
        """Across a geodesic refresh the projected moment's COORDINATES are
        unchanged (parallel transport = identity in the rotating frame; the
        retraction is a rigid frame rotation, pinned at the projector level).
        The new grad's EMA contribution is projected with the NEW basis.
        """

        weight = torch.nn.Parameter(torch.randn(8, 5))
        opt = UsuiTrack(
            [weight],
            lr=0.01,
            rank=2,
            grassmann_step_size=0.01,
            basis_refresh_interval=1,
            moment_mode="ema",
            # This test asserts exact moment math against the raw grad; keep the
            # per-tensor clip out of the way (a randn(8,5) grad's norm ~6 exceeds
            # the 2.5 default rail).
            grad_clip_norm=None,
            grassmann_aim="eigh",
        )

        weight.grad = torch.randn_like(weight)
        opt.step()
        state = opt.state[weight]
        old_basis = state["basis"].clone()
        old_moment = state["projected_exp_avg"].clone()

        second_grad = torch.randn_like(weight)
        weight.grad = second_grad.clone()
        opt.step()

        new_basis = state["basis"]
        self.assertFalse(torch.allclose(new_basis, old_basis))
        beta = opt.param_groups[0]["beta"]
        self.assertTrue(torch.allclose(state["projected_exp_avg"], beta * old_moment + (1.0 - beta) * (second_grad @ new_basis.mT), atol=1e-5))
        self.assertEqual(tuple(state["projected_exp_avg"].shape), (8, 2))

    def test_nonfinite_grad_is_zeroed_not_propagated(self):
        weight = torch.nn.Parameter(torch.randn(8, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2, basis_refresh_interval=3, grassmann_aim="eigh")
        opt.diagnostics_enabled = True

        weight.grad = torch.randn_like(weight)
        opt.step()
        self.assertEqual(opt.last_step_diagnostics["nonfinite_grad_tensors"], 0.0)

        poisoned = torch.randn_like(weight)
        poisoned[0, 0] = float("nan")
        poisoned[1, 1] = float("inf")
        weight.grad = poisoned
        opt.step()

        self.assertEqual(opt.last_step_diagnostics["nonfinite_grad_tensors"], 1.0)
        self.assertTrue(torch.isfinite(weight).all())
        state = opt.state[weight]
        self.assertTrue(torch.isfinite(state["projected_exp_avg"]).all())
        self.assertTrue(torch.isfinite(state["adafactor_row_var"]).all())

        # Ride through a refresh boundary on clean grads: the retraction after a
        # sanitized batch must stay finite and orthonormal.
        for _ in range(3):
            weight.grad = torch.randn_like(weight)
            opt.step()
        projector = opt._projector_from_state(weight, opt.param_groups[0], opt.state[weight])
        self.assertTrue(torch.isfinite(projector.basis).all())
        self.assertLess(float(projector.orthonormality_error()), 1e-4)

    def test_adafactor_broadcast_scaling_matches_outer_product_reconstruction(self):
        torch.manual_seed(89)
        grad = torch.randn(13, 7)
        state = {}
        group = {"adafactor_beta2": 0.99, "adafactor_eps": 1e-30}

        actual = UsuiTrack._adafactor_dampen_full_grad(grad, group, state)

        grad_sq = grad.float().square() + group["adafactor_eps"]
        row_hat = grad_sq.mean(dim=1)
        col_hat = grad_sq.mean(dim=0)
        factor = (row_hat.unsqueeze(1) @ col_hat.unsqueeze(0)) / row_hat.mean()
        expected = grad.float() / factor.sqrt()
        expected.mul_(grad.float().square().mean().sqrt())

        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def test_compilable_oja_adafactor_prepare_matches_component_math_on_both_sides(self):
        torch.manual_seed(91)
        grad = torch.randn(13, 7)
        beta = 0.95
        beta2 = 0.99
        eps = 1e-30
        clip_norm = 1.0
        step = 4

        for side in ("right", "left"):
            rank = 3
            if side == "right":
                basis = torch.linalg.qr(torch.randn(grad.shape[1], rank), mode="reduced").Q.mT.contiguous()
                moment = torch.randn(grad.shape[0], rank)
                prepare = UsuiTrack._prepare_oja_adafactor_right_tensors
            else:
                basis = torch.linalg.qr(torch.randn(grad.shape[0], rank), mode="reduced").Q.contiguous()
                moment = torch.randn(rank, grad.shape[1])
                prepare = UsuiTrack._prepare_oja_adafactor_left_tensors
            initial_moment = moment.clone()
            row_var = torch.rand(grad.shape[0])
            col_var = torch.rand(grad.shape[1])
            expected_state = {
                "adafactor_step": step - 1,
                "adafactor_row_var": row_var.clone(),
                "adafactor_col_var": col_var.clone(),
            }

            conditioned, tangent, raw_norm, projected_norm, moment_norm = prepare(
                grad.clone(),
                basis,
                row_var,
                col_var,
                moment,
                step,
                clip_norm,
                beta2,
                eps,
                beta,
            )

            sanitized = torch.nan_to_num(grad)
            expected_raw_norm = sanitized.float().norm()
            sanitized.mul_((sanitized.new_tensor(clip_norm) / expected_raw_norm).clamp(max=1.0))
            expected_conditioned = UsuiTrack._adafactor_dampen_full_grad(
                sanitized,
                {"adafactor_beta2": beta2, "adafactor_eps": eps},
                expected_state,
            )
            if side == "right":
                frame = basis.float().mT
                expected_projected = expected_conditioned @ basis.mT
                action = expected_conditioned.float().mT @ expected_projected.float()
            else:
                frame = basis.float()
                expected_projected = basis.mT @ expected_conditioned
                action = expected_conditioned.float() @ expected_projected.float().mT
            rayleigh = frame.mT @ action
            rayleigh = 0.5 * (rayleigh + rayleigh.mT)
            expected_tangent = (action - frame @ rayleigh) / rayleigh.diagonal().mean().clamp_min(1e-12)
            expected_moment = initial_moment.mul(beta).add(expected_projected, alpha=1.0 - beta)

            torch.testing.assert_close(conditioned, expected_conditioned)
            torch.testing.assert_close(tangent, expected_tangent)
            torch.testing.assert_close(raw_norm, expected_raw_norm)
            torch.testing.assert_close(projected_norm, expected_projected.float().norm())
            torch.testing.assert_close(moment_norm, initial_moment.float().norm())
            torch.testing.assert_close(moment, expected_moment)
            torch.testing.assert_close(row_var, expected_state["adafactor_row_var"])
            torch.testing.assert_close(col_var, expected_state["adafactor_col_var"])

    def test_batched_aurora_health_matches_scalar_diagnostics(self):
        torch.manual_seed(97)
        params = [torch.nn.Parameter(torch.randn(8, 6)) for _ in range(3)]
        opt = UsuiTrack(params, rank=3, side="right", moment_mode="ema", grad_clip_norm=None)
        entries = []
        updates = []
        for param in params:
            projector = SubspaceProjector(rank=3, side="right")
            projector.fit(torch.randn_like(param))
            moment = torch.randn(8, 3)
            entries.append(MatrixUpdate(param, projector, moment, tuple(param.shape)))
            updates.append(torch.randn_like(moment))
        opt.diagnostics_enabled = True
        diagnostics = opt._new_diagnostics()
        assert diagnostics is not None

        opt._accumulate_aurora_health(diagnostics, entries, updates)

        expected_alignment = sum(opt._aurora_alignment(entry.projected_exp_avg, update) for entry, update in zip(entries, updates, strict=True))
        expected_erank = sum(opt._effective_rank(entry.projected_exp_avg) for entry in entries)
        self.assertAlmostEqual(float(diagnostics["aurora_alignment_sum"]), expected_alignment, places=5)
        self.assertAlmostEqual(float(diagnostics["aurora_erank_sum"]), expected_erank, places=5)
        self.assertEqual(diagnostics["aurora_health_tensors"], 3)


if __name__ == "__main__":
    unittest.main()
