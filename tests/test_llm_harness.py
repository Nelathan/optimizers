import math
import sys
import types
import unittest
from unittest import mock

import torch

from usuitrack import UsuiTrack

from experiments.llm_synth_smoke import BackwardMemoryTrace, DEFAULT_MODEL, DEFAULT_SOURCE_HF_DATASET, FALLBACK_LR_RATIO, FP32StateAdamW, build_parser, build_usuitrack_param_groups, install_projected_activation_backend, maybe_compile_training_model, packed_text_limit, partition_usuitrack_params, projected_activation_param_ids, repair_lfm2_gradient_checkpointing, select_trainable_params, validate_gradient_release_contract, validate_projected_activation_contract, wandb_log
from experiments.llm_synth_smoke import cce_causal_lm_loss, gradient_norm_statistics, make_packed_batches, make_right_padded_batches, synth_masked_examples


def assert_param_membership(test_case, param, params, expected: bool) -> None:
    present = any(candidate is param for candidate in params)
    if expected:
        test_case.assertTrue(present)
    else:
        test_case.assertFalse(present)


def expected_projected_grad(opt: UsuiTrack, param: torch.nn.Parameter, full_grad: torch.Tensor) -> torch.Tensor:
    state = opt.state[param]
    basis = state["basis"]
    if state.get("projection_side_is_right", False):
        return full_grad @ basis.mT
    return basis.mT @ full_grad


class TinyTopology(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.conv = torch.nn.Conv1d(1, 2, 3)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class TinyLfmMlp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Linear(4, 6, bias=False, dtype=torch.float64)
        self.w3 = torch.nn.Linear(4, 6, bias=False, dtype=torch.float64)
        self.w2 = torch.nn.Linear(6, 4, bias=False, dtype=torch.float64)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TinyLfmAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
        self.k_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
        self.v_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
        self.out_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)

    def forward(self, x):
        return self.out_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class TinyLfmShortConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 4, kernel_size=1, groups=4, bias=False, dtype=torch.float64)
        self.in_proj = torch.nn.Linear(4, 12, bias=False, dtype=torch.float64)
        self.out_proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)

    def forward(self, x):
        b, c, x_branch = self.in_proj(x).chunk(3, dim=-1)
        conv_in = (b * x_branch).transpose(-1, -2)
        conv_out = self.conv(conv_in).transpose(-1, -2)
        return self.out_proj(c * conv_out)


class CountingCheckpointLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x.sin()


class Lfm2Model(torch.nn.Module):
    def __init__(self, layers: int = 2):
        super().__init__()
        self.gradient_checkpointing = True
        self.layers = torch.nn.ModuleList([CountingCheckpointLayer() for _ in range(layers)])


class TinyLfmForCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Lfm2Model()


class LlmHarnessParamScopeTest(unittest.TestCase):
    def test_compile_training_model_compiles_layers_in_place_only(self):
        model = TinyLfmForCausalLM()
        compiled = []

        def record_compile(layer, *args, **kwargs):
            compiled.append(layer)

        with mock.patch.object(torch.nn.Module, "compile", autospec=True, side_effect=record_compile):
            returned = maybe_compile_training_model(model, True)

        self.assertIs(returned, model)
        self.assertEqual(compiled, list(model.model.layers))
        self.assertEqual(model.model._usuitrack_compiled_layer_count, 2)

    def test_compile_training_model_disabled_leaves_layers_eager(self):
        model = TinyLfmForCausalLM()

        self.assertIs(maybe_compile_training_model(model, False), model)
        self.assertFalse(hasattr(model.model, "_usuitrack_compiled_layer_count"))

    def test_usuitrack_partition_leaves_only_matrices_in_matrix_optimizer(self):
        model = TinyTopology()
        named = [(name, param) for name, param in model.named_parameters()]

        matrix, fallback = partition_usuitrack_params(named)

        self.assertTrue(matrix)
        self.assertTrue(fallback)
        self.assertTrue(all(param.ndim == 2 for _name, param in matrix))
        self.assertTrue(all(param.ndim != 2 for _name, param in fallback))
        self.assertEqual({id(param) for _name, param in named}, {id(param) for _name, param in matrix + fallback})

    def test_fp32_state_adamw_matches_existing_fallback_math(self):
        torch.manual_seed(91)
        initial = torch.randn(7, dtype=torch.bfloat16)
        old_param = torch.nn.Parameter(initial.clone())
        split_param = torch.nn.Parameter(initial.clone())
        lr = 1.5e-4
        old = UsuiTrack([old_param], lr=lr, fallback_betas=(0.9, 0.99))
        split = FP32StateAdamW([split_param], lr=lr, betas=(0.9, 0.99))

        for _ in range(3):
            grad = torch.randn_like(initial)
            old_param.grad = grad.clone()
            split_param.grad = grad.clone()
            old.step()
            split.step()

        torch.testing.assert_close(split_param, old_param, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(split.state[split_param][key], old.state[old_param][key], rtol=0, atol=0)
        self.assertEqual(split.state[split_param]["exp_avg"].dtype, torch.float32)
        self.assertEqual(split.state[split_param]["exp_avg_sq"].dtype, torch.float32)
        self.assertEqual(FALLBACK_LR_RATIO, 0.5)

    def test_backward_memory_trace_brackets_registered_preparation_hook(self):
        param = torch.nn.Parameter(torch.randn(3, 2))
        trace = BackwardMemoryTrace(torch.device("cpu"))
        trace.install_before([("weight", param)])
        middle = param.register_post_accumulate_grad_hook(lambda completed: setattr(completed, "grad", None))
        trace.install_after([("weight", param)])

        trace.start()
        param.square().sum().backward()
        trace.stop()

        self.assertEqual([event["phase"] for event in trace.events], ["before_prepare", "after_prepare"])
        self.assertTrue(trace.events[0]["grad_present"])
        self.assertFalse(trace.events[1]["grad_present"])
        self.assertGreater(trace.events[0]["grad_bytes"], 0)
        self.assertEqual(trace.events[1]["grad_bytes"], 0)
        self.assertEqual(trace.summary()["memory_trace_event_count"], 2)
        middle.remove()
        trace.close()

    def test_gradient_norm_statistics_describe_the_per_tensor_clip_population(self):
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(1))
        first.grad = torch.tensor([3.0, 4.0])
        second.grad = torch.tensor([10.0])

        global_norm, median, clipped_fraction = gradient_norm_statistics([first, second], clip_norm=6.0)

        torch.testing.assert_close(global_norm, torch.tensor(125.0).sqrt())
        torch.testing.assert_close(median, torch.tensor(5.0))
        torch.testing.assert_close(clipped_fraction, torch.tensor(0.5))

    def test_wandb_omits_unavailable_metrics_but_preserves_numeric_nan(self):
        class Run:
            def __init__(self):
                self.calls = []

            def log(self, data, *, step):
                self.calls.append((data, step))

        run = Run()
        wandb_log(run, {"eval/target_loss": 1.0, "eval/source_loss": None, "opt/numeric_problem": float("nan")}, step=10)

        self.assertEqual(len(run.calls), 1)
        data, step = run.calls[0]
        self.assertEqual(step, 10)
        self.assertEqual(data["eval/target_loss"], 1.0)
        self.assertNotIn("eval/source_loss", data)
        self.assertTrue(math.isnan(data["opt/numeric_problem"]))

    def test_cli_defaults_encode_current_baseline(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.param_scope, "broad-no-embeddings")
        self.assertEqual(args.seq_len, 1024)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.batching, "synth_right_padded_no_mask")
        self.assertEqual(args.max_steps, 1000)
        self.assertEqual(args.retention_hf_dataset, DEFAULT_SOURCE_HF_DATASET)
        self.assertEqual(args.rank, 128)
        self.assertEqual(args.projection_side_policy, "residual-facing")
        self.assertEqual(args.usuitrack_lr, 3e-4)
        self.assertEqual(args.lr_warmup_steps, 50)
        self.assertEqual(args.beta, 0.95)
        self.assertEqual(args.projected_activation_backend, "off")
        self.assertEqual(args.basis_refresh_schedule, "burst")
        self.assertEqual(args.moment_mode, "adafactor_ema")
        self.assertEqual(args.basis_refresh_interval, 10)
        self.assertEqual(args.projected_grad_clip_norm, 0.0)
        self.assertEqual(args.projected_grad_clip_ratio, 0.0)
        self.assertEqual(args.grad_clip_norm, 1.0)
        self.assertEqual(args.grassmann_aim, "oja")
        self.assertEqual(args.oja_step_schedule, "mature")
        self.assertEqual(build_parser().parse_args(["--oja-step-schedule", "mature"]).oja_step_schedule, "mature")
        self.assertEqual(build_parser().parse_args(["--grassmann-aim", "eigh"]).grassmann_aim, "eigh")
        self.assertIsNone(args.grassmann_rotate_rank)
        self.assertEqual(args.grassmann_step_size, 0.25)
        self.assertFalse(hasattr(args, "grassmann_step_schedule"))
        self.assertFalse(hasattr(args, "no_grassmann_accumulate"))
        self.assertEqual(args.val_blocks, 8)
        self.assertEqual(args.retention_val_blocks, 8)
        self.assertEqual(args.wandb_log_every, 25)
        self.assertEqual(args.eval_every, 100)
        self.assertEqual(args.aurora_pp_iterations, 1)
        self.assertEqual(args.polar_ns_steps, 5)
        self.assertEqual(args.basis_init, "eigh")
        self.assertEqual(args.attn_implementation, "sdpa")
        self.assertTrue(args.activation_checkpointing)
        self.assertTrue(args.torch_compile)
        self.assertFalse(build_parser().parse_args(["--no-torch-compile"]).torch_compile)
        self.assertFalse(args.skip_validation)
        self.assertFalse(args.keep_grads_after_step)
        self.assertFalse(args.release_matrix_grads)
        self.assertFalse(args.trace_backward_memory)
        self.assertFalse(hasattr(args, "shadow_target_probe"))

    def test_oja_rejects_projected_activation_backend_before_setup(self):
        args = build_parser().parse_args(["--projected-activation-backend", "lfm"])

        with self.assertRaisesRegex(ValueError, "requires full matrix gradients"):
            validate_projected_activation_contract(args)

        validate_projected_activation_contract(build_parser().parse_args(["--projected-activation-backend", "lfm", "--grassmann-aim", "eigh"]))

    def test_gradient_release_rejects_incompatible_harness_contracts(self):
        validate_gradient_release_contract(build_parser().parse_args(["--release-matrix-grads"]))
        for incompatible in (
            ["--release-matrix-grads", "--grad-accum-steps", "2"],
            ["--release-matrix-grads", "--keep-grads-after-step"],
            ["--release-matrix-grads", "--projected-activation-backend", "lfm", "--grassmann-aim", "eigh"],
        ):
            with self.assertRaises(ValueError):
                validate_gradient_release_contract(build_parser().parse_args(incompatible))

    def test_gradient_statistics_include_released_matrix_norms(self):
        live = torch.nn.Parameter(torch.zeros(3))
        live.grad = torch.tensor([3.0, 4.0, 0.0])
        total, median, clipped = gradient_norm_statistics([live], 4.0, (torch.tensor(12.0),))

        self.assertEqual(float(total), 13.0)
        self.assertEqual(float(median), 5.0)
        self.assertEqual(float(clipped), 1.0)

    def test_cli_has_no_loss_or_padding_option_garden(self):
        option_strings = {option for action in build_parser()._actions for option in action.option_strings}

        self.assertNotIn("--loss-impl", option_strings)
        self.assertNotIn("--chunked-lm-loss-tokens", option_strings)
        self.assertNotIn("--pad-to-max-length", option_strings)
        self.assertNotIn("--val-texts", option_strings)
        self.assertNotIn("--retention-val-texts", option_strings)
        self.assertNotIn("--print-shape-summary", option_strings)
        self.assertNotIn("--log-grad-norm", option_strings)
        self.assertNotIn("--log-norms", option_strings)

    def test_packed_text_limit_scales_with_requested_tokens(self):
        self.assertEqual(packed_text_limit(blocks=1, batch_size=1, seq_len=128), 16)
        self.assertEqual(packed_text_limit(blocks=4, batch_size=2, seq_len=1024), 16)
        self.assertEqual(packed_text_limit(blocks=200, batch_size=4, seq_len=1024), 1600)

    def test_broad_no_embeddings_includes_fallback_topology_without_embeddings_or_head(self):
        model = TinyTopology()

        trainable, stats = select_trainable_params(model, "broad-no-embeddings")

        assert_param_membership(self, model.linear.weight, trainable, True)
        assert_param_membership(self, model.linear.bias, trainable, True)
        assert_param_membership(self, model.norm.weight, trainable, True)
        assert_param_membership(self, model.norm.bias, trainable, True)
        assert_param_membership(self, model.conv.weight, trainable, True)
        assert_param_membership(self, model.embed_tokens.weight, trainable, False)
        assert_param_membership(self, model.lm_head.weight, trainable, False)
        self.assertGreater(stats["selected_matrix_params"], 0)
        self.assertGreater(stats["selected_fallback_params"], 0)
        self.assertGreater(stats["excluded_embedding_params"], 0)

    def test_matrices_no_embeddings_still_excludes_fallback_topology(self):
        model = TinyTopology()

        trainable, stats = select_trainable_params(model, "matrices-no-embeddings")

        assert_param_membership(self, model.linear.weight, trainable, True)
        assert_param_membership(self, model.linear.bias, trainable, False)
        assert_param_membership(self, model.norm.weight, trainable, False)
        assert_param_membership(self, model.conv.weight, trainable, False)
        assert_param_membership(self, model.embed_tokens.weight, trainable, False)
        self.assertEqual(stats["selected_fallback_params"], 0)
        self.assertGreater(stats["excluded_3d_params"], 0)

    def test_residual_facing_policy_uses_backbone_axes_in_pytorch_storage(self):
        up = torch.nn.Parameter(torch.randn(16, 4))
        down = torch.nn.Parameter(torch.randn(4, 16))
        q = torch.nn.Parameter(torch.randn(4, 4))
        named = [
            ("model.layers.0.mlp.up_proj.weight", up),
            ("model.layers.0.mlp.down_proj.weight", down),
            ("model.layers.0.self_attn.q_proj.weight", q),
        ]

        groups, stats = build_usuitrack_param_groups(
            named,
            rank=4,
            projection_side_policy="residual-facing",
        )
        group_by_param = {id(param): group for group in groups for param in group["params"]}

        self.assertEqual(group_by_param[id(up)]["side"], "right")  # up_proj weight is [expanded, hidden]
        self.assertEqual(group_by_param[id(down)]["side"], "left")  # down_proj weight is [hidden, expanded]
        self.assertEqual(group_by_param[id(q)]["side"], "right")
        self.assertEqual(stats["side_policy_right_tensors"], 2)
        self.assertEqual(stats["side_policy_left_tensors"], 1)

    def test_right_policy_forces_all_matrix_params_to_storage_right(self):
        up = torch.nn.Parameter(torch.randn(16, 4))
        down = torch.nn.Parameter(torch.randn(4, 16))
        q = torch.nn.Parameter(torch.randn(4, 4))
        named = [
            ("model.layers.0.mlp.up_proj.weight", up),
            ("model.layers.0.mlp.down_proj.weight", down),
            ("model.layers.0.self_attn.q_proj.weight", q),
        ]

        groups, stats = build_usuitrack_param_groups(
            named,
            rank=4,
            projection_side_policy="right",
        )

        self.assertEqual({group["side"] for group in groups}, {"right"})
        self.assertEqual(stats["side_policy_right_tensors"], 3)
        self.assertEqual(stats["side_policy_left_tensors"], 0)

    def test_projected_activation_param_ids_do_not_override_residual_facing_side(self):
        gate = torch.nn.Parameter(torch.randn(16, 4))
        down = torch.nn.Parameter(torch.randn(4, 16))
        named = [
            ("model.layers.0.feed_forward.w1.weight", gate),
            ("model.layers.0.feed_forward.w2.weight", down),
        ]

        groups, stats = build_usuitrack_param_groups(
            named,
            rank=4,
            projection_side_policy="residual-facing",
            activation_projected_param_ids={id(gate), id(down)},
        )
        group_by_param = {id(param): group for group in groups for param in group["params"]}

        self.assertEqual(group_by_param[id(gate)]["side"], "right")
        self.assertEqual(group_by_param[id(down)]["side"], "left")
        self.assertEqual(stats["side_policy_right_tensors"], 1)
        self.assertEqual(stats["side_policy_left_tensors"], 1)

    def test_layer_staggered_refresh_schedule_adds_layer_offsets_without_changing_sides(self):
        first = torch.nn.Parameter(torch.randn(16, 4))
        second = torch.nn.Parameter(torch.randn(16, 4))
        named = [
            ("model.layers.2.feed_forward.w1.weight", first),
            ("model.layers.5.feed_forward.w3.weight", second),
        ]

        groups, _stats = build_usuitrack_param_groups(
            named,
            rank=4,
            projection_side_policy="right",
            basis_refresh_schedule="layer-staggered",
        )
        group = groups[0]

        self.assertEqual(group["side"], "right")
        self.assertEqual(group["basis_refresh_offsets"], {id(first): 2, id(second): 5})

    def test_lfm2_gradient_checkpointing_repair_wraps_decoder_layers(self):
        model = TinyLfmForCausalLM()

        wrapped = repair_lfm2_gradient_checkpointing(model)

        self.assertEqual(wrapped, 2)
        self.assertEqual(repair_lfm2_gradient_checkpointing(model), 0)
        self.assertTrue(all(getattr(layer, "_usuitrack_checkpoint_wrapped", False) for layer in model.model.layers))

        x = torch.randn(4, requires_grad=True)
        y = model.model.layers[0](x).sum()
        y.backward()

        self.assertEqual(model.model.layers[0].calls, 2)

    def test_lfm2_gradient_checkpointing_repair_uses_direct_forward_without_grad(self):
        model = TinyLfmForCausalLM()
        repair_lfm2_gradient_checkpointing(model)

        with torch.no_grad():
            model.model.layers[0](torch.randn(4))

        self.assertEqual(model.model.layers[0].calls, 1)

    def test_lfm_projected_activation_backend_falls_back_until_basis_ready_then_queues(self):
        model = torch.nn.Module()
        model.feed_forward = TinyLfmMlp()
        named = list(model.named_parameters())
        activation_projected_ids = projected_activation_param_ids(model, "lfm")
        groups, _stats = build_usuitrack_param_groups(
            named,
            rank=2,
            projection_side_policy="right",
            activation_projected_param_ids=activation_projected_ids,
        )
        opt = UsuiTrack(groups, lr=0.01, rank=2, basis_refresh_interval=100, moment_mode="ema", grassmann_aim="eigh")

        installed = install_projected_activation_backend(model, opt, "lfm")

        self.assertEqual(installed, 1)
        x = torch.randn(3, 5, 4, dtype=torch.float64)
        loss = model.feed_forward(x).square().mean()
        loss.backward()
        self.assertIsNotNone(model.feed_forward.w1.weight.grad)
        self.assertIsNotNone(model.feed_forward.w2.weight.grad)
        opt.step()
        opt.zero_grad(set_to_none=True)

        reference = torch.nn.Module()
        reference.feed_forward = TinyLfmMlp()
        reference.load_state_dict(model.state_dict())
        loss = model.feed_forward(x).square().mean()
        reference_loss = reference.feed_forward(x).square().mean()
        reference_loss.backward()
        loss.backward()

        self.assertIsNone(model.feed_forward.w1.weight.grad)
        self.assertIsNone(model.feed_forward.w3.weight.grad)
        self.assertIsNone(model.feed_forward.w2.weight.grad)
        self.assertEqual(set(opt._queued_projected_grads), {model.feed_forward.w1.weight, model.feed_forward.w3.weight, model.feed_forward.w2.weight})
        self.assertTrue(torch.allclose(opt._queued_projected_grads[model.feed_forward.w1.weight], reference.feed_forward.w1.weight.grad @ opt.state[model.feed_forward.w1.weight]["basis"].mT, atol=1e-12))
        self.assertTrue(torch.allclose(opt._queued_projected_grads[model.feed_forward.w3.weight], reference.feed_forward.w3.weight.grad @ opt.state[model.feed_forward.w3.weight]["basis"].mT, atol=1e-12))
        self.assertTrue(torch.allclose(opt._queued_projected_grads[model.feed_forward.w2.weight], reference.feed_forward.w2.weight.grad @ opt.state[model.feed_forward.w2.weight]["basis"].mT, atol=1e-12))
        opt.step()
        self.assertEqual(opt._queued_projected_grads, {})

    def test_lfm_projected_activation_backend_falls_back_on_refresh_due_layer(self):
        model = torch.nn.Module()
        model.feed_forward = TinyLfmMlp()
        named = list(model.named_parameters())
        activation_projected_ids = projected_activation_param_ids(model, "lfm")
        groups, _stats = build_usuitrack_param_groups(
            named,
            rank=2,
            projection_side_policy="residual-facing",
            activation_projected_param_ids=activation_projected_ids,
        )
        opt = UsuiTrack(groups, lr=0.01, rank=2, basis_refresh_interval=100, grassmann_aim="eigh")
        install_projected_activation_backend(model, opt, "lfm")
        x = torch.randn(3, 5, 4, dtype=torch.float64)

        model.feed_forward(x).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:
            group["basis_refresh_step"] = 100

        model.feed_forward(x).square().mean().backward()

        self.assertIsNotNone(model.feed_forward.w1.weight.grad)
        self.assertIsNotNone(model.feed_forward.w3.weight.grad)
        self.assertIsNotNone(model.feed_forward.w2.weight.grad)
        self.assertEqual(opt._queued_projected_grads, {})

    def test_lfm_projected_activation_backend_wraps_standard_attention_linears(self):
        model = torch.nn.Module()
        model.self_attn = TinyLfmAttention()
        named = list(model.named_parameters())
        activation_projected_ids = projected_activation_param_ids(model, "lfm")
        groups, _stats = build_usuitrack_param_groups(
            named,
            rank=2,
            projection_side_policy="residual-facing",
            activation_projected_param_ids=activation_projected_ids,
        )
        opt = UsuiTrack(groups, lr=0.01, rank=2, basis_refresh_interval=100, grassmann_aim="eigh")

        installed = install_projected_activation_backend(model, opt, "lfm")

        self.assertEqual(installed, 4)
        x = torch.randn(3, 5, 4, dtype=torch.float64)
        model.self_attn(x).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        reference = torch.nn.Module()
        reference.self_attn = TinyLfmAttention()
        reference.load_state_dict(model.state_dict())
        reference.self_attn(x).square().mean().backward()
        model.self_attn(x).square().mean().backward()

        weights = {model.self_attn.q_proj.weight, model.self_attn.k_proj.weight, model.self_attn.v_proj.weight, model.self_attn.out_proj.weight}
        self.assertTrue(all(weight.grad is None for weight in weights))
        self.assertEqual(set(opt._queued_projected_grads), weights)
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            projected_weight = getattr(model.self_attn, name).weight
            reference_weight = getattr(reference.self_attn, name).weight
            expected = expected_projected_grad(opt, projected_weight, reference_weight.grad)
            self.assertTrue(torch.allclose(opt._queued_projected_grads[projected_weight], expected, atol=1e-12))

    def test_lfm_projected_activation_backend_wraps_short_conv_projection_linears(self):
        model = torch.nn.Module()
        model.conv = TinyLfmShortConv()
        named = list(model.named_parameters())
        activation_projected_ids = projected_activation_param_ids(model, "lfm")
        groups, _stats = build_usuitrack_param_groups(
            named,
            rank=2,
            projection_side_policy="residual-facing",
            activation_projected_param_ids=activation_projected_ids,
        )
        opt = UsuiTrack(groups, lr=0.01, rank=2, basis_refresh_interval=100, grassmann_aim="eigh")

        installed = install_projected_activation_backend(model, opt, "lfm")

        self.assertEqual(installed, 2)
        x = torch.randn(3, 5, 4, dtype=torch.float64)
        model.conv(x).square().mean().backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        reference = torch.nn.Module()
        reference.conv = TinyLfmShortConv()
        reference.load_state_dict(model.state_dict())
        reference.conv(x).square().mean().backward()
        model.conv(x).square().mean().backward()

        projected_weights = {model.conv.in_proj.weight, model.conv.out_proj.weight}
        self.assertTrue(all(weight.grad is None for weight in projected_weights))
        self.assertIsNotNone(model.conv.conv.weight.grad)
        self.assertEqual(set(opt._queued_projected_grads), projected_weights)
        for name in ("in_proj", "out_proj"):
            projected_weight = getattr(model.conv, name).weight
            reference_weight = getattr(reference.conv, name).weight
            expected = expected_projected_grad(opt, projected_weight, reference_weight.grad)
            self.assertTrue(torch.allclose(opt._queued_projected_grads[projected_weight], expected, atol=1e-12))

    def test_uniform_rank_clamps_to_matrix_dimension(self):
        first = torch.nn.Parameter(torch.randn(16, 4))
        second = torch.nn.Parameter(torch.randn(16, 4))
        named = [("a.up_proj.weight", first), ("b.up_proj.weight", second)]

        groups, stats = build_usuitrack_param_groups(
            named,
            rank=8,
            projection_side_policy="auto",
        )

        self.assertEqual(stats["effective_rank_min"], 4)
        self.assertEqual(stats["effective_rank_max"], 4)
        self.assertTrue(all(group["rank"] == 8 for group in groups))

    def test_rank_must_be_positive(self):
        named = [("a.up_proj.weight", torch.nn.Parameter(torch.randn(16, 4)))]

        with self.assertRaises(ValueError):
            build_usuitrack_param_groups(named, rank=0, projection_side_policy="residual-facing")

    def test_packed_batches_omit_attention_mask_for_sdpa_flash_path(self):
        class TokenizerStub:
            eos_token_id = 99

            def __call__(self, text, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("tokenizer stub called with unexpected kwargs")
                token_ids = [ord(char) % 10 for char in text]
                return {"input_ids": token_ids}

        batches = make_packed_batches(TokenizerStub(), ["abc", "def"], torch.device("cpu"), batch_size=2, seq_len=3, min_batches=1)

        self.assertEqual(len(batches), 1)
        self.assertIn("input_ids", batches[0])
        self.assertIn("labels", batches[0])
        self.assertNotIn("attention_mask", batches[0])
        self.assertTrue(torch.equal(batches[0]["input_ids"], batches[0]["labels"]))
        self.assertTrue((batches[0]["input_ids"] == 99).any())

    def test_right_padded_synth_batches_mask_context_without_attention_mask(self):
        class TokenizerStub:
            bos_token_id = 11
            eos_token_id = 99
            pad_token_id = 0

            vocab = {"Q": 1, "\n": 2, "-": 3, "A": 4, "B": 5}

            def __call__(self, text, add_special_tokens):
                if add_special_tokens:
                    raise AssertionError("tokenizer stub called with unexpected kwargs")
                return {"input_ids": [self.vocab[char] for char in text]}

        examples = synth_masked_examples(["Q\n\nAB"])
        batches = make_right_padded_batches(TokenizerStub(), examples, torch.device("cpu"), batch_size=1, seq_len=10, min_batches=1)

        self.assertNotIn("attention_mask", batches[0])
        self.assertTrue(torch.equal(batches[0]["input_ids"], torch.tensor([[11, 1, 2, 3, 3, 3, 2, 4, 5, 99]])))
        self.assertTrue(torch.equal(batches[0]["labels"], torch.tensor([[-100, -100, -100, -100, -100, -100, -100, 4, 5, 99]])))

    def test_cce_causal_lm_loss_uses_hidden_states_without_full_logits(self):
        captured = {}

        def fake_linear_cross_entropy(hidden_states, weight, targets, ignore_index, shift):
            captured["hidden_states"] = hidden_states
            captured["weight"] = weight
            captured["targets"] = targets
            captured["ignore_index"] = ignore_index
            captured["shift"] = shift
            return hidden_states.sum() * 0.0

        fake_module = types.ModuleType("cut_cross_entropy")
        fake_module.linear_cross_entropy = fake_linear_cross_entropy
        previous = sys.modules.get("cut_cross_entropy")
        sys.modules["cut_cross_entropy"] = fake_module
        try:
            class BaseStub(torch.nn.Module):
                def forward(self, input_ids, use_cache=False):
                    if use_cache:
                        raise AssertionError("base stub received unexpected cache flag")
                    hidden = torch.arange(24, dtype=torch.float32).view(1, 3, 8)
                    return types.SimpleNamespace(last_hidden_state=hidden)

            class ModelStub(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = BaseStub()
                    self.lm_head = torch.nn.Linear(8, 4, bias=False)

            model = types.SimpleNamespace(_orig_mod=ModelStub())
            batch = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, 3]]),
            }

            loss = cce_causal_lm_loss(model, batch)
        finally:
            if previous is None:
                sys.modules.pop("cut_cross_entropy", None)
            else:
                sys.modules["cut_cross_entropy"] = previous

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(captured["ignore_index"], -100)
        self.assertTrue(captured["shift"])
        self.assertTrue(torch.equal(captured["targets"], torch.tensor([[-100, 2, 3]])))


if __name__ == "__main__":
    unittest.main()
