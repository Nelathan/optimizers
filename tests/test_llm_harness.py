import unittest

import torch

from experiments.llm_synth_smoke import build_parser, build_sumotrack_param_groups, select_trainable_params


def assert_param_membership(test_case, param, params, expected: bool) -> None:
    present = any(candidate is param for candidate in params)
    if expected:
        test_case.assertTrue(present)
    else:
        test_case.assertFalse(present)


class TinyTopology(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(8, 4)
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.conv = torch.nn.Conv1d(1, 2, 3)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)


class LlmHarnessParamScopeTest(unittest.TestCase):
    def test_cli_defaults_encode_current_baseline(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.param_scope, "broad-no-embeddings")
        self.assertEqual(args.rank, 64)
        self.assertEqual(args.projection_side_policy, "residual-facing")
        self.assertEqual(args.aurora_pp_iterations, 2)
        self.assertEqual(args.polar_ns_steps, 5)
        self.assertFalse(args.activation_checkpointing)
        self.assertFalse(args.torch_compile)
        self.assertFalse(args.pad_to_max_length)
        self.assertFalse(args.pack_sequences)
        self.assertEqual(args.loss_impl, "hf")
        self.assertEqual(args.chunked_lm_loss_tokens, 0)
        self.assertFalse(args.skip_validation)
        self.assertFalse(args.keep_grads_after_step)

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

        groups, stats = build_sumotrack_param_groups(
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

    def test_uniform_rank_clamps_to_matrix_dimension(self):
        first = torch.nn.Parameter(torch.randn(16, 4))
        second = torch.nn.Parameter(torch.randn(16, 4))
        named = [("a.up_proj.weight", first), ("b.up_proj.weight", second)]

        groups, stats = build_sumotrack_param_groups(
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
            build_sumotrack_param_groups(named, rank=0, projection_side_policy="residual-facing")


if __name__ == "__main__":
    unittest.main()
