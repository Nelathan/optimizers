import unittest

import torch

from experiments.llm_synth_smoke import build_sumotrack_param_groups, select_trainable_params


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

    def test_module_role_side_policy_uses_hidden_state_facing_axes(self):
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
            rank_policy="uniform",
            projection_side_policy="module-role",
            min_rank=1,
            max_rank=None,
            optimizer_state_budget_mb=0.0,
        )
        group_by_param = {id(param): group for group in groups for param in group["params"]}

        self.assertEqual(group_by_param[id(up)]["side"], "right")
        self.assertEqual(group_by_param[id(down)]["side"], "left")
        self.assertEqual(group_by_param[id(q)]["side"], "right")
        self.assertEqual(stats["side_policy_right_tensors"], 2)
        self.assertEqual(stats["side_policy_left_tensors"], 1)

    def test_rank_policy_can_clamp_to_matrix_state_budget(self):
        first = torch.nn.Parameter(torch.randn(16, 4))
        second = torch.nn.Parameter(torch.randn(16, 4))
        named = [("a.up_proj.weight", first), ("b.up_proj.weight", second)]
        min_cost_mb = 2 * (16 + 4) * first.element_size() / 1024 / 1024

        groups, stats = build_sumotrack_param_groups(
            named,
            rank=4,
            rank_policy="uniform",
            projection_side_policy="auto",
            min_rank=1,
            max_rank=None,
            optimizer_state_budget_mb=min_cost_mb,
        )

        self.assertEqual(stats["rank_policy_min_rank"], 1)
        self.assertEqual(stats["rank_policy_max_rank"], 1)
        self.assertTrue(all(group["rank"] == 1 for group in groups))


if __name__ == "__main__":
    unittest.main()
