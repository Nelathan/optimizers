import unittest

import torch

from experiments.llm_synth_smoke import select_trainable_params


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


if __name__ == "__main__":
    unittest.main()
