import sys
import types
import unittest

import torch

from experiments.llm_synth_smoke import DEFAULT_MODEL, build_parser, build_sumotrack_param_groups, packed_text_limit, select_trainable_params
from experiments.llm_synth_smoke import cce_causal_lm_loss, make_packed_batches, make_right_padded_batches, synth_masked_examples


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

        self.assertEqual(args.model, DEFAULT_MODEL)
        self.assertEqual(args.param_scope, "broad-no-embeddings")
        self.assertEqual(args.seq_len, 1024)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.batching, "synth_right_padded_no_mask")
        self.assertEqual(args.rank, 64)
        self.assertEqual(args.projection_side_policy, "residual-facing")
        self.assertEqual(args.val_blocks, 8)
        self.assertEqual(args.retention_val_blocks, 8)
        self.assertEqual(args.wandb_log_every, 20)
        self.assertEqual(args.aurora_pp_iterations, 2)
        self.assertEqual(args.polar_ns_steps, 5)
        self.assertEqual(args.basis_init, "eigh")
        self.assertEqual(args.attn_implementation, "sdpa")
        self.assertFalse(args.activation_checkpointing)
        self.assertFalse(args.torch_compile)
        self.assertFalse(args.skip_validation)
        self.assertFalse(args.keep_grads_after_step)

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
