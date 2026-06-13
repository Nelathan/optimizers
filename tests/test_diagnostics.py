import unittest

import torch

from sumotrack import SumoTrack, optimizer_state_bytes, optimizer_state_bytes_by_category


class DiagnosticsTest(unittest.TestCase):
    def test_state_bytes_are_split_between_matrix_and_fallback(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        bias = torch.nn.Parameter(torch.randn(4))
        opt = SumoTrack([weight, bias], lr=0.01, rank=2)

        (weight.square().mean() + bias.square().mean()).backward()
        opt.step()

        state_bytes = optimizer_state_bytes_by_category(opt)
        self.assertGreater(state_bytes["matrix"], 0)
        self.assertGreater(state_bytes["fallback"], 0)
        self.assertEqual(state_bytes["total"], state_bytes["matrix"] + state_bytes["fallback"])
        self.assertEqual(optimizer_state_bytes(opt), state_bytes["total"])


if __name__ == "__main__":
    unittest.main()
