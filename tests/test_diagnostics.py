import unittest

import torch

from usuitrack import UsuiTrack, optimizer_state_bytes, optimizer_state_bytes_by_category


class DiagnosticsTest(unittest.TestCase):
    def test_matrix_optimizer_has_only_matrix_state(self):
        weight = torch.nn.Parameter(torch.randn(6, 4))
        opt = UsuiTrack([weight], lr=0.01, rank=2)

        weight.square().mean().backward()
        opt.step()

        state_bytes = optimizer_state_bytes_by_category(opt)
        self.assertGreater(state_bytes["matrix"], 0)
        self.assertEqual(state_bytes["fallback"], 0)
        self.assertEqual(state_bytes["total"], state_bytes["matrix"] + state_bytes["fallback"])
        self.assertEqual(optimizer_state_bytes(opt), state_bytes["total"])


if __name__ == "__main__":
    unittest.main()
