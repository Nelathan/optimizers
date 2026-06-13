import unittest

import torch


class HeavyBallECCTest(unittest.TestCase):
    def test_heavyball_adamw_accepts_bf16_ecc_options(self):
        import heavyball
        from heavyball import utils

        old_compile_mode = utils.compile_mode
        utils.compile_mode = None
        try:
            param = torch.nn.Parameter(torch.randn(3, 3, dtype=torch.bfloat16))
            opt = heavyball.AdamW([param], lr=0.001, ecc="bf16+8", param_ecc="bf16+8", compile_step=False)

            loss = param.float().square().mean()
            loss.backward()
            opt.step()

            self.assertEqual(param.dtype, torch.bfloat16)
        finally:
            utils.compile_mode = old_compile_mode


if __name__ == "__main__":
    unittest.main()
