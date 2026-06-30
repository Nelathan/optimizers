import unittest

import torch

from sumotrack.projected_activation import ProjectedActivationGradientSink, projected_activation_gated_mlp, projected_activation_linear


def _orthonormal_rows(features: int, rank: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    q, _r = torch.linalg.qr(torch.randn(features, rank, dtype=dtype), mode="reduced")
    return q.mT.contiguous()


class ProjectedActivationTest(unittest.TestCase):
    def test_linear_emits_activation_facing_projected_grad_without_weight_grad(self):
        torch.manual_seed(20)
        batch, in_features, out_features, rank = 5, 7, 4, 3
        basis = _orthonormal_rows(in_features, rank)
        x = torch.randn(batch, in_features, dtype=torch.float64)
        target = torch.randn(batch, out_features, dtype=torch.float64)

        reference = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float64)
        projected = torch.nn.Linear(in_features, out_features, bias=True, dtype=torch.float64)
        projected.load_state_dict(reference.state_dict())
        sink = ProjectedActivationGradientSink()

        ref_x = x.detach().clone().requires_grad_(True)
        proj_x = x.detach().clone().requires_grad_(True)
        ref_loss = (reference(ref_x) - target).square().mean()
        proj_out = projected_activation_linear(proj_x, projected.weight, basis, sink, projected.weight, projected.bias)
        proj_loss = (proj_out - target).square().mean()

        ref_loss.backward()
        proj_loss.backward()

        expected = reference.weight.grad @ basis.mT
        self.assertIsNone(projected.weight.grad)
        self.assertIsNotNone(projected.bias.grad)
        self.assertTrue(torch.allclose(sink.projected_grads[projected.weight], expected, atol=1e-12))
        self.assertTrue(torch.allclose(proj_x.grad, ref_x.grad, atol=1e-12))
        self.assertTrue(torch.allclose(projected.bias.grad, reference.bias.grad, atol=1e-12))

    def test_linear_accumulates_projected_grads_across_multiple_uses(self):
        torch.manual_seed(21)
        in_features, out_features, rank = 6, 5, 2
        basis = _orthonormal_rows(in_features, rank)
        x1 = torch.randn(3, in_features, dtype=torch.float64)
        x2 = torch.randn(4, in_features, dtype=torch.float64)
        target1 = torch.randn(3, out_features, dtype=torch.float64)
        target2 = torch.randn(4, out_features, dtype=torch.float64)
        reference = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float64)
        projected = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float64)
        projected.load_state_dict(reference.state_dict())
        sink = ProjectedActivationGradientSink()

        ref_loss = (reference(x1) - target1).square().mean() + (reference(x2) - target2).square().mean()
        proj_loss = (
            projected_activation_linear(x1, projected.weight, basis, sink, "shared") - target1
        ).square().mean() + (
            projected_activation_linear(x2, projected.weight, basis, sink, "shared") - target2
        ).square().mean()

        ref_loss.backward()
        proj_loss.backward()

        self.assertIsNone(projected.weight.grad)
        self.assertTrue(torch.allclose(sink.projected_grads["shared"], reference.weight.grad @ basis.mT, atol=1e-12))

    def test_gated_mlp_projected_grads_match_full_activation_facing_projection(self):
        torch.manual_seed(22)
        batch, hidden, intermediate, rank_hidden, rank_intermediate = 4, 6, 9, 3, 4
        x = torch.randn(batch, hidden, dtype=torch.float64)
        target = torch.randn(batch, hidden, dtype=torch.float64)
        q_hidden = _orthonormal_rows(hidden, rank_hidden)
        q_intermediate = _orthonormal_rows(intermediate, rank_intermediate)

        reference = _TinyGatedMlp(hidden, intermediate, dtype=torch.float64)
        projected = _TinyProjectedGatedMlp(hidden, intermediate, q_hidden, q_intermediate, dtype=torch.float64)
        projected.load_from(reference)

        ref_x = x.detach().clone().requires_grad_(True)
        proj_x = x.detach().clone().requires_grad_(True)
        ref_loss = (reference(ref_x) - target).square().mean()
        proj_loss = (projected(proj_x) - target).square().mean()

        ref_loss.backward()
        proj_loss.backward()

        self.assertTrue(torch.allclose(projected.last_output, reference.last_output, atol=1e-12))
        self.assertTrue(torch.allclose(proj_x.grad, ref_x.grad, atol=1e-12))
        self.assertIsNone(projected.gate.weight.grad)
        self.assertIsNone(projected.up.weight.grad)
        self.assertIsNone(projected.down.weight.grad)
        self.assertTrue(torch.allclose(projected.sink.projected_grads[projected.gate.weight], reference.gate.weight.grad @ q_hidden.mT, atol=1e-12))
        self.assertTrue(torch.allclose(projected.sink.projected_grads[projected.up.weight], reference.up.weight.grad @ q_hidden.mT, atol=1e-12))
        self.assertTrue(torch.allclose(projected.sink.projected_grads[projected.down.weight], reference.down.weight.grad @ q_intermediate.mT, atol=1e-12))

    def test_fused_gated_mlp_matches_full_activation_facing_projection(self):
        torch.manual_seed(23)
        batch, hidden, intermediate, rank_hidden, rank_intermediate = 4, 6, 9, 3, 4
        x = torch.randn(batch, hidden, dtype=torch.float64)
        target = torch.randn(batch, hidden, dtype=torch.float64)
        q_hidden = _orthonormal_rows(hidden, rank_hidden)
        q_intermediate = _orthonormal_rows(intermediate, rank_intermediate)

        reference = _TinyGatedMlp(hidden, intermediate, dtype=torch.float64)
        projected = _TinyFusedProjectedGatedMlp(hidden, intermediate, q_hidden, q_intermediate, dtype=torch.float64)
        projected.load_from(reference)

        ref_x = x.detach().clone().requires_grad_(True)
        proj_x = x.detach().clone().requires_grad_(True)
        ref_loss = (reference(ref_x) - target).square().mean()
        proj_loss = (projected(proj_x) - target).square().mean()

        ref_loss.backward()
        proj_loss.backward()

        self.assertTrue(torch.allclose(projected.last_output, reference.last_output, atol=1e-12))
        self.assertTrue(torch.allclose(proj_x.grad, ref_x.grad, atol=1e-12))
        self.assertIsNone(projected.gate.weight.grad)
        self.assertIsNone(projected.up.weight.grad)
        self.assertIsNone(projected.down.weight.grad)
        self.assertTrue(torch.allclose(projected.sink.projected_grads[projected.gate.weight], reference.gate.weight.grad @ q_hidden.mT, atol=1e-12))
        self.assertTrue(torch.allclose(projected.sink.projected_grads[projected.up.weight], reference.up.weight.grad @ q_hidden.mT, atol=1e-12))
        self.assertTrue(torch.allclose(projected.sink.projected_grads[projected.down.weight], reference.down.weight.grad @ q_intermediate.mT, atol=1e-12))


class _TinyGatedMlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down = torch.nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.last_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))
        self.last_output = output.detach()
        return output


class _TinyProjectedGatedMlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, q_hidden: torch.Tensor, q_intermediate: torch.Tensor, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down = torch.nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.register_buffer("q_hidden", q_hidden)
        self.register_buffer("q_intermediate", q_intermediate)
        self.sink = ProjectedActivationGradientSink()
        self.last_output = None

    def load_from(self, other: _TinyGatedMlp) -> None:
        self.gate.weight.data.copy_(other.gate.weight)
        self.up.weight.data.copy_(other.up.weight)
        self.down.weight.data.copy_(other.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = projected_activation_linear(x, self.gate.weight, self.q_hidden, self.sink, self.gate.weight)
        up = projected_activation_linear(x, self.up.weight, self.q_hidden, self.sink, self.up.weight)
        hidden = torch.nn.functional.silu(gate) * up
        output = projected_activation_linear(hidden, self.down.weight, self.q_intermediate, self.sink, self.down.weight)
        self.last_output = output.detach()
        return output


class _TinyFusedProjectedGatedMlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, q_hidden: torch.Tensor, q_intermediate: torch.Tensor, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.up = torch.nn.Linear(hidden, intermediate, bias=False, dtype=dtype)
        self.down = torch.nn.Linear(intermediate, hidden, bias=False, dtype=dtype)
        self.register_buffer("q_hidden", q_hidden)
        self.register_buffer("q_intermediate", q_intermediate)
        self.sink = ProjectedActivationGradientSink()
        self.last_output = None

    def load_from(self, other: _TinyGatedMlp) -> None:
        self.gate.weight.data.copy_(other.gate.weight)
        self.up.weight.data.copy_(other.up.weight)
        self.down.weight.data.copy_(other.down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = projected_activation_gated_mlp(
            x,
            self.gate.weight,
            self.up.weight,
            self.down.weight,
            self.q_hidden,
            self.q_intermediate,
            self.sink,
            self.gate.weight,
            self.up.weight,
            self.down.weight,
        )
        self.last_output = output.detach()
        return output


if __name__ == "__main__":
    unittest.main()
