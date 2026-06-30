from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack.projected_activation import ProjectedActivationGradientSink, projected_activation_gated_mlp


@dataclass
class GatedSmokeResult:
    mode: str
    after_forward_allocated: int
    peak_allocated: int
    after_backward_allocated: int
    weight_grad_bytes: int
    projected_grad_bytes: int
    source_grad_max_abs_diff: float | None = None
    gate_projected_grad_max_abs_diff: float | None = None
    up_projected_grad_max_abs_diff: float | None = None
    down_projected_grad_max_abs_diff: float | None = None


class GatedMlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.up = torch.nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.down = torch.nn.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))


class ProjectedGatedMlp(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, rank_hidden: int, rank_intermediate: int, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.up = torch.nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.down = torch.nn.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)
        self.register_buffer("q_hidden", orthonormal_rows(hidden, rank_hidden, device=device, dtype=dtype))
        self.register_buffer("q_intermediate", orthonormal_rows(intermediate, rank_intermediate, device=device, dtype=dtype))
        self.sink = ProjectedActivationGradientSink()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return projected_activation_gated_mlp(
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


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def orthonormal_rows(features: int, rank: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    q, _r = torch.linalg.qr(torch.randn(features, rank, device=device, dtype=work_dtype), mode="reduced")
    return q.mT.to(dtype=dtype).contiguous()


def make_source_and_previous(tokens: int, source_features: int, hidden: int, *, device: torch.device, dtype: torch.dtype):
    source = torch.randn(tokens, source_features, device=device, dtype=dtype, requires_grad=True)
    previous = torch.nn.Linear(source_features, hidden, bias=False, device=device, dtype=dtype)
    return source, previous


def run_ordinary(args: argparse.Namespace, dtype: torch.dtype) -> tuple[GatedSmokeResult, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    source, previous = make_source_and_previous(args.tokens, args.source_features, args.hidden, device=device, dtype=dtype)
    mlp = GatedMlp(args.hidden, args.intermediate, device=device, dtype=dtype)

    x = previous(source)
    output = mlp(x)
    loss = output.float().square().mean()
    del x
    torch.cuda.synchronize()
    after_forward = torch.cuda.memory_allocated()

    loss.backward()
    torch.cuda.synchronize()
    assert source.grad is not None
    grads = {
        "gate": mlp.gate.weight.grad.detach().float().cpu(),
        "up": mlp.up.weight.grad.detach().float().cpu(),
        "down": mlp.down.weight.grad.detach().float().cpu(),
    }
    weights = {
        "previous": previous.weight.detach().clone(),
        "gate": mlp.gate.weight.detach().clone(),
        "up": mlp.up.weight.detach().clone(),
        "down": mlp.down.weight.detach().clone(),
    }
    result = GatedSmokeResult(
        mode="ordinary",
        after_forward_allocated=after_forward,
        peak_allocated=torch.cuda.max_memory_allocated(),
        after_backward_allocated=torch.cuda.memory_allocated(),
        weight_grad_bytes=tensor_bytes(mlp.gate.weight.grad) + tensor_bytes(mlp.up.weight.grad) + tensor_bytes(mlp.down.weight.grad),
        projected_grad_bytes=0,
    )
    return result, source.grad.detach().float().cpu(), grads, weights


def run_projected(
    args: argparse.Namespace,
    dtype: torch.dtype,
    reference_source_grad: torch.Tensor,
    reference_grads: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
) -> GatedSmokeResult:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    source, previous = make_source_and_previous(args.tokens, args.source_features, args.hidden, device=device, dtype=dtype)
    mlp = ProjectedGatedMlp(args.hidden, args.intermediate, args.rank_hidden, args.rank_intermediate, device=device, dtype=dtype)
    previous.weight.data.copy_(weights["previous"])
    mlp.gate.weight.data.copy_(weights["gate"])
    mlp.up.weight.data.copy_(weights["up"])
    mlp.down.weight.data.copy_(weights["down"])

    x = previous(source)
    output = mlp(x)
    loss = output.float().square().mean()
    del x
    torch.cuda.synchronize()
    after_forward = torch.cuda.memory_allocated()

    loss.backward()
    torch.cuda.synchronize()
    assert source.grad is not None
    sink = mlp.sink.projected_grads
    expected_gate = reference_grads["gate"].to(device=device, dtype=mlp.q_hidden.dtype) @ mlp.q_hidden.mT
    expected_up = reference_grads["up"].to(device=device, dtype=mlp.q_hidden.dtype) @ mlp.q_hidden.mT
    expected_down = reference_grads["down"].to(device=device, dtype=mlp.q_intermediate.dtype) @ mlp.q_intermediate.mT
    projected_bytes = sum(tensor_bytes(value) for value in sink.values())
    result = GatedSmokeResult(
        mode="projected",
        after_forward_allocated=after_forward,
        peak_allocated=torch.cuda.max_memory_allocated(),
        after_backward_allocated=torch.cuda.memory_allocated(),
        weight_grad_bytes=tensor_bytes(mlp.gate.weight.grad) + tensor_bytes(mlp.up.weight.grad) + tensor_bytes(mlp.down.weight.grad),
        projected_grad_bytes=projected_bytes,
        source_grad_max_abs_diff=float((source.grad.detach().float().cpu() - reference_source_grad).abs().max()),
        gate_projected_grad_max_abs_diff=float((sink[mlp.gate.weight].float() - expected_gate.float()).abs().max().detach().cpu()),
        up_projected_grad_max_abs_diff=float((sink[mlp.up.weight].float() - expected_up.float()).abs().max().detach().cpu()),
        down_projected_grad_max_abs_diff=float((sink[mlp.down.weight].float() - expected_down.float()).abs().max().detach().cpu()),
    )
    return result


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for projected gated-MLP memory smoke")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    ordinary, source_grad, reference_grads, weights = run_ordinary(args, dtype)
    torch.manual_seed(args.seed)
    projected = run_projected(args, dtype, source_grad, reference_grads, weights)
    return {
        "config": vars(args),
        "ordinary": asdict(ordinary),
        "projected": asdict(projected),
        "delta": {
            "after_forward_allocated": ordinary.after_forward_allocated - projected.after_forward_allocated,
            "peak_allocated": ordinary.peak_allocated - projected.peak_allocated,
            "after_backward_allocated": ordinary.after_backward_allocated - projected.after_backward_allocated,
            "weight_grad_bytes": ordinary.weight_grad_bytes - projected.weight_grad_bytes,
            "projected_replacement_bytes": projected.projected_grad_bytes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA smoke for fused projected-activation gated MLP backward")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--source-features", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=8192)
    parser.add_argument("--rank-hidden", type=int, default=64)
    parser.add_argument("--rank-intermediate", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=456)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
