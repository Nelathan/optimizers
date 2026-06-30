from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack.projected_activation import ProjectedActivationGradientSink, projected_activation_linear


@dataclass
class SmokeResult:
    mode: str
    after_forward_allocated: int
    peak_allocated: int
    after_backward_allocated: int
    target_weight_grad_bytes: int
    projected_grad_bytes: int
    input_grad_max_abs_diff: float | None = None
    projected_grad_max_abs_diff: float | None = None


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def orthonormal_rows(features: int, rank: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    work_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    q, _r = torch.linalg.qr(torch.randn(features, rank, device=device, dtype=work_dtype), mode="reduced")
    return q.mT.to(dtype=dtype).contiguous()


def make_inputs(tokens: int, source_features: int, in_features: int, out_features: int, *, device: torch.device, dtype: torch.dtype):
    source = torch.randn(tokens, source_features, device=device, dtype=dtype, requires_grad=True)
    previous = torch.nn.Linear(source_features, in_features, bias=False, device=device, dtype=dtype)
    target = torch.nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    return source, previous, target


def run_ordinary(tokens: int, source_features: int, in_features: int, out_features: int, dtype: torch.dtype) -> tuple[SmokeResult, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    source, previous, target = make_inputs(tokens, source_features, in_features, out_features, device=device, dtype=dtype)

    hidden = previous(source)
    output = target(hidden)
    loss = output.float().square().mean()
    del hidden
    torch.cuda.synchronize()
    after_forward = torch.cuda.memory_allocated()

    loss.backward()
    torch.cuda.synchronize()
    result = SmokeResult(
        mode="ordinary",
        after_forward_allocated=after_forward,
        peak_allocated=torch.cuda.max_memory_allocated(),
        after_backward_allocated=torch.cuda.memory_allocated(),
        target_weight_grad_bytes=tensor_bytes(target.weight.grad),
        projected_grad_bytes=0,
    )
    assert source.grad is not None
    assert target.weight.grad is not None
    return result, source.grad.detach().float().cpu(), target.weight.grad.detach().float().cpu()


def run_projected(
    tokens: int,
    source_features: int,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    reference_source_grad: torch.Tensor,
    reference_weight_grad: torch.Tensor,
) -> SmokeResult:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    source, previous, target = make_inputs(tokens, source_features, in_features, out_features, device=device, dtype=dtype)
    basis = orthonormal_rows(in_features, rank, device=device, dtype=dtype)
    sink = ProjectedActivationGradientSink()

    # Reuse deterministic weights and source by reseeding around both runs rather
    # than copying tensors across modes; this keeps the smoke simple and lets the
    # equivalence check compare the actual projected gradient for this run.
    hidden = previous(source)
    output = projected_activation_linear(hidden, target.weight, basis, sink, target.weight)
    loss = output.float().square().mean()
    del hidden
    torch.cuda.synchronize()
    after_forward = torch.cuda.memory_allocated()

    loss.backward()
    torch.cuda.synchronize()
    projected_grad = sink.projected_grads[target.weight]
    expected_projected_grad = reference_weight_grad.to(device=device, dtype=projected_grad.dtype) @ basis.mT
    assert source.grad is not None
    source_grad = source.grad.detach().float().cpu()
    result = SmokeResult(
        mode="projected",
        after_forward_allocated=after_forward,
        peak_allocated=torch.cuda.max_memory_allocated(),
        after_backward_allocated=torch.cuda.memory_allocated(),
        target_weight_grad_bytes=tensor_bytes(target.weight.grad),
        projected_grad_bytes=tensor_bytes(projected_grad),
        input_grad_max_abs_diff=float((source_grad - reference_source_grad).abs().max()),
        projected_grad_max_abs_diff=float((projected_grad.float() - expected_projected_grad.float()).abs().max().detach().cpu()),
    )
    return result


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for projected-activation memory smoke")
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(args.seed)
    ordinary, source_grad, weight_grad = run_ordinary(args.tokens, args.source_features, args.in_features, args.out_features, dtype)
    torch.manual_seed(args.seed)
    projected = run_projected(
        args.tokens,
        args.source_features,
        args.in_features,
        args.out_features,
        args.rank,
        dtype,
        source_grad,
        weight_grad,
    )

    return {
        "config": vars(args),
        "ordinary": asdict(ordinary),
        "projected": asdict(projected),
        "delta": {
            "after_forward_allocated": ordinary.after_forward_allocated - projected.after_forward_allocated,
            "peak_allocated": ordinary.peak_allocated - projected.peak_allocated,
            "after_backward_allocated": ordinary.after_backward_allocated - projected.after_backward_allocated,
            "target_weight_grad_bytes": ordinary.target_weight_grad_bytes - projected.target_weight_grad_bytes,
            "projected_replacement_bytes": projected.projected_grad_bytes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA smoke for activation-facing projected Linear backward")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--source-features", type=int, default=1024)
    parser.add_argument("--in-features", type=int, default=4096)
    parser.add_argument("--out-features", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
