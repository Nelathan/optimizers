from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usuitrack.projected_activation import (
    ProjectedActivationGradientSink,
    projected_activation_gated_mlp_side_aware,
    set_projected_activation_compile,
)

MODEL_NAME = "LiquidAI/LFM2.5-350M-Base"
TOKEN_COUNTS = (16384, 65536)
RANKS = (64, 256)
WARMUP_ITERS = 10
TIMED_ITERS = 50
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def orthonormal_rows(features: int, rank: int) -> torch.Tensor:
    q, _r = torch.linalg.qr(torch.randn(features, rank, device=DEVICE, dtype=torch.float32), mode="reduced")
    return q.mT.to(dtype=DTYPE).contiguous()


def orthonormal_columns(features: int, rank: int) -> torch.Tensor:
    q, _r = torch.linalg.qr(torch.randn(features, rank, device=DEVICE, dtype=torch.float32), mode="reduced")
    return q.to(dtype=DTYPE).contiguous()


def make_weights(hidden: int, intermediate: int) -> dict[str, torch.Tensor]:
    gate = torch.nn.Linear(hidden, intermediate, bias=False, device=DEVICE, dtype=DTYPE)
    up = torch.nn.Linear(hidden, intermediate, bias=False, device=DEVICE, dtype=DTYPE)
    down = torch.nn.Linear(intermediate, hidden, bias=False, device=DEVICE, dtype=DTYPE)
    return {"gate": gate.weight.detach().clone(), "up": up.weight.detach().clone(), "down": down.weight.detach().clone()}


def baseline_step(x: torch.Tensor, weights: dict[str, torch.Tensor], q_gate: torch.Tensor, q_up: torch.Tensor, p_down: torch.Tensor) -> None:
    gate_w = weights["gate"].clone().requires_grad_(True)
    up_w = weights["up"].clone().requires_grad_(True)
    down_w = weights["down"].clone().requires_grad_(True)

    gate_pre = x @ gate_w.mT
    up = x @ up_w.mT
    hidden = torch.nn.functional.silu(gate_pre) * up
    output = hidden @ down_w.mT
    loss = output.float().square().mean()
    loss.backward()

    full_dwg = gate_w.grad
    full_dwu = up_w.grad
    full_dwd = down_w.grad
    _dwg_proj = full_dwg @ q_gate.mT
    _dwu_proj = full_dwu @ q_up.mT
    _dwd_proj = p_down.mT @ full_dwd


def compiled_baseline_step(compiled_fn, x: torch.Tensor, weights: dict[str, torch.Tensor], q_gate: torch.Tensor, q_up: torch.Tensor, p_down: torch.Tensor) -> None:
    gate_w = weights["gate"].clone().requires_grad_(True)
    up_w = weights["up"].clone().requires_grad_(True)
    down_w = weights["down"].clone().requires_grad_(True)

    output = compiled_fn(x, gate_w, up_w, down_w)
    loss = output.float().square().mean()
    loss.backward()

    full_dwg = gate_w.grad
    full_dwu = up_w.grad
    full_dwd = down_w.grad
    _dwg_proj = full_dwg @ q_gate.mT
    _dwu_proj = full_dwu @ q_up.mT
    _dwd_proj = p_down.mT @ full_dwd


def _gated_mlp_forward(x: torch.Tensor, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    gate_pre = x @ gate_w.mT
    up = x @ up_w.mT
    hidden = torch.nn.functional.silu(gate_pre) * up
    return hidden @ down_w.mT


def fused_step(x: torch.Tensor, weights: dict[str, torch.Tensor], q_gate: torch.Tensor, q_up: torch.Tensor, p_down: torch.Tensor, sink: dict) -> None:
    output = projected_activation_gated_mlp_side_aware(
        x, weights["gate"], weights["up"], weights["down"], q_gate, q_up, p_down, sink, "gate", "up", "down"
    )
    loss = output.float().square().mean()
    loss.backward()


def time_steps(step_fn, warmup: int, timed: int) -> float:
    for _ in range(warmup):
        step_fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(timed):
        step_fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / timed * 1000.0


def peak_bytes(step_fn) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step_fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def kernel_count(step_fn) -> int:
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step_fn()
    torch.cuda.synchronize()
    return sum(1 for event in prof.key_averages() if event.device_type == torch.profiler.DeviceType.CUDA)


def bench_contestant(name: str, step_fn) -> tuple[str, float, int, int]:
    step_ms = time_steps(step_fn, WARMUP_ITERS, TIMED_ITERS)
    peak = peak_bytes(step_fn)
    kernels = kernel_count(step_fn)
    return name, step_ms, peak, kernels


def run_shape(hidden: int, intermediate: int, tokens: int, rank: int) -> None:
    torch.manual_seed(0)
    weights = make_weights(hidden, intermediate)
    q_gate = orthonormal_rows(hidden, rank)
    q_up = orthonormal_rows(hidden, rank)
    p_down = orthonormal_columns(hidden, rank)
    x = torch.randn(tokens, hidden, device=DEVICE, dtype=DTYPE, requires_grad=True)

    compiled_fn = torch.compile(_gated_mlp_forward)

    rows = []

    set_projected_activation_compile(False)
    rows.append(bench_contestant("baseline-eager", lambda: baseline_step(x, weights, q_gate, q_up, p_down)))
    rows.append(bench_contestant("baseline-compiled", lambda: compiled_baseline_step(compiled_fn, x, weights, q_gate, q_up, p_down)))

    set_projected_activation_compile(True)
    sink: dict = {}

    def fused_call():
        sink.clear()
        fused_step(x, weights, q_gate, q_up, p_down, sink)

    rows.append(bench_contestant("fused-projected", fused_call))
    set_projected_activation_compile(False)

    print(f"\n### T={tokens}, rank={rank}\n")
    print("| contestant | step_ms | peak_bytes | cuda_kernel_count |")
    print("|---|---|---|---|")
    for name, step_ms, peak, kernels in rows:
        print(f"| {name} | {step_ms:.3f} | {peak} | {kernels} |")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the side-aware fused MLP microbench")

    config = AutoConfig.from_pretrained(MODEL_NAME)
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    print(f"model={MODEL_NAME} hidden_size={hidden} intermediate_size={intermediate}")

    for tokens in TOKEN_COUNTS:
        for rank in RANKS:
            run_shape(hidden, intermediate, tokens, rank)


if __name__ == "__main__":
    main()
