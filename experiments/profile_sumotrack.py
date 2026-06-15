from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SumoTrack


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )


def make_model(device: torch.device) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(64, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 64),
        torch.nn.GELU(),
        torch.nn.Linear(64, 16),
    ).to(device)


def make_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(4)
    return torch.randn(32, 64, device=device), torch.randn(32, 16, device=device)


def run_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor) -> float:
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def benchmark(refresh_budget: int, device: torch.device, steps: int = 20) -> dict[str, float | int]:
    torch.manual_seed(5)
    model = make_model(device)
    x, y = make_batch(device)
    optimizer = SumoTrack(
        model.parameters(),
        lr=0.001,
        rank=8,
        beta=0.9,
        subspace_refresh_budget=refresh_budget,
        grassmann_step_size=0.01,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(3):
        run_step(model, optimizer, x, y)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times = []
    final_loss = 0.0
    for _ in range(steps):
        start = time.perf_counter()
        final_loss = run_step(model, optimizer, x, y)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)

    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "final_loss": final_loss,
        "avg_step_seconds": sum(times) / len(times),
        "max_step_seconds": max(times),
        "min_step_seconds": min(times),
        "state_bytes": optimizer_state_bytes(optimizer),
        "peak_cuda_bytes": peak_memory,
    }


def profile_cuda_events(device: torch.device) -> int:
    torch.manual_seed(6)
    model = make_model(device)
    x, y = make_batch(device)
    optimizer = SumoTrack(
        model.parameters(),
        lr=0.001,
        rank=8,
        subspace_refresh_budget=1,
    )
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, acc_events=True) as prof:
        run_step(model, optimizer, x, y)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return sum(1 for event in prof.events() if "CUDA" in str(getattr(event, "device_type", "")))


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    budgeted = benchmark(refresh_budget=1, device=device)
    full_refresh = benchmark(refresh_budget=4, device=device)
    cuda_event_count = profile_cuda_events(device)
    spike_ratio = full_refresh["avg_step_seconds"] / budgeted["avg_step_seconds"]

    print(f"device={device}")
    print(f"budgeted_avg_step_seconds={budgeted['avg_step_seconds']:.6f}")
    print(f"budgeted_max_step_seconds={budgeted['max_step_seconds']:.6f}")
    print(f"budgeted_min_step_seconds={budgeted['min_step_seconds']:.6f}")
    print(f"full_refresh_avg_step_seconds={full_refresh['avg_step_seconds']:.6f}")
    print(f"refresh_spike_ratio={spike_ratio:.3f}")
    print(f"state_bytes={budgeted['state_bytes']}")
    print(f"peak_cuda_bytes={budgeted['peak_cuda_bytes']}")
    print(f"profiler_cuda_event_count={cuda_event_count}")
    if device.type == "cuda" and cuda_event_count == 0:
        raise SystemExit("expected CUDA profiler events on CUDA device")


if __name__ == "__main__":
    main()
