from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SumoTrack


def make_problem() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(2)
    x = torch.randn(128, 16)
    teacher = torch.randn(16, 8)
    y = x @ teacher + 0.05 * torch.randn(128, 8)
    return x, y


def run(method: str) -> tuple[float, float, float]:
    torch.manual_seed(3)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 24),
        torch.nn.Tanh(),
        torch.nn.Linear(24, 8),
    )
    x, y = make_problem()
    opt = SumoTrack(
        model.parameters(),
        lr=0.01,
        rank=4,
        beta=0.9,
        subspace_refresh_budget=1,
        subspace_update_method=method,
        grassmann_step_size=0.01,
    )
    loss_fn = torch.nn.MSELoss()
    initial = float(loss_fn(model(x), y).detach())
    start = time.perf_counter()
    for _step in range(80):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    elapsed = time.perf_counter() - start
    final = float(loss_fn(model(x), y).detach())
    return initial, final, elapsed


def main() -> None:
    svd_initial, svd_final, svd_seconds = run("svd_refresh")
    grass_initial, grass_final, grass_seconds = run("grassmann")
    print(f"svd_initial_loss={svd_initial:.6f}")
    print(f"svd_final_loss={svd_final:.6f}")
    print(f"svd_seconds={svd_seconds:.6f}")
    print(f"grassmann_initial_loss={grass_initial:.6f}")
    print(f"grassmann_final_loss={grass_final:.6f}")
    print(f"grassmann_seconds={grass_seconds:.6f}")
    if not svd_final < svd_initial:
        raise SystemExit("SVD refresh comparison failed: loss did not descend")
    if not grass_final < grass_initial:
        raise SystemExit("Grassmann comparison failed: loss did not descend")


if __name__ == "__main__":
    main()
