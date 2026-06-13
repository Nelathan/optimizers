from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SumoTrack


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def make_problem() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(128, 16)
    teacher = torch.randn(16, 8)
    y = x @ teacher + 0.05 * torch.randn(128, 8)
    return x, y


def train(optimizer_name: str) -> tuple[float, float, int]:
    torch.manual_seed(1)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 24),
        torch.nn.Tanh(),
        torch.nn.Linear(24, 8),
    )
    x, y = make_problem()

    if optimizer_name == "sumotrack":
        optimizer = SumoTrack(model.parameters(), lr=0.01, rank=4, beta=0.9, subspace_refresh_budget=1)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    else:  # pragma: no cover - local script guard
        raise ValueError(optimizer_name)

    loss_fn = torch.nn.MSELoss()
    initial_loss = float(loss_fn(model(x), y).detach())
    for _step in range(80):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    final_loss = float(loss_fn(model(x), y).detach())
    return initial_loss, final_loss, optimizer_state_bytes(optimizer)


def main() -> None:
    sumo_initial, sumo_final, sumo_state_bytes = train("sumotrack")
    adam_initial, adam_final, adam_state_bytes = train("adamw")
    print(f"sumotrack_initial_loss={sumo_initial:.6f}")
    print(f"sumotrack_final_loss={sumo_final:.6f}")
    print(f"adamw_initial_loss={adam_initial:.6f}")
    print(f"adamw_final_loss={adam_final:.6f}")
    print(f"sumotrack_state_bytes={sumo_state_bytes}")
    print(f"adamw_state_bytes={adam_state_bytes}")
    if not sumo_final < sumo_initial:
        raise SystemExit("SumoTrack smoke failed: loss did not descend")
    if not sumo_state_bytes < adam_state_bytes:
        raise SystemExit("SumoTrack smoke failed: state bytes were not below AdamW")


if __name__ == "__main__":
    main()
