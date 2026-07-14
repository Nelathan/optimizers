from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usuitrack import UsuiTrack


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

    if optimizer_name == "usuitrack":
        optimizer = UsuiTrack(model.parameters(), lr=0.01, rank=4, beta=0.9, basis_refresh_interval=1)
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
    usui_initial, usui_final, usui_state_bytes = train("usuitrack")
    adam_initial, adam_final, adam_state_bytes = train("adamw")
    print(f"usuitrack_initial_loss={usui_initial:.6f}")
    print(f"usuitrack_final_loss={usui_final:.6f}")
    print(f"adamw_initial_loss={adam_initial:.6f}")
    print(f"adamw_final_loss={adam_final:.6f}")
    print(f"usuitrack_state_bytes={usui_state_bytes}")
    print(f"adamw_state_bytes={adam_state_bytes}")
    if not usui_final < usui_initial:
        raise SystemExit("UsuiTrack smoke failed: loss did not descend")
    if not usui_state_bytes < adam_state_bytes:
        raise SystemExit("UsuiTrack smoke failed: state bytes were not below AdamW")


if __name__ == "__main__":
    main()
