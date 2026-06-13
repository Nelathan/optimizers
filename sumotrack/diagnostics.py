from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import torch


def tensor_state_bytes(state: Mapping) -> int:
    total = 0
    for value in state.values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
        elif isinstance(value, Mapping):
            total += tensor_state_bytes(value)
    return total


def optimizer_state_bytes_by_category(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """Return optimizer state bytes split by SumoTrack matrix/fallback semantics.

    SumoTrack deliberately uses different state shapes for matrix and fallback
    parameters. Keep the accounting coupled to those state keys so broad-scope
    experiments cannot hide full AdamW fallback state inside a single attractive
    total.
    """

    totals: defaultdict[str, int] = defaultdict(int)
    for state in optimizer.state.values():
        state_bytes = tensor_state_bytes(state)
        totals["total"] += state_bytes
        if "projected_exp_avg" in state or "basis" in state:
            totals["matrix"] += state_bytes
        elif state:
            totals["fallback"] += state_bytes

    return {"matrix": totals["matrix"], "fallback": totals["fallback"], "total": totals["total"]}


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return optimizer_state_bytes_by_category(optimizer)["total"]
