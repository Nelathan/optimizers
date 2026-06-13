from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass
class RoundRobinRefreshScheduler:
    """Deterministic round-robin scheduler for sparse basis refreshes.

    This scheduler only decides which eligible matrix states should refresh their
    subspace basis on a step. It deliberately does not model ordinary optimizer
    updates; those must still run for every parameter each step.
    """

    num_items: int
    subspace_refresh_budget: int | None = 1
    target_refresh_interval: int | None = None
    cursor: int = 0

    def __post_init__(self) -> None:
        if self.num_items < 0:
            raise ValueError(f"num_items must be non-negative, got {self.num_items}")
        if self.target_refresh_interval is not None and self.target_refresh_interval <= 0:
            raise ValueError(
                f"target_refresh_interval must be positive, got {self.target_refresh_interval}"
            )
        if self.subspace_refresh_budget is None:
            if self.target_refresh_interval is None:
                raise ValueError("subspace_refresh_budget and target_refresh_interval cannot both be None")
            self.subspace_refresh_budget = self._budget_for_target_interval()
        if self.subspace_refresh_budget <= 0:
            raise ValueError(f"subspace_refresh_budget must be positive, got {self.subspace_refresh_budget}")
        if self.num_items == 0:
            self.cursor = 0
        elif not 0 <= self.cursor < self.num_items:
            raise ValueError(f"cursor must be in [0, {self.num_items}), got {self.cursor}")

    @classmethod
    def from_target_interval(cls, num_items: int, target_refresh_interval: int, cursor: int = 0) -> "RoundRobinRefreshScheduler":
        return cls(
            num_items=num_items,
            subspace_refresh_budget=None,
            target_refresh_interval=target_refresh_interval,
            cursor=cursor,
        )

    def next_refresh_indices(self) -> tuple[int, ...]:
        """Return the next basis-refresh indices and advance the cursor."""

        if self.num_items == 0:
            return ()

        budget = min(self.subspace_refresh_budget, self.num_items)
        indices = tuple((self.cursor + offset) % self.num_items for offset in range(budget))
        self.cursor = (self.cursor + budget) % self.num_items
        return indices

    def all_update_indices(self) -> tuple[int, ...]:
        """Return all ordinary per-step update indices.

        Keeping this separate from ``next_refresh_indices`` makes the intended
        invariant concrete: round-robin controls rotations, not optimization.
        """

        return tuple(range(self.num_items))

    def _budget_for_target_interval(self) -> int:
        if self.num_items == 0:
            return 1
        assert self.target_refresh_interval is not None
        return max(1, ceil(self.num_items / self.target_refresh_interval))
