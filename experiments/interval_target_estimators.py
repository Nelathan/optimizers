from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProbeConfig:
    dimension: int = 64
    rank: int = 10
    interval: int = 10
    rows_per_gradient: int = 32
    intervals: int = 40
    trials: int = 8
    controller_step: float = 0.25
    rotation_degrees: float = 60.0
    signal_top: float = 8.0
    signal_cutoff: float = 3.0
    noise_floor: float = 1.0
    seed: int = 17
    factor_multipliers: tuple[int, ...] = (1, 2, 4)
    oja_steps: tuple[float, ...] = (0.03125, 0.0625, 0.125, 0.1875)
    direct_oja_steps: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05)

    def validate(self) -> None:
        if self.dimension < 2 * self.rank:
            raise ValueError("dimension must be at least twice rank so the abrupt world has an orthogonal replacement")
        if self.rank < 1 or self.interval < 1 or self.rows_per_gradient < 1:
            raise ValueError("rank, interval, and rows_per_gradient must be positive")
        if self.intervals < 2 or self.trials < 1:
            raise ValueError("intervals must be at least two and trials must be positive")
        if not 0.0 < self.controller_step <= 1.0:
            raise ValueError("controller_step must be in (0, 1]")
        if not self.signal_top >= self.signal_cutoff > self.noise_floor > 0.0:
            raise ValueError("signal strengths must satisfy signal_top >= signal_cutoff > noise_floor > 0")
        if any(multiplier < 1 for multiplier in self.factor_multipliers):
            raise ValueError("factor multipliers must be positive")
        if any(step <= 0.0 for step in self.oja_steps):
            raise ValueError("Oja steps must be positive")
        if any(step <= 0.0 for step in self.direct_oja_steps):
            raise ValueError("direct Oja steps must be positive")
        for multiplier in self.factor_multipliers:
            width = multiplier * self.rank
            if width % self.interval:
                raise ValueError(
                    f"factor width {width} must divide evenly across interval {self.interval}; "
                    "choose rank/interval/multipliers that preserve equal per-step contact"
                )
            if width // self.interval > self.rows_per_gradient:
                raise ValueError("factor rows per step cannot exceed rows_per_gradient")


@dataclass
class EstimatorState:
    basis: Tensor
    oja_frame: Tensor | None = None


@dataclass
class MetricSeries:
    target_to_population: list[float]
    target_to_interval: list[float]
    basis_to_population: list[float]


def orthonormalize(matrix: Tensor) -> Tensor:
    return torch.linalg.qr(matrix, mode="reduced").Q


def top_frame(covariance: Tensor, rank: int) -> Tensor:
    return torch.linalg.eigh(0.5 * (covariance + covariance.mT)).eigenvectors[:, -rank:]


def principal_angle_mass(frame_a: Tensor, frame_b: Tensor) -> float:
    singular_values = torch.linalg.svdvals(frame_a.mT @ frame_b)
    return float(torch.acos(singular_values.clamp(-1.0, 1.0)).sum())


def geodesic_toward(frame: Tensor, target: Tensor, fraction: float) -> Tensor:
    left, cosine, right_h = torch.linalg.svd(frame.mT @ target)
    right = right_h.mT
    residual = target @ right - frame @ (left * cosine)
    sine = residual.norm(dim=0)
    angles = torch.atan2(sine, cosine.clamp_min(0.0))
    normal = residual / sine.clamp_min(1e-12)
    moved = (frame @ left) * torch.cos(fraction * angles) + normal * torch.sin(fraction * angles)
    return orthonormalize(moved @ left.mT)


def normalized_covariance(gradient: Tensor) -> Tensor:
    normalized = gradient / gradient.norm().clamp_min(1e-12)
    return normalized.mT @ normalized


def rayleigh_normalized_oja_step(frame: Tensor, covariance: Tensor, step_size: float) -> Tensor:
    rayleigh = frame.mT @ covariance @ frame
    tangent = covariance @ frame - frame @ (0.5 * (rayleigh + rayleigh.mT))
    mean_retained_energy = rayleigh.diagonal().mean().clamp_min(1e-12)
    return orthonormalize(frame + (step_size / mean_retained_energy) * tangent)


def rayleigh_normalized_geodesic_oja_step(frame: Tensor, covariance: Tensor, step_size: float) -> Tensor:
    rayleigh = frame.mT @ covariance @ frame
    tangent = covariance @ frame - frame @ (0.5 * (rayleigh + rayleigh.mT))
    tangent = tangent / rayleigh.diagonal().mean().clamp_min(1e-12)
    normal, singular_values, right_h = torch.linalg.svd(tangent, full_matrices=False)
    right = right_h.mT
    moved = (frame @ right) * torch.cos(step_size * singular_values)
    moved = moved + normal * torch.sin(step_size * singular_values)
    return orthonormalize(moved @ right_h)


def orthogonal_factor(gradient: Tensor, rows: int, generator: torch.Generator) -> Tensor:
    random_columns = torch.randn((gradient.shape[0], rows), generator=generator, dtype=gradient.dtype)
    projection = orthonormalize(random_columns).mT
    projection.mul_(math.sqrt(gradient.shape[0] / rows))
    return projection @ (gradient / gradient.norm().clamp_min(1e-12))


def make_world_basis(config: ProbeConfig, generator: torch.Generator) -> Tensor:
    matrix = torch.randn((config.dimension, config.dimension), generator=generator, dtype=torch.float64)
    return torch.linalg.qr(matrix, mode="complete").Q


def population_covariance(world: str, interval_index: int, config: ProbeConfig, world_basis: Tensor) -> Tensor:
    rank = config.rank
    primary = world_basis[:, :rank]
    replacement = world_basis[:, rank : 2 * rank]
    signal = torch.linspace(config.signal_top, config.signal_cutoff, rank, dtype=torch.float64)
    identity = config.noise_floor * torch.eye(config.dimension, dtype=torch.float64)

    if world == "stationary":
        frame = primary
    elif world == "rotating":
        progress = interval_index / max(1, config.intervals - 1)
        angle = math.radians(config.rotation_degrees) * progress
        frame = primary * math.cos(angle) + replacement * math.sin(angle)
    elif world == "abrupt":
        frame = primary if interval_index < config.intervals // 2 else replacement
    elif world == "cutoff_crossing":
        progress = interval_index / max(1, config.intervals - 1)
        fixed = primary[:, : rank - 1]
        falling = primary[:, rank - 1 : rank]
        rising = replacement[:, :1]
        fixed_signal = signal[: rank - 1]
        excursion = 0.5 * (config.signal_cutoff - config.noise_floor)
        falling_value = config.signal_cutoff + excursion * (1.0 - 2.0 * progress)
        rising_value = config.signal_cutoff - excursion * (1.0 - 2.0 * progress)
        covariance = identity
        covariance = covariance + fixed @ torch.diag(fixed_signal - config.noise_floor) @ fixed.mT
        covariance = covariance + (falling_value - config.noise_floor) * (falling @ falling.mT)
        covariance = covariance + (rising_value - config.noise_floor) * (rising @ rising.mT)
        return covariance
    else:
        raise ValueError(f"unknown world: {world}")

    return identity + frame @ torch.diag(signal - config.noise_floor) @ frame.mT


def sample_gradients(covariance: Tensor, config: ProbeConfig, generator: torch.Generator) -> list[Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    root = (eigenvectors * eigenvalues.clamp_min(0.0).sqrt()) @ eigenvectors.mT
    return [
        torch.randn((config.rows_per_gradient, config.dimension), generator=generator, dtype=torch.float64) @ root
        for _ in range(config.interval)
    ]


def estimator_names(config: ProbeConfig) -> list[str]:
    names = ["boundary_eigh", "interval_eigh", "warm_block"]
    names.extend(f"factor_{multiplier}r" for multiplier in config.factor_multipliers)
    names.extend(f"oja_{step:g}" for step in config.oja_steps)
    names.extend(f"direct_oja_{step:g}" for step in config.direct_oja_steps)
    return names


def initialize_states(initial_frame: Tensor, config: ProbeConfig) -> dict[str, EstimatorState]:
    states: dict[str, EstimatorState] = {}
    for name in estimator_names(config):
        frame = initial_frame.clone()
        states[name] = EstimatorState(basis=frame, oja_frame=frame.clone() if name.startswith("oja_") else None)
    return states


def estimate_targets(
    gradients: list[Tensor],
    states: dict[str, EstimatorState],
    config: ProbeConfig,
    generator: torch.Generator,
) -> tuple[dict[str, Tensor], Tensor]:
    covariances = [normalized_covariance(gradient) for gradient in gradients]
    interval_covariance = torch.stack(covariances).mean(dim=0)
    interval_target = top_frame(interval_covariance, config.rank)
    targets = {
        "boundary_eigh": top_frame(covariances[-1], config.rank),
        "interval_eigh": interval_target,
    }

    warm_basis = states["warm_block"].basis
    targets["warm_block"] = orthonormalize(interval_covariance @ warm_basis)

    for multiplier in config.factor_multipliers:
        rows_per_step = multiplier * config.rank // config.interval
        factors = [orthogonal_factor(gradient, rows_per_step, generator) for gradient in gradients]
        factor = torch.cat(factors, dim=0)
        targets[f"factor_{multiplier}r"] = top_frame(factor.mT @ factor, config.rank)

    for step_size in config.oja_steps:
        name = f"oja_{step_size:g}"
        frame = states[name].oja_frame
        assert frame is not None
        for covariance in covariances:
            frame = rayleigh_normalized_oja_step(frame, covariance, step_size)
        states[name].oja_frame = frame
        targets[name] = frame

    for step_size in config.direct_oja_steps:
        name = f"direct_oja_{step_size:g}"
        frame = states[name].basis
        for covariance in covariances:
            frame = rayleigh_normalized_geodesic_oja_step(frame, covariance, step_size)
        states[name].basis = frame
        targets[name] = frame

    return targets, interval_target


def run_trial(world: str, config: ProbeConfig, seed: int) -> dict[str, MetricSeries]:
    generator = torch.Generator().manual_seed(seed)
    world_basis = make_world_basis(config, generator)
    initial_covariance = population_covariance(world, 0, config, world_basis)
    initial_gradient = sample_gradients(initial_covariance, config, generator)[-1]
    initial_frame = top_frame(normalized_covariance(initial_gradient), config.rank)
    states = initialize_states(initial_frame, config)
    metrics = {
        name: MetricSeries(target_to_population=[], target_to_interval=[], basis_to_population=[])
        for name in estimator_names(config)
    }

    for interval_index in range(config.intervals):
        covariance = population_covariance(world, interval_index, config, world_basis)
        population_target = top_frame(covariance, config.rank)
        gradients = sample_gradients(covariance, config, generator)
        targets, interval_target = estimate_targets(gradients, states, config, generator)

        for name, target in targets.items():
            state = states[name]
            if not name.startswith("direct_oja_"):
                state.basis = geodesic_toward(state.basis, target, config.controller_step)
            series = metrics[name]
            series.target_to_population.append(principal_angle_mass(target, population_target))
            series.target_to_interval.append(principal_angle_mass(target, interval_target))
            series.basis_to_population.append(principal_angle_mass(state.basis, population_target))

    return metrics


def summarize(config: ProbeConfig) -> dict[str, object]:
    config.validate()
    worlds = ("stationary", "rotating", "abrupt", "cutoff_crossing")
    worlds_summary: dict[str, dict[str, dict[str, object]]] = {}
    summary: dict[str, object] = {"config": asdict(config), "worlds": worlds_summary}

    for world_index, world in enumerate(worlds):
        trials = [run_trial(world, config, config.seed + 10_000 * world_index + trial) for trial in range(config.trials)]
        world_summary: dict[str, dict[str, object]] = {}
        window = max(1, config.intervals // 4)
        for name in estimator_names(config):
            target_population = torch.tensor(
                [value for trial in trials for value in trial[name].target_to_population[-window:]], dtype=torch.float64
            )
            target_interval = torch.tensor(
                [value for trial in trials for value in trial[name].target_to_interval[-window:]], dtype=torch.float64
            )
            basis_population = torch.tensor(
                [value for trial in trials for value in trial[name].basis_to_population[-window:]], dtype=torch.float64
            )
            world_summary[name] = {
                "target_population_last_quarter_mean": float(target_population.mean()),
                "target_interval_last_quarter_mean": float(target_interval.mean()),
                "basis_population_last_quarter_mean": float(basis_population.mean()),
                "basis_population_final_mean": float(
                    torch.tensor([trial[name].basis_to_population[-1] for trial in trials], dtype=torch.float64).mean()
                ),
                "target_population_curve_mean": torch.tensor(
                    [trial[name].target_to_population for trial in trials], dtype=torch.float64
                ).mean(dim=0).tolist(),
                "target_interval_curve_mean": torch.tensor(
                    [trial[name].target_to_interval for trial in trials], dtype=torch.float64
                ).mean(dim=0).tolist(),
                "basis_population_curve_mean": torch.tensor(
                    [trial[name].basis_to_population for trial in trials], dtype=torch.float64
                ).mean(dim=0).tolist(),
            }
        worlds_summary[world] = world_summary

    return summary


def print_summary(summary: dict[str, object]) -> None:
    worlds = summary["worlds"]
    assert isinstance(worlds, dict)
    for world, estimators in worlds.items():
        print(f"\n{world}")
        print("estimator          target->pop  target->interval  basis->pop  final basis")
        assert isinstance(estimators, dict)
        for name, metrics in estimators.items():
            assert isinstance(metrics, dict)
            target_population = metrics["target_population_last_quarter_mean"]
            target_interval = metrics["target_interval_last_quarter_mean"]
            basis_population = metrics["basis_population_last_quarter_mean"]
            final_basis = metrics["basis_population_final_mean"]
            assert isinstance(target_population, float)
            assert isinstance(target_interval, float)
            assert isinstance(basis_population, float)
            assert isinstance(final_basis, float)
            print(
                f"{name:<18}"
                f"{target_population:>11.4f}"
                f"{target_interval:>18.4f}"
                f"{basis_population:>12.4f}"
                f"{final_basis:>13.4f}"
            )


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split(",") if item)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def main() -> None:
    defaults = ProbeConfig()
    parser = argparse.ArgumentParser(description="Compare interval covariance target estimators in hostile synthetic worlds.")
    parser.add_argument("--dimension", type=int, default=defaults.dimension)
    parser.add_argument("--rank", type=int, default=defaults.rank)
    parser.add_argument("--interval", type=int, default=defaults.interval)
    parser.add_argument("--rows-per-gradient", type=int, default=defaults.rows_per_gradient)
    parser.add_argument("--intervals", type=int, default=defaults.intervals)
    parser.add_argument("--trials", type=int, default=defaults.trials)
    parser.add_argument("--controller-step", type=float, default=defaults.controller_step)
    parser.add_argument("--rotation-degrees", type=float, default=defaults.rotation_degrees)
    parser.add_argument("--signal-top", type=float, default=defaults.signal_top)
    parser.add_argument("--signal-cutoff", type=float, default=defaults.signal_cutoff)
    parser.add_argument("--noise-floor", type=float, default=defaults.noise_floor)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--factor-multipliers", type=parse_int_tuple, default=defaults.factor_multipliers)
    parser.add_argument("--oja-steps", type=parse_float_tuple, default=defaults.oja_steps)
    parser.add_argument("--direct-oja-steps", type=parse_float_tuple, default=defaults.direct_oja_steps)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    config = ProbeConfig(
        dimension=args.dimension,
        rank=args.rank,
        interval=args.interval,
        rows_per_gradient=args.rows_per_gradient,
        intervals=args.intervals,
        trials=args.trials,
        controller_step=args.controller_step,
        rotation_degrees=args.rotation_degrees,
        signal_top=args.signal_top,
        signal_cutoff=args.signal_cutoff,
        noise_floor=args.noise_floor,
        seed=args.seed,
        factor_multipliers=args.factor_multipliers,
        oja_steps=args.oja_steps,
        direct_oja_steps=args.direct_oja_steps,
    )
    result = summarize(config)
    print_summary(result)
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
