from __future__ import annotations

import argparse
import csv
import gc
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usuitrack import UsuiTrack  # noqa: E402

from experiments.llm_synth_smoke import (  # noqa: E402
    DEFAULT_MODEL,
    batch_tokens,
    build_usuitrack_param_groups,
    cce_causal_lm_loss,
    load_model_and_tokenizer,
    make_batches,
    packed_text_limit,
    projected_activation_param_ids,
    scalar,
    select_trainable_named_params,
    synth_texts,
    transformer_layer_index,
    transformer_matrix_role,
)


def tensor_percentile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float(), q).cpu())


def short_param_name(name: str) -> str:
    parts = name.split(".")
    if "layers" in parts:
        index = parts.index("layers")
        return ".".join(parts[index : min(len(parts), index + 5)])
    return name


def collect_projected_grad_rows(
    optimizer: UsuiTrack,
    named_params: list[tuple[str, torch.nn.Parameter]],
    step: int,
    clip_norm: float,
    clip_ratio: float,
) -> list[dict[str, str | int | float]]:
    rows = []
    group_by_param = {param: group for group in optimizer.param_groups for param in group["params"]}
    for name, param in named_params:
        grad = param.grad
        if param.ndim != 2 or grad is None:
            continue
        state = optimizer.state.get(param, {})
        if state.get("basis") is None:
            continue
        group = group_by_param[param]
        projector = optimizer._projector_from_state(param, group, state)  # diagnostic script; keep optimizer semantics untouched
        projected_grad = projector.project(grad)
        grad_norm = float(projected_grad.float().norm().cpu())
        exp_avg = state.get("projected_exp_avg")
        exp_avg_norm = float(exp_avg.float().norm().cpu()) if exp_avg is not None else 0.0
        ratio = grad_norm / max(exp_avg_norm, 1e-12)
        clipped_by_norm = clip_norm > 0 and grad_norm > clip_norm
        clipped_by_ratio = clip_ratio > 0 and exp_avg is not None and ratio > clip_ratio
        role = transformer_matrix_role(name, param)
        rows.append(
            {
                "step": step,
                "name": name,
                "short_name": short_param_name(name),
                "role": role,
                "layer": transformer_layer_index(name),
                "shape": "x".join(str(dim) for dim in param.shape),
                "side": "right" if state.get("projection_side_is_right", False) else "left",
                "projected_shape": "x".join(str(dim) for dim in projected_grad.shape),
                "projected_grad_norm_preclip": grad_norm,
                "projected_exp_avg_norm_preupdate": exp_avg_norm,
                "grad_to_moment_ratio": ratio,
                "clipped_by_norm": int(clipped_by_norm),
                "clipped_by_ratio": int(clipped_by_ratio),
                "clipped": int(clipped_by_norm or clipped_by_ratio),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_steps(rows: list[dict]) -> list[dict[str, float | int]]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    summaries = []
    for step, step_rows in sorted(by_step.items()):
        norms = torch.tensor([float(row["projected_grad_norm_preclip"]) for row in step_rows])
        ratios = torch.tensor([float(row["grad_to_moment_ratio"]) for row in step_rows])
        summaries.append(
            {
                "step": step,
                "tensor_count": len(step_rows),
                "clipped_count": sum(int(row["clipped"]) for row in step_rows),
                "clipped_by_norm_count": sum(int(row["clipped_by_norm"]) for row in step_rows),
                "clipped_by_ratio_count": sum(int(row["clipped_by_ratio"]) for row in step_rows),
                "norm_p50": tensor_percentile(norms, 0.50),
                "norm_p90": tensor_percentile(norms, 0.90),
                "norm_p99": tensor_percentile(norms, 0.99),
                "norm_max": float(norms.max().cpu()),
                "ratio_p50": tensor_percentile(ratios, 0.50),
                "ratio_p90": tensor_percentile(ratios, 0.90),
                "ratio_p99": tensor_percentile(ratios, 0.99),
                "ratio_max": float(ratios.max().cpu()),
            }
        )
    return summaries


def summarize_params(rows: list[dict]) -> list[dict[str, str | int | float]]:
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_name[str(row["name"])].append(row)
    summaries = []
    for name, param_rows in by_name.items():
        norms = torch.tensor([float(row["projected_grad_norm_preclip"]) for row in param_rows])
        ratios = torch.tensor([float(row["grad_to_moment_ratio"]) for row in param_rows])
        clipped_count = sum(int(row["clipped"]) for row in param_rows)
        clipped_by_norm_count = sum(int(row["clipped_by_norm"]) for row in param_rows)
        clipped_by_ratio_count = sum(int(row["clipped_by_ratio"]) for row in param_rows)
        first = param_rows[0]
        summaries.append(
            {
                "name": name,
                "short_name": first["short_name"],
                "role": first["role"],
                "layer": first["layer"],
                "shape": first["shape"],
                "side": first["side"],
                "samples": len(param_rows),
                "clipped_count": clipped_count,
                "clipped_by_norm_count": clipped_by_norm_count,
                "clipped_by_ratio_count": clipped_by_ratio_count,
                "clip_fraction": clipped_count / max(1, len(param_rows)),
                "norm_median": tensor_percentile(norms, 0.50),
                "norm_p90": tensor_percentile(norms, 0.90),
                "norm_max": float(norms.max().cpu()),
                "ratio_median": tensor_percentile(ratios, 0.50),
                "ratio_p90": tensor_percentile(ratios, 0.90),
                "ratio_max": float(ratios.max().cpu()),
            }
        )
    summaries.sort(key=lambda row: float(row["norm_median"]))
    return summaries


def role_color(role: str) -> str:
    return {
        "mlp_up_gate": "#1f77b4",
        "mlp_down": "#ff7f0e",
        "attention_qkv": "#2ca02c",
        "attention_out": "#d62728",
        "other_matrix": "#9467bd",
    }.get(role, "#7f7f7f")


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color("#444444")


def save_spectrum_plot(output_dir: Path, step_rows: list[dict], param_rows: list[dict], clip_norm: float) -> None:
    steps = [int(row["step"]) for row in step_rows]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
    fig.suptitle("Projected-gradient spectrum before clipping", fontsize=15)

    ax = axes[0][0]
    style_axes(ax)
    ax.plot(steps, [row["norm_p50"] for row in step_rows], label="p50", color="#1f77b4")
    ax.plot(steps, [row["norm_p90"] for row in step_rows], label="p90", color="#ff7f0e")
    ax.plot(steps, [row["norm_p99"] for row in step_rows], label="p99", color="#2ca02c")
    ax.plot(steps, [row["norm_max"] for row in step_rows], label="max", color="#d62728")
    if clip_norm > 0:
        ax.axhline(clip_norm, color="black", linestyle="--", linewidth=1, label=f"clip={clip_norm:g}")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("projected grad norm")
    ax.legend(frameon=True, facecolor="white")

    ax = axes[0][1]
    style_axes(ax)
    ax.bar(steps, [row["clipped_by_ratio_count"] for row in step_rows], color="#1f77b4", label="ratio")
    ax.bar(steps, [row["clipped_by_norm_count"] for row in step_rows], color="#d62728", alpha=0.75, label="norm")
    ax.set_xlabel("step")
    ax.set_ylabel("tensors clipped")
    ax.legend(frameon=True, facecolor="white")

    ax = axes[1][0]
    style_axes(ax)
    xs = list(range(len(param_rows)))
    colors = [role_color(str(row["role"])) for row in param_rows]
    ax.scatter(xs, [row["norm_median"] for row in param_rows], c=colors, s=18, alpha=0.85)
    if clip_norm > 0:
        ax.axhline(clip_norm, color="black", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("matrix tensor, sorted by median norm")
    ax.set_ylabel("median projected grad norm")

    ax = axes[1][1]
    style_axes(ax)
    ax.plot(steps, [row["ratio_p50"] for row in step_rows], label="p50", color="#1f77b4")
    ax.plot(steps, [row["ratio_p90"] for row in step_rows], label="p90", color="#ff7f0e")
    ax.plot(steps, [row["ratio_p99"] for row in step_rows], label="p99", color="#2ca02c")
    ax.plot(steps, [row["ratio_max"] for row in step_rows], label="max", color="#d62728")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("projected grad / moment norm")
    ax.legend(frameon=True, facecolor="white")

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"projected_grad_spectrum.{suffix}", dpi=160, facecolor="white", edgecolor="white")
    plt.close(fig)


def write_markdown_summary(output_dir: Path, step_rows: list[dict], param_rows: list[dict], clip_norm: float, clip_ratio: float) -> None:
    clipped_steps = sum(1 for row in step_rows if int(row["clipped_count"]) > 0)
    total_clips = sum(int(row["clipped_count"]) for row in step_rows)
    norm_clips = sum(int(row["clipped_by_norm_count"]) for row in step_rows)
    ratio_clips = sum(int(row["clipped_by_ratio_count"]) for row in step_rows)
    by_role = Counter(str(row["role"]) for row in param_rows if float(row["clip_fraction"]) > 0)
    top_norm = sorted(param_rows, key=lambda row: float(row["norm_max"]), reverse=True)[:12]
    top_ratio = sorted(param_rows, key=lambda row: float(row["ratio_max"]), reverse=True)[:12]
    lines = [
        "# Projected-gradient spectrum",
        "",
        f"Clip norm: `{clip_norm:g}`" if clip_norm > 0 else "Clip norm: disabled",
        f"Clip ratio: `{clip_ratio:g}`" if clip_ratio > 0 else "Clip ratio: disabled",
        f"Steps: `{len(step_rows)}`",
        f"Matrix tensors: `{len(param_rows)}`",
        f"Steps with any clipping: `{clipped_steps}`",
        f"Total tensor clips: `{total_clips}`",
        f"Norm-triggered clips: `{norm_clips}`",
        f"Ratio-triggered clips: `{ratio_clips}`",
        f"Roles clipped: `{dict(by_role)}`",
        "",
        "## Largest projected-gradient norms",
        "",
        "| name | role | median | p90 | max | clipped | norm | ratio | ratio max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_norm:
        lines.append(
            f"| `{row['short_name']}` | {row['role']} | {float(row['norm_median']):.4g} | {float(row['norm_p90']):.4g} | {float(row['norm_max']):.4g} | {int(row['clipped_count'])}/{int(row['samples'])} | {int(row['clipped_by_norm_count'])} | {int(row['clipped_by_ratio_count'])} | {float(row['ratio_max']):.4g} |"
        )
    lines.extend([
        "",
        "## Largest grad/moment ratios",
        "",
        "| name | role | ratio median | ratio p90 | ratio max | norm max | clipped | norm | ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in top_ratio:
        lines.append(
            f"| `{row['short_name']}` | {row['role']} | {float(row['ratio_median']):.4g} | {float(row['ratio_p90']):.4g} | {float(row['ratio_max']):.4g} | {float(row['norm_max']):.4g} | {int(row['clipped_count'])}/{int(row['samples'])} | {int(row['clipped_by_norm_count'])} | {int(row['clipped_by_ratio_count'])} |"
        )
    (output_dir / "projected_grad_spectrum_summary.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect per-matrix projected-gradient norm spectra for UsuiTrack")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default="/home/djg/.cache/nanochat/base_data_synth")
    parser.add_argument("--output-dir", default="/tmp/opencode/projected_grad_spectrum")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--projected-grad-clip-norm", type=float, default=2.0)
    parser.add_argument("--projected-grad-clip-ratio", type=float, default=5.0)
    parser.add_argument("--activation-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-activation-checkpointing", dest="activation_checkpointing", action="store_false")
    parser.add_argument("--attn-implementation", default="sdpa")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    if args.warmup_steps < 1:
        raise ValueError("at least one warmup step is required to initialize bases")
    if args.projected_grad_clip_norm < 0:
        raise ValueError("projected_grad_clip_norm must be non-negative")
    if args.projected_grad_clip_ratio < 0:
        raise ValueError("projected_grad_clip_ratio must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    train_blocks = args.warmup_steps + args.steps
    train_texts = synth_texts(Path(args.data_dir), "train", limit=packed_text_limit(train_blocks, args.batch_size, args.seq_len))
    model, tokenizer = load_model_and_tokenizer(args.model, device, args.activation_checkpointing, args.attn_implementation)
    trainable_named, _stats = select_trainable_named_params(model, "broad-no-embeddings")
    trainable = [param for _name, param in trainable_named]
    groups, policy_stats = build_usuitrack_param_groups(
        trainable_named,
        rank=args.rank,
        projection_side_policy="residual-facing",
        activation_projected_param_ids=projected_activation_param_ids(model, "off"),
        basis_refresh_schedule="burst",
    )
    batches = make_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len, train_blocks, "synth_right_padded_no_mask", "synth")
    optimizer = UsuiTrack(
        groups,
        lr=args.lr,
        beta=args.beta,
        basis_init="eigh",
        grassmann_aim="eigh",
        basis_refresh_interval=100,
        projected_grad_clip_norm=args.projected_grad_clip_norm if args.projected_grad_clip_norm > 0 else None,
        projected_grad_clip_ratio=args.projected_grad_clip_ratio if args.projected_grad_clip_ratio > 0 else None,
    )

    print(f"device={device}")
    print(f"model={args.model}")
    print(f"batch_size={args.batch_size} seq_len={args.seq_len} rank={args.rank} steps={args.steps} warmup_steps={args.warmup_steps}")
    print(f"policy_stats={policy_stats}")
    print(f"output_dir={output_dir}")

    for step in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = cce_causal_lm_loss(model, batches[step])
        loss.backward()
        optimizer.step()

    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for offset in range(args.steps):
        step = offset + 1
        batch = batches[args.warmup_steps + offset]
        optimizer.zero_grad(set_to_none=True)
        loss = cce_causal_lm_loss(model, batch)
        loss.backward()
        rows.extend(collect_projected_grad_rows(optimizer, trainable_named, step, args.projected_grad_clip_norm, args.projected_grad_clip_ratio))
        optimizer.step()
        if step == 1 or step == args.steps or step % 10 == 0:
            print(f"step={step} loss={scalar(loss):.6f}")

    step_summary = summarize_steps(rows)
    param_summary = summarize_params(rows)
    write_csv(output_dir / "projected_grad_spectrum_raw.csv", rows)
    write_csv(output_dir / "projected_grad_spectrum_by_step.csv", step_summary)
    write_csv(output_dir / "projected_grad_spectrum_by_param.csv", param_summary)
    save_spectrum_plot(output_dir, step_summary, param_summary, args.projected_grad_clip_norm)
    write_markdown_summary(output_dir, step_summary, param_summary, args.projected_grad_clip_norm, args.projected_grad_clip_ratio)

    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    print(f"peak_cuda_bytes={peak}")
    print(f"tokens_per_step={batch_tokens(batches[0])}")
    print(f"artifacts={output_dir}")
    del optimizer, model, tokenizer, batches, trainable
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
