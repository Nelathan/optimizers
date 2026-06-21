from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SumoTrack, optimizer_state_bytes_by_category


DEFAULT_MODEL = "LiquidAI/LFM2.5-350M-Base"

ParamScope = Literal["full", "broad-no-embeddings", "matrices-no-embeddings"]
ProjectionSidePolicy = Literal["auto", "residual-facing"]


@torch.no_grad()
def tensor_global_norm(tensors) -> float:
    norm_sq = 0.0
    for tensor in tensors:
        if tensor is None:
            continue
        norm = tensor.detach().float().norm()
        norm_sq += float(norm.square().cpu())
    return norm_sq**0.5


@torch.no_grad()
def parameter_norm(params: list[torch.nn.Parameter]) -> float:
    return tensor_global_norm(params)


@torch.no_grad()
def gradient_norm(params: list[torch.nn.Parameter]) -> float:
    return tensor_global_norm([param.grad for param in params if param.grad is not None])


def optimizer_update_norm(optimizer: torch.optim.Optimizer) -> float:
    diagnostics = getattr(optimizer, "last_step_diagnostics", None)
    if not diagnostics:
        return float("nan")
    return float(diagnostics.get("update_norm", float("nan")))


def optimizer_projected_leverage_stats(optimizer: torch.optim.Optimizer) -> tuple[float, float, float]:
    diagnostics = getattr(optimizer, "last_step_diagnostics", None)
    if not diagnostics:
        return float("nan"), float("nan"), float("nan")
    return (
        float(diagnostics.get("mean_projected_leverage_cv", float("nan"))),
        float(diagnostics.get("mean_projected_leverage_min_ratio", float("nan"))),
        float(diagnostics.get("mean_projected_leverage_max_ratio", float("nan"))),
    )


def synth_texts(data_dir: Path, split: str, limit: int) -> list[str]:
    shards = sorted(data_dir.glob("synth_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No SYNTH parquet shards found in {data_dir}")
    shard = shards[-1] if split == "val" else shards[0]
    table = pq.ParquetFile(shard).read_row_group(0, columns=["query", "synthetic_reasoning", "synthetic_answer"])
    rows = zip(
        table.column("query").to_pylist(),
        table.column("synthetic_reasoning").to_pylist(),
        table.column("synthetic_answer").to_pylist(),
    )
    texts = []
    for query, reasoning, answer in rows:
        parts = [part for part in (query, reasoning, answer) if part]
        texts.append("\n\n".join(parts))
        if len(texts) >= limit:
            break
    return texts


def packed_text_limit(blocks: int, batch_size: int, seq_len: int) -> int:
    # SYNTH rows are often shorter than 1k tokens, so row count must scale with
    # requested token blocks. The packer still verifies enough real tokens exist.
    return max(blocks * batch_size, (blocks * batch_size * seq_len + 511) // 512, 16)


def load_model_and_tokenizer(model_name: str, device: torch.device, activation_checkpointing: bool, attn_implementation: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_kwargs = {}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        **model_kwargs,
    )
    model.config.use_cache = False
    if activation_checkpointing:
        if not hasattr(model, "gradient_checkpointing_enable"):
            raise RuntimeError(f"Model {model_name} does not expose gradient_checkpointing_enable()")
        model.gradient_checkpointing_enable()
    model.to(device)
    return model, tokenizer


def is_embedding_like_name(name: str) -> bool:
    lowered = name.lower()
    return any(part in lowered for part in ("embed", "embedding", "wte", "wpe", "lm_head"))


def select_trainable_named_params(model: torch.nn.Module, param_scope: ParamScope) -> tuple[list[tuple[str, torch.nn.Parameter]], dict[str, int]]:
    trainable = []
    stats = {
        "selected_tensors": 0,
        "selected_params": 0,
        "selected_matrix_tensors": 0,
        "selected_matrix_params": 0,
        "selected_fallback_tensors": 0,
        "selected_fallback_params": 0,
        "excluded_embedding_tensors": 0,
        "excluded_embedding_params": 0,
        "excluded_3d_tensors": 0,
        "excluded_3d_params": 0,
    }

    for name, param in model.named_parameters():
        if param_scope == "full":
            wants_param = True
        elif param_scope in {"broad-no-embeddings", "matrices-no-embeddings"}:
            is_embedding = is_embedding_like_name(name)
            wants_param = not is_embedding if param_scope == "broad-no-embeddings" else param.ndim == 2 and not is_embedding
            if is_embedding:
                stats["excluded_embedding_tensors"] += 1
                stats["excluded_embedding_params"] += param.numel()
            if param.ndim == 3 and not wants_param:
                stats["excluded_3d_tensors"] += 1
                stats["excluded_3d_params"] += param.numel()
        else:  # pragma: no cover - argparse constrains this
            raise ValueError(f"unknown param scope: {param_scope}")

        param.requires_grad_(wants_param)
        if not wants_param:
            continue
        trainable.append((name, param))
        stats["selected_tensors"] += 1
        stats["selected_params"] += param.numel()
        if param.ndim == 2:
            stats["selected_matrix_tensors"] += 1
            stats["selected_matrix_params"] += param.numel()
        else:
            stats["selected_fallback_tensors"] += 1
            stats["selected_fallback_params"] += param.numel()

    if not trainable:
        raise RuntimeError(f"No trainable parameters selected for param_scope={param_scope}")
    return trainable, stats


def select_trainable_params(model: torch.nn.Module, param_scope: ParamScope) -> tuple[list[torch.nn.Parameter], dict[str, int]]:
    trainable, stats = select_trainable_named_params(model, param_scope)
    return [param for _name, param in trainable], stats


def transformer_matrix_role(name: str, param: torch.nn.Parameter) -> str:
    lowered = name.lower()
    if param.ndim != 2:
        return "fallback"
    if any(part in lowered for part in ("gate_proj", "up_proj", "w1", "w3", "fc1")):
        return "mlp_up_gate"
    if any(part in lowered for part in ("down_proj", "w2", "fc2")):
        return "mlp_down"
    if any(part in lowered for part in ("q_proj", "k_proj", "v_proj", "query", "key", "value")):
        return "attention_qkv"
    if any(part in lowered for part in ("o_proj", "out_proj", "dense")):
        return "attention_out"
    return "other_matrix"


def storage_side_for_residual_axis(role: str, policy: ProjectionSidePolicy) -> str:
    """Return the storage side that faces the residual/backbone stream.

    `nn.Linear.weight` is stored as `[out_features, in_features]`. For MLP
    up/gate and attention q/k/v matrices, the residual stream is the forward
    input activation axis, i.e. the right/column side in storage. For MLP down
    and attention output projections, the residual stream is the output/top side,
    i.e. the left/row side in storage.
    """

    if policy == "auto":
        return "auto"
    if role in {"mlp_up_gate", "attention_qkv"}:
        return "right"
    if role in {"mlp_down", "attention_out"}:
        return "left"
    return "auto"


def build_sumotrack_param_groups(
    named_params: list[tuple[str, torch.nn.Parameter]],
    rank: int,
    projection_side_policy: ProjectionSidePolicy,
) -> tuple[list[dict], dict[str, int]]:
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    policy_stats = {"effective_rank_min": 0, "effective_rank_max": 0, "side_policy_left_tensors": 0, "side_policy_right_tensors": 0, "side_policy_auto_tensors": 0}

    grouped: dict[str, list[torch.nn.Parameter]] = {}
    for name, param in named_params:
        role = transformer_matrix_role(name, param)
        side = storage_side_for_residual_axis(role, projection_side_policy)
        grouped.setdefault(side, []).append(param)
        if param.ndim == 2:
            effective_rank = min(rank, *param.shape)
            policy_stats["effective_rank_min"] = effective_rank if not policy_stats["effective_rank_min"] else min(policy_stats["effective_rank_min"], effective_rank)
            policy_stats["effective_rank_max"] = max(policy_stats["effective_rank_max"], effective_rank)
            policy_stats[f"side_policy_{side}_tensors"] += 1

    return [{"params": params, "rank": rank, "side": side} for side, params in grouped.items()], policy_stats


def make_packed_batches(tokenizer, texts: list[str], device: torch.device, batch_size: int, seq_len: int, min_batches: int):
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise RuntimeError("tokenizer must define eos_token_id for packed no-mask LM batches")
    stream: list[int] = []
    needed = max(min_batches, 1) * batch_size * seq_len
    for text in texts:
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        stream.extend(token_ids)
        stream.append(eos_token_id)
        if len(stream) >= needed:
            break
    if len(stream) < needed:
        raise RuntimeError(f"not enough real tokens to build packed batches: need {needed}, got {len(stream)}")
    tokens = torch.tensor(stream[:needed], dtype=torch.long).view(max(min_batches, 1), batch_size, seq_len).to(device)
    return [{"input_ids": batch, "labels": batch.clone()} for batch in tokens]


def batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["input_ids"].numel())


def _inner_causal_lm_modules(model):
    inner = getattr(model, "_orig_mod", model)
    base = getattr(inner, "model", None)
    lm_head = getattr(inner, "lm_head", None)
    if base is None or lm_head is None:
        raise RuntimeError("CCE LM loss requires a causal LM with .model and .lm_head modules")
    return base, lm_head


def cce_causal_lm_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    from cut_cross_entropy import linear_cross_entropy

    base, lm_head = _inner_causal_lm_modules(model)
    outputs = base(
        input_ids=batch["input_ids"],
        use_cache=False,
    )
    return linear_cross_entropy(outputs.last_hidden_state, lm_head.weight, batch["input_ids"], ignore_index=-100, shift=True)


@torch.no_grad()
def evaluate_loss(model, batches) -> float:
    was_training = model.training
    model.eval()
    try:
        losses = []
        for batch in batches:
            losses.append(cce_causal_lm_loss(model, batch).detach().float())
        return float(torch.stack(losses).mean().cpu())
    finally:
        if was_training:
            model.train()


def train_step(
    model,
    optimizer: torch.optim.Optimizer,
    trainable: list[torch.nn.Parameter],
    batches,
    start_index: int,
    grad_accum_steps: int,
    log_norms: bool,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for offset in range(grad_accum_steps):
        batch = batches[(start_index + offset) % len(batches)]
        loss = cce_causal_lm_loss(model, batch)
        (loss / grad_accum_steps).backward()
        losses.append(loss.detach().float())
    grad_norm = gradient_norm(trainable) if log_norms else float("nan")
    param_norm = parameter_norm(trainable) if log_norms else float("nan")
    optimizer.step()
    update_norm = optimizer_update_norm(optimizer) if log_norms else float("nan")
    leverage_cv, leverage_min_ratio, leverage_max_ratio = optimizer_projected_leverage_stats(optimizer) if log_norms else (float("nan"), float("nan"), float("nan"))
    return {
        "loss": float(torch.stack(losses).mean().cpu()),
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "update_norm": update_norm,
        "update_to_param_ratio": update_norm / param_norm if param_norm > 0 else float("nan"),
        "projected_leverage_cv": leverage_cv,
        "projected_leverage_min_ratio": leverage_min_ratio,
        "projected_leverage_max_ratio": leverage_max_ratio,
    }


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if value == value]


def mean_or_nan(values: list[float]) -> float:
    finite = finite_values(values)
    return sum(finite) / len(finite) if finite else float("nan")


def run_optimizer(
    args,
    optimizer_name: str,
    model_name: str,
    train_texts: list[str],
    val_texts: list[str],
    retention_texts: list[str] | None,
    device: torch.device,
) -> dict[str, float | int | str]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    model, tokenizer = load_model_and_tokenizer(model_name, device, args.activation_checkpointing, args.attn_implementation)
    trainable_named, param_stats = select_trainable_named_params(model, args.param_scope)
    trainable = [param for _name, param in trainable_named]
    sumotrack_param_groups, policy_stats = build_sumotrack_param_groups(
        trainable_named,
        args.rank,
        args.projection_side_policy,
    )
    train_batches = make_packed_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len, args.warmup_steps + args.measure_steps)
    val_batches = make_packed_batches(tokenizer, val_texts, device, args.batch_size, args.seq_len, args.val_blocks)
    retention_batches = make_packed_batches(tokenizer, retention_texts, device, args.batch_size, args.seq_len, args.retention_val_blocks) if retention_texts else None

    if args.torch_compile:
        model = torch.compile(model)

    if optimizer_name == "subspace":
        optimizer_name = "sumotrack"

    if optimizer_name == "sumotrack":
        optimizer = SumoTrack(
            sumotrack_param_groups,
            lr=args.sumotrack_lr,
            beta=args.beta,
            basis_init=args.basis_init,
            grassmann_step_size=args.grassmann_step_size,
            basis_refresh_interval=args.basis_refresh_interval,
            aurora_pp_iterations=args.aurora_pp_iterations,
            polar_ns_steps=args.polar_ns_steps,
            consume_grad=not args.keep_grads_after_step,
        )
        optimizer.diagnostics_enabled = args.log_norms
    elif optimizer_name in {"adamw", "torch_adamw"}:
        optimizer = torch.optim.AdamW(trainable, lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=device.type == "cuda")
    else:  # pragma: no cover
        raise ValueError(optimizer_name)

    initial_val = evaluate_loss(model, val_batches) if not args.skip_validation else float("nan")
    initial_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None and not args.skip_validation else float("nan")
    measured_steps = []

    for step in range(args.warmup_steps):
        batch_index = step * args.grad_accum_steps
        train_step(model, optimizer, trainable, train_batches, batch_index, args.grad_accum_steps, args.log_norms)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for step in range(args.measure_steps):
        batch_index = (args.warmup_steps + step) * args.grad_accum_steps
        measured_steps.append(train_step(model, optimizer, trainable, train_batches, batch_index, args.grad_accum_steps, args.log_norms))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_elapsed = time.perf_counter() - start_time
    final_val = evaluate_loss(model, val_batches) if not args.skip_validation else float("nan")
    final_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None and not args.skip_validation else float("nan")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    measured_losses = [step["loss"] for step in measured_steps]
    measured_grad_norms = [step["grad_norm"] for step in measured_steps]
    measured_param_norms = [step["param_norm"] for step in measured_steps]
    measured_update_norms = [step["update_norm"] for step in measured_steps]
    measured_update_to_param_ratios = [step["update_to_param_ratio"] for step in measured_steps]
    measured_leverage_cvs = [step["projected_leverage_cv"] for step in measured_steps]
    measured_leverage_min_ratios = [step["projected_leverage_min_ratio"] for step in measured_steps]
    measured_leverage_max_ratios = [step["projected_leverage_max_ratio"] for step in measured_steps]
    state_bytes = optimizer_state_bytes_by_category(optimizer)
    result = {
        "optimizer": optimizer_name,
        "projection_side_policy": args.projection_side_policy if optimizer_name == "sumotrack" else "n/a",
        "rank": args.rank if optimizer_name == "sumotrack" else 0,
        "effective_rank_min": policy_stats["effective_rank_min"] if optimizer_name == "sumotrack" else 0,
        "effective_rank_max": policy_stats["effective_rank_max"] if optimizer_name == "sumotrack" else 0,
        "side_policy_left_tensors": policy_stats["side_policy_left_tensors"] if optimizer_name == "sumotrack" else 0,
        "side_policy_right_tensors": policy_stats["side_policy_right_tensors"] if optimizer_name == "sumotrack" else 0,
        "side_policy_auto_tensors": policy_stats["side_policy_auto_tensors"] if optimizer_name == "sumotrack" else 0,
        "basis_init": args.basis_init if optimizer_name == "sumotrack" else "n/a",
        "basis_refresh_interval": args.basis_refresh_interval if optimizer_name == "sumotrack" else 0,
        "aurora_pp_iterations": args.aurora_pp_iterations if optimizer_name == "sumotrack" else 0,
        "polar_ns_steps": args.polar_ns_steps if optimizer_name == "sumotrack" else 0,
        "consume_grad": (not args.keep_grads_after_step) if optimizer_name == "sumotrack" else False,
        "activation_checkpointing": args.activation_checkpointing,
        "torch_compile": args.torch_compile,
        "attn_implementation": getattr(getattr(model, "config", None), "_attn_implementation", "n/a"),
        "batching": "eos_packed_no_mask",
        "loss_impl": "cce",
        "skip_validation": args.skip_validation,
        "actual_sequence_tokens": batch_tokens(train_batches[0]),
        "tokens_per_optimizer_step": batch_tokens(train_batches[0]) * args.grad_accum_steps,
        "measured_tokens_per_second": (batch_tokens(train_batches[0]) * args.grad_accum_steps * args.measure_steps) / measured_elapsed,
        "matrix_state_bytes": state_bytes["matrix"],
        "fallback_state_bytes": state_bytes["fallback"],
        "state_bytes": state_bytes["total"],
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "initial_retention_val_loss": initial_retention_val,
        "final_retention_val_loss": final_retention_val,
        "retention_val_loss_delta": final_retention_val - initial_retention_val,
        "last_measured_train_loss": measured_losses[-1],
        "mean_measured_grad_norm": mean_or_nan(measured_grad_norms),
        "mean_measured_param_norm": mean_or_nan(measured_param_norms),
        "mean_measured_update_norm": mean_or_nan(measured_update_norms),
        "mean_measured_update_to_param_ratio": mean_or_nan(measured_update_to_param_ratios),
        "mean_measured_projected_leverage_cv": mean_or_nan(measured_leverage_cvs),
        "mean_measured_projected_leverage_min_ratio": mean_or_nan(measured_leverage_min_ratios),
        "mean_measured_projected_leverage_max_ratio": mean_or_nan(measured_leverage_max_ratios),
        "measured_elapsed_seconds": measured_elapsed,
        "measured_step_seconds": measured_elapsed / args.measure_steps,
        "peak_cuda_bytes": peak,
    }
    del optimizer, model, tokenizer, train_batches, val_batches, retention_batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short pretrained-LLM SYNTH smoke for SumoTrack")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model name; default = {DEFAULT_MODEL}")
    parser.add_argument("--data-dir", default="/home/djg/.cache/nanochat/base_data_synth")
    parser.add_argument("--retention-data-dir", default="", help="optional SYNTH-format source/retention parquet directory")
    parser.add_argument("--optimizers", default="sumotrack", help="comma-separated: sumotrack,torch_adamw")
    parser.add_argument("--param-scope", choices=("full", "broad-no-embeddings", "matrices-no-embeddings"), default="broad-no-embeddings")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-blocks", type=int, default=8, help="number of packed validation blocks")
    parser.add_argument("--retention-val-blocks", type=int, default=8, help="number of packed retention validation blocks")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--projection-side-policy", choices=("auto", "residual-facing"), default="residual-facing")
    parser.add_argument("--basis-init", choices=("svd", "random"), default="svd")
    parser.add_argument("--sumotrack-lr", type=float, default=0.0025)
    parser.add_argument("--adamw-lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--grassmann-step-size", type=float, default=0.01)
    parser.add_argument("--basis-refresh-interval", type=int, default=100)
    parser.add_argument("--aurora-pp-iterations", type=int, default=2)
    parser.add_argument("--polar-ns-steps", type=int, default=5)
    parser.add_argument("--activation-checkpointing", action="store_true", help="enable model gradient checkpointing before training")
    parser.add_argument("--torch-compile", action="store_true", help="compile the model forward/backward with torch.compile")
    parser.add_argument("--attn-implementation", default="", help="optional Transformers attention implementation, e.g. flash_attention_2, sdpa, eager")
    parser.add_argument("--skip-validation", action="store_true", help="skip initial/final validation for throughput-only runs")
    parser.add_argument("--keep-grads-after-step", action="store_true", help="leave p.grad populated after optimizer.step(); default consumes grads once projected")
    parser.add_argument("--log-norms", action="store_true", help="log grad/param norms and SumoTrack update norms; adds reduction overhead")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.measure_steps <= 0:
        raise ValueError("measure_steps must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if args.seq_len <= 1:
        raise ValueError("seq_len must be greater than 1")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.val_blocks <= 0:
        raise ValueError("val_blocks must be positive")
    if args.retention_val_blocks <= 0:
        raise ValueError("retention_val_blocks must be positive")
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    if args.basis_refresh_interval <= 0:
        raise ValueError("basis_refresh_interval must be positive")
    if args.aurora_pp_iterations <= 0:
        raise ValueError("aurora_pp_iterations must be positive")
    if not 1 <= args.polar_ns_steps <= 5:
        raise ValueError("polar_ns_steps must be in [1, 5]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model

    total_train_steps = args.warmup_steps + args.measure_steps
    train_blocks = max(total_train_steps * args.grad_accum_steps, 1)
    val_blocks = args.val_blocks
    train_texts = synth_texts(Path(args.data_dir), "train", limit=packed_text_limit(train_blocks, args.batch_size, args.seq_len))
    val_texts = synth_texts(Path(args.data_dir), "val", limit=packed_text_limit(val_blocks, args.batch_size, args.seq_len))
    retention_texts = None
    if args.retention_data_dir:
        retention_blocks = args.retention_val_blocks
        retention_texts = synth_texts(Path(args.retention_data_dir), "val", limit=packed_text_limit(retention_blocks, args.batch_size, args.seq_len))
    print(f"device={device}")
    print(f"model={model_name}")
    print(f"data_dir={args.data_dir}")
    print(f"retention_data_dir={args.retention_data_dir or 'none'}")
    print(f"train_texts={len(train_texts)} val_texts={len(val_texts)} retention_texts={len(retention_texts) if retention_texts else 0}")
    print(
        f"seq_len={args.seq_len} batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
        f"warmup_steps={args.warmup_steps} measure_steps={args.measure_steps} param_scope={args.param_scope} "
        f"rank={args.rank} projection_side_policy={args.projection_side_policy} "
        f"basis_init={args.basis_init} basis_refresh_interval={args.basis_refresh_interval} "
        f"orthogonalization=aurora aurora_pp_iterations={args.aurora_pp_iterations} polar_ns_steps={args.polar_ns_steps} "
        f"activation_checkpointing={args.activation_checkpointing} torch_compile={args.torch_compile} attn_implementation={args.attn_implementation or 'default'} "
        f"batching=eos_packed_no_mask loss_impl=cce "
        f"skip_validation={args.skip_validation} consume_grad={not args.keep_grads_after_step}"
    )

    for optimizer_name in [name.strip() for name in args.optimizers.split(",") if name.strip()]:
        if optimizer_name == "subspace":
            optimizer_name = "sumotrack"
        result = run_optimizer(args, optimizer_name, model_name, train_texts, val_texts, retention_texts, device)
        prefix = optimizer_name
        for key, value in result.items():
            if key == "optimizer":
                continue
            if isinstance(value, float):
                print(f"{prefix}_{key}={value:.6f}")
            else:
                print(f"{prefix}_{key}={value}")


if __name__ == "__main__":
    main()
