from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
import torch
import heavyball
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SumoTrack, optimizer_state_bytes_by_category


PREFERRED_MODELS = (
    "LiquidAI/LFM2.5-1.2B-Base",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3-4B",
)

ParamScope = Literal["full", "broad-no-embeddings", "matrices-no-embeddings"]
ProjectionSidePolicy = Literal["auto", "module-role"]
RankPolicy = Literal["uniform", "size", "module-role"]


def cached_model_snapshots(model_name: str) -> list[Path]:
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not cache_dir.exists():
        return []
    return sorted([path for path in cache_dir.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)


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


def choose_model(explicit: str | None) -> str:
    candidates = (explicit,) if explicit else PREFERRED_MODELS
    errors = []
    for name in candidates:
        try:
            AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=True)
            has_weights = any(snapshot.glob("*.safetensors") for snapshot in cached_model_snapshots(name)) or any(
                snapshot.glob("*.bin") for snapshot in cached_model_snapshots(name)
            )
            if not has_weights:
                raise FileNotFoundError(f"tokenizer/config cached but no local weight files for {name}")
            return name
        except Exception as error:  # noqa: BLE001 - printed as terrain
            errors.append(f"{name}: {type(error).__name__}: {error}")
    raise RuntimeError("No preferred model tokenizer is cached locally:\n" + "\n".join(errors))


def load_model_and_tokenizer(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
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


def module_role(name: str, param: torch.nn.Parameter) -> str:
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


def side_for_policy(role: str, policy: ProjectionSidePolicy) -> str:
    if policy == "auto":
        return "auto"
    if role in {"mlp_up_gate", "attention_qkv"}:
        return "right"
    if role in {"mlp_down", "attention_out"}:
        return "left"
    return "auto"


def rank_for_policy(param: torch.nn.Parameter, role: str, rank: int, policy: RankPolicy, min_rank: int, max_rank: int | None, median_matrix_params: float) -> int:
    if param.ndim != 2:
        return rank
    if policy == "uniform":
        proposed = rank
    elif policy == "size":
        proposed = round(rank * (param.numel() / median_matrix_params) ** 0.5)
    else:
        multiplier = {
            "mlp_up_gate": 1.5,
            "mlp_down": 1.25,
            "attention_qkv": 1.0,
            "attention_out": 1.0,
            "other_matrix": 0.75,
        }.get(role, 1.0)
        proposed = round(rank * multiplier)
    cap = min(param.shape) if max_rank is None else min(max_rank, min(param.shape))
    return max(1, min(cap, max(min_rank, proposed)))


def matrix_state_cost_per_rank(param: torch.nn.Parameter) -> int:
    return (param.shape[0] + param.shape[1]) * param.element_size()


def apply_matrix_state_budget(param_options: dict[int, dict], named_params: list[tuple[str, torch.nn.Parameter]], budget_mb: float, min_rank: int) -> None:
    if budget_mb <= 0:
        return
    budget_bytes = int(budget_mb * 1024 * 1024)
    matrix_params = [param for _name, param in named_params if param.ndim == 2]
    min_cost = sum(min(min_rank, min(param.shape)) * matrix_state_cost_per_rank(param) for param in matrix_params)
    if min_cost > budget_bytes:
        raise ValueError(f"optimizer_state_budget_mb={budget_mb} is below minimum matrix state cost {min_cost / 1024 / 1024:.2f} MB")

    def total_cost() -> int:
        return sum(param_options[id(param)]["rank"] * matrix_state_cost_per_rank(param) for param in matrix_params)

    while total_cost() > budget_bytes:
        candidates = [param for param in matrix_params if param_options[id(param)]["rank"] > min(min_rank, min(param.shape))]
        if not candidates:
            break
        target = max(candidates, key=matrix_state_cost_per_rank)
        param_options[id(target)]["rank"] -= 1


def build_sumotrack_param_groups(
    named_params: list[tuple[str, torch.nn.Parameter]],
    rank: int,
    rank_policy: RankPolicy,
    projection_side_policy: ProjectionSidePolicy,
    min_rank: int,
    max_rank: int | None,
    optimizer_state_budget_mb: float,
) -> tuple[list[dict], dict[str, int]]:
    matrix_sizes = sorted(param.numel() for _name, param in named_params if param.ndim == 2)
    median_matrix_params = float(matrix_sizes[len(matrix_sizes) // 2]) if matrix_sizes else 1.0
    param_options = {}
    policy_stats = {"rank_policy_min_rank": 0, "rank_policy_max_rank": 0, "side_policy_left_tensors": 0, "side_policy_right_tensors": 0, "side_policy_auto_tensors": 0}

    for name, param in named_params:
        role = module_role(name, param)
        side = side_for_policy(role, projection_side_policy)
        param_rank = rank_for_policy(param, role, rank, rank_policy, min_rank, max_rank, median_matrix_params)
        param_options[id(param)] = {"rank": param_rank, "side": side}

    apply_matrix_state_budget(param_options, named_params, optimizer_state_budget_mb, min_rank)

    grouped: dict[tuple[int, str], list[torch.nn.Parameter]] = {}
    for _name, param in named_params:
        options = param_options[id(param)]
        grouped.setdefault((options["rank"], options["side"]), []).append(param)
        if param.ndim == 2:
            policy_stats["rank_policy_min_rank"] = options["rank"] if not policy_stats["rank_policy_min_rank"] else min(policy_stats["rank_policy_min_rank"], options["rank"])
            policy_stats["rank_policy_max_rank"] = max(policy_stats["rank_policy_max_rank"], options["rank"])
            policy_stats[f"side_policy_{options['side']}_tensors"] += 1

    return [{"params": params, "rank": group_rank, "side": group_side} for (group_rank, group_side), params in grouped.items()], policy_stats


def make_batches(tokenizer, texts: list[str], device: torch.device, batch_size: int, seq_len: int):
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()})
    return batches


@torch.no_grad()
def evaluate_loss(model, batches) -> float:
    model.eval()
    losses = []
    for batch in batches:
        losses.append(float(model(**batch).loss.detach().float().cpu()))
    model.train()
    return sum(losses) / len(losses)


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
        loss = model(**batch).loss
        (loss / grad_accum_steps).backward()
        losses.append(float(loss.detach().float().cpu()))
    grad_norm = gradient_norm(trainable) if log_norms else float("nan")
    param_norm = parameter_norm(trainable) if log_norms else float("nan")
    optimizer.step()
    update_norm = optimizer_update_norm(optimizer) if log_norms else float("nan")
    leverage_cv, leverage_min_ratio, leverage_max_ratio = optimizer_projected_leverage_stats(optimizer) if log_norms else (float("nan"), float("nan"), float("nan"))
    return {
        "loss": sum(losses) / len(losses),
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
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    trainable_named, param_stats = select_trainable_named_params(model, args.param_scope)
    trainable = [param for _name, param in trainable_named]
    sumotrack_param_groups, policy_stats = build_sumotrack_param_groups(
        trainable_named,
        args.rank,
        args.rank_policy,
        args.projection_side_policy,
        args.min_rank,
        args.max_rank,
        args.optimizer_state_budget_mb,
    )
    train_batches = make_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len)
    val_batches = make_batches(tokenizer, val_texts, device, args.batch_size, args.seq_len)
    retention_batches = make_batches(tokenizer, retention_texts, device, args.batch_size, args.seq_len) if retention_texts else None

    if optimizer_name == "subspace":
        optimizer_name = "sumotrack"

    if optimizer_name == "sumotrack":
        optimizer = SumoTrack(
            sumotrack_param_groups,
            lr=args.sumotrack_lr,
            beta=args.beta,
            projection_mode=args.projection_mode,
            orthogonalization=args.orthogonalization,
            orthogonalization_scale_mode=args.orthogonalization_scale_mode,
            heavyball_orthogonalization_mode=args.heavyball_orthogonalization_mode,
            aurora_pp_iterations=args.aurora_pp_iterations,
            aurora_pp_beta=args.aurora_pp_beta,
            subspace_init=args.subspace_init,
            subspace_update_method=args.subspace_update_method,
            grassmann_step_size=args.grassmann_step_size,
            subspace_refresh_budget=args.subspace_refresh_budget,
        )
        optimizer.diagnostics_enabled = args.log_norms
    elif optimizer_name in {"adamw", "torch_adamw"}:
        optimizer = torch.optim.AdamW(trainable, lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=device.type == "cuda")
    elif optimizer_name in {"heavyball_adamw", "hb_adamw"}:
        optimizer = heavyball.AdamW(trainable, lr=args.adamw_lr, betas=(0.9, 0.99), weight_decay=0.0, compile_step=False)
    else:  # pragma: no cover
        raise ValueError(optimizer_name)

    initial_val = evaluate_loss(model, val_batches)
    initial_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None else float("nan")
    warmup_steps = []
    measured_steps = []

    for step in range(args.warmup_steps):
        batch_index = step * args.grad_accum_steps
        warmup_steps.append(train_step(model, optimizer, trainable, train_batches, batch_index, args.grad_accum_steps, args.log_norms))

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
    final_val = evaluate_loss(model, val_batches)
    final_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None else float("nan")
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    warmup_losses = [step["loss"] for step in warmup_steps]
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
        "param_scope": args.param_scope,
        "rank_policy": args.rank_policy if optimizer_name == "sumotrack" else "n/a",
        "projection_side_policy": args.projection_side_policy if optimizer_name == "sumotrack" else "n/a",
        "rank_policy_min_rank": policy_stats["rank_policy_min_rank"] if optimizer_name == "sumotrack" else 0,
        "rank_policy_max_rank": policy_stats["rank_policy_max_rank"] if optimizer_name == "sumotrack" else 0,
        "side_policy_left_tensors": policy_stats["side_policy_left_tensors"] if optimizer_name == "sumotrack" else 0,
        "side_policy_right_tensors": policy_stats["side_policy_right_tensors"] if optimizer_name == "sumotrack" else 0,
        "side_policy_auto_tensors": policy_stats["side_policy_auto_tensors"] if optimizer_name == "sumotrack" else 0,
        "subspace_init": args.subspace_init if optimizer_name == "sumotrack" else "n/a",
        "projection_mode": args.projection_mode if optimizer_name == "sumotrack" else "n/a",
        "subspace_update_method": args.subspace_update_method if optimizer_name == "sumotrack" else "n/a",
        "orthogonalization": args.orthogonalization if optimizer_name == "sumotrack" else "n/a",
        "orthogonalization_scale_mode": args.orthogonalization_scale_mode if optimizer_name == "sumotrack" else "n/a",
        "heavyball_orthogonalization_mode": args.heavyball_orthogonalization_mode if optimizer_name == "sumotrack" else "n/a",
        "grad_accum_steps": args.grad_accum_steps,
        "norm_logging": int(args.log_norms),
        "tokens_per_optimizer_step": args.batch_size * args.seq_len * args.grad_accum_steps,
        "trainable_params": param_stats["selected_params"],
        "trainable_tensors": param_stats["selected_tensors"],
        "trainable_matrix_params": param_stats["selected_matrix_params"],
        "trainable_matrix_tensors": param_stats["selected_matrix_tensors"],
        "trainable_fallback_params": param_stats["selected_fallback_params"],
        "trainable_fallback_tensors": param_stats["selected_fallback_tensors"],
        "excluded_embedding_params": param_stats["excluded_embedding_params"],
        "excluded_3d_params": param_stats["excluded_3d_params"],
        "matrix_state_bytes": state_bytes["matrix"],
        "fallback_state_bytes": state_bytes["fallback"],
        "state_bytes": state_bytes["total"],
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "initial_retention_val_loss": initial_retention_val,
        "final_retention_val_loss": final_retention_val,
        "retention_val_loss_delta": final_retention_val - initial_retention_val,
        "first_warmup_loss": warmup_losses[0] if warmup_losses else float("nan"),
        "last_warmup_loss": warmup_losses[-1] if warmup_losses else float("nan"),
        "first_measured_train_loss": measured_losses[0],
        "last_measured_train_loss": measured_losses[-1],
        "mean_measured_train_loss": sum(measured_losses) / len(measured_losses),
        "min_measured_train_loss": min(measured_losses),
        "max_measured_train_loss": max(measured_losses),
        "mean_measured_grad_norm": mean_or_nan(measured_grad_norms),
        "mean_measured_param_norm": mean_or_nan(measured_param_norms),
        "mean_measured_update_norm": mean_or_nan(measured_update_norms),
        "mean_measured_update_to_param_ratio": mean_or_nan(measured_update_to_param_ratios),
        "mean_measured_projected_leverage_cv": mean_or_nan(measured_leverage_cvs),
        "mean_measured_projected_leverage_min_ratio": mean_or_nan(measured_leverage_min_ratios),
        "mean_measured_projected_leverage_max_ratio": mean_or_nan(measured_leverage_max_ratios),
        "last_measured_grad_norm": measured_grad_norms[-1],
        "last_measured_param_norm": measured_param_norms[-1],
        "last_measured_update_norm": measured_update_norms[-1],
        "last_measured_update_to_param_ratio": measured_update_to_param_ratios[-1],
        "last_measured_projected_leverage_cv": measured_leverage_cvs[-1],
        "last_measured_projected_leverage_min_ratio": measured_leverage_min_ratios[-1],
        "last_measured_projected_leverage_max_ratio": measured_leverage_max_ratios[-1],
        "measured_elapsed_seconds": measured_elapsed,
        "measured_step_seconds": measured_elapsed / args.measure_steps,
        "peak_cuda_bytes": peak,
    }
    del optimizer, model, tokenizer, train_batches, val_batches, retention_batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def local_safetensor_shapes(model_name: str) -> dict[str, tuple[int, ...]]:
    for snapshot in cached_model_snapshots(model_name):
        index_path = snapshot / "model.safetensors.index.json"
        if index_path.exists():
            with index_path.open("r", encoding="utf-8") as handle:
                index = json.load(handle)
            # Index files usually omit shape metadata; fall through to shard headers.
            safetensors_paths = sorted({snapshot / file_name for file_name in index.get("weight_map", {}).values()})
        else:
            safetensors_paths = sorted(snapshot.glob("*.safetensors"))
        if not safetensors_paths:
            continue

        shapes: dict[str, tuple[int, ...]] = {}
        for path in safetensors_paths:
            with path.open("rb") as handle:
                header_size = int.from_bytes(handle.read(8), byteorder="little")
                header = json.loads(handle.read(header_size))
            for tensor_name, tensor_info in header.items():
                if tensor_name == "__metadata__":
                    continue
                shapes[tensor_name] = tuple(tensor_info["shape"])
        return shapes
    return {}


def print_shapes(model_name: str, shapes: dict[str, tuple[int, ...]]) -> None:
    shapes_by_dim: dict[int, tuple[int, int]] = {}
    total_params = 0
    for shape in shapes.values():
        dim = len(shape)
        count, params = shapes_by_dim.get(dim, (0, 0))
        param_count = 1
        for size in shape:
            param_count *= size
        total_params += param_count
        shapes_by_dim[dim] = (count + 1, params + param_count)

    print(f"shape_summary_model={model_name}")
    print(f"shape_summary_total_params={total_params}")
    for dim, (count, params) in sorted(shapes_by_dim.items()):
        print(f"shape_summary_dim_{dim}_tensors={count}")
        print(f"shape_summary_dim_{dim}_params={params}")
    for tensor_name, shape in sorted(shapes.items()):
        if len(shape) == 3:
            print(f"shape_summary_3d={tensor_name}:{shape}")


def print_shape_summary(model_name: str) -> None:
    local_shapes = local_safetensor_shapes(model_name)
    if local_shapes:
        print_shapes(model_name, local_shapes)
        return

    from huggingface_hub import get_safetensors_metadata

    metadata = get_safetensors_metadata(model_name)
    remote_shapes = {}
    for tensor_name, tensor_info in metadata.tensors.items():
        remote_shapes[tensor_name] = tuple(tensor_info.shape)
    print_shapes(model_name, remote_shapes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Short cached pretrained-LLM SYNTH smoke for SumoTrack")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Base", help="HF model name; default = cached LFM 1.2B")
    parser.add_argument("--data-dir", default="/home/djg/.cache/nanochat/base_data_synth")
    parser.add_argument("--retention-data-dir", default="", help="optional SYNTH-format source/retention parquet directory")
    parser.add_argument("--optimizers", default="sumotrack", help="comma-separated: sumotrack,torch_adamw,heavyball_adamw")
    parser.add_argument("--param-scope", choices=("full", "broad-no-embeddings", "matrices-no-embeddings"), default="matrices-no-embeddings")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-texts", type=int, default=8)
    parser.add_argument("--retention-val-texts", type=int, default=8)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--rank-policy", choices=("uniform", "size", "module-role"), default="uniform")
    parser.add_argument("--min-rank", type=int, default=1)
    parser.add_argument("--max-rank", type=int, default=0, help="0 means cap only by tensor dimension")
    parser.add_argument("--optimizer-state-budget-mb", type=float, default=0.0, help="optional matrix-state budget for rank clamping")
    parser.add_argument("--projection-mode", choices=("one_sided", "two_sided"), default="one_sided")
    parser.add_argument("--projection-side-policy", choices=("auto", "module-role"), default="auto")
    parser.add_argument("--subspace-init", choices=("svd", "random"), default="svd")
    parser.add_argument("--subspace-update-method", choices=("none", "svd_refresh", "grassmann"), default="grassmann")
    parser.add_argument("--orthogonalization", choices=("none", "svd", "heavyball", "aurora"), default="aurora")
    parser.add_argument("--orthogonalization-scale-mode", choices=("none", "scale", "graft", "muon"), default="muon")
    parser.add_argument("--heavyball-orthogonalization-mode", default="", help="empty = HeavyBall default; e.g. newtonschulz or thinky_polar_express")
    parser.add_argument("--aurora-pp-iterations", type=int, default=2)
    parser.add_argument("--aurora-pp-beta", type=float, default=0.5)
    parser.add_argument("--sumotrack-lr", type=float, default=0.0025)
    parser.add_argument("--subspace-lr", dest="sumotrack_lr", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--adamw-lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--grassmann-step-size", type=float, default=0.01)
    parser.add_argument("--subspace-refresh-budget", type=int, default=1)
    parser.add_argument("--log-norms", action="store_true", help="log grad/param norms and SumoTrack update norms; adds reduction overhead")
    parser.add_argument("--print-shape-summary", action="store_true", help="print HF safetensor shape metadata and exit")
    args = parser.parse_args()

    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.measure_steps <= 0:
        raise ValueError("measure_steps must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if args.min_rank <= 0:
        raise ValueError("min_rank must be positive")
    args.max_rank = args.max_rank or None
    args.heavyball_orthogonalization_mode = args.heavyball_orthogonalization_mode or None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = choose_model(args.model or None)
    if args.print_shape_summary:
        print_shape_summary(model_name)
        return

    total_train_steps = args.warmup_steps + args.measure_steps
    train_texts = synth_texts(Path(args.data_dir), "train", limit=max(total_train_steps * args.grad_accum_steps, 4) * args.batch_size)
    val_texts = synth_texts(Path(args.data_dir), "val", limit=max(args.val_texts, 1) * args.batch_size)
    retention_texts = None
    if args.retention_data_dir:
        retention_texts = synth_texts(Path(args.retention_data_dir), "val", limit=max(args.retention_val_texts, 1) * args.batch_size)
    print(f"device={device}")
    print(f"model={model_name}")
    print(f"data_dir={args.data_dir}")
    print(f"retention_data_dir={args.retention_data_dir or 'none'}")
    print(f"train_texts={len(train_texts)} val_texts={len(val_texts)} retention_texts={len(retention_texts) if retention_texts else 0}")
    print(
        f"seq_len={args.seq_len} batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
        f"warmup_steps={args.warmup_steps} measure_steps={args.measure_steps} param_scope={args.param_scope} "
        f"rank={args.rank} rank_policy={args.rank_policy} projection_side_policy={args.projection_side_policy} "
        f"subspace_init={args.subspace_init} orthogonalization={args.orthogonalization}"
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
