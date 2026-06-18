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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SumoTrack, optimizer_state_bytes_by_category


PREFERRED_MODELS = (
    "LiquidAI/LFM2.5-1.2B-Base",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3-4B",
)

ParamScope = Literal["full", "broad-no-embeddings", "matrices-no-embeddings"]
ProjectionSidePolicy = Literal["auto", "residual-facing"]
LossImpl = Literal["hf", "chunked", "cce"]


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


def load_model_and_tokenizer(model_name: str, device: torch.device, activation_checkpointing: bool, attn_implementation: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {}
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
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
    encoded = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)["input_ids"].flatten()
    needed = max(min_batches, 1) * batch_size * seq_len
    if encoded.numel() < 2:
        raise RuntimeError("not enough tokens to build packed batches")
    repeats = (needed + encoded.numel() - 1) // encoded.numel()
    tokens = encoded.repeat(repeats)[:needed].view(max(min_batches, 1), batch_size, seq_len).to(device)
    return [{"input_ids": batch, "labels": batch.clone()} for batch in tokens]


def make_batches(tokenizer, texts: list[str], device: torch.device, batch_size: int, seq_len: int, pad_to_max_length: bool):
    batches = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding="max_length" if pad_to_max_length else True,
            truncation=True,
            max_length=seq_len,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()})
    return batches


def batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["input_ids"].numel())


@torch.no_grad()
def evaluate_loss(model, batches) -> float:
    model.eval()
    losses = []
    for batch in batches:
        losses.append(float(model(**batch).loss.detach().float().cpu()))
    model.train()
    return sum(losses) / len(losses)


def _inner_causal_lm_modules(model):
    inner = getattr(model, "_orig_mod", model)
    base = getattr(inner, "model", None)
    lm_head = getattr(inner, "lm_head", None)
    if base is None or lm_head is None:
        raise RuntimeError("chunked LM loss requires a causal LM with .model and .lm_head modules")
    return base, lm_head


def cce_causal_lm_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    from cut_cross_entropy import linear_cross_entropy

    base, lm_head = _inner_causal_lm_modules(model)
    outputs = base(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        use_cache=False,
    )
    targets = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        targets = targets.masked_fill(attention_mask == 0, -100)
    return linear_cross_entropy(outputs.last_hidden_state, lm_head.weight, targets, ignore_index=-100, shift=True)


def chunked_causal_lm_loss(model, batch: dict[str, torch.Tensor], chunk_tokens: int, loss_impl: LossImpl) -> torch.Tensor:
    if loss_impl == "hf":
        return model(**batch).loss
    if loss_impl == "cce":
        return cce_causal_lm_loss(model, batch)
    if chunk_tokens <= 0:
        raise ValueError("chunked loss requires chunked_lm_loss_tokens > 0")
    base, lm_head = _inner_causal_lm_modules(model)
    outputs = base(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        use_cache=False,
    )
    hidden = outputs.last_hidden_state
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    total_loss = hidden.new_zeros(())
    total_items = 0
    for start in range(0, hidden.shape[1] - 1, chunk_tokens):
        end = min(start + chunk_tokens, hidden.shape[1] - 1)
        logits = lm_head(hidden[:, start:end, :]).float()
        labels = input_ids[:, start + 1 : end + 1]
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask[:, start + 1 : end + 1] == 0, -100)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1).to(logits.device),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss = total_loss + loss
        total_items += int((labels != -100).sum().detach().cpu())
    if total_items == 0:
        raise RuntimeError("chunked LM loss saw zero non-padding target tokens")
    return total_loss / total_items


@torch.no_grad()
def evaluate_loss_chunked(model, batches, chunk_tokens: int, loss_impl: LossImpl) -> float:
    if loss_impl == "hf":
        return evaluate_loss(model, batches)
    model.eval()
    losses = []
    for batch in batches:
        losses.append(float(chunked_causal_lm_loss(model, batch, chunk_tokens, loss_impl).detach().float().cpu()))
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
    chunked_lm_loss_tokens: int,
    loss_impl: LossImpl,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for offset in range(grad_accum_steps):
        batch = batches[(start_index + offset) % len(batches)]
        loss = chunked_causal_lm_loss(model, batch, chunked_lm_loss_tokens, loss_impl)
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
    model, tokenizer = load_model_and_tokenizer(model_name, device, args.activation_checkpointing, args.attn_implementation)
    trainable_named, param_stats = select_trainable_named_params(model, args.param_scope)
    trainable = [param for _name, param in trainable_named]
    sumotrack_param_groups, policy_stats = build_sumotrack_param_groups(
        trainable_named,
        args.rank,
        args.projection_side_policy,
    )
    if args.pack_sequences:
        train_batches = make_packed_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len, args.warmup_steps + args.measure_steps)
        val_batches = make_packed_batches(tokenizer, val_texts, device, args.batch_size, args.seq_len, max(args.val_texts, 1))
        retention_batches = make_packed_batches(tokenizer, retention_texts, device, args.batch_size, args.seq_len, max(args.retention_val_texts, 1)) if retention_texts else None
    else:
        train_batches = make_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len, args.pad_to_max_length)
        val_batches = make_batches(tokenizer, val_texts, device, args.batch_size, args.seq_len, args.pad_to_max_length)
        retention_batches = make_batches(tokenizer, retention_texts, device, args.batch_size, args.seq_len, args.pad_to_max_length) if retention_texts else None

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

    initial_val = evaluate_loss_chunked(model, val_batches, args.chunked_lm_loss_tokens, args.loss_impl) if not args.skip_validation else float("nan")
    initial_retention_val = evaluate_loss_chunked(model, retention_batches, args.chunked_lm_loss_tokens, args.loss_impl) if retention_batches is not None and not args.skip_validation else float("nan")
    measured_steps = []

    for step in range(args.warmup_steps):
        batch_index = step * args.grad_accum_steps
        train_step(model, optimizer, trainable, train_batches, batch_index, args.grad_accum_steps, args.log_norms, args.chunked_lm_loss_tokens, args.loss_impl)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for step in range(args.measure_steps):
        batch_index = (args.warmup_steps + step) * args.grad_accum_steps
        measured_steps.append(train_step(model, optimizer, trainable, train_batches, batch_index, args.grad_accum_steps, args.log_norms, args.chunked_lm_loss_tokens, args.loss_impl))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_elapsed = time.perf_counter() - start_time
    final_val = evaluate_loss_chunked(model, val_batches, args.chunked_lm_loss_tokens, args.loss_impl) if not args.skip_validation else float("nan")
    final_retention_val = evaluate_loss_chunked(model, retention_batches, args.chunked_lm_loss_tokens, args.loss_impl) if retention_batches is not None and not args.skip_validation else float("nan")
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
        "pad_to_max_length": args.pad_to_max_length,
        "pack_sequences": args.pack_sequences,
        "loss_impl": args.loss_impl,
        "chunked_lm_loss_tokens": args.chunked_lm_loss_tokens,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short cached pretrained-LLM SYNTH smoke for SumoTrack")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-1.2B-Base", help="HF model name; default = cached LFM 1.2B")
    parser.add_argument("--data-dir", default="/home/djg/.cache/nanochat/base_data_synth")
    parser.add_argument("--retention-data-dir", default="", help="optional SYNTH-format source/retention parquet directory")
    parser.add_argument("--optimizers", default="sumotrack", help="comma-separated: sumotrack,torch_adamw")
    parser.add_argument("--param-scope", choices=("full", "broad-no-embeddings", "matrices-no-embeddings"), default="broad-no-embeddings")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-texts", type=int, default=8)
    parser.add_argument("--retention-val-texts", type=int, default=8)
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
    parser.add_argument("--pad-to-max-length", action="store_true", help="pad batches to seq_len so throughput runs really use the requested token count")
    parser.add_argument("--pack-sequences", action="store_true", help="pack text into full seq_len blocks without an attention_mask so SDPA can use causal flash kernels")
    parser.add_argument("--loss-impl", choices=("hf", "chunked", "cce"), default="hf", help="causal LM loss implementation")
    parser.add_argument("--chunked-lm-loss-tokens", type=int, default=0, help="compute causal LM head/loss in token chunks to avoid full seq*vocab logits allocation")
    parser.add_argument("--skip-validation", action="store_true", help="skip initial/final validation for throughput-only runs")
    parser.add_argument("--keep-grads-after-step", action="store_true", help="leave p.grad populated after optimizer.step(); default consumes grads once projected")
    parser.add_argument("--log-norms", action="store_true", help="log grad/param norms and SumoTrack update norms; adds reduction overhead")
    parser.add_argument("--print-shape-summary", action="store_true", help="print HF safetensor shape metadata and exit")
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
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    if args.basis_refresh_interval <= 0:
        raise ValueError("basis_refresh_interval must be positive")
    if args.aurora_pp_iterations <= 0:
        raise ValueError("aurora_pp_iterations must be positive")
    if not 1 <= args.polar_ns_steps <= 5:
        raise ValueError("polar_ns_steps must be in [1, 5]")
    if args.chunked_lm_loss_tokens < 0:
        raise ValueError("chunked_lm_loss_tokens must be non-negative")
    if args.loss_impl == "chunked" and args.chunked_lm_loss_tokens <= 0:
        raise ValueError("--loss-impl chunked requires --chunked-lm-loss-tokens > 0")

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
        f"rank={args.rank} projection_side_policy={args.projection_side_policy} "
        f"basis_init={args.basis_init} basis_refresh_interval={args.basis_refresh_interval} "
        f"orthogonalization=aurora aurora_pp_iterations={args.aurora_pp_iterations} polar_ns_steps={args.polar_ns_steps} "
        f"activation_checkpointing={args.activation_checkpointing} torch_compile={args.torch_compile} attn_implementation={args.attn_implementation or 'default'} "
        f"pad_to_max_length={args.pad_to_max_length} pack_sequences={args.pack_sequences} loss_impl={args.loss_impl} chunked_lm_loss_tokens={args.chunked_lm_loss_tokens} "
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
