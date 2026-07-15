from __future__ import annotations

import argparse
import gc
import sys
import time
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from usuitrack import UsuiTrack, optimizer_state_bytes_by_category
from usuitrack.projected_activation import (
    OptimizerProjectedGradientSink,
    projected_activation_gated_mlp,
    projected_activation_linear,
    set_projected_activation_compile,
)


DEFAULT_MODEL = "LiquidAI/LFM2.5-350M-Base"
DEFAULT_SOURCE_HF_DATASET = "HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"

ParamScope = Literal["full", "broad-no-embeddings", "matrices-no-embeddings"]
ProjectionSidePolicy = Literal["auto", "residual-facing", "right"]
ProjectedActivationBackend = Literal["off", "lfm"]
BasisRefreshSchedule = Literal["burst", "layer-staggered"]
BatchingMode = Literal["eos_packed_no_mask", "synth_right_padded_no_mask"]
DatasetFormat = Literal["auto", "synth", "profile_text", "text"]
SYNTH_DIVIDER = "\n---\n"
PROFILE_TEXT_DIVIDER = "\n\n---\n\n"


@torch.no_grad()
def tensor_global_norm(tensors) -> torch.Tensor:
    norm_sq = None
    for tensor in tensors:
        if tensor is None:
            continue
        norm = tensor.detach().float().norm()
        norm_sq = norm.square() if norm_sq is None else norm_sq + norm.square()
    if norm_sq is None:
        return torch.tensor(float("nan"))
    return norm_sq.sqrt()


@torch.no_grad()
def parameter_norm(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return tensor_global_norm(params)


@torch.no_grad()
def gradient_norm(params: list[torch.nn.Parameter]) -> torch.Tensor:
    return tensor_global_norm([param.grad for param in params if param.grad is not None])


@torch.no_grad()
def gradient_norm_statistics(
    params: list[torch.nn.Parameter], clip_norm: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    norms = torch.stack([param.grad.detach().float().norm() for param in params if param.grad is not None])
    clipped_fraction = (norms > clip_norm).float().mean() if clip_norm > 0 else None
    return norms.norm(), norms.median(), clipped_fraction


def optimizer_update_norm(optimizer: torch.optim.Optimizer) -> float | None:
    diagnostics = getattr(optimizer, "last_step_diagnostics", None)
    if not diagnostics:
        return None
    value = diagnostics.get("update_norm")
    return float(value) if value is not None else None


def scalar(value: float | int | torch.Tensor) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(value.detach().float().cpu())


def scalar_or_none(value: float | int | torch.Tensor | None) -> float | None:
    return scalar(value) if value is not None else None


def last_finite_scalar(values: list[float | int | torch.Tensor | None]) -> float:
    """Last non-NaN value, e.g. the last real basis refresh event rather than
    whatever step happened to be logged last (which is NaN on non-refresh steps).
    """

    for value in reversed(values):
        if value is None:
            continue
        number = scalar(value)
        if number == number:
            return number
    return float("nan")


def optimizer_rotation_angle(optimizer: torch.optim.Optimizer) -> float | None:
    diagnostics = getattr(optimizer, "last_step_diagnostics", None)
    if not diagnostics:
        return None
    value = diagnostics.get("mean_rotation_angle")
    return float(value) if value is not None else None


def optimizer_diagnostic(optimizer: torch.optim.Optimizer, key: str) -> float | None:
    diagnostics = getattr(optimizer, "last_step_diagnostics", None)
    if not diagnostics:
        return None
    value = diagnostics.get(key)
    return float(value) if value is not None else None


def parquet_table_to_texts(table, dataset_format: DatasetFormat = "auto") -> list[str]:
    names = set(table.column_names)
    if dataset_format == "auto":
        if {"query", "synthetic_reasoning", "synthetic_answer"}.issubset(names):
            dataset_format = "synth"
        elif {"profile", "text"}.issubset(names):
            dataset_format = "profile_text"
        elif "text" in names:
            dataset_format = "text"

    if dataset_format == "text" and "text" in names:
        return [text for text in table.column("text").to_pylist() if text]
    if dataset_format == "profile_text" and {"profile", "text"}.issubset(names):
        rows = zip(table.column("profile").to_pylist(), table.column("text").to_pylist())
        texts = []
        for profile, text in rows:
            if text:
                profile_text = str(profile).strip() if profile else ""
                body = str(text).strip()
                if profile_text and body:
                    texts.append(f"{profile_text}{PROFILE_TEXT_DIVIDER}{body}")
                elif body:
                    texts.append(body)
        return texts
    if dataset_format == "synth" and {"query", "synthetic_reasoning", "synthetic_answer"}.issubset(names):
        rows = zip(
            table.column("query").to_pylist(),
            table.column("synthetic_reasoning").to_pylist(),
            table.column("synthetic_answer").to_pylist(),
        )
        texts = []
        for query, reasoning, answer in rows:
            parts = [part for part in (query, reasoning, answer) if part]
            if parts:
                texts.append("\n\n".join(parts))
        return texts
    raise ValueError(f"Unsupported parquet schema/format, format={dataset_format}, columns={table.column_names}")


def parquet_texts(parquet_path: Path, limit: int, dataset_format: DatasetFormat = "auto", offset: int = 0) -> list[str]:
    pf = pq.ParquetFile(parquet_path)
    texts = []
    seen = 0
    for row_group in range(pf.num_row_groups):
        for text in parquet_table_to_texts(pf.read_row_group(row_group), dataset_format):
            if seen < offset:
                seen += 1
                continue
            texts.append(text)
            if len(texts) >= limit:
                return texts
    return texts


def synth_texts(data_dir: Path, split: str, limit: int) -> list[str]:
    shards = sorted(data_dir.glob("synth_*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No SYNTH parquet shards found in {data_dir}")
    shard = shards[-1] if split == "val" else shards[0]
    return parquet_texts(shard, limit, "synth")


def hf_first_parquet_texts(repo_id: str, limit: int, dataset_format: DatasetFormat = "auto", offset: int = 0) -> tuple[list[str], str]:
    from huggingface_hub import hf_hub_download, list_repo_files

    parquets = sorted(path for path in list_repo_files(repo_id, repo_type="dataset") if path.endswith(".parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquet files found in Hugging Face dataset {repo_id}")
    first_parquet = parquets[0]
    local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=first_parquet)
    return parquet_texts(Path(local_path), limit, dataset_format, offset), first_parquet


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
        repair_lfm2_gradient_checkpointing(model)
    model.to(device)
    return model, tokenizer


def repair_lfm2_gradient_checkpointing(model: torch.nn.Module) -> int:
    """Wrap LFM2 decoder layers when the HF checkpointing flag is not used.

    The current Transformers LFM2 model advertises gradient checkpointing and
    accepts `gradient_checkpointing_enable()`, but `Lfm2Model.forward()` calls
    decoder layers directly instead of routing through `_gradient_checkpointing_func`.
    Keep this repair narrow: only patch the LFM2 base model and only when its
    checkpointing flag is enabled.
    """

    base_model = getattr(model, "model", model)
    if base_model.__class__.__name__ != "Lfm2Model":
        return 0
    if not getattr(base_model, "gradient_checkpointing", False):
        return 0
    layers = getattr(base_model, "layers", None)
    if layers is None:
        return 0

    wrapped = 0
    for layer in layers:
        if getattr(layer, "_usuitrack_checkpoint_wrapped", False):
            continue
        original_forward = layer.forward

        def checkpointed_forward(self, *args, __original_forward=original_forward, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(__original_forward, *args, use_reentrant=False, **kwargs)
            return __original_forward(*args, **kwargs)

        layer.forward = types.MethodType(checkpointed_forward, layer)
        layer._usuitrack_checkpoint_wrapped = True
        wrapped += 1

    base_model._usuitrack_checkpoint_wrapped_layers = wrapped
    return wrapped


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
    if policy == "right":
        return "right"
    if role in {"mlp_up_gate", "attention_qkv"}:
        return "right"
    if role in {"mlp_down", "attention_out"}:
        return "left"
    return "auto"


def build_usuitrack_param_groups(
    named_params: list[tuple[str, torch.nn.Parameter]],
    rank: int,
    projection_side_policy: ProjectionSidePolicy,
    activation_projected_param_ids: set[int] | None = None,
    basis_refresh_schedule: BasisRefreshSchedule = "burst",
) -> tuple[list[dict], dict[str, int]]:
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if basis_refresh_schedule not in {"burst", "layer-staggered"}:  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown basis refresh schedule: {basis_refresh_schedule}")
    policy_stats = {"effective_rank_min": 0, "effective_rank_max": 0, "side_policy_left_tensors": 0, "side_policy_right_tensors": 0, "side_policy_auto_tensors": 0}

    grouped: dict[str, list[torch.nn.Parameter]] = {}
    refresh_offsets_by_side: dict[str, dict[int, int]] = {}
    for name, param in named_params:
        role = transformer_matrix_role(name, param)
        side = storage_side_for_residual_axis(role, projection_side_policy)
        grouped.setdefault(side, []).append(param)
        if param.ndim == 2:
            if basis_refresh_schedule == "layer-staggered":
                refresh_offsets_by_side.setdefault(side, {})[id(param)] = transformer_layer_index(name)
            effective_rank = min(rank, *param.shape)
            policy_stats["effective_rank_min"] = effective_rank if not policy_stats["effective_rank_min"] else min(policy_stats["effective_rank_min"], effective_rank)
            policy_stats["effective_rank_max"] = max(policy_stats["effective_rank_max"], effective_rank)
            policy_stats[f"side_policy_{side}_tensors"] += 1

    groups = []
    for side, params in grouped.items():
        group = {"params": params, "rank": rank, "side": side}
        if basis_refresh_schedule == "layer-staggered" and refresh_offsets_by_side.get(side):
            group["basis_refresh_offsets"] = refresh_offsets_by_side[side]
        groups.append(group)
    return groups, policy_stats


def transformer_layer_index(name: str) -> int:
    parts = name.split(".")
    for layer_token in ("layers", "h", "blocks"):
        if layer_token in parts:
            index = parts.index(layer_token) + 1
            if index < len(parts):
                try:
                    return int(parts[index])
                except ValueError:
                    return 0
    return 0


def lfm_mlp_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    modules = []
    for _name, module in model.named_modules():
        w1 = getattr(module, "w1", None)
        w2 = getattr(module, "w2", None)
        w3 = getattr(module, "w3", None)
        if all(isinstance(linear, torch.nn.Linear) for linear in (w1, w2, w3)):
            modules.append(module)
    return modules


def lfm_projected_linear_modules(model: torch.nn.Module) -> list[torch.nn.Linear]:
    linears: list[torch.nn.Linear] = []
    for _name, module in model.named_modules():
        attention_linears = tuple(getattr(module, name, None) for name in ("q_proj", "k_proj", "v_proj", "out_proj"))
        if all(isinstance(linear, torch.nn.Linear) for linear in attention_linears):
            linears.extend(attention_linears)
            continue

        in_proj = getattr(module, "in_proj", None)
        out_proj = getattr(module, "out_proj", None)
        conv = getattr(module, "conv", None)
        if isinstance(in_proj, torch.nn.Linear) and isinstance(out_proj, torch.nn.Linear) and isinstance(conv, torch.nn.Conv1d):
            linears.extend((in_proj, out_proj))
    return linears


def projected_activation_param_ids(model: torch.nn.Module, backend: ProjectedActivationBackend) -> set[int]:
    if backend == "off":
        return set()
    if backend != "lfm":  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown projected activation backend: {backend}")
    ids: set[int] = set()
    for module in lfm_mlp_modules(model):
        ids.add(id(module.w1.weight))
        ids.add(id(module.w2.weight))
        ids.add(id(module.w3.weight))
    for linear in lfm_projected_linear_modules(model):
        ids.add(id(linear.weight))
    return ids


def install_projected_activation_backend(model: torch.nn.Module, optimizer: UsuiTrack, backend: ProjectedActivationBackend) -> int:
    if backend == "off":
        return 0
    if backend != "lfm":  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown projected activation backend: {backend}")
    sink = OptimizerProjectedGradientSink(optimizer)
    installed = 0
    for module in lfm_mlp_modules(model):
        original_forward = module.forward
        module.forward = _make_projected_activation_lfm_mlp_forward(module, optimizer, sink, original_forward)  # type: ignore[method-assign]
        installed += 1
    for linear in lfm_projected_linear_modules(model):
        original_forward = linear.forward
        linear.forward = _make_projected_activation_linear_forward(linear, optimizer, sink, original_forward)  # type: ignore[method-assign]
        installed += 1
    if installed == 0:
        raise RuntimeError("projected activation backend lfm did not find any LFM MLP, attention, or short-conv projection modules")
    return installed


def _make_projected_activation_linear_forward(
    linear: torch.nn.Linear,
    optimizer: UsuiTrack,
    sink: OptimizerProjectedGradientSink,
    fallback_forward: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def forward(x: torch.Tensor) -> torch.Tensor:
        projected = _projected_activation_linear_basis(linear, optimizer)
        if projected is None:
            return fallback_forward(x)
        basis, side = projected
        return projected_activation_linear(x, linear.weight, basis, sink, linear.weight, linear.bias, side=side)

    return forward


def _make_projected_activation_lfm_mlp_forward(
    module: torch.nn.Module,
    optimizer: UsuiTrack,
    sink: OptimizerProjectedGradientSink,
    fallback_forward: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def forward(x: torch.Tensor) -> torch.Tensor:
        bases = _projected_activation_lfm_mlp_bases(module, optimizer)
        if bases is None:
            return fallback_forward(x)
        gate_basis, up_basis, down_basis = bases
        return projected_activation_gated_mlp(
            x,
            module.w1.weight,
            module.w3.weight,
            module.w2.weight,
            gate_basis,
            up_basis,
            down_basis,
            sink,
            module.w1.weight,
            module.w3.weight,
            module.w2.weight,
        )

    return forward


def _projected_activation_lfm_mlp_bases(module: torch.nn.Module, optimizer: UsuiTrack) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    params = (module.w1.weight, module.w3.weight, module.w2.weight)
    bases = []
    for param in params:
        group = _optimizer_group_for_param(optimizer, param)
        if group is None or _usuitrack_param_refresh_due(group, param):
            return None
        state = optimizer.state.get(param, {})
        basis = state.get("basis")
        if basis is None or not state.get("projection_side_is_right", False):
            return None
        if basis.ndim != 2 or basis.shape[1] != param.shape[1]:
            return None
        bases.append(basis)
    return bases[0], bases[1], bases[2]


def _projected_activation_linear_basis(linear: torch.nn.Linear, optimizer: UsuiTrack) -> tuple[torch.Tensor, str] | None:
    group = _optimizer_group_for_param(optimizer, linear.weight)
    if group is None or _usuitrack_param_refresh_due(group, linear.weight):
        return None
    state = optimizer.state.get(linear.weight, {})
    basis = state.get("basis")
    if basis is None:
        return None
    side = "right" if state.get("projection_side_is_right", False) else "left"
    if side == "right" and (basis.ndim != 2 or basis.shape[1] != linear.weight.shape[1]):
        return None
    if side == "left" and (basis.ndim != 2 or basis.shape[0] != linear.weight.shape[0]):
        return None
    return basis, side


def _optimizer_group_for_param(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> dict | None:
    for group in optimizer.param_groups:
        if any(param is candidate for candidate in group["params"]):
            return group
    return None


def _usuitrack_param_refresh_due(group: dict, param: torch.nn.Parameter) -> bool:
    step = group.get("basis_refresh_step", 0)
    interval = group.get("basis_refresh_interval", 0)
    if interval <= 0 or step <= 0:
        return False
    offsets = group.get("basis_refresh_offsets")
    if offsets is None:
        return step % interval == 0
    if step < interval:
        return False
    return (step - offsets.get(id(param), 0)) % interval == 0


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


def synth_masked_examples(texts: list[str]) -> list[tuple[str, str]]:
    examples = []
    for text in texts:
        prefix, body = masked_prefix_and_body(text)
        if prefix is None or body is None:
            continue
        examples.append((f"{prefix}{body}", prefix))
    return examples


def masked_prefix_and_body(text: str) -> tuple[str | None, str | None]:
    if PROFILE_TEXT_DIVIDER in text:
        query, separator, body = text.partition(PROFILE_TEXT_DIVIDER)
        divider = PROFILE_TEXT_DIVIDER
    elif SYNTH_DIVIDER in text:
        query, separator, body = text.partition(SYNTH_DIVIDER)
        divider = SYNTH_DIVIDER
    else:
        query, separator, body = text.partition("\n\n")
        divider = SYNTH_DIVIDER
    if not separator or not query.strip() or not body.strip():
        return None, None
    return f"{query}{divider}", body


@torch.no_grad()
def print_final_eval_sample(model, tokenizer, val_texts: list[str], args) -> None:
    if not args.final_sample:
        return
    if args.target_hf_dataset and args.target_format != "synth" and not args.allow_dirty_final_sample:
        print("final_sample_skipped=target_hf_dataset_requires_explicit_allow_dirty_final_sample")
        return
    if not val_texts:
        print("final_sample_skipped=no_val_texts")
        return

    prompt = None
    for offset in range(len(val_texts)):
        candidate_prompt, _body = masked_prefix_and_body(val_texts[(args.final_sample_row + offset) % len(val_texts)])
        if candidate_prompt:
            prompt = candidate_prompt
            break
    if prompt is None:
        print("final_sample_skipped=no_masked_eval_prompt")
        return

    generator = getattr(model, "_orig_mod", model)
    was_training = generator.training
    generator.eval()
    try:
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if tokenizer.bos_token_id is not None:
            bos = torch.tensor([[tokenizer.bos_token_id]], dtype=prompt_ids.dtype)
            prompt_ids = torch.cat([bos, prompt_ids], dim=1)
        if prompt_ids.shape[1] >= args.final_sample_max_seq_len:
            prompt_ids = prompt_ids[:, : args.final_sample_max_seq_len - 1]
        prompt_ids = prompt_ids.to(next(generator.parameters()).device)
        max_new_tokens = max(1, args.final_sample_max_seq_len - int(prompt_ids.shape[1]))
        output_ids = generator.generate(
            input_ids=prompt_ids,
            do_sample=True,
            temperature=args.final_sample_temperature,
            top_k=args.final_sample_top_k,
            top_p=args.final_sample_top_p,
            repetition_penalty=args.final_sample_repetition_penalty,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )[0]
        prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
        completion_text = tokenizer.decode(output_ids[prompt_ids.shape[1] :], skip_special_tokens=True)
        print("final_sample_prompt_begin")
        print(prompt_text)
        print("final_sample_prompt_end")
        print("final_sample_completion_begin")
        print(completion_text)
        print("final_sample_completion_end")
    finally:
        if was_training:
            generator.train()


def full_lm_examples(texts: list[str]) -> list[tuple[str, str]]:
    return [(text, "") for text in texts if text]


def make_right_padded_batches(
    tokenizer,
    examples: list[tuple[str, str]],
    device: torch.device,
    batch_size: int,
    seq_len: int,
    min_batches: int,
):
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    if eos_token_id is None or pad_token_id is None:
        raise RuntimeError("tokenizer must define eos_token_id and pad_token_id or eos fallback for right-padded LM batches")

    rows = []
    needed_rows = max(min_batches, 1) * batch_size
    bos_token_id = tokenizer.bos_token_id
    for text, masked_prefix in examples:
        text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not text_ids:
            continue
        prefix_ids = tokenizer(masked_prefix, add_special_tokens=False)["input_ids"] if masked_prefix else []
        input_ids = ([bos_token_id] if bos_token_id is not None else []) + text_ids + [eos_token_id]
        labels = input_ids.copy()
        masked_until = (1 if bos_token_id is not None else 0) + min(len(prefix_ids), len(text_ids))
        labels[:masked_until] = [-100] * masked_until
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
        if not any(label != -100 for label in labels):
            continue
        padding = seq_len - len(input_ids)
        if padding > 0:
            input_ids.extend([pad_token_id] * padding)
            labels.extend([-100] * padding)
        rows.append((input_ids, labels))
        if len(rows) >= needed_rows:
            break

    if len(rows) < needed_rows:
        raise RuntimeError(f"not enough usable examples to build right-padded batches: need {needed_rows}, got {len(rows)}")

    input_tensor = torch.tensor([row[0] for row in rows], dtype=torch.long).view(max(min_batches, 1), batch_size, seq_len).to(device)
    label_tensor = torch.tensor([row[1] for row in rows], dtype=torch.long).view(max(min_batches, 1), batch_size, seq_len).to(device)
    return [{"input_ids": input_batch, "labels": label_batch} for input_batch, label_batch in zip(input_tensor, label_tensor)]


def make_batches(
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    seq_len: int,
    min_batches: int,
    batching: BatchingMode,
    stream: Literal["synth", "source"],
):
    if batching == "eos_packed_no_mask":
        return make_packed_batches(tokenizer, texts, device, batch_size, seq_len, min_batches)
    if batching == "synth_right_padded_no_mask":
        examples = synth_masked_examples(texts) if stream == "synth" else full_lm_examples(texts)
        return make_right_padded_batches(tokenizer, examples, device, batch_size, seq_len, min_batches)
    raise ValueError(f"unknown batching mode: {batching}")


def batch_tokens(batch: dict[str, torch.Tensor]) -> int:
    return int(batch["input_ids"].numel())


def batch_supervised_tokens(batch: dict[str, torch.Tensor]) -> int:
    return int((batch["labels"] != -100).sum().item())


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
    return linear_cross_entropy(outputs.last_hidden_state, lm_head.weight, batch.get("labels", batch["input_ids"]), ignore_index=-100, shift=True)


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
    grad_clip_norm: float,
    collect_norms: bool,
    collect_basis: bool,
) -> dict[str, float | torch.Tensor | None]:
    if hasattr(optimizer, "diagnostics_enabled"):
        optimizer.diagnostics_enabled = collect_norms or collect_basis
    if hasattr(optimizer, "diagnostics_leverage_enabled"):
        optimizer.diagnostics_leverage_enabled = False
    if hasattr(optimizer, "diagnostics_basis_enabled"):
        optimizer.diagnostics_basis_enabled = collect_basis
    if hasattr(optimizer, "diagnostics_aurora_health_enabled"):
        optimizer.diagnostics_aurora_health_enabled = collect_norms
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for offset in range(grad_accum_steps):
        batch = batches[(start_index + offset) % len(batches)]
        loss = cce_causal_lm_loss(model, batch)
        (loss / grad_accum_steps).backward()
        losses.append(loss.detach().float())
    if collect_norms:
        grad_norm, grad_norm_median_tensor, grad_clip_fraction = gradient_norm_statistics(
            trainable, grad_clip_norm
        )
    else:
        grad_norm = float("nan")
        grad_norm_median_tensor = float("nan")
        grad_clip_fraction = None
    param_norm = parameter_norm(trainable) if collect_norms else float("nan")
    optimizer.step()
    update_norm = optimizer_update_norm(optimizer) if collect_norms else float("nan")
    projected_grad_norm = optimizer_diagnostic(optimizer, "mean_projected_grad_norm") if collect_norms else float("nan")
    projected_grad_to_moment_ratio = optimizer_diagnostic(optimizer, "mean_projected_grad_to_moment_ratio") if collect_norms else float("nan")
    rotation_angle = optimizer_rotation_angle(optimizer) if collect_basis else float("nan")
    basis_target_angle_mass = optimizer_diagnostic(optimizer, "mean_basis_target_angle_mass") if collect_basis else float("nan")
    basis_step_angle_mass = rotation_angle
    basis_capture = optimizer_diagnostic(optimizer, "mean_basis_capture") if collect_norms else float("nan")
    aurora_alignment = optimizer_diagnostic(optimizer, "mean_aurora_alignment") if collect_norms else float("nan")
    moment_erank = optimizer_diagnostic(optimizer, "mean_aurora_erank") if collect_norms else float("nan")
    moment_erank_pct = optimizer_diagnostic(optimizer, "mean_aurora_erank_pct") if collect_norms else float("nan")
    param_norm_scalar = scalar(param_norm) if collect_norms else float("nan")
    update_to_param_ratio = update_norm / param_norm_scalar if update_norm is not None and param_norm_scalar > 0 else None
    return {
        "loss": torch.stack(losses).mean(),
        "grad_norm": grad_norm,
        "grad_norm_median_tensor": grad_norm_median_tensor,
        "grad_clip_fraction": grad_clip_fraction,
        "param_norm": param_norm,
        "update_norm": update_norm,
        "update_to_param_ratio": update_to_param_ratio,
        "projected_grad_norm": projected_grad_norm,
        "projected_grad_to_moment_ratio": projected_grad_to_moment_ratio,
        "basis_target_angle_mass": basis_target_angle_mass,
        "basis_step_angle_mass": basis_step_angle_mass,
        "basis_capture": basis_capture,
        "aurora_alignment": aurora_alignment,
        "moment_erank": moment_erank,
        "moment_erank_pct": moment_erank_pct,
    }


def wandb_log(wandb_run: Any | None, data: Mapping[str, float | int | str | None], step: int) -> None:
    if wandb_run is not None:
        # ``None`` means this run does not define the metric (for example source
        # validation without a retention corpus). Keep it absent: NaN is reserved
        # for a numeric result that actually became non-finite.
        wandb_run.log({key: value for key, value in data.items() if value is not None}, step=step)


def optimizer_base_lrs(optimizer: torch.optim.Optimizer) -> list[float]:
    return [float(group["lr"]) for group in optimizer.param_groups]


def apply_lr_warmup(optimizer: torch.optim.Optimizer, base_lrs: Sequence[float], optimizer_step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    scale = min(1.0, optimizer_step / warmup_steps)
    for group, base_lr in zip(optimizer.param_groups, base_lrs, strict=True):
        group["lr"] = base_lr * scale
    return scale


def should_compile_projected_activation(args, optimizer_name: str) -> bool:
    return args.torch_compile and optimizer_name == "usuitrack" and args.projected_activation_backend != "off"


def validate_projected_activation_contract(args) -> None:
    if args.projected_activation_backend != "off" and args.grassmann_aim == "oja":
        raise ValueError("--projected-activation-backend is incompatible with --grassmann-aim oja, which requires full matrix gradients every step")


def maybe_compile_training_model(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    """Compile the module that the CCE training loss actually calls.

    `cce_causal_lm_loss()` bypasses the CausalLM wrapper and calls the inner
    base model plus `lm_head` directly to avoid materializing full logits. If we
    only compile the outer wrapper, the hot transformer path remains eager.
    """

    if not enabled:
        return model
    base = getattr(model, "model", None)
    lm_head = getattr(model, "lm_head", None)
    if isinstance(base, torch.nn.Module) and isinstance(lm_head, torch.nn.Module):
        model.model = torch.compile(base)  # type: ignore[assignment]
        return model
    return torch.compile(model)


def run_optimizer(
    args,
    optimizer_name: str,
    model_name: str,
    train_texts: list[str],
    val_texts: list[str],
    retention_texts: list[str] | None,
    device: torch.device,
    wandb_run: Any | None = None,
) -> dict[str, float | int | str]:
    if optimizer_name == "usuitrack":
        validate_projected_activation_contract(args)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    model, tokenizer = load_model_and_tokenizer(model_name, device, args.activation_checkpointing, args.attn_implementation)
    trainable_named, param_stats = select_trainable_named_params(model, args.param_scope)
    trainable = [param for _name, param in trainable_named]
    activation_projected_param_ids = projected_activation_param_ids(model, args.projected_activation_backend)
    usuitrack_param_groups, policy_stats = build_usuitrack_param_groups(
        trainable_named,
        args.rank,
        args.projection_side_policy,
        activation_projected_param_ids,
        args.basis_refresh_schedule,
    )
    train_batch_count = (args.warmup_steps + args.max_steps) * args.grad_accum_steps
    train_batches = make_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len, train_batch_count, args.batching, "synth")
    val_batches = make_batches(tokenizer, val_texts, device, args.batch_size, args.seq_len, args.val_blocks, args.batching, "synth")
    retention_batches = make_batches(tokenizer, retention_texts, device, args.batch_size, args.seq_len, args.retention_val_blocks, args.batching, "source") if retention_texts else None

    set_projected_activation_compile(should_compile_projected_activation(args, optimizer_name))
    if optimizer_name == "usuitrack":
        optimizer = UsuiTrack(
            usuitrack_param_groups,
            lr=args.usuitrack_lr,
            beta=args.beta,
            basis_init=args.basis_init,
            moment_mode=args.moment_mode,
            adafactor_beta2=args.adafactor_beta2,
            grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
            grassmann_step_size=args.grassmann_step_size,
            grassmann_rotate_rank=args.grassmann_rotate_rank,
            grassmann_aim=args.grassmann_aim,
            oja_step_schedule=args.oja_step_schedule,
            basis_refresh_interval=args.basis_refresh_interval,
            aurora_pp_iterations=args.aurora_pp_iterations,
            polar_ns_steps=args.polar_ns_steps,
            projected_grad_clip_norm=args.projected_grad_clip_norm if args.projected_grad_clip_norm > 0 else None,
            projected_grad_clip_ratio=args.projected_grad_clip_ratio if args.projected_grad_clip_ratio > 0 else None,
            consume_grad=not args.keep_grads_after_step,
            compile_tensor_kernels=args.torch_compile,
        )
        projected_activation_modules = install_projected_activation_backend(model, optimizer, args.projected_activation_backend)
        model = maybe_compile_training_model(model, args.torch_compile)
    elif optimizer_name in {"adamw", "torch_adamw"}:
        if args.projected_activation_backend != "off":
            raise RuntimeError("projected activation backend requires the usuitrack optimizer")
        optimizer = torch.optim.AdamW(trainable, lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=device.type == "cuda")
        projected_activation_modules = 0
        model = maybe_compile_training_model(model, args.torch_compile)
    else:  # pragma: no cover
        raise ValueError(optimizer_name)
    base_lrs = optimizer_base_lrs(optimizer)

    initial_val = evaluate_loss(model, val_batches) if not args.skip_validation else float("nan")
    initial_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None and not args.skip_validation else float("nan")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    wandb_log(
        wandb_run,
        {
            "eval/target_loss": initial_val if not args.skip_validation else None,
            "eval/source_loss": initial_retention_val if retention_batches is not None and not args.skip_validation else None,
        },
        step=0,
    )
    measured_steps = []
    loss_window = []
    last_eval_val = float("nan")

    for step in range(args.warmup_steps):
        batch_index = step * args.grad_accum_steps
        apply_lr_warmup(optimizer, base_lrs, step + 1, args.lr_warmup_steps)
        train_step(model, optimizer, trainable, train_batches, batch_index, args.grad_accum_steps, args.grad_clip_norm, False, False)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    eval_elapsed = 0.0
    for step in range(args.max_steps):
        batch_index = (args.warmup_steps + step) * args.grad_accum_steps
        global_step = step + 1
        apply_lr_warmup(optimizer, base_lrs, args.warmup_steps + global_step, args.lr_warmup_steps)
        should_log_train = args.wandb_log_every > 0 and global_step % args.wandb_log_every == 0
        should_eval = args.eval_every > 0 and global_step % args.eval_every == 0
        collect_norms = should_log_train
        collect_basis = should_log_train
        step_result = train_step(
            model,
            optimizer,
            trainable,
            train_batches,
            batch_index,
            args.grad_accum_steps,
            args.grad_clip_norm,
            collect_norms,
            collect_basis,
        )
        measured_steps.append(step_result)
        loss_window.append(step_result["loss"])
        train_loss = None
        if should_log_train:
            train_loss = scalar(torch.stack(loss_window).mean())
            loss_window.clear()
            train_metrics = {
                "train/loss": train_loss,
                "train/grad_norm": scalar(step_result["grad_norm"]),
                "train/grad_norm_median_tensor": scalar(step_result["grad_norm_median_tensor"]),
                "train/grad_clip_fraction": scalar_or_none(step_result["grad_clip_fraction"]),
                "train/update_norm": scalar_or_none(step_result["update_norm"]),
                "train/update_to_param_ratio": scalar_or_none(step_result["update_to_param_ratio"]),
                "opt/projected_grad_norm": scalar_or_none(step_result["projected_grad_norm"]),
                "opt/projected_grad_to_moment_ratio": scalar_or_none(step_result["projected_grad_to_moment_ratio"]),
                "opt/basis_capture": scalar_or_none(step_result["basis_capture"]),
                "opt/aurora_alignment": scalar_or_none(step_result["aurora_alignment"]),
                "opt/moment_erank": scalar_or_none(step_result["moment_erank"]),
                "opt/moment_erank_pct": scalar_or_none(step_result["moment_erank_pct"]),
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            # Omit unavailable frame metrics rather than logging NaN. Oja moves
            # every step, while EIGH/tangent emit motion only at boundaries.
            for metric_name in ("basis_target_angle_mass", "basis_step_angle_mass"):
                metric = scalar_or_none(step_result[metric_name])
                if metric is not None and metric == metric:
                    train_metrics[f"opt/{metric_name}"] = metric
            wandb_log(
                wandb_run,
                train_metrics,
                step=global_step,
            )
        if should_eval and not args.skip_validation:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            eval_start = time.perf_counter()
            eval_val = evaluate_loss(model, val_batches)
            eval_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None else None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            eval_elapsed += time.perf_counter() - eval_start
            last_eval_val = eval_val
            wandb_log(
                wandb_run,
                {
                    "eval/target_loss": eval_val,
                    "eval/source_loss": eval_retention_val,
                },
                step=global_step,
            )
        if should_log_train or should_eval:
            avg_step_seconds = (time.perf_counter() - start_time - eval_elapsed) / global_step
            console_train_loss = train_loss if train_loss is not None else scalar(step_result["loss"])
            print(
                f"step={global_step} train_loss={console_train_loss:.6f} eval_loss={last_eval_val:.6f} "
                f"avg_step_seconds={avg_step_seconds:.4f}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_elapsed = time.perf_counter() - start_time
    training_elapsed = measured_elapsed - eval_elapsed
    training_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    training_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    final_val = evaluate_loss(model, val_batches) if not args.skip_validation else float("nan")
    final_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None and not args.skip_validation else float("nan")
    if not args.skip_validation:
        print_final_eval_sample(model, tokenizer, val_texts, args)
        wandb_log(
            wandb_run,
            {
                "eval/target_loss": final_val,
                "eval/source_loss": final_retention_val if retention_batches is not None else None,
            },
            step=global_step,
        )
    post_eval_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    post_eval_peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    measured_losses = [step["loss"] for step in measured_steps]
    measured_grad_norms = [step["grad_norm"] for step in measured_steps]
    measured_param_norms = [step["param_norm"] for step in measured_steps]
    measured_update_norms = [step["update_norm"] for step in measured_steps]
    measured_update_to_param_ratios = [step["update_to_param_ratio"] for step in measured_steps]
    measured_projected_grad_norms = [step["projected_grad_norm"] for step in measured_steps]
    measured_projected_grad_to_moment_ratios = [step["projected_grad_to_moment_ratio"] for step in measured_steps]
    measured_basis_target_angle_mass = [step["basis_target_angle_mass"] for step in measured_steps]
    measured_basis_step_angle_mass = [step["basis_step_angle_mass"] for step in measured_steps]
    measured_basis_capture = [step["basis_capture"] for step in measured_steps]
    measured_aurora_alignment = [step["aurora_alignment"] for step in measured_steps]
    measured_moment_erank = [step["moment_erank"] for step in measured_steps]
    measured_moment_erank_pct = [step["moment_erank_pct"] for step in measured_steps]
    state_bytes = optimizer_state_bytes_by_category(optimizer)
    result = {
        "optimizer": optimizer_name,
        "projection_side_policy": args.projection_side_policy if optimizer_name == "usuitrack" else "n/a",
        "projected_activation_backend": args.projected_activation_backend if optimizer_name == "usuitrack" else "n/a",
        "projected_activation_modules": projected_activation_modules if optimizer_name == "usuitrack" else 0,
        "rank": args.rank if optimizer_name == "usuitrack" else 0,
        "effective_rank_min": policy_stats["effective_rank_min"] if optimizer_name == "usuitrack" else 0,
        "effective_rank_max": policy_stats["effective_rank_max"] if optimizer_name == "usuitrack" else 0,
        "side_policy_left_tensors": policy_stats["side_policy_left_tensors"] if optimizer_name == "usuitrack" else 0,
        "side_policy_right_tensors": policy_stats["side_policy_right_tensors"] if optimizer_name == "usuitrack" else 0,
        "side_policy_auto_tensors": policy_stats["side_policy_auto_tensors"] if optimizer_name == "usuitrack" else 0,
        "basis_init": args.basis_init if optimizer_name == "usuitrack" else "n/a",
        "boundary_ablation_refresh_interval": args.basis_refresh_interval if optimizer_name == "usuitrack" else 0,
        "boundary_ablation_refresh_schedule": args.basis_refresh_schedule if optimizer_name == "usuitrack" else "n/a",
        "grassmann_aim": args.grassmann_aim if optimizer_name == "usuitrack" else "n/a",
        "oja_step_schedule": args.oja_step_schedule if optimizer_name == "usuitrack" else "n/a",
        "lr_warmup_steps": args.lr_warmup_steps,
        "aurora_pp_iterations": args.aurora_pp_iterations if optimizer_name == "usuitrack" else 0,
        "polar_ns_steps": args.polar_ns_steps if optimizer_name == "usuitrack" else 0,
        "projected_grad_clip_norm": args.projected_grad_clip_norm if optimizer_name == "usuitrack" else 0.0,
        "projected_grad_clip_ratio": args.projected_grad_clip_ratio if optimizer_name == "usuitrack" else 0.0,
        "consume_grad": (not args.keep_grads_after_step) if optimizer_name == "usuitrack" else False,
        "activation_checkpointing": args.activation_checkpointing,
        "torch_compile": args.torch_compile,
        "attn_implementation": getattr(getattr(model, "config", None), "_attn_implementation", "n/a"),
        "batching": args.batching,
        "loss_impl": "cce",
        "skip_validation": args.skip_validation,
        "actual_sequence_tokens": batch_tokens(train_batches[0]),
        "actual_supervised_tokens": batch_supervised_tokens(train_batches[0]),
        "tokens_per_optimizer_step": batch_tokens(train_batches[0]) * args.grad_accum_steps,
        "supervised_tokens_per_optimizer_step": batch_supervised_tokens(train_batches[0]) * args.grad_accum_steps,
        "measured_tokens_per_second": (batch_tokens(train_batches[0]) * args.grad_accum_steps * args.max_steps) / training_elapsed,
        "measured_tokens_per_second_with_eval": (batch_tokens(train_batches[0]) * args.grad_accum_steps * args.max_steps) / measured_elapsed,
        "matrix_state_bytes": state_bytes["matrix"],
        "fallback_state_bytes": state_bytes["fallback"],
        "state_bytes": state_bytes["total"],
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "initial_retention_val_loss": initial_retention_val,
        "final_retention_val_loss": final_retention_val,
        "retention_val_loss_delta": final_retention_val - initial_retention_val,
        "last_measured_train_loss": scalar(measured_losses[-1]),
        "last_logged_grad_norm": last_finite_scalar(measured_grad_norms),
        "last_logged_param_norm": last_finite_scalar(measured_param_norms),
        "last_logged_update_norm": last_finite_scalar(measured_update_norms),
        "last_logged_update_to_param_ratio": last_finite_scalar(measured_update_to_param_ratios),
        "last_logged_projected_grad_norm": last_finite_scalar(measured_projected_grad_norms),
        "last_logged_projected_grad_to_moment_ratio": last_finite_scalar(measured_projected_grad_to_moment_ratios),
        "last_logged_basis_target_angle_mass": last_finite_scalar(measured_basis_target_angle_mass),
        "last_logged_basis_step_angle_mass": last_finite_scalar(measured_basis_step_angle_mass),
        "last_logged_basis_capture": last_finite_scalar(measured_basis_capture),
        "last_logged_aurora_alignment": last_finite_scalar(measured_aurora_alignment),
        "last_logged_moment_erank": last_finite_scalar(measured_moment_erank),
        "last_logged_moment_erank_pct": last_finite_scalar(measured_moment_erank_pct),
        "measured_elapsed_seconds": measured_elapsed,
        "measured_train_elapsed_seconds": training_elapsed,
        "measured_eval_elapsed_seconds": eval_elapsed,
        "measured_step_seconds": training_elapsed / args.max_steps,
        "measured_step_seconds_with_eval": measured_elapsed / args.max_steps,
        "peak_cuda_bytes": training_peak,
        "peak_cuda_reserved_bytes": training_peak_reserved,
        "post_eval_peak_cuda_bytes": post_eval_peak,
        "post_eval_peak_cuda_reserved_bytes": post_eval_peak_reserved,
    }
    del optimizer, model, tokenizer, train_batches, val_batches, retention_batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Faithful pretrained-LLM SYNTH harness for UsuiTrack")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model name; default = {DEFAULT_MODEL}")
    parser.add_argument("--data-dir", default="/home/djg/.cache/nanochat/base_data_synth")
    parser.add_argument("--target-hf-dataset", default="", help="optional Hugging Face target dataset; first parquet shard only")
    parser.add_argument("--target-format", choices=("auto", "synth", "profile_text", "text"), default="auto", help="target dataset row formatter; profile_text masks profile+divider and trains only on text")
    parser.add_argument("--target-val-offset", type=int, default=9000, help="row offset for validation when --target-hf-dataset is used")
    parser.add_argument("--retention-data-dir", default="", help="optional SYNTH-format source/retention parquet directory")
    parser.add_argument("--retention-hf-dataset", default=DEFAULT_SOURCE_HF_DATASET, help=f"Hugging Face source/retention dataset; first parquet shard only; default = {DEFAULT_SOURCE_HF_DATASET}; pass an empty string to disable")
    parser.add_argument("--optimizers", default="usuitrack", help="comma-separated: usuitrack,torch_adamw")
    parser.add_argument("--param-scope", choices=("full", "broad-no-embeddings", "matrices-no-embeddings"), default="broad-no-embeddings")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000, help="ceiling on measured optimizer steps; default is the current 1k quality contract; clamped down with a warning if the dataset can't supply this many rows")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-blocks", type=int, default=8, help="number of validation batches/blocks to build")
    parser.add_argument("--retention-val-blocks", type=int, default=8, help="number of source validation batches/blocks to build")
    parser.add_argument("--batching", choices=("synth_right_padded_no_mask", "eos_packed_no_mask"), default="synth_right_padded_no_mask", help="batch construction policy; default is faithful SYNTH diagnostics; choose eos_packed_no_mask explicitly for throughput")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--projection-side-policy", choices=("auto", "residual-facing", "right"), default="residual-facing")
    parser.add_argument(
        "--projected-activation-backend",
        choices=("off", "lfm"),
        default="off",
        help="experimental UsuiTrack-only activation-projected backward backend; incompatible with Oja because it queues projected gradients instead of supplying the full matrix gradients Oja requires every step",
    )
    parser.add_argument("--basis-init", choices=("eigh", "random"), default="eigh")
    parser.add_argument("--usuitrack-lr", type=float, default=3e-4)
    parser.add_argument("--adamw-lr", type=float, default=2e-5)
    parser.add_argument("--lr-warmup-steps", type=int, default=50, help="linearly ramp optimizer learning rates over this many optimizer steps; 0 disables")
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument(
        "--moment-mode",
        choices=("ema", "adafactor_ema"),
        default="adafactor_ema",
        help="projected moment path: adafactor_ema (default) dampens the full gradient with a row/col factored second moment before basis tracking and projection, then feeds the result through the same first-moment EMA (--beta) as ema mode; ema is the plain first-moment path, kept as a comparator. A moment_mode ablation (none/second_moment/plain adafactor) found adafactor_ema beats plain ema on both target and source loss at matched LR/rank/steps; see commit db62ca2 for the losing arms' code",
    )
    parser.add_argument("--adafactor-beta2", type=float, default=0.99, help="EMA beta for --moment-mode adafactor_ema's row/col factored second-moment tracking")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="clip the RAW gradient PER TENSOR to this norm before adafactor/basis/projection; 0 disables. Protects adafactor's row/col second moment from blip batches before they can poison tracking or moment state. The released rank-128 lane uses 1.0; this is upstream of everything, unlike the moment-only projected-grad clip.")
    parser.add_argument("--grassmann-step-size", type=float, default=0.25, help="ablation-only boundary step: EIGH target fraction (1.0 = snap) or tangent multiplier. Oja ignores this control")
    parser.add_argument("--grassmann-rotate-rank", type=int, default=None, help="ablation-only number of planes rotated by EIGH/tangent boundary updates. None rotates all planes; Oja always rotates every tracked plane and ignores this control")
    parser.add_argument("--grassmann-aim", choices=("tangent", "eigh", "oja"), default="oja", help="basis update law. oja (default) updates the one live frame from every full gradient; eigh is the fixed-.25 boundary position-control ablation; tangent is the historical SubTrack ablation")
    parser.add_argument("--oja-step-schedule", choices=("fixed", "mature"), default="mature", help="Oja step law. mature (default) uses 1/2, 1/3, ... down to the 0.01 floor after EIGH initialization; fixed is the steady-0.01 ablation")
    parser.add_argument("--basis-refresh-interval", type=int, default=10, help="ablation-only cadence for EIGH/tangent boundary updates; Oja updates every gradient and ignores this interval")
    parser.add_argument(
        "--basis-refresh-schedule",
        choices=("burst", "layer-staggered"),
        default="burst",
        help="ablation-only EIGH/tangent boundary timing; Oja updates every gradient and ignores this schedule",
    )
    parser.add_argument("--aurora-pp-iterations", type=int, default=1)
    parser.add_argument("--polar-ns-steps", type=int, default=5)
    parser.add_argument("--projected-grad-clip-norm", type=float, default=0.0, help="per-matrix projected-gradient norm clip before the projected moment update; 0 disables. OFF now: raw-grad clipping (--grad-clip-norm) bounds the gradient upstream of adafactor/projection, which makes this downstream moment-only clip redundant (and it could not stop a blip from poisoning adafactor's second moment anyway -- that damage is upstream). Re-enable only if a specific moment-scale failure reappears.")
    parser.add_argument("--projected-grad-clip-ratio", type=float, default=0.0, help="per-matrix projected-gradient/moment norm ratio clip before the projected moment update; 0 disables. adafactor_ema's projected-grad norm is stable so this rail is unnecessary there; --moment-mode ema needs it re-enabled, e.g. 6.0")
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="model gradient checkpointing; default ON (bs16@seq1024 OOMs a 12GB card without it); --no-activation-checkpointing to disable")
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile the model forward/backward and UsuiTrack tensor kernels with torch.compile (default on); the Python optimizer step remains eager",
    )
    parser.add_argument("--attn-implementation", default="sdpa", help="Transformers attention implementation; local default is sdpa until real flash kernels are available")
    parser.add_argument("--skip-validation", action="store_true", help="skip initial/final validation for throughput-only runs")
    parser.add_argument("--keep-grads-after-step", action="store_true", help="leave p.grad populated after optimizer.step(); default consumes grads once projected")
    parser.add_argument("--eval-every", type=int, default=100, help="periodically log target/source validation loss every N measured steps; default 100 for the 1k quality contract; use 50 for a 200-step sensor; 0 disables")
    parser.add_argument("--no-final-sample", dest="final_sample", action="store_false", help="disable final qualitative generation from a target eval prompt")
    parser.set_defaults(final_sample=True)
    parser.add_argument("--final-sample-row", type=int, default=0, help="target eval row index used for final qualitative generation")
    parser.add_argument("--final-sample-max-seq-len", type=int, default=4096, help="maximum prompt+generated token length for final qualitative generation")
    parser.add_argument("--final-sample-temperature", type=float, default=0.6)
    parser.add_argument("--final-sample-top-k", type=int, default=20)
    parser.add_argument("--final-sample-top-p", type=float, default=0.95)
    parser.add_argument("--final-sample-repetition-penalty", type=float, default=1.1, help="mild repetition penalty for final qualitative generation; 1.0 disables")
    parser.add_argument("--allow-dirty-final-sample", action="store_true", help="allow printing target-HF samples for non-SYNTH formats; never enable for dirty/NSFW datasets")
    parser.add_argument("--wandb-run", default="", help="wandb run name; empty disables wandb")
    parser.add_argument("--wandb-entity", default="pink-marker")
    parser.add_argument("--wandb-project", default="usuitrack")
    parser.add_argument("--wandb-log-every", type=int, default=25, help="log train loss and optimizer diagnostics every N measured steps; basis_capture always measures the held frame before that step's basis movement")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_projected_activation_contract(args)

    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
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
    if args.eval_every < 0:
        raise ValueError("eval_every must be non-negative")
    if args.final_sample_row < 0:
        raise ValueError("final_sample_row must be non-negative")
    if args.final_sample_max_seq_len <= 1:
        raise ValueError("final_sample_max_seq_len must be greater than 1")
    if args.final_sample_temperature <= 0:
        raise ValueError("final_sample_temperature must be positive")
    if args.final_sample_top_k <= 0:
        raise ValueError("final_sample_top_k must be positive")
    if not 0 < args.final_sample_top_p <= 1:
        raise ValueError("final_sample_top_p must be in (0, 1]")
    if args.final_sample_repetition_penalty <= 0:
        raise ValueError("final_sample_repetition_penalty must be positive")
    if args.wandb_log_every < 0:
        raise ValueError("wandb_log_every must be non-negative")
    if args.retention_data_dir and args.retention_hf_dataset:
        raise ValueError("Use either --retention-data-dir or --retention-hf-dataset, not both")
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    if args.basis_refresh_interval <= 0:
        raise ValueError("basis_refresh_interval must be positive")
    if args.projected_grad_clip_norm < 0:
        raise ValueError("projected_grad_clip_norm must be non-negative")
    if args.projected_grad_clip_ratio < 0:
        raise ValueError("projected_grad_clip_ratio must be non-negative")
    if args.lr_warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")
    if not 0 <= args.adafactor_beta2 < 1:
        raise ValueError("adafactor_beta2 must be in [0, 1)")
    if args.moment_mode == "adafactor_ema" and args.projected_activation_backend != "off":
        raise ValueError("--moment-mode adafactor_ema dampens the full gradient before projection and is incompatible with --projected-activation-backend")
    if args.aurora_pp_iterations <= 0:
        raise ValueError("aurora_pp_iterations must be positive")
    if not 1 <= args.polar_ns_steps <= 5:
        raise ValueError("polar_ns_steps must be in [1, 5]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model_name = args.model

    total_train_steps = args.warmup_steps + args.max_steps
    train_blocks = max(total_train_steps * args.grad_accum_steps, 1)
    val_blocks = args.val_blocks
    target_source = args.data_dir
    if args.target_hf_dataset:
        train_limit = packed_text_limit(train_blocks, args.batch_size, args.seq_len)
        if train_limit > args.target_val_offset:
            while train_blocks > 1 and packed_text_limit(train_blocks, args.batch_size, args.seq_len) > args.target_val_offset:
                train_blocks -= 1
            train_limit = packed_text_limit(train_blocks, args.batch_size, args.seq_len)
            clamped_total_steps = max(train_blocks // args.grad_accum_steps, 1)
            clamped_max_steps = max(clamped_total_steps - args.warmup_steps, 1)
            print(
                f"warning: --max-steps {args.max_steps} needs {packed_text_limit(max(total_train_steps * args.grad_accum_steps, 1), args.batch_size, args.seq_len)} "
                f"training rows but only {args.target_val_offset} are available before --target-val-offset; "
                f"clamping to --max-steps {clamped_max_steps}",
                flush=True,
            )
            args.max_steps = clamped_max_steps
            total_train_steps = args.warmup_steps + args.max_steps
        train_texts, target_train_parquet = hf_first_parquet_texts(
            args.target_hf_dataset,
            limit=train_limit,
            dataset_format=args.target_format,
        )
        val_texts, target_val_parquet = hf_first_parquet_texts(
            args.target_hf_dataset,
            limit=packed_text_limit(val_blocks, args.batch_size, args.seq_len),
            dataset_format=args.target_format,
            offset=args.target_val_offset,
        )
        target_source = f"{args.target_hf_dataset}:{target_train_parquet}:format={args.target_format}:val_offset={args.target_val_offset}"
        if target_train_parquet != target_val_parquet:
            target_source += f":val_parquet={target_val_parquet}"
    else:
        train_texts = synth_texts(Path(args.data_dir), "train", limit=packed_text_limit(train_blocks, args.batch_size, args.seq_len))
        val_texts = synth_texts(Path(args.data_dir), "val", limit=packed_text_limit(val_blocks, args.batch_size, args.seq_len))
    retention_texts = None
    retention_source = "none"
    if args.retention_data_dir:
        retention_blocks = args.retention_val_blocks
        retention_texts = synth_texts(Path(args.retention_data_dir), "val", limit=packed_text_limit(retention_blocks, args.batch_size, args.seq_len))
        retention_source = args.retention_data_dir
    elif args.retention_hf_dataset:
        retention_blocks = args.retention_val_blocks
        retention_texts, first_parquet = hf_first_parquet_texts(args.retention_hf_dataset, limit=packed_text_limit(retention_blocks, args.batch_size, args.seq_len))
        retention_source = f"{args.retention_hf_dataset}:{first_parquet}"
    print(f"device={device}")
    print(f"model={model_name}")
    print(f"target_source={target_source}")
    print(f"retention_source={retention_source}")
    print(f"train_texts={len(train_texts)} val_texts={len(val_texts)} retention_texts={len(retention_texts) if retention_texts else 0}")
    print(
        f"seq_len={args.seq_len} batch_size={args.batch_size} grad_accum_steps={args.grad_accum_steps} "
        f"warmup_steps={args.warmup_steps} max_steps={args.max_steps} param_scope={args.param_scope} "
        f"rank={args.rank} projection_side_policy={args.projection_side_policy} "
        f"basis_init={args.basis_init} grassmann_aim={args.grassmann_aim} oja_step_schedule={args.oja_step_schedule} "
        f"boundary_ablation_refresh_interval={args.basis_refresh_interval} boundary_ablation_refresh_schedule={args.basis_refresh_schedule} "
        f"lr_warmup_steps={args.lr_warmup_steps} "
        f"orthogonalization=aurora aurora_pp_iterations={args.aurora_pp_iterations} polar_ns_steps={args.polar_ns_steps} "
        f"activation_checkpointing={args.activation_checkpointing} torch_compile={args.torch_compile} attn_implementation={args.attn_implementation or 'default'} "
        f"batching={args.batching} loss_impl=cce "
        f"skip_validation={args.skip_validation} eval_every={args.eval_every} "
        f"final_sample={args.final_sample} final_sample_max_seq_len={args.final_sample_max_seq_len} "
        f"final_sample_temperature={args.final_sample_temperature} final_sample_top_k={args.final_sample_top_k} final_sample_top_p={args.final_sample_top_p} "
        f"final_sample_repetition_penalty={args.final_sample_repetition_penalty} "
        f"wandb_run={args.wandb_run or 'none'} wandb_entity={args.wandb_entity} consume_grad={not args.keep_grads_after_step}"
    )

    wandb_run = None
    if args.wandb_run:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run, config=vars(args))

    try:
        for optimizer_name in [name.strip() for name in args.optimizers.split(",") if name.strip()]:
            result = run_optimizer(args, optimizer_name, model_name, train_texts, val_texts, retention_texts, device, wandb_run=wandb_run)
            prefix = optimizer_name
            for key, value in result.items():
                if key == "optimizer":
                    continue
                if isinstance(value, float):
                    print(f"{prefix}_{key}={value:.6f}")
                else:
                    print(f"{prefix}_{key}={value}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
