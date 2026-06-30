from __future__ import annotations

import argparse
import gc
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SumoTrack, optimizer_state_bytes_by_category
from sumotrack.projected_activation import OptimizerProjectedGradientSink, projected_activation_gated_mlp


DEFAULT_MODEL = "LiquidAI/LFM2.5-350M-Base"
DEFAULT_SOURCE_HF_DATASET = "HuggingFaceFW/finepdfs_50BT-dclm_30BT-fineweb_edu_20BT-shuffled"

ParamScope = Literal["full", "broad-no-embeddings", "matrices-no-embeddings"]
ProjectionSidePolicy = Literal["auto", "residual-facing"]
ProjectedActivationBackend = Literal["off", "lfm-mlp"]
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


def optimizer_update_norm(optimizer: torch.optim.Optimizer) -> float:
    diagnostics = getattr(optimizer, "last_step_diagnostics", None)
    if not diagnostics:
        return float("nan")
    return float(diagnostics.get("update_norm", float("nan")))


def scalar(value: float | int | torch.Tensor) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    return float(value.detach().float().cpu())


def mean_scalar(values: list[float | int | torch.Tensor]) -> float:
    finite = []
    for value in values:
        number = scalar(value)
        if number == number:
            finite.append(number)
    return sum(finite) / len(finite) if finite else float("nan")


def optimizer_basis_rotation(optimizer: torch.optim.Optimizer) -> float:
    diagnostics = getattr(optimizer, "last_step_diagnostics", None)
    if not diagnostics:
        return float("nan")
    return float(diagnostics.get("mean_basis_rotation_chordal", float("nan")))


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
    activation_projected_param_ids = activation_projected_param_ids or set()
    for name, param in named_params:
        role = transformer_matrix_role(name, param)
        side = "right" if id(param) in activation_projected_param_ids and param.ndim == 2 else storage_side_for_residual_axis(role, projection_side_policy)
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


def projected_activation_param_ids(model: torch.nn.Module, backend: ProjectedActivationBackend) -> set[int]:
    if backend == "off":
        return set()
    if backend != "lfm-mlp":  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown projected activation backend: {backend}")
    ids: set[int] = set()
    for module in lfm_mlp_modules(model):
        ids.add(id(module.w1.weight))
        ids.add(id(module.w2.weight))
        ids.add(id(module.w3.weight))
    return ids


def install_projected_activation_backend(model: torch.nn.Module, optimizer: SumoTrack, backend: ProjectedActivationBackend) -> int:
    if backend == "off":
        return 0
    if backend != "lfm-mlp":  # pragma: no cover - argparse constrains this
        raise ValueError(f"unknown projected activation backend: {backend}")
    sink = OptimizerProjectedGradientSink(optimizer)
    installed = 0
    for module in lfm_mlp_modules(model):
        original_forward = module.forward
        module.forward = _make_projected_activation_lfm_mlp_forward(module, optimizer, sink, original_forward)  # type: ignore[method-assign]
        installed += 1
    if installed == 0:
        raise RuntimeError("projected activation backend lfm-mlp did not find any LFM MLP modules with w1/w2/w3 Linear children")
    return installed


def _make_projected_activation_lfm_mlp_forward(
    module: torch.nn.Module,
    optimizer: SumoTrack,
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


def _projected_activation_lfm_mlp_bases(module: torch.nn.Module, optimizer: SumoTrack) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    params = (module.w1.weight, module.w3.weight, module.w2.weight)
    bases = []
    for param in params:
        group = _optimizer_group_for_param(optimizer, param)
        if group is None or _sumotrack_param_refresh_due(group, param):
            return None
        state = optimizer.state.get(param, {})
        basis = state.get("basis")
        if basis is None or not state.get("projection_side_is_right", False):
            return None
        if basis.ndim != 2 or basis.shape[1] != param.shape[1]:
            return None
        bases.append(basis)
    return bases[0], bases[1], bases[2]


def _optimizer_group_for_param(optimizer: torch.optim.Optimizer, param: torch.nn.Parameter) -> dict | None:
    for group in optimizer.param_groups:
        if any(param is candidate for candidate in group["params"]):
            return group
    return None


def _sumotrack_param_refresh_due(group: dict, param: torch.nn.Parameter) -> bool:
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
    collect_norms: bool,
    collect_basis: bool,
) -> dict[str, float | torch.Tensor]:
    if hasattr(optimizer, "diagnostics_enabled"):
        optimizer.diagnostics_enabled = collect_norms or collect_basis
    if hasattr(optimizer, "diagnostics_leverage_enabled"):
        optimizer.diagnostics_leverage_enabled = False
    if hasattr(optimizer, "diagnostics_basis_enabled"):
        optimizer.diagnostics_basis_enabled = collect_basis
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for offset in range(grad_accum_steps):
        batch = batches[(start_index + offset) % len(batches)]
        loss = cce_causal_lm_loss(model, batch)
        (loss / grad_accum_steps).backward()
        losses.append(loss.detach().float())
    grad_norm = gradient_norm(trainable) if collect_norms else float("nan")
    param_norm = parameter_norm(trainable) if collect_norms else float("nan")
    optimizer.step()
    update_norm = optimizer_update_norm(optimizer) if collect_norms else float("nan")
    basis_rotation_chordal = optimizer_basis_rotation(optimizer) if collect_basis else float("nan")
    param_norm_scalar = scalar(param_norm) if collect_norms else float("nan")
    update_to_param_ratio = update_norm / param_norm_scalar if param_norm_scalar > 0 else float("nan")
    return {
        "loss": torch.stack(losses).mean(),
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "update_norm": update_norm,
        "update_to_param_ratio": update_to_param_ratio,
        "basis_rotation_chordal": basis_rotation_chordal,
    }


def wandb_log(wandb_run: Any | None, data: Mapping[str, float | int | str], step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(data, step=step)


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
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    model, tokenizer = load_model_and_tokenizer(model_name, device, args.activation_checkpointing, args.attn_implementation)
    trainable_named, param_stats = select_trainable_named_params(model, args.param_scope)
    trainable = [param for _name, param in trainable_named]
    activation_projected_param_ids = projected_activation_param_ids(model, args.projected_activation_backend)
    sumotrack_param_groups, policy_stats = build_sumotrack_param_groups(
        trainable_named,
        args.rank,
        args.projection_side_policy,
        activation_projected_param_ids,
        args.basis_refresh_schedule,
    )
    train_batch_count = (args.warmup_steps + args.measure_steps) * args.grad_accum_steps
    train_batches = make_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len, train_batch_count, args.batching, "synth")
    val_batches = make_batches(tokenizer, val_texts, device, args.batch_size, args.seq_len, args.val_blocks, args.batching, "synth")
    retention_batches = make_batches(tokenizer, retention_texts, device, args.batch_size, args.seq_len, args.retention_val_blocks, args.batching, "source") if retention_texts else None

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
            compile_tensor_kernels=args.torch_compile,
        )
        projected_activation_modules = install_projected_activation_backend(model, optimizer, args.projected_activation_backend)
    elif optimizer_name in {"adamw", "torch_adamw"}:
        if args.projected_activation_backend != "off":
            raise RuntimeError("projected activation backend requires the sumotrack optimizer")
        optimizer = torch.optim.AdamW(trainable, lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=device.type == "cuda")
        projected_activation_modules = 0
    else:  # pragma: no cover
        raise ValueError(optimizer_name)

    initial_val = evaluate_loss(model, val_batches) if not args.skip_validation else float("nan")
    initial_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None and not args.skip_validation else float("nan")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    wandb_log(
        wandb_run,
        {
            f"{optimizer_name}/target_val_loss": initial_val,
            f"{optimizer_name}/source_val_loss": initial_retention_val,
        },
        step=0,
    )
    measured_steps = []
    loss_window = []

    for step in range(args.warmup_steps):
        batch_index = step * args.grad_accum_steps
        train_step(model, optimizer, trainable, train_batches, batch_index, args.grad_accum_steps, False, False)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for step in range(args.measure_steps):
        batch_index = (args.warmup_steps + step) * args.grad_accum_steps
        global_step = step + 1
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
            collect_norms,
            collect_basis,
        )
        measured_steps.append(step_result)
        loss_window.append(step_result["loss"])
        if should_log_train:
            train_loss = scalar(torch.stack(loss_window).mean())
            loss_window.clear()
            train_metrics = {
                f"{optimizer_name}/train_loss": train_loss,
                f"{optimizer_name}/grad_norm": scalar(step_result["grad_norm"]),
                f"{optimizer_name}/update_norm": scalar(step_result["update_norm"]),
                f"{optimizer_name}/update_to_param_ratio": scalar(step_result["update_to_param_ratio"]),
                f"{optimizer_name}/basis_rotation_chordal": scalar(step_result["basis_rotation_chordal"]),
            }
            wandb_log(
                wandb_run,
                train_metrics,
                step=global_step,
            )
        if should_eval and not args.skip_validation:
            eval_val = evaluate_loss(model, val_batches)
            eval_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None else float("nan")
            print(f"{optimizer_name}_step={global_step} target_val_loss={eval_val:.6f} source_val_loss={eval_retention_val:.6f}")
            wandb_log(
                wandb_run,
                {
                    f"{optimizer_name}/target_val_loss": eval_val,
                    f"{optimizer_name}/source_val_loss": eval_retention_val,
                },
                step=global_step,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    measured_elapsed = time.perf_counter() - start_time
    final_val = evaluate_loss(model, val_batches) if not args.skip_validation else float("nan")
    final_retention_val = evaluate_loss(model, retention_batches) if retention_batches is not None and not args.skip_validation else float("nan")
    if not args.skip_validation:
        print_final_eval_sample(model, tokenizer, val_texts, args)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    measured_losses = [step["loss"] for step in measured_steps]
    measured_grad_norms = [step["grad_norm"] for step in measured_steps]
    measured_param_norms = [step["param_norm"] for step in measured_steps]
    measured_update_norms = [step["update_norm"] for step in measured_steps]
    measured_update_to_param_ratios = [step["update_to_param_ratio"] for step in measured_steps]
    measured_basis_rotation_chordal = [step["basis_rotation_chordal"] for step in measured_steps]
    state_bytes = optimizer_state_bytes_by_category(optimizer)
    result = {
        "optimizer": optimizer_name,
        "projection_side_policy": args.projection_side_policy if optimizer_name == "sumotrack" else "n/a",
        "projected_activation_backend": args.projected_activation_backend if optimizer_name == "sumotrack" else "n/a",
        "projected_activation_modules": projected_activation_modules if optimizer_name == "sumotrack" else 0,
        "rank": args.rank if optimizer_name == "sumotrack" else 0,
        "effective_rank_min": policy_stats["effective_rank_min"] if optimizer_name == "sumotrack" else 0,
        "effective_rank_max": policy_stats["effective_rank_max"] if optimizer_name == "sumotrack" else 0,
        "side_policy_left_tensors": policy_stats["side_policy_left_tensors"] if optimizer_name == "sumotrack" else 0,
        "side_policy_right_tensors": policy_stats["side_policy_right_tensors"] if optimizer_name == "sumotrack" else 0,
        "side_policy_auto_tensors": policy_stats["side_policy_auto_tensors"] if optimizer_name == "sumotrack" else 0,
        "basis_init": args.basis_init if optimizer_name == "sumotrack" else "n/a",
        "basis_refresh_interval": args.basis_refresh_interval if optimizer_name == "sumotrack" else 0,
        "basis_refresh_schedule": args.basis_refresh_schedule if optimizer_name == "sumotrack" else "n/a",
        "aurora_pp_iterations": args.aurora_pp_iterations if optimizer_name == "sumotrack" else 0,
        "polar_ns_steps": args.polar_ns_steps if optimizer_name == "sumotrack" else 0,
        "consume_grad": (not args.keep_grads_after_step) if optimizer_name == "sumotrack" else False,
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
        "measured_tokens_per_second": (batch_tokens(train_batches[0]) * args.grad_accum_steps * args.measure_steps) / measured_elapsed,
        "matrix_state_bytes": state_bytes["matrix"],
        "fallback_state_bytes": state_bytes["fallback"],
        "state_bytes": state_bytes["total"],
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "initial_retention_val_loss": initial_retention_val,
        "final_retention_val_loss": final_retention_val,
        "retention_val_loss_delta": final_retention_val - initial_retention_val,
        "last_measured_train_loss": scalar(measured_losses[-1]),
        "mean_logged_grad_norm": mean_scalar(measured_grad_norms),
        "mean_logged_param_norm": mean_scalar(measured_param_norms),
        "mean_logged_update_norm": mean_scalar(measured_update_norms),
        "mean_logged_update_to_param_ratio": mean_scalar(measured_update_to_param_ratios),
        "mean_logged_basis_rotation_chordal": mean_scalar(measured_basis_rotation_chordal),
        "measured_elapsed_seconds": measured_elapsed,
        "measured_step_seconds": measured_elapsed / args.measure_steps,
        "peak_cuda_bytes": peak,
        "peak_cuda_reserved_bytes": peak_reserved,
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
    parser.add_argument("--target-hf-dataset", default="", help="optional Hugging Face target dataset; first parquet shard only")
    parser.add_argument("--target-format", choices=("auto", "synth", "profile_text", "text"), default="auto", help="target dataset row formatter; profile_text masks profile+divider and trains only on text")
    parser.add_argument("--target-val-offset", type=int, default=9000, help="row offset for validation when --target-hf-dataset is used")
    parser.add_argument("--retention-data-dir", default="", help="optional SYNTH-format source/retention parquet directory")
    parser.add_argument("--retention-hf-dataset", default="", help=f"optional Hugging Face source/retention dataset; first parquet shard only, e.g. {DEFAULT_SOURCE_HF_DATASET}")
    parser.add_argument("--optimizers", default="sumotrack", help="comma-separated: sumotrack,torch_adamw")
    parser.add_argument("--param-scope", choices=("full", "broad-no-embeddings", "matrices-no-embeddings"), default="broad-no-embeddings")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--val-blocks", type=int, default=8, help="number of validation batches/blocks to build")
    parser.add_argument("--retention-val-blocks", type=int, default=8, help="number of source validation batches/blocks to build")
    parser.add_argument("--batching", choices=("synth_right_padded_no_mask", "eos_packed_no_mask"), default="synth_right_padded_no_mask", help="batch construction policy; default is faithful SYNTH diagnostics; choose eos_packed_no_mask explicitly for throughput")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--projection-side-policy", choices=("auto", "residual-facing"), default="residual-facing")
    parser.add_argument(
        "--projected-activation-backend",
        choices=("off", "lfm-mlp"),
        default="off",
        help="experimental SumoTrack-only activation-projected backward backend; lfm-mlp wraps LFM w1/w3/w2 MLPs after basis warmup",
    )
    parser.add_argument("--basis-init", choices=("eigh", "random"), default="eigh")
    parser.add_argument("--sumotrack-lr", type=float, default=0.0025)
    parser.add_argument("--adamw-lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--grassmann-step-size", type=float, default=0.01)
    parser.add_argument("--basis-refresh-interval", type=int, default=100)
    parser.add_argument(
        "--basis-refresh-schedule",
        choices=("burst", "layer-staggered"),
        default="burst",
        help="basis refresh timing; burst preserves the default all-due refresh, layer-staggered offsets refresh by transformer layer index",
    )
    parser.add_argument("--aurora-pp-iterations", type=int, default=2)
    parser.add_argument("--polar-ns-steps", type=int, default=5)
    parser.add_argument("--activation-checkpointing", action="store_true", help="enable model gradient checkpointing before training")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="compile the model forward/backward and SumoTrack tensor kernels with torch.compile; the Python optimizer step remains eager",
    )
    parser.add_argument("--attn-implementation", default="sdpa", help="Transformers attention implementation; local default is sdpa until real flash kernels are available")
    parser.add_argument("--skip-validation", action="store_true", help="skip initial/final validation for throughput-only runs")
    parser.add_argument("--keep-grads-after-step", action="store_true", help="leave p.grad populated after optimizer.step(); default consumes grads once projected")
    parser.add_argument("--eval-every", type=int, default=0, help="periodically log target/source validation loss every N measured steps; 0 disables")
    parser.add_argument("--no-final-sample", dest="final_sample", action="store_false", help="disable final qualitative generation from a target eval prompt")
    parser.set_defaults(final_sample=True)
    parser.add_argument("--final-sample-row", type=int, default=0, help="target eval row index used for final qualitative generation")
    parser.add_argument("--final-sample-max-seq-len", type=int, default=4096, help="maximum prompt+generated token length for final qualitative generation")
    parser.add_argument("--final-sample-temperature", type=float, default=0.6)
    parser.add_argument("--final-sample-top-k", type=int, default=20)
    parser.add_argument("--final-sample-top-p", type=float, default=0.95)
    parser.add_argument("--allow-dirty-final-sample", action="store_true", help="allow printing target-HF samples for non-SYNTH formats; never enable for dirty/NSFW datasets")
    parser.add_argument("--wandb-run", default="", help="wandb run name; empty disables wandb")
    parser.add_argument("--wandb-entity", default="pink-marker")
    parser.add_argument("--wandb-project", default="sumotrack")
    parser.add_argument("--wandb-log-every", type=int, default=20, help="log train loss and core grad/update norms every N measured steps")
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
    if args.wandb_log_every < 0:
        raise ValueError("wandb_log_every must be non-negative")
    if args.retention_data_dir and args.retention_hf_dataset:
        raise ValueError("Use either --retention-data-dir or --retention-hf-dataset, not both")
    if args.rank <= 0:
        raise ValueError("rank must be positive")
    if args.basis_refresh_interval <= 0:
        raise ValueError("basis_refresh_interval must be positive")
    if args.aurora_pp_iterations <= 0:
        raise ValueError("aurora_pp_iterations must be positive")
    if not 1 <= args.polar_ns_steps <= 5:
        raise ValueError("polar_ns_steps must be in [1, 5]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model_name = args.model

    total_train_steps = args.warmup_steps + args.measure_steps
    train_blocks = max(total_train_steps * args.grad_accum_steps, 1)
    val_blocks = args.val_blocks
    target_source = args.data_dir
    if args.target_hf_dataset:
        train_texts, target_train_parquet = hf_first_parquet_texts(
            args.target_hf_dataset,
            limit=packed_text_limit(train_blocks, args.batch_size, args.seq_len),
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
        f"warmup_steps={args.warmup_steps} measure_steps={args.measure_steps} param_scope={args.param_scope} "
        f"rank={args.rank} projection_side_policy={args.projection_side_policy} "
        f"basis_init={args.basis_init} basis_refresh_interval={args.basis_refresh_interval} basis_refresh_schedule={args.basis_refresh_schedule} "
        f"orthogonalization=aurora aurora_pp_iterations={args.aurora_pp_iterations} polar_ns_steps={args.polar_ns_steps} "
        f"activation_checkpointing={args.activation_checkpointing} torch_compile={args.torch_compile} attn_implementation={args.attn_implementation or 'default'} "
        f"batching={args.batching} loss_impl=cce "
        f"skip_validation={args.skip_validation} eval_every={args.eval_every} "
        f"final_sample={args.final_sample} final_sample_max_seq_len={args.final_sample_max_seq_len} "
        f"final_sample_temperature={args.final_sample_temperature} final_sample_top_k={args.final_sample_top_k} final_sample_top_p={args.final_sample_top_p} "
        f"wandb_run={args.wandb_run or 'none'} wandb_entity={args.wandb_entity} consume_grad={not args.keep_grads_after_step}"
    )

    wandb_run = None
    if args.wandb_run:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_run, config=vars(args))

    try:
        for optimizer_name in [name.strip() for name in args.optimizers.split(",") if name.strip()]:
            if optimizer_name == "subspace":
                optimizer_name = "sumotrack"
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
