from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sumotrack import SubspaceMuon


PREFERRED_MODELS = (
    "LiquidAI/LFM2.5-1.2B-Base",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3-4B",
)


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
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
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{name.replace('/', '--')}"
            has_weights = any((cache_dir / "snapshots").glob("*/*.safetensors")) or any((cache_dir / "snapshots").glob("*/*.bin"))
            if not has_weights:
                raise FileNotFoundError(f"tokenizer/config cached but no local weight files under {cache_dir}")
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


def select_trainable_attention_params(model: torch.nn.Module, last_layers: int) -> list[torch.nn.Parameter]:
    attention_layer_ids = set()
    for name, _param in model.named_parameters():
        marker = ".layers."
        if marker in name and ".self_attn." in name:
            suffix = name.split(marker, 1)[1]
            attention_layer_ids.add(int(suffix.split(".", 1)[0]))
    if not attention_layer_ids:
        raise RuntimeError("Could not infer transformer layer IDs from model parameter names")
    selected_ids = sorted(attention_layer_ids)[-last_layers:]
    layer_prefixes = [f".layers.{layer_id}.self_attn." for layer_id in selected_ids]

    trainable = []
    for name, param in model.named_parameters():
        wants_param = any(prefix in name for prefix in layer_prefixes) and param.ndim == 2
        param.requires_grad_(wants_param)
        if wants_param:
            trainable.append(param)
    if not trainable:
        raise RuntimeError("No trainable attention matrix parameters selected")
    return trainable


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


def run_optimizer(args, optimizer_name: str, model_name: str, train_texts: list[str], val_texts: list[str], device: torch.device) -> dict[str, float | int | str]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    trainable = select_trainable_attention_params(model, args.last_layers)
    train_batches = make_batches(tokenizer, train_texts, device, args.batch_size, args.seq_len)
    val_batches = make_batches(tokenizer, val_texts, device, args.batch_size, args.seq_len)

    if optimizer_name == "subspace":
        optimizer = SubspaceMuon(
            trainable,
            lr=args.subspace_lr,
            rank=args.rank,
            beta=args.beta,
            subspace_update_method="grassmann",
            grassmann_step_size=args.grassmann_step_size,
            subspace_refresh_budget=args.subspace_refresh_budget,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=args.adamw_lr, betas=(0.9, 0.95), weight_decay=0.0, fused=device.type == "cuda")
    else:  # pragma: no cover
        raise ValueError(optimizer_name)

    initial_val = evaluate_loss(model, val_batches)
    losses = []
    start_time = time.perf_counter()
    for step in range(args.steps):
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    final_val = evaluate_loss(model, val_batches)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    result = {
        "optimizer": optimizer_name,
        "trainable_params": sum(p.numel() for p in trainable),
        "state_bytes": optimizer_state_bytes(optimizer),
        "initial_val_loss": initial_val,
        "final_val_loss": final_val,
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "elapsed_seconds": elapsed,
        "peak_cuda_bytes": peak,
    }
    del optimizer, model, tokenizer, train_batches, val_batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Short cached pretrained-LLM SYNTH smoke for SUMOTrack")
    parser.add_argument("--model", default="", help="HF model name; empty = prefer cached LiquidAI, Qwen3.5, Qwen3")
    parser.add_argument("--data-dir", default="/home/djg/.cache/nanochat/base_data_synth")
    parser.add_argument("--optimizers", default="subspace", help="comma-separated: subspace,adamw")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--last-layers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--subspace-lr", type=float, default=2e-5)
    parser.add_argument("--adamw-lr", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--grassmann-step-size", type=float, default=0.01)
    parser.add_argument("--subspace-refresh-budget", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = choose_model(args.model or None)
    train_texts = synth_texts(Path(args.data_dir), "train", limit=max(args.steps, 4) * args.batch_size)
    val_texts = synth_texts(Path(args.data_dir), "val", limit=2 * args.batch_size)
    print(f"device={device}")
    print(f"model={model_name}")
    print(f"data_dir={args.data_dir}")
    print(f"train_texts={len(train_texts)} val_texts={len(val_texts)}")
    print(f"seq_len={args.seq_len} batch_size={args.batch_size} steps={args.steps} last_layers={args.last_layers} rank={args.rank}")

    for optimizer_name in [name.strip() for name in args.optimizers.split(",") if name.strip()]:
        result = run_optimizer(args, optimizer_name, model_name, train_texts, val_texts, device)
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
