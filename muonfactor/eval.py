import torch, wandb, random, re, ftfy
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from trl import pack_dataset
from tqdm.auto import trange, tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    Adafactor,
)
from .hybrid_muon_adafactor_bs1 import HybridMuonAdaFactorBS1

# ------------------- CONFIG -------------------
MODEL_ID = "Qwen/Qwen3-0.6B"
CTX = 4096
BS = 1
LOG_EVERY = 10
# ----------------------------------------------


from .data_utils import get_sugarquill
from .eval_utils import evaluate_model


def main_training_run():
    """
    Main experiment script for bs=1 full finetuning with HybridMuonAdaFactorBS1.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="kernels-community/flash-attn",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # 2. Load and prepare dataset using get_sugarquill and trl.packing
    raw_dataset = get_sugarquill(tokenizer, max_seq_length=CTX)
    packed_dataset = pack_dataset(raw_dataset, seq_length=CTX, strategy="bfd")

    packed_dataset.set_format("torch")

    split_dataset = packed_dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BS)

    # 3. Setup optimizer and scheduler
    total_steps = len(train_loader)
    warmup_steps = int(0.05 * total_steps)
    eval_steps = {int(p * total_steps) for p in [0.25, 0.5, 0.75]}

    # opt = HybridMuonAdaFactorBS1(
    #     model.named_parameters(),
    #     lr_hidden=5e-4,
    #     lr_other_scale=0.5,
    #     half_life_tokens_hidden=2_000_000,
    #     half_life_tokens_other=4_000_000,
    #     tokens_per_step=CTX,
    #     ns_steps=5,
    #     clip_update_rms=1.0,
    #     wd_other=2e-3,
    #     include_embeddings_in_muon=False,
    #     stochastic_bf16=True,
    #     wandb_log=True,
    #     wandb_prefix="hybrid",
    # )

    opt = Adafactor(
        model.parameters(),
        lr=1e-4,
        decay_rate=-0.9,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    print("--- Starting Training ---")
    print(f"Model: {MODEL_ID}")
    print(
        f"Dataset: sugarquill (packed) | Train Items: {len(train_dataset)} | Test Items: {len(test_dataset)}"
    )
    print(f"Total Steps: {total_steps} | Warmup Steps: {warmup_steps}")
    print(f"Context Length: {CTX} | Batch Size: {BS}")
    print("-------------------------")

    # 4. Training loop
    model.train()
    losses = []
    it = iter(train_loader)
    for step in trange(total_steps, desc="Training"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"].clone()

        out = model(**batch)
        loss = out.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        losses.append(loss.item())

        if (step + 1) % LOG_EVERY == 0:
            avg_loss = sum(losses[-LOG_EVERY:]) / LOG_EVERY
            max_allocated_gb = torch.cuda.max_memory_allocated() / 1e9
            max_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
            wandb.log(
                {
                    "train/loss": avg_loss,
                    "vram/max_allocated_gb": max_allocated_gb,
                    "vram/max_reserved_gb": max_reserved_gb,
                    "progress/step": step + 1,
                    "progress/percent": (step + 1) / total_steps * 100,
                },
                step=step + 1,
            )
            tqdm.write(f"Step {step+1}/{total_steps} | Loss: {avg_loss:.4f}")

        if (step + 1) in eval_steps:
            eval_loss = evaluate_model(model, test_loader)
            tqdm.write(f"--- Evaluation at step {step+1} ---")
            tqdm.write(f"Eval Loss: {eval_loss:.4f}")
            wandb.log({"train/eval_loss": eval_loss}, step=step + 1)
            model.train()

    # 5. Final logging
    final_train_loss = sum(losses[-LOG_EVERY:]) / LOG_EVERY
    final_vram = torch.cuda.max_memory_reserved() / 1e9
    print(
        f"Final Train Loss: {final_train_loss:.4f}, Max Reserved VRAM: {final_vram:.2f} GB"
    )
    wandb.log(
        {
            "train/loss": final_train_loss,
            "vram/max_reserved_gb": final_vram,
        }
    )

    # 6. Final Evaluation
    print("--- Running Final Evaluation ---")
    eval_loss = evaluate_model(model, test_loader)
    print(f"Final Evaluation Loss: {eval_loss:.4f}")
    wandb.log({"train/eval_loss": eval_loss})


if __name__ == "__main__":
    wandb.init(
        project="bs1-hybrid-muon-hard-test",
        config={
            "model": MODEL_ID,
            "ctx": CTX,
            "bs": BS,
            "dataset": "sugarquill_packed",
        },
    )
    main_training_run()
    wandb.finish()
    print("--- Training Complete ---")
