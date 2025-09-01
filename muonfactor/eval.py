# run_qwen06b_bs1_optuna.py
import os, math, optuna, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor
from hybrid_muon_adafactor_bs1 import HybridMuonAdaFactorBS1
import wandb

MODEL_ID = "Qwen/Qwen2.5-0.5B"  # or Qwen3-0.6B if available in your env
CTX = 8192
BS = 1
STEPS = 500
WARMUP = 50
LOG_EVERY = 25

def build_dataloader(tokenizer):
    ds = load_dataset("tiiuae/falcon-refinedweb", split="train[:1%]").shuffle(seed=42).select(range(2000))
    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=CTX, return_tensors=None)
    ds = ds.map(tok, remove_columns=ds.column_names, num_proc=4)
    def collate(batch):
        input_ids = [torch.tensor(x["input_ids"][:CTX]) for x in batch]
        # pack best-fit decreasing-ish
        input_ids = sorted(input_ids, key=lambda t: -t.numel())
        x = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attn = (x != tokenizer.pad_token_id).to(torch.long)
        return {"input_ids": x, "attention_mask": attn, "labels": x.clone()}
    return torch.utils.data.DataLoader(ds, batch_size=BS, collate_fn=collate)

def make_optimizer(name, model, trial=None):
    params = [p for n,p in model.named_parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-2, momentum=0.0)
    if name == "adamw":
        lr = trial.suggest_float("adamw_lr", 1e-4, 1e-3, log=True) if trial else 5e-4
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    if name == "adafactor":
        lr = trial.suggest_float("af_lr", 1e-4, 1e-3, log=True) if trial else 5e-4
        return Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False, warmup_init=False)
    if name == "muon":
        lr = trial.suggest_float("muon_lr", 1e-4, 1e-3, log=True) if trial else 5e-4
        # Muon for all 2D; others fall back to Muon skipping NS internally (PyTorch MoE warns on 1D)
        return torch.optim.Muon(model.parameters(), lr=lr, weight_decay=0.0, momentum=0.0, nesterov=False,
                                ns_steps=5, adjust_lr_fn="match_rms_adamw")
    if name == "hybrid":
        lr_hidden = trial.suggest_float("hy_lr_hidden", 2e-4, 1e-3, log=True) if trial else 5e-4
        lr_other_scale = trial.suggest_float("hy_lr_other_scale", 0.25, 0.5) if trial else 0.5
        htok_hidden = trial.suggest_categorical("hy_H_hidden", [1_000_000, 2_000_000, 4_000_000]) if trial else 2_000_000
        htok_other  = trial.suggest_categorical("hy_H_other",  [2_000_000, 4_000_000, 8_000_000]) if trial else 4_000_000
        return HybridMuonAdaFactorBS1(
            model.named_parameters(),
            lr_hidden=lr_hidden,
            lr_other_scale=lr_other_scale,
            half_life_tokens_hidden=htok_hidden,
            half_life_tokens_other=htok_other,
            tokens_per_step=CTX,
            ns_steps=5,
            clip_update_rms=1.0,
            wd_other=2e-3,
            include_embeddings_in_muon=False,
            stochastic_bf16=True,
            wandb_log=True,
            wandb_prefix=f"{name}",
        )
    raise ValueError(name)

def train_once(opt_name, trial=None):
    torch.cuda.empty_cache()
    HybridMuonAdaFactorBS1.reset_cuda_peak()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    ).cuda()
    model.gradient_checkpointing_enable()

    loader = build_dataloader(tokenizer)
    opt = make_optimizer(opt_name, model, trial)
    # unify scheduler
    total_steps = min(STEPS, len(loader))
    sched = get_linear_schedule_with_warmup(opt, WARMUP, total_steps)
    scaler = None  # bf16 no need

    model.train()
    it = iter(loader)
    losses = []
    for step in range(total_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if (step + 1) % LOG_EVERY == 0:
            mb = HybridMuonAdaFactorBS1.max_cuda_bytes()
            wandb.log({f"{opt_name}/loss": sum(losses[-LOG_EVERY:]) / LOG_EVERY,
                       f"{opt_name}/max_allocated": mb["max_allocated"],
                       f"{opt_name}/max_reserved": mb["max_reserved"]}, step=step+1)
    mb = HybridMuonAdaFactorBS1.max_cuda_bytes()
    return (sum(losses[-LOG_EVERY:]) / LOG_EVERY), mb["max_reserved"]

def objective(trial):
    opt_name = trial.suggest_categorical("optimizer", ["hybrid", "adamw", "adafactor", "muon", "sgd"])
    loss, vram = train_once(opt_name, trial)
    trial.set_user_attr("max_reserved_bytes", int(vram))
    return loss

if __name__ == "__main__":
    wandb.init(project="bs1-hybrid-muon-adafactor", config={"model": MODEL_ID, "ctx": CTX, "bs": BS})
    # Quick baseline runs without Optuna
    for name in ["sgd", "adamw", "adafactor", "muon", "hybrid"]:
        loss, vram = train_once(name, trial=None)
        print(f"{name}: loss={loss:.4f}, max_reserved={vram/1e9:.2f} GB")
        wandb.log({f"summary/{name}_loss": loss, f"summary/{name}_max_reserved": vram})

    # Optuna sweep focusing on hybrid
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10, timeout=3600)
    print("Best trial:", study.best_trial.params, "loss:", study.best_value,
          "vram:", study.best_trial.user_attrs.get("max_reserved_bytes", 0)/1e9, "GB")
    wandb.finish()
