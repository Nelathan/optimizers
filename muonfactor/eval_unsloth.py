import unsloth
import torch, wandb
from trl import SFTTrainer, SFTConfig, pack_dataset
from unsloth import FastLanguageModel, get_chat_template
from transformers import AutoTokenizer
from datasets import load_dataset
from .data_utils import get_sugarquill
from .hybrid_muon_adafactor_bs1 import HybridMuonAdaFactorBS1

# ------------------- CONFIG -------------------
MODEL_ID = "Qwen/Qwen3-0.6B"
CTX = 1024
BS = 1
LOG_EVERY = 20
# ----------------------------------------------

# 1. Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID,
    max_seq_length=1024 * 8,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=True,
    device_map="auto",
    attn_implementation="kernels-community/flash-attn",
)

CHAT_TEMPLATE = """
{%- for message in messages %}
    {%- if message['role'] == 'assistant' -%}
        {{ '<|im_start|>model\n' }}
        {%- generation -%}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- endif -%}
        {{ '<|im_end|>\n' }}
        {%- endgeneration -%}
    {%- else -%}
        {{ '<|im_start|>' + message['role'] + '\n' }}
        {%- if message['content'] is string -%}
            {{ message['content'] | trim }}
        {%- elif message['content'] is iterable -%}
            {%- for item in message['content'] -%}
                {%- if item['type'] == 'text' -%}
                    {{ item['text'] | trim }}
                {%- endif -%}
            {%- endfor -%}
        {%- else -%}
            {{ raise_exception("Invalid content type") }}
        {%- endif -%}
        {{ '<|im_end|>\n' }}
    {%- endif -%}
{%- endfor %}
{%- if add_generation_prompt -%}
    {{'<|im_start|>model\n' }}
{%- endif -%}"""

eos_token = "<|im_end|>"

tokenizer = get_chat_template(tokenizer, chat_template=(CHAT_TEMPLATE, eos_token))

# 2. Load and prepare dataset
raw_dataset = get_sugarquill(tokenizer, max_seq_length=CTX)
# packed_dataset = pack_dataset(raw_dataset, seq_length=CTX, strategy="bfd")
raw_dataset.set_format("torch")
split_dataset = raw_dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = pack_dataset(split_dataset["train"], seq_length=CTX, strategy="bfd")
test_dataset = split_dataset["test"]

# 3. Setup Trainer
args = SFTConfig(
    output_dir=f"outputs/{MODEL_ID}-unsloth",
    report_to="wandb",
    num_train_epochs=1,
    per_device_train_batch_size=BS,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=BS,
    weight_decay=0.01,
    learning_rate=1e-5,
    lr_scheduler_type="linear",  # a reasonable default
    max_grad_norm=1.0,
    warmup_ratio=0.03,
    logging_steps=LOG_EVERY,
    eval_strategy="steps",
    eval_steps=int(len(train_dataset) / (BS * 4)),  # eval 4 times per epoch
    save_total_limit=1,
)

opt = HybridMuonAdaFactorBS1(
    model.named_parameters(),
    lr_hidden=5e-4,
    lr_other_scale=0.5,
    half_life_tokens_hidden=2_000_000,
    half_life_tokens_other=4_000_000,
    tokens_per_step=CTX,
    ns_steps=5,
    clip_update_rms=1.0,
    wd_other=2e-3,
    include_embeddings_in_muon=False,
    stochastic_bf16=True,
    wandb_log=True,
    wandb_prefix="hybrid",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=args,
    optimizers=(opt, None),  # Pass the optimizer and scheduler (None for default)
)

# 4. Train
wandb.init(project="bs1-hybrid-muon-hard-test", config=vars(args))
trainer.train()
wandb.finish()

print("--- Training Complete ---")
