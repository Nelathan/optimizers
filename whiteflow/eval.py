"""
WhiteFlow vs AdamW Comparison on SmolLM2-135M with FineWeb Data

Test setup for comparing WhiteFlow optimizer against fused AdamW
on continued pretraining scenario.
"""

from os import sep
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import time
import wandb
from whiteflow import WhiteFlow
import gc


def create_dataloader(tokenizer, num_samples=12000, context_length=1024, batch_size=8):
    """Create proper DataLoader using transformers utilities"""
    print(f"Loading wikitext-2-raw-v1 dataset (first {num_samples} samples)...")

    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{num_samples}]")

    # Filter out empty texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    # Tokenize the dataset
    def tokenize_function(examples):
        # Add BOS/EOS tokens to each text in the batch
        texts_with_tokens = [
            tokenizer.bos_token + text + tokenizer.eos_token
            for text in examples["text"]
        ]
        return tokenizer(
            texts_with_tokens,
            truncation=True,
            padding=False,  # Let DataCollator handle padding
            max_length=context_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8,  # Efficient for GPU
    )

    # Create DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=0     # Single process to avoid memory fragmentation
    )

    print(f"Created DataLoader with {len(tokenized_dataset)} samples, batch_size={batch_size}")
    print(f"Total batches: {len(dataloader)}")
    return dataloader


def get_layer_parameters(model):
    """Get only transformer layer parameters (skip embeddings and final layer norm)"""
    layer_params = []
    layer_names = []

    for name, param in model.named_parameters():
        # Only include transformer layers, skip embeddings and final norm
        if 'layers.' in name and param.requires_grad:
            layer_params.append(param)
            layer_names.append(name)

    print(f"Training on {len(layer_params)} layer parameters")
    return layer_params


def run_training_comparison():
    """Compare WhiteFlow vs AdamW on SmolLM2-135M with FineWeb"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model and tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M"
    print(f"Loading {model_name}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    layer_params = get_layer_parameters(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Layer parameters: {sum(p.numel() for p in layer_params) / 1e6:.1f}M")

    # Create DataLoader with original parameters
    train_dataloader = create_dataloader(tokenizer, num_samples=12000, context_length=1024, batch_size=8)

    # Training configurations
    configs = [
        {
            'name': 'WhiteFlow',
            'optimizer': WhiteFlow(layer_params, lr=1e-1, rank=256, orthogonalize=True, weight_decay=1e-2, gradient_clipping=1.0),
        },
        {
            'name': 'AdamW_Fused',
            'optimizer': torch.optim.AdamW(layer_params, lr=1e-4, fused=True, weight_decay=1e-2),
        },
    ]

    # Run training for each optimizer
    for config in configs:
        print(f"\n=== Training with {config['name']} ===")

        # Initialize wandb
        wandb.init(
            project="whiteflow-comparison",
            entity="pink-marker",  # Set the wandb entity here
            name=f"SmolLM2-135M-{config['name']}",
            config={
                "model": model_name,
                "optimizer": config['name'],
                "batch_size": 16,
                "context_length": 1024,
                "dataset": "wikitext-2-raw-v1",
                "num_samples": 20000,
                "gradient_checkpointing": True,
                "optimizer_config": str(config['optimizer'])
            }
        )

        # Reset model state and clear memory
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"Pre-training GPU memory: {torch.cuda.memory_reserved() / 1e9:.2f}GB reserved")

        optimizer = config['optimizer']
        print(f"Optimizer: {optimizer}")

        # Training loop
        step = 0
        total_batches = len(train_dataloader)
        print(f"Training for {total_batches} batches...")

        for batch in train_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Optimizer step (only on layer parameters)
            optimizer.step()
            optimizer.zero_grad()

            # Log to wandb
            if torch.cuda.is_available():
                allocated_gb = torch.cuda.memory_allocated()
            else:
                allocated_gb = 0

            # Check for NaN/Inf in loss
            loss_val = loss.item()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"💥 CRITICAL: NaN/Inf loss at step {step}!")
                print(f"   Loss tensor: {loss}")
                print(f"   Input IDs shape: {input_ids.shape}")
                print(f"   Attention mask shape: {attention_mask.shape}")

                # Check model parameters for NaN/Inf
                for name, param in model.named_parameters():
                    if param.requires_grad and ('layers.' in name):
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            print(f"   💥 NaN/Inf in parameter: {name}")

                raise RuntimeError(f"Training failed due to NaN/Inf loss at step {step}")

            wandb.log({
                "train/loss": loss_val,
                "system/GPU Memory Allocated (Bytes)": allocated_gb,
            })

            if step % 10 == 0:  # More frequent logging
                print(f"Step {step:4d}: Loss={loss_val:.6f}")

                # Log gradient norms for first few steps
                if step <= 50:
                    total_grad_norm = 0.0
                    param_count = 0
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None and 'layers.' in name:
                            grad_norm = torch.norm(param.grad).item()
                            total_grad_norm += grad_norm
                            param_count += 1
                            if param_count <= 3:  # Log first 3 layer gradients
                                print(f"   {name[:50]}: grad_norm={grad_norm:.6f}")

                    if param_count > 0:
                        avg_grad_norm = total_grad_norm / param_count
                        print(f"   Average grad norm: {avg_grad_norm:.6f}")

            if step % 50 == 0:
                progress = (step / total_batches) * 100
                if torch.cuda.is_available():
                    print(f"Step {step:4d}/{total_batches}: Loss={loss.item():.4f}, GPU={(allocated_gb / 1e9):.2f}GB alloc")

                    # Show memory breakdown during training
                    if step == 100:
                        model_params = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
                        optimizer_states = sum(p.numel() * 8 for p in layer_params) / 1e9  # AdamW: 8 bytes per param
                        gradients = sum(p.numel() * p.element_size() for p in layer_params) / 1e9
                        kv_cache_est = (8 * 1024 * 1024 * 2) / 1e9  # batch * seq * bf16 (rough estimate)

                        print(f"  Memory breakdown at step {step}:")
                        print(f"    Model parameters: {model_params:.2f}GB")
                        print(f"    Optimizer states: {optimizer_states:.2f}GB")
                        print(f"    Gradients: {gradients:.2f}GB")
                        print(f"    KV cache (est): {kv_cache_est:.2f}GB")
                else:
                    print(f"Step {step:4d}/{total_batches}: Loss={loss.item():.4f}, Progress={progress:.1f}%")

            step += 1

        wandb.finish()
        print(f"Completed training with {config['name']}")

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    print("WhiteFlow Optimizer Test Suite")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU (will be slow)")
    else:
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem, total_mem_check = torch.cuda.mem_get_info()
        free_mem_gb = free_mem / 1e9
        used_mem_gb = (total_mem_check - free_mem) / 1e9
        print(f"GPU Memory: {total_mem:.1f}GB total")
        print(f"Memory usage: {used_mem_gb:.2f}GB used, {free_mem_gb:.2f}GB free ({used_mem_gb/total_mem*100:.1f}% used)")

    try:
        run_training_comparison()
        print("\n✅ Training comparison completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
