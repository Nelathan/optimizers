import torch
from tqdm.auto import tqdm

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    losses = []
    print("Running evaluation...")
    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        batch["labels"] = batch["input_ids"].clone()
        out = model(**batch)
        losses.append(out.loss.item())
    model.train()  # Set model back to train mode
    return sum(losses) / len(losses)
