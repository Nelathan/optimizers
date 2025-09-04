# test.py
import torch
import torch.nn as nn
from .hybrid_muon_adafactor_bs1 import HybridMuonAdaFactorBS1

# Enable TF32 fast math on matmul and cuDNN (Ampere+)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Let PyTorch pick TF32 kernels where available for float32 matmuls
torch.set_float32_matmul_precision("high")  # "high" uses TF32, "highest" disables TF32

# Import the optimizer and helpers from your module
# from your_module import HybridMuonAdaFactorBS1, collect_attention_head_meta, is_embedding_param_name


class TinyAttn(nn.Module):
    def __init__(self, hidden=64, heads=8, kv_heads=4):
        super().__init__()
        self.num_heads = heads
        self.num_key_value_heads = kv_heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        # trivial linear to produce grads
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        o = self.o_proj(q)  # arbitrary usage
        return o + k.mean(dim=-1, keepdim=True) + v.mean(dim=-1, keepdim=True)


class TinyMLP(nn.Module):
    def __init__(self, hidden=64, ff=128):
        super().__init__()
        self.up_proj = nn.Linear(hidden, ff, bias=False)
        self.down_proj = nn.Linear(ff, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


class TinyModel(nn.Module):
    def __init__(self, vocab=1000, hidden=64, heads=8, kv_heads=4, ff=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.self_attn = TinyAttn(hidden, heads, kv_heads)
        self.mlp = TinyMLP(hidden, ff)
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, idx):
        x = self.embed_tokens(idx)
        x = self.self_attn(x)
        x = self.mlp(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyModel().to(device).bfloat16()
    model.train()

    named_params = [(n, p) for n, p in model.named_parameters()]
    # Build optimizer; pass model to collect attention head meta
    opt = HybridMuonAdaFactorBS1(
        named_params,
        model=model,
        lr_hidden=1e-5,
        lr_embed_1d=1e-6,
        stochastic_bf16=True,
        compile_loops=False,  # easier to debug; set True to test compile path
    )

    # Check partition sanity
    muonable = next(g for g in opt.param_groups if g["name"] == "muonable")["params"]
    fallback = next(g for g in opt.param_groups if g["name"] == "fallback_adamw")[
        "params"
    ]
    assert any(
        id(p) in opt.attn_head_meta for p in muonable
    ), "attention with head meta should be in muonable"
    assert any(
        "embed_tokens.weight" in opt.param_names[id(p)] for p in fallback
    ), "embeddings should be fallback"
    assert any(
        "lm_head.weight" in opt.param_names[id(p)] for p in fallback
    ), "lm_head should be fallback"

    # Dummy data and loss
    B, T = 4, 16
    idx = torch.randint(0, model.embed_tokens.num_embeddings, (B, T), device=device)
    target = torch.randint(
        0, model.lm_head.out_features, (B, T, model.lm_head.out_features), device=device
    ).bfloat16()

    out = model(idx)
    loss = (out - target).float().pow(2).mean()
    loss.backward()

    # Step
    opt.step()
    model.zero_grad(set_to_none=True)

    # Ensure shapes unchanged and params finite
    for n, p in model.named_parameters():
        assert torch.isfinite(p.float()).all(), f"param has NaN/Inf: {n}"
    print("test OK")


if __name__ == "__main__":
    main()
