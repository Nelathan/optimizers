import math
from typing import Dict, Iterable, Tuple, Optional, Any, Callable
import torch
from torch.optim import Optimizer


def _adjust_lr(
    lr: float, adjust_lr_fn: Optional[str], param_shape: torch.Size
) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


@torch.no_grad()
def to_bf16_stochastic_in_graph(x32: torch.Tensor) -> torch.Tensor:
    """
    Compile-friendly stochastic rounding fp32->bf16.
    Keeps everything as pure tensor ops so Inductor can fuse.
    """
    x32 = x32.to(torch.float32).contiguous()
    xi = x32.view(torch.int32)
    rnd = torch.randint(0, 1 << 16, xi.shape, device=xi.device, dtype=torch.int32)
    mask = torch.tensor(-65536, device=xi.device, dtype=torch.int32)  # 0xFFFF0000
    yi = (xi + rnd) & mask
    y32 = yi.view(torch.float32)
    return y32.to(torch.bfloat16)


def is_2d_weight(p: torch.nn.Parameter) -> bool:
    return p.ndim == 2


def is_embedding_param_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("embed", "embedding", "lm_head", "tok_embeddings"))


def is_stacked_attention_weight_name(name: str) -> bool:
    n = name.lower()
    # conservative: route classic attention projections and fused qkv to AdamW
    return any(
        k in n
        for k in ("qkv", "in_proj", "out_proj", "o_proj", "q_proj", "k_proj", "v_proj")
    )


import torch.nn as nn


# Minimal predicate: attention modules usually expose these attributes
def _is_attention_module(mod: nn.Module) -> bool:
    has_proj = all(hasattr(mod, n) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))
    has_heads = hasattr(mod, "num_heads") or hasattr(mod, "num_attention_heads")
    return has_proj and has_heads


def _get_attr_any(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def collect_attention_head_meta(model: nn.Module) -> Dict[int, Dict[str, int]]:
    """
    Returns: dict[id(param)] -> {
        'proj': one of {'q','k','v','o'},
        'n_heads': int,          # n_heads for q/o, n_kv_heads for k/v
        'head_dim': int,         # per-head dimension
        'in_features': int,      # linear in_features
        'out_features': int,     # linear out_features (== n_heads*head_dim)
    }
    Works for LLaMA/Mistral/Gemma/Qwen classes in HF.
    """
    meta: Dict[int, Dict[str, int]] = {}
    for mod in model.modules():
        if not _is_attention_module(mod):
            continue

        # Try to read standard attributes across models
        hidden_size = _get_attr_any(mod, ("hidden_size",), None)
        num_heads = _get_attr_any(mod, ("num_heads", "num_attention_heads"), None)
        num_kv_heads = _get_attr_any(
            mod, ("num_key_value_heads", "num_kv_heads"), num_heads
        )
        head_dim = _get_attr_any(
            mod,
            ("head_dim", "attention_head_size"),
            hidden_size // num_heads if (hidden_size and num_heads) else None,
        )

        if not (num_heads and head_dim):
            # Fallback: infer from q_proj weight
            if hasattr(mod, "q_proj") and isinstance(mod.q_proj, nn.Linear):
                out_f, in_f = mod.q_proj.weight.shape
                # assume out_f = num_heads * head_dim
                # pick a plausible divisor
                for h in (num_heads or [],):
                    if h and out_f % h == 0:
                        head_dim = out_f // h
                        break
                if hidden_size is None:
                    hidden_size = in_f
                if num_heads is None:
                    # last resort: try gcd with hidden_size
                    import math

                    g = math.gcd(out_f, hidden_size)
                    num_heads = out_f // g
                    head_dim = g

        # Register q/k/v/o weights
        for proj_name, proj_tag, hcount in (
            ("q_proj", "q", num_heads),
            ("k_proj", "k", num_kv_heads),
            ("v_proj", "v", num_kv_heads),
            ("o_proj", "o", num_heads),
        ):
            if not hasattr(mod, proj_name):
                continue
            lin = getattr(mod, proj_name)
            if not isinstance(lin, nn.Linear):
                continue
            W = lin.weight
            out_f, in_f = W.shape
            nh = int(hcount)
            hd = int(head_dim)
            # Sanity: out_f should equal nh*hd for q/k/v; for o, in_f should equal nh*hd
            if proj_tag in ("q", "k", "v"):
                if out_f != nh * hd:
                    # skip if shape doesn't match expectation
                    continue
            else:  # 'o'
                if in_f != nh * hd:
                    continue
            meta[id(W)] = {
                "proj": proj_tag,
                "n_heads": nh,
                "head_dim": hd,
                "in_features": int(in_f),
                "out_features": int(out_f),
            }
    return meta


def view_heads(W: torch.Tensor, meta: Dict[str, int]):
    nh, hd, in_f, out_f, proj = (
        meta["n_heads"],
        meta["head_dim"],
        meta["in_features"],
        meta["out_features"],
        meta["proj"],
    )
    if proj in ("q", "k", "v"):
        # [nh, hd, in_f], per-head mat = [:, :, :]
        return W.view(nh, hd, in_f), ("row_dim", 1), ("col_dim", 2)
    else:  # 'o'
        # [out_f, nh, hd], per-head mat across columns
        return W.view(out_f, nh, hd), ("row_dim", 0), ("col_dim", 2)


# -------------------- Beta2 schedule --------------------


class TokenHalfLife:
    """
    Maintains beta2 from token half-life. beta2 = exp(-ln2 / H_steps).
    Supports linear ramp over warmup_steps and upper cap (<= 0.9999).
    """

    def __init__(
        self,
        half_life_tokens: int,
        tokens_per_step: int,
        warmup_steps: int = 256,
        cap: float = 0.9999,
    ):
        self.base_h_tokens = max(int(half_life_tokens), 1)
        self.tokens_per_step = max(int(tokens_per_step), 1)
        self.warmup_steps = warmup_steps
        self.cap = min(cap, 0.9999)

    def beta2_at(self, step: int) -> float:
        H_steps = max(self.base_h_tokens // self.tokens_per_step, 1)
        target = math.exp(-math.log(2.0) / H_steps)
        if step < self.warmup_steps:
            start = max(target - 0.01, 0.90)
            t = step / max(self.warmup_steps, 1)
            val = start + (target - start) * t
        else:
            val = target
        return min(val, self.cap)


class AdamWMoments:
    def __init__(self, beta1: float = 0.9, eps: float = 1e-8):
        self.beta1 = float(beta1)
        self.eps = float(eps)
        self.state: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = id(p)
        if sid not in self.state:
            self.state[sid] = (
                torch.zeros_like(p, dtype=torch.float32),
                torch.zeros_like(p, dtype=torch.float32),
            )
        return self.state[sid]


@torch.no_grad()
def _zeropower_via_newtonschulz_batched(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
) -> torch.Tensor:
    """
    Batched Newton–Schulz 'zeroth power' update.
    Accepts shape [..., M, N] or [M, N]. Uses transpose(-1, -2) and batched matmul.
    """
    if ns_steps >= 100:
        raise ValueError("Number of steps must be < 100")
    if grad.ndim < 2:
        raise ValueError("grad must be at least 2D")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must have 3 values")

    a, b, c = ns_coefficients
    X = grad.float()
    transposed = False

    M, N = X.shape[-2], X.shape[-1]
    if M > N:
        X = X.transpose(-1, -2)
        M, N = N, M
        transposed = True

    # spectral norm <= 1 (use per-matrix Frobenius norm as in your reference)
    # norm over last two dims
    norms = X.square().sum(dim=(-2, -1)).sqrt().clamp_min(eps)  # shape [...]
    X = X / norms[..., None, None]

    for _ in range(ns_steps):
        G = X @ X.transpose(-1, -2)  # [..., M, M]
        GG = G @ G  # [..., M, M]
        G_update = b * G + c * GG  # [..., M, M]
        X = a * X + G_update @ X  # [..., M, N]

    if transposed:
        X = X.transpose(-1, -2)
    return X


# -------------------- AdaFactor EMA (compile-friendly) --------------------


class FactoredEMA:
    def __init__(self, beta2_sched: Callable[[int], float], eps2: float = 1e-30):
        self.beta2_sched = beta2_sched
        self.eps2 = eps2
        # tensor-wise: id(p) -> (vr[R], vc[C], boot[1])
        self.state: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        # head-wise: id(p) -> (vr[H,R], vc[H,C], boot[1])
        self.state_head: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def _get(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sid = id(p)
        if sid not in self.state:
            r, c = p.shape
            dev = p.device
            self.state[sid] = (
                torch.zeros(r, dtype=torch.float32, device=dev),
                torch.zeros(c, dtype=torch.float32, device=dev),
                torch.zeros(1, dtype=torch.uint8, device=dev),  # boot=0
            )
        return self.state[sid]

    def _get_headwise(self, p: torch.Tensor, nh: int, rows: int, cols: int):
        sid = id(p)
        if sid not in self.state_head:
            dev = p.device
            self.state_head[sid] = (
                torch.zeros((nh, rows), dtype=torch.float32, device=dev),  # vr[H,R]
                torch.zeros((nh, cols), dtype=torch.float32, device=dev),  # vc[H,C]
                torch.zeros(1, dtype=torch.uint8, device=dev),  # boot=0
            )
        else:
            vr, vc, boot = self.state_head[sid]
            # resize on shape change (rare)
            if vr.shape != (nh, rows) or vc.shape != (nh, cols):
                dev = p.device
                vr = torch.zeros((nh, rows), dtype=torch.float32, device=dev)
                vc = torch.zeros((nh, cols), dtype=torch.float32, device=dev)
                boot = torch.zeros(1, dtype=torch.uint8, device=dev)
                self.state_head[sid] = (vr, vc, boot)
        return self.state_head[sid]

    @torch.no_grad()
    def denom_inv(self, p: torch.Tensor, g32: torch.Tensor, step: int) -> torch.Tensor:
        vr, vc, boot = self._get(p)
        beta2 = self.beta2_sched(step)
        beta2_t = torch.tensor(beta2, dtype=vr.dtype, device=vr.device)
        one_m = 1.0 - beta2_t
        boot_b = boot.to(torch.bool)

        g2 = g32 * g32  # [R,C]
        row_mean = g2.mean(dim=1)  # [R]
        col_mean = g2.mean(dim=0)  # [C]

        vr_new = torch.where(boot_b, beta2_t * vr + one_m * row_mean, row_mean)
        vc_new = torch.where(boot_b, beta2_t * vc + one_m * col_mean, col_mean)

        vr.copy_(vr_new)
        vc.copy_(vc_new)
        boot.fill_(True)

        vr_eps = vr + self.eps2
        vc_eps = vc + self.eps2
        r = vr_eps.rsqrt()
        c = vc_eps.rsqrt()
        scale = vr_eps.mean().rsqrt()
        return torch.outer(r, c).mul_(scale)

    @torch.no_grad()
    def denom_inv_headwise(
        self,
        p: torch.Tensor,
        g_head: torch.Tensor,  # [H, R, C]
        step: int,
    ) -> torch.Tensor:
        nh, rows, cols = g_head.shape
        vr, vc, boot = self._get_headwise(p, nh, rows, cols)
        beta2 = self.beta2_sched(step)
        beta2_t = torch.tensor(beta2, dtype=vr.dtype, device=vr.device)
        one_m = 1.0 - beta2_t
        boot_b = boot.to(torch.bool)

        g2 = g_head * g_head  # [H,R,C]
        row_mean = g2.mean(dim=2)  # [H,R]
        col_mean = g2.mean(dim=1)  # [H,C]

        vr_new = torch.where(boot_b, beta2_t * vr + one_m * row_mean, row_mean)
        vc_new = torch.where(boot_b, beta2_t * vc + one_m * col_mean, col_mean)

        vr.copy_(vr_new)
        vc.copy_(vc_new)
        boot.fill_(True)

        vr_eps = vr + self.eps2  # [H,R]
        vc_eps = vc + self.eps2  # [H,C]
        r = vr_eps.rsqrt()  # [H,R]
        c = vc_eps.rsqrt()  # [H,C]
        scale = vr_eps.mean(dim=1).rsqrt().view(nh, 1, 1)  # [H,1,1]
        denom = r[:, :, None] * c[:, None, :]
        return denom * scale

    def state_dict(self):
        sd = {
            k: (v[0].clone(), v[1].clone(), bool(v[2].item()))
            for k, v in self.state.items()
        }
        sd_head = {
            k: (vh[0].clone(), vh[1].clone(), bool(vh[2].item()))
            for k, vh in self.state_head.items()
        }
        return {"tensor": sd, "head": sd_head}

    def load_state_dict(self, sd):
        self.state, self.state_head = {}, {}
        for k, (vr, vc, boot_bool) in sd.get("tensor", {}).items():
            dev = vr.device
            boot = torch.tensor(1 if boot_bool else 0, dtype=torch.uint8, device=dev)
            self.state[k] = (vr.to(device=dev), vc.to(device=dev), boot)
        for k, (vr, vc, boot_bool) in sd.get("head", {}).items():
            dev = vr.device
            boot = torch.tensor(1 if boot_bool else 0, dtype=torch.uint8, device=dev)
            self.state_head[k] = (vr.to(device=dev), vc.to(device=dev), boot)


# -------------------- Optimizer --------------------


class HybridMuonAdaFactorBS1(Optimizer):
    """
    Hybrid optimizer for bs=1 full finetuning (compile-friendly).
    - 2D "hidden" weights: Muon update with Newton–Schulz preconditioning via factored AdaFactor.
    - "Other" params: AdamW (decoupled weight decay).
    - Stochastic rounding to bf16 performed in-graph (no Triton, no graph breaks).
    """

    def __init__(
        self,
        named_params: Iterable[Tuple[str, torch.nn.Parameter]],
        *,
        model: Optional[torch.nn.Module] = None,  # optional: to collect head meta
        lr_hidden: float = 1e-5,  # Muonable and attention LR
        lr_embed_1d: float = 1e-6,  # Embeddings and 1D LR (fallback)
        half_life_tokens_hidden: int = 2_000_000,
        half_life_tokens_other: int = 4_000_000,
        tokens_per_step: int = 8192,
        ns_steps: int = 5,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        adjust_lr_fn: str = "match_rms_adamw",
        clip_update_rms: Optional[float] = 1.0,
        wd_other: float = 2e-3,  # decoupled WD for fallback group
        stochastic_bf16: bool = True,
        compile_loops: bool = True,
    ):
        # Optional per-head meta for attention params
        self.attn_head_meta: Dict[int, Dict[str, int]] = {}
        if model is not None:
            try:
                self.attn_head_meta = collect_attention_head_meta(model)
            except Exception:
                self.attn_head_meta = {}

        # Partition params
        params_muonable, params_fallback = [], []
        self.param_names: Dict[int, str] = {}

        for name, p in named_params:
            if not p.requires_grad:
                continue
            self.param_names[id(p)] = name

            is_embed = is_embedding_param_name(name)
            is_attn = is_stacked_attention_weight_name(name)

            if p.ndim == 2 and not is_embed:
                if is_attn and (id(p) not in self.attn_head_meta):
                    # Attention but no head meta → fallback (AdamW)
                    params_fallback.append(p)
                else:
                    # 2D MLP or attention with head meta → Muonable
                    params_muonable.append(p)
            else:
                # Embeddings/lm_head/1D/others → fallback (AdamW)
                params_fallback.append(p)

        param_groups = [
            {"params": params_muonable, "lr": lr_hidden, "name": "muonable"},
            {
                "params": params_fallback,
                "lr": lr_hidden,  # unused for embeddings/1D; we pick per-param LR inside step
                "name": "fallback_adamw",
                "weight_decay": wd_other,
            },
        ]
        super().__init__(param_groups, {})

        # Muon params
        self.ns_steps = ns_steps
        self.ns_coefficients = ns_coefficients
        self.eps = eps
        self.adjust_lr_fn = adjust_lr_fn

        # Beta2 schedules
        self.beta2_hidden_sched = TokenHalfLife(
            half_life_tokens_hidden, tokens_per_step
        ).beta2_at
        self.beta2_other_sched = TokenHalfLife(
            half_life_tokens_other, tokens_per_step
        ).beta2_at

        # Preconditioners (tensor-wise + head-wise)
        self.af_factored = FactoredEMA(self.beta2_hidden_sched)

        # AdamW state (fallback)
        self.adamw_beta1 = 0.9
        self.adamw_eps = 1e-8
        self.adamw = AdamWMoments(beta1=self.adamw_beta1, eps=self.adamw_eps)

        self.clip_update_rms = clip_update_rms
        self.stochastic_bf16 = stochastic_bf16
        self.lr_hidden = float(lr_hidden)
        self.lr_embed_1d = float(lr_embed_1d)

        # Step index
        self.step_idx = torch.zeros((), dtype=torch.long)

        # Optionally compile loops
        if compile_loops:
            self._step_muonable_impl = torch.compile(
                self._step_muonable_impl, fullgraph=False
            )
            self._step_fallback_impl = torch.compile(
                self._step_fallback_impl, fullgraph=False
            )

    # ---------- group loops (kept small and pure) ----------

    @torch.no_grad()
    def _step_muonable_impl(self):
        step = int(self.step_idx.item())
        for group in self.param_groups:
            if group.get("name") != "muonable":
                continue
            lr = float(group["lr"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                pid = id(p)
                g32 = p.grad.to(torch.float32)

                meta = self.attn_head_meta.get(pid, None)
                if meta is not None:
                    # Head-wise path for attention
                    proj = meta["proj"]  # 'q','k','v','o'
                    nh = int(meta["n_heads"])
                    hd = int(meta["head_dim"])
                    in_f = int(meta["in_features"])
                    out_f = int(meta["out_features"])

                    if proj in ("q", "k", "v"):
                        # [H, R=hd, C=in_f]
                        g_head = g32.view(nh, hd, in_f)
                    else:
                        # o-proj: [H, R=out_f, C=hd]
                        g_head = g32.view(out_f, nh, hd).permute(1, 0, 2)

                    denom_inv = self.af_factored.denom_inv_headwise(p, g_head, step)
                    g_pre_head = g_head * denom_inv

                    if self.clip_update_rms is not None:
                        rms = g_pre_head.pow(2).mean().sqrt()
                        if rms > self.clip_update_rms:
                            g_pre_head.mul_(self.clip_update_rms / (rms + 1e-12))

                    update_head = _zeropower_via_newtonschulz_batched(
                        g_pre_head, self.ns_coefficients, self.ns_steps, self.eps
                    )

                    if proj in ("q", "k", "v"):
                        update = update_head.reshape(nh * hd, in_f)
                    else:
                        update = update_head.permute(1, 0, 2).reshape(out_f, nh * hd)

                else:
                    # Tensor-wise path (MLP)
                    denom_inv = self.af_factored.denom_inv(p, g32, step)
                    g_pre = g32 * denom_inv

                    if self.clip_update_rms is not None:
                        rms = g_pre.pow(2).mean().sqrt()
                        if rms > self.clip_update_rms:
                            g_pre.mul_(self.clip_update_rms / (rms + 1e-12))

                    # Call batched NS on 2D
                    update = _zeropower_via_newtonschulz_batched(
                        g_pre, self.ns_coefficients, self.ns_steps, self.eps
                    )

                adjusted_lr = _adjust_lr(lr, self.adjust_lr_fn, p.shape)
                if p.dtype == torch.bfloat16 and self.stochastic_bf16:
                    x = p.to(torch.float32) - adjusted_lr * update
                    p.copy_(to_bf16_stochastic_in_graph(x))
                else:
                    p.add_(update, alpha=-adjusted_lr)

    @torch.no_grad()
    def _step_fallback_impl(self):
        step = int(self.step_idx.item())
        for group in self.param_groups:
            if group.get("name") != "fallback_adamw":
                continue

            wd = float(group.get("weight_decay", 0.0))

            for p in group["params"]:
                if p.grad is None:
                    continue

                name = self.param_names.get(id(p), "")
                g32 = p.grad.to(torch.float32)

                # LR selection: embeddings/lm_head or 1D → lr_embed_1d, else lr_hidden
                is_embed_like = is_embedding_param_name(name) or (p.ndim == 1)
                lr_local = self.lr_embed_1d if is_embed_like else self.lr_hidden

                # Beta2 schedule: use "other" for embed/1D; "hidden" for attention/no-meta etc.
                beta2 = (
                    self.beta2_other_sched(step)
                    if is_embed_like
                    else self.beta2_hidden_sched(step)
                )

                # AdamW with bias correction
                m, v = self.adamw.get(p)
                bc1 = 1.0 - (self.adamw_beta1 ** (step + 1))
                bc2 = 1.0 - (beta2 ** (step + 1))

                m.mul_(self.adamw_beta1).add_(g32, alpha=1.0 - self.adamw_beta1)
                v.mul_(beta2).addcmul_(g32, g32, value=1.0 - beta2)

                denom = v.sqrt().add_(self.adamw_eps)
                step_size = lr_local * (bc2**0.5) / bc1

                if p.dtype == torch.bfloat16 and self.stochastic_bf16:
                    x = p.to(torch.float32)
                    if wd != 0.0:
                        x = x * (1.0 - lr_local * wd)  # decoupled WD with lr_local
                    x = x - step_size * (m / denom)
                    p.copy_(to_bf16_stochastic_in_graph(x))
                else:
                    if wd != 0.0:
                        p.mul_(1.0 - lr_local * wd)
                    p.addcdiv_(m, denom, value=-step_size)

    # ---------- public API ----------

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_muonable_impl()
        self._step_fallback_impl()
        self.step_idx += 1
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        # Name-keyed AdaFactor (tensor-wise and head-wise) + AdamW moments
        af_tensor: Dict[str, Dict[str, torch.Tensor | bool]] = {}
        af_head: Dict[str, Dict[str, torch.Tensor | bool]] = {}
        for sid, (vr, vc, boot) in self.af_factored.state.items():
            name = self.param_names.get(sid)
            if name is None:
                continue
            af_tensor[name] = {
                "vr": vr.detach().clone(),
                "vc": vc.detach().clone(),
                "boot": bool(boot.item()),
            }
        for sid, (vrh, vch, boot) in self.af_factored.state_head.items():
            name = self.param_names.get(sid)
            if name is None:
                continue
            af_head[name] = {
                "vr": vrh.detach().clone(),
                "vc": vch.detach().clone(),
                "boot": bool(boot.item()),
            }

        adamw_name: Dict[str, Dict[str, torch.Tensor]] = {}
        for sid, (m, v) in self.adamw.state.items():
            name = self.param_names.get(sid)
            if name is None:
                continue
            adamw_name[name] = {
                "m": m.detach().clone(),
                "v": v.detach().clone(),
            }

        return {
            "af_factored": {"tensor": af_tensor, "head": af_head},
            "adamw": adamw_name,
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        # Build name -> Parameter
        name_to_param: Dict[str, torch.nn.Parameter] = {}
        for group in self.param_groups:
            for p in group["params"]:
                n = self.param_names.get(id(p))
                if n is not None:
                    name_to_param[n] = p

        # Reset current optimizer state
        self.af_factored.state = {}
        self.af_factored.state_head = {}
        self.adamw.state = {}

        # Load AdaFactor tensor-wise (strict)
        af_sd = sd.get("af_factored", {})
        af_tensor = af_sd.get("tensor", {})
        for name, payload in af_tensor.items():
            if name not in name_to_param:
                raise KeyError(
                    f"AdaFactor tensor-wise state refers to unknown param '{name}'"
                )
            p = name_to_param[name]
            if p.ndim != 2:
                raise ValueError(
                    f"AdaFactor tensor-wise state expects 2D param for '{name}', got ndim={p.ndim}"
                )
            vr = payload["vr"].to(device=p.device, dtype=torch.float32).clone()
            vc = payload["vc"].to(device=p.device, dtype=torch.float32).clone()
            boot_bool = bool(payload.get("boot", True))
            if vr.shape != (p.shape[0],) or vc.shape != (p.shape[1],):
                raise ValueError(
                    f"AdaFactor tensor-wise shapes mismatch for '{name}': "
                    f"vr{tuple(vr.shape)} vc{tuple(vc.shape)} vs param {tuple(p.shape)}"
                )
            boot = torch.tensor(
                1 if boot_bool else 0, dtype=torch.uint8, device=p.device
            )
            self.af_factored.state[id(p)] = (vr, vc, boot)

        # Load AdaFactor head-wise (strict, requires meta)
        af_head = af_sd.get("head", {})
        for name, payload in af_head.items():
            if name not in name_to_param:
                raise KeyError(
                    f"AdaFactor head-wise state refers to unknown param '{name}'"
                )
            p = name_to_param[name]
            meta = self.attn_head_meta.get(id(p))
            if meta is None:
                raise ValueError(
                    f"Head-wise state provided for '{name}' but no attention head meta is available"
                )

            nh = int(meta["n_heads"])
            if meta["proj"] in ("q", "k", "v"):
                rows = int(meta["head_dim"])
                cols = int(meta["in_features"])
            else:  # 'o'
                rows = int(meta["out_features"])
                cols = int(meta["head_dim"])

            vrh = payload["vr"].to(device=p.device, dtype=torch.float32).clone()
            vch = payload["vc"].to(device=p.device, dtype=torch.float32).clone()
            boot_bool = bool(payload.get("boot", True))

            if vrh.shape != (nh, rows) or vch.shape != (nh, cols):
                raise ValueError(
                    f"AdaFactor head-wise shapes mismatch for '{name}': "
                    f"vr{tuple(vrh.shape)} vc{tuple(vch.shape)} vs expected (H,R,C)=({nh},{rows},{cols})"
                )
            boot = torch.tensor(
                1 if boot_bool else 0, dtype=torch.uint8, device=p.device
            )
            self.af_factored.state_head[id(p)] = (vrh, vch, boot)

        # Load AdamW moments (strict)
        aw = sd.get("adamw", {})
        for name, payload in aw.items():
            if name not in name_to_param:
                raise KeyError(f"AdamW state refers to unknown param '{name}'")
            p = name_to_param[name]
            m = payload["m"].to(device=p.device, dtype=torch.float32).clone()
            v = payload["v"].to(device=p.device, dtype=torch.float32).clone()
            if m.shape != p.shape or v.shape != p.shape:
                raise ValueError(
                    f"AdamW moment shapes mismatch for '{name}': m{tuple(m.shape)} v{tuple(v.shape)} vs param {tuple(p.shape)}"
                )
            self.adamw.state[id(p)] = (m, v)
