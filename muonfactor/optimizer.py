# hybrid_muon_adafactor_bs1.py
import math
from typing import Dict, Iterable, Tuple, Optional, Any, Callable
import torch
from torch.optim import Optimizer

def is_2d_weight(p: torch.nn.Parameter) -> bool:
    return p.requires_grad and p.dim() == 2

@torch.no_grad()
def stochastic_round_to_bf16_(x: torch.Tensor):
    if x.dtype != torch.bfloat16:
        return x
    x32 = x.to(torch.float32)
    x_q = x32.to(torch.bfloat16).to(torch.float32)
    next_up = torch.nextafter(x_q, torch.full_like(x_q, float('inf')))
    ulp = (next_up - x_q).abs()
    res = (x32 - x_q)
    mask = ulp > 0
    prob = torch.zeros_like(x32)
    prob[mask] = (res[mask] / ulp[mask]).clamp_(0.0, 1.0)
    bump = (torch.rand_like(x32) < prob).to(x32) * ulp
    x.copy_((x_q + bump).to(x.dtype))
    return x

class TokenHalfLife:
    """
    Maintains beta2 from token half-life. beta2 = exp(-ln2 / H_steps).
    Supports linear ramp over warmup_steps and upper cap (<= 0.9999).
    """
    def __init__(self, half_life_tokens: int, tokens_per_step: int, warmup_steps: int = 256, cap: float = 0.9999):
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

class FactoredEMA:
    def __init__(self, beta2_sched: Callable[[int], float], eps2: float = 1e-30):
        self.beta2_sched = beta2_sched
        self.eps2 = eps2
        self.state: Dict[int, Tuple[torch.Tensor, torch.Tensor, bool]] = {}

    def _get(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        sid = id(p)
        if sid not in self.state:
            r, c = p.shape
            dev = p.device
            self.state[sid] = (
                torch.zeros(r, dtype=torch.float32, device=dev),
                torch.zeros(c, dtype=torch.float32, device=dev),
                False,  # bootstrapped?
            )
        return self.state[sid]

    @torch.no_grad()
    def denom_inv(self, p: torch.Tensor, g32: torch.Tensor, step: int) -> torch.Tensor:
        vr, vc, boot = self._get(p)
        g2 = g32.mul(g32)
        row_mean = g2.mean(dim=1)
        col_mean = g2.mean(dim=0)
        if not boot:
            vr.copy_(row_mean)
            vc.copy_(col_mean)
            self.state[id(p)] = (vr, vc, True)
        beta2 = self.beta2_sched(step)
        one_m = 1.0 - beta2
        vr.mul_(beta2).add_(row_mean, alpha=one_m)
        vc.mul_(beta2).add_(col_mean, alpha=one_m)
        vr_eps = vr.add(self.eps2)
        vc_eps = vc.add(self.eps2)
        r = vr_eps.rsqrt()
        c = vc_eps.rsqrt()
        scale = (vr_eps.mean()).rsqrt()
        return torch.outer(r, c).mul_(scale)

    def state_dict(self):
        return {k: (v[0].clone(), v[1].clone(), v[2]) for k, v in self.state.items()}

    def load_state_dict(self, sd):
        self.state = sd

class UnfactoredEMA:
    def __init__(self, beta2_sched: Callable[[int], float], eps2: float = 1e-30):
        self.beta2_sched = beta2_sched
        self.eps2 = eps2
        self.state: Dict[int, Tuple[torch.Tensor, bool]] = {}

    def _get(self, p: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        sid = id(p)
        if sid not in self.state:
            self.state[sid] = (torch.zeros_like(p, dtype=torch.float32), False)
        return self.state[sid]

    @torch.no_grad()
    def denom_inv(self, p: torch.Tensor, g32: torch.Tensor, step: int) -> torch.Tensor:
        v, boot = self._get(p)
        g2 = g32.mul(g32)
        if not boot:
            v.copy_(g2)
            self.state[id(p)] = (v, True)
        beta2 = self.beta2_sched(step)
        v.mul_(beta2).add_(g2, alpha=(1.0 - beta2))
        return (v + self.eps2).rsqrt()

    def state_dict(self):
        return {k: (v[0].clone(), v[1]) for k, v in self.state.items()}

    def load_state_dict(self, sd):
        self.state = sd

class HybridMuonAdaFactorBS1(Optimizer):
    """
    Hybrid optimizer for bs=1 full finetuning.

    - Hidden 2D weights (non-embedding linears):
        AdaFactor preconditioning -> Muon (NS) update
        momentum=0, nesterov=False, weight_decay=0.0 (decoupled)
        LR adjusted via Muon adjust_lr_fn, defaults match AdamW RMS
        Grad/updates computed in fp32; bf16 stochastic rounding on write

    - Embeddings + 1D (bias, norms, heads):
        AdaFactor (factored for 2D embeddings, unfactored for 1D)
        lr scaled (lr_other_scale) and decoupled WD=2e-3
        bf16 stochastic rounding on write

    Stability for bs=1:
        - Token-based beta2 with warmup ramp, cap 0.9999
        - EMA bootstrap from first observed g^2
        - Optional RMS clip on updates

    Exposes:
        - param_groups (for schedulers)
        - state_dict / load_state_dict
        - noise_report() summary
        - CUDA VRAM helpers (static)

    Notes:
        - Muon is only applied to 2D hidden layers. Embeddings are excluded on purpose.
        - Set include_embeddings_in_muon=True if you really want them in Muon (not recommended).
    """
    def __init__(
        self,
        named_params: Iterable[Tuple[str, torch.nn.Parameter]],
        *,
        lr_hidden: float = 5e-4,
        lr_other_scale: float = 0.5,
        half_life_tokens_hidden: int = 2_000_000,
        half_life_tokens_other: int = 4_000_000,
        tokens_per_step: int = 8192,
        ns_steps: int = 5,
        ns_coefficients = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        adjust_lr_fn: str = "match_rms_adamw",
        clip_update_rms: Optional[float] = 1.0,
        wd_other: float = 2e-3,
        include_embeddings_in_muon: bool = False,
        stochastic_bf16: bool = True,
        wandb_log: bool = False,
        wandb_prefix: str = "opt",
    ):
        # Partition params
        params_2d, params_other = [], []
        self.param_names: Dict[int, str] = {}
        for name, p in named_params:
            if not p.requires_grad:
                continue
            self.param_names[id(p)] = name
            is_embed = ("embed" in name) or ("embedding" in name) or ("lm_head" in name)
            if is_2d_weight(p) and (include_embeddings_in_muon is True or not is_embed):
                params_2d.append(p)
            else:
                params_other.append(p)

        param_groups = [
            {"params": params_2d, "lr": lr_hidden, "name": "hidden_2d"},
            {"params": params_other, "lr": lr_hidden * lr_other_scale, "name": "other", "weight_decay": wd_other},
        ]
        super().__init__(param_groups, {})

        # Muon for 2D
        self.muon = torch.optim.Muon(
            params_2d,
            lr=lr_hidden,
            weight_decay=0.0,
            momentum=0.0,
            nesterov=False,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
        )

        # Beta2 schedules
        self.beta2_hidden_sched = TokenHalfLife(half_life_tokens_hidden, tokens_per_step).beta2_at
        self.beta2_other_sched  = TokenHalfLife(half_life_tokens_other,  tokens_per_step).beta2_at

        # Preconditioners
        self.af_factored   = FactoredEMA(self.beta2_hidden_sched)
        self.af_unfactored = UnfactoredEMA(self.beta2_other_sched)
        self.clip_update_rms = clip_update_rms

        self.stochastic_bf16 = stochastic_bf16
        self._params_2d = params_2d
        self._params_other = params_other

        # Hooks
        self._pre_hook  = self.muon.register_step_pre_hook(self._pre_muon_hook)
        self._post_hook = self.muon.register_step_post_hook(self._post_muon_hook)

        # Noise tracking
        self.register_buffer("step_idx", torch.zeros((), dtype=torch.long))
        self.noise_window = 128
        self.grad_norm_hist, self.update_rms_hist, self.cos_sim_hist = [], [], []
        self.prev_g2d_flat: Optional[torch.Tensor] = None

        # Logging
        self._wandb = None
        self._wandb_on = bool(wandb_log)
        self._wandb_prefix = wandb_prefix
        if self._wandb_on:
            try:
                import wandb  # noqa
                self._wandb = wandb
            except Exception:
                self._wandb_on = False

    @torch.no_grad()
    def _pre_muon_hook(self, optim: Optimizer, args, kwargs):
        prev = []
        curr = []
        step = int(self.step_idx.item())
        for p in self._params_2d:
            g = p.grad
            if g is None:
                continue
            g32 = g.to(torch.float32)
            denom_inv = self.af_factored.denom_inv(p, g32, step)
            g_pre = g32 * denom_inv
            if self.clip_update_rms is not None:
                rms = g_pre.pow(2).mean().sqrt()
                if rms > self.clip_update_rms:
                    g_pre.mul_(self.clip_update_rms / (rms + 1e-12))
            curr.append(g_pre.reshape(-1))
            g.copy_(g_pre.to(g.dtype))
        if curr:
            gflat = torch.cat(curr)
            self.grad_norm_hist.append(gflat.norm().item())
            if self.prev_g2d_flat is not None:
                cos = torch.nn.functional.cosine_similarity(gflat, self.prev_g2d_flat, dim=0).item()
                self.cos_sim_hist.append(cos)
            self.prev_g2d_flat = gflat.detach().clone()
        return None

    @torch.no_grad()
    def _post_muon_hook(self, optim: Optimizer, args, kwargs):
        if not self.stochastic_bf16:
            return
        for p in self._params_2d:
            if p.dtype == torch.bfloat16:
                stochastic_round_to_bf16_(p.data)

    @torch.no_grad()
    def _step_other(self):
        step = int(self.step_idx.item())
        for group in self.param_groups:
            if group.get("name") != "other":
                continue
            lr = float(group["lr"])
            wd = float(group.get("weight_decay", 0.0))
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                g32 = g.to(torch.float32)
                if p.dim() == 2:
                    denom_inv = self.af_factored.denom_inv(p, g32, step)
                else:
                    denom_inv = self.af_unfactored.denom_inv(p, g32, step)
                u = g32 * denom_inv
                if self.clip_update_rms is not None:
                    rms = u.pow(2).mean().sqrt()
                    if rms > self.clip_update_rms:
                        u.mul_(self.clip_update_rms / (rms + 1e-12))
                # decoupled WD
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                # apply update in fp32 then quantize if needed
                p32 = p.data.to(torch.float32)
                p32.add_(u, alpha=-lr)
                if p.dtype == torch.bfloat16 and self.stochastic_bf16:
                    tmp = p32.clone()
                    stochastic_round_to_bf16_(tmp)
                    p.data.copy_(tmp.to(p.dtype))
                else:
                    p.data.copy_(p32.to(p.dtype))
                self.update_rms_hist.append(u.pow(2).mean().sqrt().item())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self._step_other()
        self.step_idx += 1
        # cap histories
        for hist in (self.grad_norm_hist, self.update_rms_hist, self.cos_sim_hist):
            if len(hist) > self.noise_window:
                del hist[: len(hist) - self.noise_window]
        if self._wandb_on and (int(self.step_idx.item()) % 25 == 0):
            rep = self.noise_report()
            self._wandb.log({f"{self._wandb_prefix}/{k}": v for k, v in rep.items()}, step=int(self.step_idx.item()))
        return loss

    def zero_grad(self, set_to_none: bool = True):
        for p in self._params_2d + self._params_other:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "muon": self.muon.state_dict(),
            "af_factored": self.af_factored.state_dict(),
            "af_unfactored": self.af_unfactored.state_dict(),
            "meta": {
                "step": int(self.step_idx.item()),
                "noise_window": self.noise_window,
            },
        }

    def load_state_dict(self, sd: Dict[str, Any]):
        self.muon.load_state_dict(sd["muon"])
        self.af_factored.load_state_dict(sd["af_factored"])
        self.af_unfactored.load_state_dict(sd["af_unfactored"])
        m = sd.get("meta", {})
        self.step_idx.fill_(m.get("step", 0))
        self.noise_window = m.get("noise_window", 128)

    def noise_report(self) -> Dict[str, float]:
        import numpy as np
        def stats(xs):
            if not xs:
                return float('nan'), float('nan'), float('nan')
            a = np.array(xs, dtype=float)
            return float(a.mean()), float(a.std()), float(a[-1])
        g_mean, g_std, g_last = stats(self.grad_norm_hist)
        u_mean, u_std, u_last = stats(self.update_rms_hist)
        c_mean, c_std, c_last = stats(self.cos_sim_hist)
        var_ratio = (g_std / (g_mean + 1e-8)) ** 2 if (g_mean == g_mean) else float('nan')
        return {
            "grad_norm_mean": g_mean, "grad_norm_std": g_std, "grad_norm_last": g_last,
            "update_rms_mean": u_mean, "update_rms_std": u_std, "update_rms_last": u_last,
            "cos_sim_mean": c_mean, "cos_sim_std": c_std, "cos_sim_last": c_last,
            "var_ratio": var_ratio,
        }

    @staticmethod
    def reset_cuda_peak():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def max_cuda_bytes() -> Dict[str, int]:
        if not torch.cuda.is_available():
            return {"max_allocated": 0, "max_reserved": 0}
        return {
            "max_allocated": torch.cuda.max_memory_allocated(),
            "max_reserved": torch.cuda.max_memory_reserved(),
        }
