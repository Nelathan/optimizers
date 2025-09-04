import torch

def _python_denom_inv_factored(g, vr, vc, beta2, eps2, boot):
    g2 = g.pow(2)
    row_mean = g2.mean(dim=1)
    col_mean = g2.mean(dim=0)
    if not boot[0]:
        vr.copy_(row_mean)
        vc.copy_(col_mean)
        boot[0] = True
    else:
        vr.mul_(beta2).add_(row_mean, alpha=1.0 - beta2)
        vc.mul_(beta2).add_(col_mean, alpha=1.0 - beta2)
    
    vr_eps = vr.add(eps2)
    vc_eps = vc.add(eps2)
    
    r = vr_eps.rsqrt()
    c = vc_eps.rsqrt()
    
    scale = (vr_eps.mean()).rsqrt()
    
    return torch.outer(r, c).mul_(scale)

def _python_denom_inv_unfactored(g, v, beta2, eps2, boot):
    g2 = g.pow(2)
    if not boot[0]:
        v.copy_(g2)
        boot[0] = True
    else:
        v.mul_(beta2).add_(g2, alpha=(1.0 - beta2))
    return (v + eps2).rsqrt()

def fused_factored_denom_inv(g, vr, vc, beta2, eps2, boot):
    return _python_denom_inv_factored(g, vr, vc, beta2, eps2, boot)

def fused_unfactored_denom_inv(g, v, beta2, eps2, boot):
    return _python_denom_inv_unfactored(g, v, beta2, eps2, boot)

def has_triton():
    return False
