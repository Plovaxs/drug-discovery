"""Generic ternary quantization-aware training (QAT) for any model built from nn.Linear layers, via the
Straight-Through Estimator (STE). This is the mechanism that makes TNN trainable instead of the post-hoc
probe (guidance/tnn_feasibility_probe.py) that already showed naive, untrained ternarization collapses
R2 from 0.41 to -1.50 on this project's EGNN affinity model.

How it integrates into an existing architecture (the actual answer to "how would this look"):
  1. ste_ternarize(w): forward pass applies TWN's threshold rule (Li et al. 2016) to get the ternary
     {-alpha, 0, alpha} tensor; backward pass is IDENTITY (gradient flows to the full-precision weight
     unchanged), via the standard trick `w_t = w + (ternary(w) - w).detach()` -- in the forward value this
     equals ternary(w) exactly, but autograd sees d(w_t)/d(w) = 1 because the detached term carries no
     gradient.
  2. TernaryLinear(nn.Linear): drop-in replacement whose forward() ternarizes self.weight on the fly before
     the matmul. Bias is NOT ternarized (standard TWN/TTQ practice, same as the post-hoc probe).
  3. convert_to_ternary_qat(model): walks the module tree and swaps every nn.Linear for a TernaryLinear
     that SHARES the same nn.Parameter objects (not copies) -- so an existing checkpoint's full-precision
     weights load either before or after conversion with no special-casing, and the optimizer still
     updates the full-precision shadow weights (only the forward computation sees ternary values).

This file only provides the mechanism -- it does not decide which model to apply it to or launch any
training. See guidance/tnn_qat_pilot.py for a CPU correctness/stability check before spending GPU time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def twn_ternary(w, delta_factor=0.7):
    delta = delta_factor * w.detach().abs().mean()
    mask = w.detach().abs() > delta
    if mask.sum() == 0:
        return torch.zeros_like(w)
    alpha = w.detach()[mask].abs().mean()
    return torch.where(mask, alpha * w.detach().sign(), torch.zeros_like(w))


def ste_ternarize(w, delta_factor=0.7):
    """Forward value = twn_ternary(w); backward gradient = identity w.r.t. w."""
    return w + (twn_ternary(w, delta_factor) - w).detach()


class TernaryLinear(nn.Linear):
    def forward(self, x):
        w_t = ste_ternarize(self.weight)
        return F.linear(x, w_t, self.bias)


def convert_to_ternary_qat(model):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear) and not isinstance(module, TernaryLinear):
            tl = TernaryLinear(module.in_features, module.out_features, bias=module.bias is not None)
            tl.weight = module.weight
            if module.bias is not None:
                tl.bias = module.bias
            setattr(model, name, tl)
        else:
            convert_to_ternary_qat(module)
    return model


def sparsity_report(model):
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            w_t = twn_ternary(module.weight)
            report[name] = float((w_t == 0).float().mean().item())
    return report
