"""
image_residual/losses.py
------------------------
IR1 training objective (§19). Simple, declared, fully logged. NO adversarial / LPIPS /
segmentation / diffusion / VAE / pseudo-label loss.

    total = charbonnier(missing ∩ brain)
          + 0.05 * grad_mae(missing ∩ brain)
          + 0.001 * mean|delta|(missing region)

Every term is returned separately for logging.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

CHARB_EPS = 1e-3
GRAD_W = 0.05
RESID_W = 0.001


def _masked_mean(x, m):
    s = m.sum()
    return (x * m).sum() / (s + 1e-8)


def charbonnier(pred, gt, region):
    d = pred - gt
    ch = torch.sqrt(d * d + CHARB_EPS * CHARB_EPS)
    return _masked_mean(ch, region)


def _grad_mag_3d(x):
    """Central-difference in-plane gradient magnitude on [B,1,D,H,W] (H,W axes)."""
    gh = torch.zeros_like(x); gw = torch.zeros_like(x)
    gh[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) * 0.5
    gw[..., :, 1:-1] = (x[..., :, 2:] - x[..., :, :-2]) * 0.5
    return torch.sqrt(gh * gh + gw * gw + 1e-12)


def grad_mae(pred, gt, region):
    return _masked_mean((_grad_mag_3d(pred) - _grad_mag_3d(gt)).abs(), region)


def residual_magnitude(delta, missing_mask):
    return _masked_mean(delta.abs(), missing_mask)


def ir1_loss(x_pred, delta, gt, missing_mask, brain_mask, grad_ok=True):
    """Compose the total loss on the missing ∩ brain region. Returns (total, terms).
    If grad_ok is False (verified gradient-term problem), drop the grad term rather
    than silently substituting it (§19)."""
    region = missing_mask * brain_mask
    ch = charbonnier(x_pred, gt, region)
    gm = grad_mae(x_pred, gt, region) if grad_ok else torch.zeros((), device=x_pred.device)
    rm = residual_magnitude(delta, missing_mask)
    total = ch + (GRAD_W * gm if grad_ok else 0.0) + RESID_W * rm
    terms = dict(charbonnier=float(ch.detach()),
                 grad_mae=float(gm.detach()),
                 residual_magnitude=float(rm.detach()),
                 grad_weight=GRAD_W if grad_ok else 0.0,
                 residual_weight=RESID_W,
                 total=float(total.detach()))
    return total, terms
