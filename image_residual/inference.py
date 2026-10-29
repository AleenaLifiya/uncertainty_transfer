"""
image_residual/inference.py
---------------------------
Sliding-window full-volume reconstruction for IR1. The network predicts the raw
correction field on tiles; composition (tanh-scale, anchor add, clip, hard-copy) is a
deterministic post-process applied on the full volume, so the acquired-slice hard-copy
is always exact.
"""
from __future__ import annotations
import numpy as np
import torch
from monai.inferers import sliding_window_inference

from . import inputs as IN
from . import masks as M


@torch.no_grad()
def reconstruct_volume(model, vol_hw_d, ret, D, device, roi=(48, 128, 128),
                       sw_batch=2, overlap=0.25):
    """Reconstruct a full volume for a (subject, mask). Returns (x_pred_hw_d,
    x_linear_hw_d, delta_hw_d). Acquired slices are hard-copied bit-exactly."""
    model.eval()
    x6, x_lin, known_hw_d, missing_hw_d = IN.build_input_channels(vol_hw_d, ret, D)
    x6_t = torch.from_numpy(x6)[None].to(device)                     # [1,6,D,H,W]
    raw = sliding_window_inference(x6_t, roi, sw_batch, model.raw,
                                   overlap=overlap, mode="gaussian")  # [1,1,D,H,W]
    delta = model.residual_scale * torch.tanh(raw)
    x_lin_t = torch.from_numpy(np.moveaxis(x_lin, 2, 0))[None, None].to(device)  # [1,1,D,H,W]
    miss_t = torch.from_numpy(np.moveaxis(missing_hw_d, 2, 0))[None, None].to(device)
    known_t = torch.from_numpy(np.moveaxis(known_hw_d, 2, 0))[None, None].to(device)
    x_sparse = torch.from_numpy(np.moveaxis(IN.sparse_volume(vol_hw_d, ret, D), 2, 0))[None, None].to(device)
    x = torch.clamp(x_lin_t + miss_t * delta, 0.0, 1.0)
    x = torch.where(known_t > 0.5, x_sparse, x)
    x_hw_d = np.moveaxis(x[0, 0].cpu().numpy(), 0, 2)               # (H,W,D)
    delta_hw_d = np.moveaxis((miss_t * delta)[0, 0].cpu().numpy(), 0, 2)
    return x_hw_d.astype(np.float32), x_lin.astype(np.float32), delta_hw_d.astype(np.float32)


@torch.no_grad()
def direct_forward_small(model, vol_hw_d, ret, D, device):
    """Single-window direct forward (no tiling) for the sliding-window equivalence test.
    Only valid when the volume fits one ROI."""
    model.eval()
    x6, x_lin, known_hw_d, missing_hw_d = IN.build_input_channels(vol_hw_d, ret, D)
    x6_t = torch.from_numpy(x6)[None].to(device)
    raw = model.raw(x6_t)
    delta = model.residual_scale * torch.tanh(raw)
    x_lin_t = torch.from_numpy(np.moveaxis(x_lin, 2, 0))[None, None].to(device)
    miss_t = torch.from_numpy(np.moveaxis(missing_hw_d, 2, 0))[None, None].to(device)
    x = torch.clamp(x_lin_t + miss_t * delta, 0.0, 1.0)
    return np.moveaxis(x[0, 0].cpu().numpy(), 0, 2).astype(np.float32)
