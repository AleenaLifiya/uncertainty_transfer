"""Probabilistic adaptations of the two NATIVE irregular-mask architectures (no adapter): ArSSR and DCSRN-mixed.
Added 2026-10-19 (step 3 of the user's plan). Same recipe as models.py: original mean path untouched, a separate
log-scale head (hidden ReLU layer, last layer zero-initialised -> b = B0 at start), trained MCDropout layers that are
the only modules switched on for MC inference.

ArSSR-P  encoder (RDN) unchanged -> feature lookup + coordinates -> stage_one MLP (unchanged) ->
         stage_two: L0-ReLU-[MCDropout]-L1-ReLU-[MCDropout]-L2-ReLU -> { mu: L3 (+ReLU, original)
                                                                          { b : Linear(256,32)-ReLU-Linear(32,1)
DCSRN-P  w0 + 4 densely connected BN-ELU-conv layers (unchanged; batch-statistic BN as upstream) -> concat (144 ch)
         -> [MCDropout3d] -> { mu: w2 (original conv3d 144->1) ; b: conv3d(144,32)-ReLU-conv3d(32,1) }
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models as M

P_DROP = 0.1


class MCDropout(nn.Dropout):
    """Marker: MLP dropout switched on for MC inference."""


class MCDropout3d(nn.Dropout3d):
    """Marker: 3-D channel dropout switched on for MC inference."""


def _zero_last(seq, init_b):
    nn.init.zeros_(seq[-1].weight)
    nn.init.constant_(seq[-1].bias, math.log(init_b))
    return seq


# ============================================================================ ArSSR
class ProbArSSR(nn.Module):
    def __init__(self, base, p=P_DROP):
        super().__init__()
        self.base = base
        s2 = base.decoder.stage_two                       # [L0,ReLU,L1,ReLU,L2,ReLU,L3,ReLU]
        assert len(s2) == 8, "expected decoder depth 4"
        self.d1, self.d2 = MCDropout(p), MCDropout(p)
        w = s2[4].out_features
        self.scale_head = _zero_last(nn.Sequential(nn.Linear(w, 32), nn.ReLU(), nn.Linear(32, 1)), M.B0 - M.B_MIN)

    @property
    def encoder(self):
        return self.base.encoder

    def decode(self, z):
        """z: (n, feature_dim+3) -> (mu (n,1), b (n,1))."""
        dec = self.base.decoder
        s2 = dec.stage_two
        x = z + dec.stage_one(z)
        h = self.d1(s2[1](s2[0](x)))
        h = self.d2(s2[3](s2[2](h)))
        h = s2[5](s2[4](h))
        mu = s2[7](s2[6](h))
        b = M.to_scale(self.scale_head(h))
        return mu, b

    def forward(self, img_lr, xyz):
        fm = self.base.encoder(img_lr)
        fv = F.grid_sample(fm, xyz.flip(-1).unsqueeze(1).unsqueeze(1), mode="bilinear",
                           align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        N, K = xyz.shape[:2]
        mu, b = self.decode(torch.cat([fv, xyz], -1).view(N * K, -1))
        return mu.view(N, K, 1), b.view(N, K, 1)


@torch.no_grad()
def arssr_mc(pm, gt_hw_d, ret, T, seed, D=153, device="cuda", z_chunk=32, z_margin=8, slice_batch=12, skip_acquired=True):
    """ArSSR whole-volume inference (tiles/margins/query geometry exactly as comparisons.arssr_adapter.reconstruct)
    with the encoder run ONCE per tile and the decoder T times. Returns (mu_t, b_t): (max(T,1), H, W, D)."""
    from comparisons import arssr_adapter as AR
    ret = sorted(int(r) for r in ret)
    K = len(ret)
    H, W = gt_hw_d.shape[:2]
    lr = torch.from_numpy(AR.build_lr_stack(gt_hw_d, ret)).to(device)[None, None]
    p = AR.lr_positions(ret, D)
    cy = torch.from_numpy(AR.norm_coord(np.arange(H), H)).to(device)
    cx = torch.from_numpy(AR.norm_coord(np.arange(W), W)).to(device)
    gy, gx = torch.meshgrid(cy, cx, indexing="ij")
    plane = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)
    nT = max(T, 1)
    mus = torch.zeros(nT, H, W, D, device=device)
    bs = torch.zeros_like(mus)
    filled = np.zeros(D, bool)
    if T > 0 and skip_acquired:          # acquired slices are copied back from the data and never scored
        filled[ret] = True
    M.set_mc(pm, T > 0)
    torch.manual_seed(seed)
    for a in range(0, K, z_chunk):
        b_ = min(K, a + z_chunk)
        a2, b2 = max(0, a - z_margin), min(K, b_ + z_margin)
        feat = pm.encoder(lr[:, :, a2:b2])
        nz = b2 - a2
        zs = [z for z in range(D) if ((a <= p[z] < b_) or (b_ == K and p[z] >= a)) and not filled[z]]
        for i0 in range(0, len(zs), slice_batch):
            sub = zs[i0:i0 + slice_batch]
            cz = torch.from_numpy(AR.norm_coord(p[sub] - a2, nz)).to(device)
            xyz = torch.cat([cz[:, None, None].expand(len(sub), plane.shape[0], 1),
                             plane[None].expand(len(sub), -1, -1)], -1)
            fv = F.grid_sample(feat.expand(len(sub), -1, -1, -1, -1), xyz.flip(-1).unsqueeze(1).unsqueeze(1),
                               mode="bilinear", align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
            zin = torch.cat([fv, xyz], -1).reshape(len(sub) * plane.shape[0], -1)
            for t in range(nT):
                mu, b = pm.decode(zin)
                mus[t, :, :, sub] = mu.view(len(sub), H, W).permute(1, 2, 0).float()
                bs[t, :, :, sub] = b.view(len(sub), H, W).permute(1, 2, 0).float()
            filled[sub] = True
        del feat
    assert filled.all()
    M.set_mc(pm, False)
    return mus, bs


# ============================================================================ DCSRN
class ProbDCSRN(nn.Module):
    def __init__(self, base, p=P_DROP):
        super().__init__()
        self.base = base
        c = base.w2.in_channels
        self.drop = MCDropout3d(p)
        self.scale_head = _zero_last(nn.Sequential(nn.Conv3d(c, 32, 3, padding=1), nn.ReLU(),
                                                   nn.Conv3d(32, 1, 3, padding=1)), M.B0 - M.B_MIN)

    def features(self, x):
        """Verbatim DCSRNNet.forward up to the final concatenation."""
        b = self.base
        h = b.w0(x)
        convs = []
        for layer in range(b.layers):
            last = h
            if convs:
                h = torch.cat([h] + convs, dim=1)
            h = b.convs[layer](F.elu(b.bns[layer](h)))
            convs.append(last)
        return torch.cat([h] + convs, dim=1)

    def head(self, h):
        h = self.drop(h)
        return self.base.w2(h), M.to_scale(self.scale_head(h))

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def dcsrn_mc(pm, gt_hw_d, ret, T, seed, D=153, device="cuda", roi=64, sw_batch=4, overlap=0.25, mode="tiled64"):
    """Upstream tiled inference (MONAI sliding window, 64^3, batch 4, Gaussian blending -- identical windows and
    batch composition, so the batch-statistic BN sees the same batches) with the dense block run ONCE per window
    batch and the heads T times; mu_t and b_t are blended like any output channel."""
    from monai.inferers import sliding_window_inference
    from comparisons_more import dcsrn_adapter as DC
    lr = DC.build_lr_volume(gt_hw_d, ret, D)
    x = DC.to_tensor(lr, device)
    nT = max(T, 1)
    M.set_mc(pm, T > 0)
    torch.manual_seed(seed)

    def predictor(xb):
        h = pm.features(xb)
        outs = [pm.head(h) for _ in range(nT)]
        return torch.cat([o[0] for o in outs] + [o[1] for o in outs], 1)

    if mode == "whole_volume":           # the upstream-faithful mode chosen for some strides (BN on the whole volume)
        y = predictor(x)
    else:
        y = sliding_window_inference(x, (roi,) * 3, sw_batch, predictor, overlap=overlap, mode="gaussian")
    M.set_mc(pm, False)
    y = y[0].permute(0, 2, 3, 1)                                  # (2T, H, W, D)
    return y[:nT].float(), y[nT:].float()


# ============================================================================ training samplers with supervision weights
def arssr_batch_weights(xyz, n_lr=4, tol=1e-4):
    """Weight 0 for queries that sit exactly on an acquired LR slice (p(z) integer), 1 otherwise. Inverts
    arssr_adapter.norm_coord: p = (c + 1) * n / 2 - 0.5."""
    p = (xyz[..., 0] + 1) * n_lr / 2 - 0.5
    return ((p - torch.round(p)).abs() > tol).float()[..., None]


def dcsrn_masked_batch(sampler, idxs, device="cuda"):
    """MixedMaskPatchSampler.batch, additionally returning a (N,1,p,1,1) weight that is 0 on acquired slices.
    Mirrors comparisons_more.dcsrn_adapter.MixedMaskPatchSampler.batch line by line (same RNG order)."""
    from comparisons_more import dcsrn_adapter as DC
    lrs, hrs, ws = [], [], []
    p = sampler.patch
    for i in idxs:
        gt = sampler._gt(int(i))
        H, W, D = gt.shape
        y = int(sampler.rng.randint(0, H - p + 1))
        x = int(sampler.rng.randint(0, W - p + 1))
        z = int(sampler.rng.randint(0, D - p + 1))
        ret, meta = sampler.draw_mask()
        sub = np.ascontiguousarray(gt[y:y + p, x:x + p, :])
        lr = DC.build_lr_volume(sub, ret, sampler.D)
        lrs.append(lr[:, :, z:z + p])
        hrs.append(sub[:, :, z:z + p])
        w = np.ones(p, np.float32)
        w[[r - z for r in ret if z <= r < z + p]] = 0
        ws.append(w)
    f = lambda a: torch.from_numpy(np.ascontiguousarray(np.stack(a, 0).transpose(0, 3, 1, 2))).unsqueeze(1).to(device)  # noqa: E731
    w = torch.from_numpy(np.stack(ws)).to(device)[:, None, :, None, None]
    return f(lrs), f(hrs), w
