"""Learned + geometry ('PG') models for SAINT-P and ArSSR-P (exp_pg, 2026-10-26).

Both PG variants keep the probabilistic model's mean path and trained dropout FROZEN and widen only the first layer of
its scale head by 3 zero-initialised inputs: the gap geometry of the TRUE acquisition of the slice being predicted,
  (log(1 + d_mm), log(max(g_mm, 1)), r = 1 - |2u - 1|)        -- identical to SAINT-PG (models.ProbRFNGeo / geo_maps),
so at initialisation PG == P exactly, the MC means are identical to P's (same RNG consumption), and only the scale
changes. Physical spacing: BraTS 1 mm, IXI-tiny 2.18 mm per slice.
  SAINT : models.ProbRFNGeo (existing), geometry maps from geo_feats().
  ArSSR : ArSSRGeo below; per-query geometry, appended to the decoder state that feeds the scale head.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isbi import common as C
from isbi import models as M


def geo_feats(ret, D, spacing):
    """(D, 3) float32 geometry features per slice z for the acquired set ret (acquired slices: near=0, gap=0 -> clamp)."""
    g = C.gap_geometry(ret, D)
    near = g["near_mm"] * spacing
    gap = g["gap_mm"] * spacing
    return np.stack([np.log1p(near), np.log(np.maximum(gap, 1.0)), 1 - np.abs(2 * g["t"] - 1)], 1).astype(np.float32)


def saint_geo_maps(ret, D, H, W, spacing, device):
    t = torch.from_numpy(geo_feats(ret, D, spacing)).to(device)
    return t[:, :, None, None].expand(D, 3, H, W)


class ArSSRGeo(nn.Module):
    """ProbArSSR whose scale head also receives the 3 geometry features (first Linear widened, new columns zero-init)."""

    def __init__(self, prob):
        super().__init__()
        self.base, self.d1, self.d2 = prob.base, prob.d1, prob.d2
        old = prob.scale_head                         # Linear(w,32) - ReLU - Linear(32,1)
        l0 = old[0]
        new0 = nn.Linear(l0.in_features + 3, l0.out_features)
        with torch.no_grad():
            new0.weight.zero_()
            new0.weight[:, :l0.in_features] = l0.weight
            new0.bias.copy_(l0.bias)
        self.scale_head = nn.Sequential(new0, nn.ReLU(), old[2])

    @property
    def encoder(self):
        return self.base.encoder

    def decode(self, z, geo):
        dec = self.base.decoder
        s2 = dec.stage_two
        x = z + dec.stage_one(z)
        h = self.d1(s2[1](s2[0](x)))
        h = self.d2(s2[3](s2[2](h)))
        h = s2[5](s2[4](h))
        mu = s2[7](s2[6](h))
        b = M.to_scale(self.scale_head(torch.cat([h, geo.to(h.dtype)], -1)))
        return mu, b

    def forward(self, img_lr, xyz, geo):
        fm = self.base.encoder(img_lr)
        fv = F.grid_sample(fm, xyz.flip(-1).unsqueeze(1).unsqueeze(1), mode="bilinear",
                           align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        N, K = xyz.shape[:2]
        mu, b = self.decode(torch.cat([fv, xyz], -1).view(N * K, -1), geo.reshape(N * K, 3))
        return mu.view(N, K, 1), b.view(N, K, 1)


def arssr_train_geo(xyz, metas, spacing, n_lr=4):
    """Per-query geometry for UNIFORM training slabs: in LR-index coordinates every gap has unit length, the stride R of the
    slab's mask gives the physical gap g = R * spacing, and the fractional index u gives d = min(u, 1-u) * g."""
    p = (xyz[..., 0] + 1) * n_lr / 2 - 0.5                         # inverse of arssr_adapter.norm_coord
    u = p - torch.floor(p)
    R = torch.tensor([float(m["R"]) for m in metas], device=xyz.device)[:, None]
    gap = (R * spacing).expand_as(u)
    near = torch.minimum(u, 1 - u) * gap
    return torch.stack([torch.log1p(near), torch.log(torch.clamp(gap, min=1.0)), 1 - torch.abs(2 * u - 1)], -1)


@torch.no_grad()
def arssr_geo_mc(pm, gt_hw_d, ret, T, seed, D, spacing, device="cuda", z_chunk=32, z_margin=8, slice_batch=12,
                 skip_acquired=True):
    """Copy of isbi.native.arssr_mc with the per-slice geometry appended to the scale-head input."""
    from comparisons import arssr_adapter as AR
    ret = sorted(int(r) for r in ret)
    K = len(ret)
    H, W = gt_hw_d.shape[:2]
    lr = torch.from_numpy(AR.build_lr_stack(gt_hw_d, ret)).to(device)[None, None]
    p = AR.lr_positions(ret, D)
    gz = torch.from_numpy(geo_feats(ret, D, spacing)).to(device)
    cy = torch.from_numpy(AR.norm_coord(np.arange(H), H)).to(device)
    cx = torch.from_numpy(AR.norm_coord(np.arange(W), W)).to(device)
    gy, gx = torch.meshgrid(cy, cx, indexing="ij")
    plane = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)
    nT = max(T, 1)
    mus = torch.zeros(nT, H, W, D, device=device)
    bs = torch.zeros_like(mus)
    filled = np.zeros(D, bool)
    if T > 0 and skip_acquired:
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
            gq = gz[sub][:, None, :].expand(len(sub), plane.shape[0], 3).reshape(-1, 3)
            for t in range(nT):
                mu, b = pm.decode(zin, gq)
                mus[t, :, :, sub] = mu.view(len(sub), H, W).permute(1, 2, 0).float()
                bs[t, :, :, sub] = b.view(len(sub), H, W).permute(1, 2, 0).float()
            filled[sub] = True
        del feat
    assert filled.all()
    M.set_mc(pm, False)
    return mus, bs


@torch.no_grad()
def saint_geo_mc(pg, sag, cor, ret, T, seed, spacing, slab=16):
    """Copy of isbi.recon.saint_geo_mc with physical spacing (IXI) in the geometry maps."""
    dev = next(pg.parameters()).device
    s = torch.from_numpy(np.asarray(sag, np.float32)).to(dev)
    c = torch.from_numpy(np.asarray(cor, np.float32)).to(dev)
    vol = torch.stack([s, c]).permute(3, 0, 1, 2)
    D, _, H, W = vol.shape
    geo = saint_geo_maps(ret, D, H, W, spacing, dev)
    nT = max(T, 1)
    M.set_mc(pg, T > 0)
    torch.manual_seed(seed)
    mus = torch.zeros(nT, H, W, D, device=dev)
    bs = torch.zeros_like(mus)
    for i in range(0, D, slab):
        v = vol[i:i + slab]
        cat, f1 = pg.trunk(v)
        for t in range(nT):
            mu, b = pg.head(cat, f1, v, geo[i:i + slab])
            mus[t, :, :, i:i + slab] = mu[:, 0].permute(1, 2, 0)
            bs[t, :, :, i:i + slab] = b[:, 0].permute(1, 2, 0)
    M.set_mc(pg, False)
    return mus, bs
