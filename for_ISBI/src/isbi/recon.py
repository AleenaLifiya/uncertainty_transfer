"""Reconstruction + MC inference for every method on one (volume, mask).

Adapter rule (SAINT and I3Net need an exactly uniform stride-R grid G = {0, R, ..., D-1}):
  input slices on G are LINEAR interpolations (physical z, 1 mm) of the ORIGINALLY ACQUIRED slices
  only (exact copies where a G position was acquired). The network then reconstructs the whole
  volume from G. Every originally missing slice is scored, including G positions that the adapter
  filled. For the uniform pattern G == acquired set and the adapter is the identity.
Copy-back rule (all methods): after reconstruction, originally acquired slices are replaced by the
  acquired data; metrics use originally missing slices only.
"""
from __future__ import annotations

import copy
import os

import numpy as np
import torch

from . import common as C
from . import models as M

DEV = "cuda" if torch.cuda.is_available() else "cpu"

import image_residual as IR                                   # noqa: E402
from comparisons import protocol as P                         # noqa: E402
from comparisons import i3net_adapter as I3                   # noqa: E402
import nonuniform_method_adapter as NA                        # noqa: E402


# --------------------------------------------------------------------------- classical
def linear(gt, ret):
    return IR.inputs.linear_interp_volume(gt, ret, C.D).astype(np.float32)


def bicubic(gt, ret):
    """Project `bicubic`: Keys a=-0.5 cubic convolution over the acquired stack in LR-sample
    coordinates p(z) (piecewise-linear in physical z between acquired slices)."""
    miss = [z for z in range(C.D) if z not in set(ret)]
    return NA.bicubic_z(gt, sorted(ret), miss).astype(np.float32)


def adapter_input(gt, ret, R):
    G = C.uniform_mask(R)
    return linear(gt, ret), G


def copy_back(x, gt, ret):
    x = np.array(x, np.float32, copy=True)
    x[:, :, ret] = gt[:, :, ret]
    return x


# --------------------------------------------------------------------------- loaders
def load_source_i3net(R):
    from IR3C_LPIPS import brats_methods as BM
    nets, sels = BM._load_i3net()
    for k in list(nets):
        if k != R:
            del nets[k]
    return nets[R], sels[R]


def load_source_saint():
    from IR3C_LPIPS import brats_methods as BM
    net, chunk, sel = BM._load_saint(2)
    return net, chunk, sel


def build_prob_i3net(R, ckpt=None):
    base, sel = load_source_i3net(R)
    m = M.ProbI3Net(base).to(DEV)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    m.eval()
    return m, sel


def build_prob_saint(ckpt=None):
    net, chunk, sel = load_source_saint()
    prfn = M.ProbRFN(copy.deepcopy(net.RFN)).to(DEV)
    if ckpt:
        prfn.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    prfn.eval()
    return net, prfn, chunk, sel


# --------------------------------------------------------------------------- I3Net
@torch.no_grad()
def i3net_mc(pm, vin, G, R, T, seed, window_batch=4, lr_patch=4, clamp_windows=False):
    """Window-averaged (mu_t, b_t), t = 0..T-1, on the (H,W,D) grid. T=0 -> one dropout-off pass.
    Overlapping windows are averaged exactly as upstream sliding-window inference averages the
    mean; b is averaged the same way (documented approximation)."""
    G = sorted(G)
    K = len(G)
    lr_np, pad = I3.build_lr_stack(vin, G)
    lr = torch.from_numpy(lr_np).to(DEV)
    span = (lr_patch - 1) * R + 1
    nT = max(T, 1)
    acc_m = torch.zeros(nT, 256, 256, C.D, device=DEV)
    acc_b = torch.zeros(nT, 256, 256, C.D, device=DEV)
    cnt = torch.zeros(C.D, device=DEV)
    M.set_mc(pm, T > 0)
    torch.manual_seed(seed)
    starts = list(range(0, K - lr_patch + 1))
    for b0 in range(0, len(starts), window_batch):
        ch = starts[b0:b0 + window_batch]
        x = torch.stack([lr[:, :, s:s + lr_patch] for s in ch], 0)
        res, xp = pm.features(x)
        for t in range(nT):
            mu, b = pm.tail(res, xp)
            if clamp_windows:                      # upstream deterministic inference only
                mu = mu.clamp(0.0, 1.0)
            for j, s in enumerate(ch):
                z0 = G[s]
                acc_m[t, :, :, z0:z0 + span] += mu[j].float()
                acc_b[t, :, :, z0:z0 + span] += b[j].float()
        for s in ch:
            cnt[G[s]:G[s] + span] += 1
    assert bool((cnt > 0).all())
    t, b_, l_, r_ = pad
    mu = (acc_m / cnt)[:, t:256 - b_, l_:256 - r_]
    bb = (acc_b / cnt)[:, t:256 - b_, l_:256 - r_]
    M.set_mc(pm, False)
    return mu, bb


# --------------------------------------------------------------------------- SAINT
@torch.no_grad()
def saint_marginals(net, vin, G, R, chunk):
    from comparisons_more import saint_adapter as SA
    return SA.marginals(net, vin, G, R, C.D, DEV, chunk)


@torch.no_grad()
def saint_mc(prfn, sag, cor, T, seed, slab=16):
    s = torch.from_numpy(np.asarray(sag, np.float32)).to(DEV)
    c = torch.from_numpy(np.asarray(cor, np.float32)).to(DEV)
    vol = torch.stack([s, c]).permute(3, 0, 1, 2)            # (D,2,H,W)
    nT = max(T, 1)
    M.set_mc(prfn, T > 0)
    torch.manual_seed(seed)
    mus = torch.zeros(nT, *vol.shape[2:], vol.shape[0], device=DEV)
    bs = torch.zeros_like(mus)
    for i in range(0, vol.shape[0], slab):
        v = vol[i:i + slab]
        cat, f1 = prfn.trunk(v)                       # computed once per slab
        for t in range(nT):
            mu, b = prfn.head(cat, f1, v)
            mus[t, :, :, i:i + slab] = mu[:, 0].permute(1, 2, 0)
            bs[t, :, :, i:i + slab] = b[:, 0].permute(1, 2, 0)
    M.set_mc(prfn, False)
    return mus, bs


def saint_det(net, sag, cor):
    from comparisons_more import saint_adapter as SA
    return SA.fuse(net, sag, cor, DEV, 16)


# --------------------------------------------------------------------------- moments
def moments(mu_t, b_t):
    """Predictive mean, epistemic var (across conditional means), aleatoric var (mean 2b^2)."""
    mean = mu_t.mean(0)
    epi = mu_t.var(0, unbiased=False) if mu_t.shape[0] > 1 else torch.zeros_like(mean)
    alea = (2 * b_t ** 2).mean(0)
    return mean, epi, alea


def mixture_quantile(mu, b, q, iters=40):
    """q-quantile of the equal-weight Laplace mixture sum_t 1/T Laplace(mu_t, b_t), per voxel.
    mu, b: (T, n). Bisection on the exact mixture CDF."""
    lo = (mu - 40 * b).min(0).values
    hi = (mu + 40 * b).max(0).values
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        z = (mid[None] - mu) / b
        cdf = torch.where(z < 0, 0.5 * torch.exp(z.clamp(max=0)),
                          1 - 0.5 * torch.exp(-z.clamp(min=0))).mean(0)
        go = cdf < q
        lo = torch.where(go, mid, lo)
        hi = torch.where(go, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- option 3: SAINT-PG
def build_prob_saint_geo(saint_p_ckpt, geo_ckpt=None):
    net, prfn, chunk, sel = build_prob_saint(saint_p_ckpt)
    g = M.ProbRFNGeo(prfn).to(DEV)
    if geo_ckpt:
        g.load_state_dict(torch.load(geo_ckpt, map_location="cpu", weights_only=False)["state_dict"])
    g.eval()
    return net, g, chunk


@torch.no_grad()
def saint_geo_mc(pg, sag, cor, ret, T, seed, slab=16):
    s = torch.from_numpy(np.asarray(sag, np.float32)).to(DEV)
    c = torch.from_numpy(np.asarray(cor, np.float32)).to(DEV)
    vol = torch.stack([s, c]).permute(3, 0, 1, 2)
    D, _, H, W = vol.shape
    geo = M.geo_maps(ret, D, H, W, DEV)
    nT = max(T, 1)
    M.set_mc(pg, T > 0)
    torch.manual_seed(seed)
    mus = torch.zeros(nT, H, W, D, device=DEV)
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
