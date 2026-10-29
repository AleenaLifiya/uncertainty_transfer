"""One (patient, condition) -> reconstructions of every method + UQ inputs on a fixed voxel subset.

Methods (all share the same acquired slices and the same scored voxels):
  linear, bicubic                      classical controls
  SAINT_det, I3Net_det                 source deterministic checkpoints (+ interpolation adapter)
  SAINT_P,  I3Net_P                    probabilistic adaptations (+ adapter); mean of T MC passes
Scored region: originally MISSING slices; UQ region: missing slices with GT foreground (gt > 0).
UQ voxel subset: up to N_SUB voxels of that region, drawn with a seed from (patient, condition)
only -- identical for every method.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy import ndimage

from . import common as C
from . import recon as RC
from comparisons import protocol as P
from comparisons import i3net_adapter as I3

N_SUB = 40000
FEATURES = ["mu", "grad", "dlin", "dacq", "near_mm", "gap_mm", "s_mid"]


class Models:
    def __init__(self, sel_paths):
        self.saint_net, self.saint_p, self.chunk, _ = RC.build_prob_saint(sel_paths["saint_prob"])
        self.i3_det, self.i3_p = {}, {}
        for R in C.STRIDES:
            self.i3_det[R], _ = RC.load_source_i3net(R)
            self.i3_p[R], _ = RC.build_prob_i3net(R, sel_paths[f"i3net_prob_x{R}"])


def sobel_mag(x):
    gx = ndimage.sobel(x, axis=0, mode="nearest")
    gy = ndimage.sobel(x, axis=1, mode="nearest")
    return np.sqrt(gx ** 2 + gy ** 2) / 8.0


def features(x, lin, gt, ret, geo, sub):
    """Inference-available features at the subset voxels (sub = (iy, ix, iz) arrays)."""
    iy, ix, iz = sub
    zs = np.unique(iz)
    grad = np.zeros(len(iz), np.float32)
    for z in zs:
        k = iz == z
        grad[k] = sobel_mag(x[:, :, z])[iy[k], ix[k]]
    L, Rr = geo["left"][iz], geo["right"][iz]
    t = geo["t"][iz]
    return np.stack([x[iy, ix, iz], grad, np.abs(x[iy, ix, iz] - lin[iy, ix, iz]),
                     np.abs(gt[iy, ix, L] - gt[iy, ix, Rr]), geo["near_mm"][iz], geo["gap_mm"][iz],
                     1 - np.abs(2 * t - 1)], 1).astype(np.float32)


def run_case(models, scan, patient, pattern, R, rep, T, n_sub=N_SUB, keep_volumes=False):
    timing = {}
    gt = P.prepare_one(C.scan_path(scan))["vol01"]
    ret = C.make_mask(pattern, R, patient, rep)
    geo = C.gap_geometry(ret)
    miss = np.where(~geo["acquired"])[0]
    # fixed UQ subset
    region = np.zeros(gt.shape, bool)
    region[:, :, miss] = gt[:, :, miss] > 0
    cand = np.flatnonzero(region)
    rng = np.random.RandomState(C._h("isbi-subset", patient, C.cond_key(pattern, R, rep)))
    pick = np.sort(rng.choice(cand, min(n_sub, cand.size), replace=False))
    sub = np.unravel_index(pick, gt.shape)
    y = gt[sub]

    out = dict(scan=scan, patient=patient, pattern=pattern, R=R, rep=rep, ret=np.asarray(ret),
               idx=pick.astype(np.int64), y=y.astype(np.float32), n_region=int(cand.size),
               near_mm=geo["near_mm"][sub[2]].astype(np.float32),
               gap_mm=geo["gap_mm"][sub[2]].astype(np.float32))
    recons, maps = {}, {}
    t0 = time.time()
    lin = RC.linear(gt, ret)
    recons["linear"] = lin
    recons["bicubic"] = RC.bicubic(gt, ret)
    timing["classical"] = time.time() - t0

    vin, G = RC.adapter_input(gt, ret, R)
    seed = C._h("isbi-mc", patient, C.cond_key(pattern, R, rep))
    # SAINT
    t0 = time.time()
    sag, cor = RC.saint_marginals(models.saint_net, vin, G, R, models.chunk)
    timing["SAINT_msr"] = time.time() - t0
    t0 = time.time()
    recons["SAINT_det"] = _fuse1(models.saint_net, sag, cor)
    timing["SAINT_det"] = time.time() - t0 + timing["SAINT_msr"]
    t0 = time.time()
    mu, b = RC.saint_mc(models.saint_p, sag, cor, T, seed)
    torch.cuda.synchronize()
    timing["SAINT_P"] = time.time() - t0 + timing["SAINT_msr"]
    _prob(out, recons, maps, "SAINT_P", mu, b, sub, keep_volumes)
    # I3Net
    t0 = time.time()
    recons["I3Net_det"] = I3.reconstruct(models.i3_det[R], vin, G, R, C.D, RC.DEV, 4)
    timing["I3Net_det"] = time.time() - t0
    t0 = time.time()
    mu, b = RC.i3net_mc(models.i3_p[R], vin, G, R, T, seed)
    torch.cuda.synchronize()
    timing["I3Net_P"] = time.time() - t0
    _prob(out, recons, maps, "I3Net_P", mu, b, sub, keep_volumes)

    rows = []
    for m, x in recons.items():
        x = RC.copy_back(np.clip(x, 0, 1), gt, ret)
        recons[m] = x
        ev = P.evaluate_volume(x, gt, ret, C.D, region="missing")
        rows.append(dict(scan=scan, patient=patient, pattern=pattern, R=R, rep=rep, method=m,
                         psnr=ev["psnr_masked"], ssim=ev["ssim"], mae=ev["mae"], nmse=ev.get("nmse", np.nan),
                         bg_haze=ev["bg_haze"], sec=timing.get(m, timing.get("classical")),
                         mae_fg_missing=float(np.abs(x[region] - gt[region]).mean())))
        out[f"{m}__x"] = x[sub].astype(np.float32)
        out[f"{m}__feat"] = features(x, lin, gt, ret, geo, sub)
    if keep_volumes:
        return out, rows, dict(gt=gt, recons=recons, maps=maps, ret=ret, region=region, lin=lin, geo=geo)
    return out, rows, None


@torch.no_grad()
def _fuse1(net, sag, cor):
    """Stage-2 RFN one axial slice at a time == the stored SAINT path (verified 0.0)."""
    from comparisons_more import saint_adapter as SA
    return SA.fuse(net, sag, cor, RC.DEV, 1)


def _prob(out, recons, maps, name, mu, b, sub, keep):
    mean, epi, alea = RC.moments(mu, b)
    recons[name] = mean.cpu().numpy()
    iy, ix, iz = (torch.from_numpy(np.asarray(a)).to(mu.device) for a in sub)
    out[f"{name}__mu_t"] = mu[:, iy, ix, iz].cpu().numpy().astype(np.float32)
    out[f"{name}__b_t"] = b[:, iy, ix, iz].cpu().numpy().astype(np.float32)
    out[f"{name}__epi"] = epi[iy, ix, iz].cpu().numpy().astype(np.float32)
    out[f"{name}__alea"] = alea[iy, ix, iz].cpu().numpy().astype(np.float32)
    if keep:
        maps[name + "__epi"] = epi.cpu().numpy()
        maps[name + "__alea"] = alea.cpu().numpy()
