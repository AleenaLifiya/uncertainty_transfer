"""
image_residual/evaluation.py
----------------------------
Region-masked evaluation helpers for IR1 (§22-23). Reuses the corrected-NU1 masked
metric definitions (aggregate squared error before dB, support-aware). Adds the M0-M5
region decomposition, paired IR1-vs-linear differences, and gap-stratified analysis.
"""
from __future__ import annotations
import os, sys, math
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "X-Diffusion", "scripts"))

import nu1_corrected_eval as CE
from . import masks as M

PEAK = 1.0


def region_masks(gt_hw_d, ret, D, gap_thresh=8):
    """Boolean per-voxel region masks over the volume (missing-only variants).
    Returns dict of (H,W,D) bool arrays for M0/M1/M4 and the missing/brain/gap fields.
    Tumour regions (M2/M3) are added by the SSA path where labels exist."""
    ret_s = set(int(r) for r in ret)
    known = np.zeros(D, bool)
    for r in ret_s:
        known[r] = True
    missing = ~known
    prof = M.positional_profiles(ret, D)
    gap = prof["local_gap"]                                  # [D]
    H, W, _ = gt_hw_d.shape
    miss_vol = np.broadcast_to(missing[None, None, :], (H, W, D))
    brain = gt_hw_d > 0.0
    longgap = np.broadcast_to((gap > gap_thresh)[None, None, :], (H, W, D))
    return dict(
        M0_missing=miss_vol,
        M1_missing_brain=miss_vol & brain,
        M4_missing_brain_longgap=miss_vol & brain & longgap,
        gap=gap, missing=missing, brain=brain,
    )


def _psnr_from_region(pred, gt, region):
    if region.sum() < 1000:
        return dict(psnr=float("nan"), mse=float("nan"), mae=float("nan"),
                    nmse=float("nan"), n_vox=int(region.sum()), defined=False)
    d2 = (pred[region] - gt[region]) ** 2
    mse = float(d2.mean())
    denom = float((gt[region] ** 2).sum())
    return dict(psnr=float(10 * math.log10(PEAK ** 2 / (mse + 1e-12))), mse=mse,
                mae=float(np.abs(pred[region] - gt[region]).mean()),
                nmse=float(d2.sum() / (denom + 1e-12)), n_vox=int(region.sum()), defined=True)


def evaluate_regions(pred_hw_d, gt_hw_d, ret, D):
    """M0/M1/M4 + gap-9-16 and gap>16 sub-bands (§22)."""
    rm = region_masks(gt_hw_d, ret, D)
    out = dict(M0=_psnr_from_region(pred_hw_d, gt_hw_d, rm["M0_missing"]),
               M1=_psnr_from_region(pred_hw_d, gt_hw_d, rm["M1_missing_brain"]),
               M4=_psnr_from_region(pred_hw_d, gt_hw_d, rm["M4_missing_brain_longgap"]))
    gap = rm["gap"]; missing = rm["missing"]; brain = rm["brain"]
    H, W, _ = gt_hw_d.shape
    miss_vol = np.broadcast_to(missing[None, None, :], (H, W, D))
    for name, lo, hi in [("gap_9_16", 9, 16), ("gap_gt16", 17, 10 ** 9)]:
        band = np.broadcast_to(((gap >= lo) & (gap <= hi))[None, None, :], (H, W, D))
        out[name] = _psnr_from_region(pred_hw_d, gt_hw_d, band & miss_vol & brain)
    return out


def matched_slice_metrics(pred_hw_d, gt_hw_d, evaluated_indices, lpips_fn=None, lpips_slices=4):
    """Full corrected-NU1 masked metric bundle (PSNR/SSIM/MS-SSIM/grad/laplacian/LPIPS)
    on exactly the evaluated slices."""
    return CE.masked_metrics(pred_hw_d, gt_hw_d, evaluated_indices, lpips_fn=lpips_fn,
                             max_lpips_slices=lpips_slices)


def acquired_identity_audit(pred_hw_d, vol_hw_d, ret, D, tol=1e-4):
    """M5 — verify prediction[known] == sparse_input[known] within tolerance."""
    ret = sorted(int(r) for r in ret)
    maxdiff = 0.0
    for z in ret:
        maxdiff = max(maxdiff, float(np.abs(pred_hw_d[:, :, z] - vol_hw_d[:, :, z]).max()))
    return dict(max_abs_diff_on_acquired=maxdiff, passed=bool(maxdiff <= tol), tol=tol)


def paired_ir1_vs_linear(df):
    """IR1 - linear per (subject, R, mask_family) on brain-masked missing PSNR (§23)."""
    import pandas as pd
    ir1 = df[df.method == "IR1"].set_index(["subject", "nominal_R", "mask_family"])
    lin = df[df.method == "linear"].set_index(["subject", "nominal_R", "mask_family"])
    rows = []
    for k in ir1.index:
        if k not in lin.index:
            continue
        a = ir1.loc[k]; b = lin.loc[k]
        if isinstance(a, pd.DataFrame): a = a.iloc[0]
        if isinstance(b, pd.DataFrame): b = b.iloc[0]
        if not (np.isfinite(a["psnr_masked"]) and np.isfinite(b["psnr_masked"])):
            continue
        rows.append(dict(subject=k[0], nominal_R=k[1], mask_family=k[2],
                         ir1_psnr=float(a["psnr_masked"]), linear_psnr=float(b["psnr_masked"]),
                         delta_psnr=float(a["psnr_masked"] - b["psnr_masked"]),
                         ir1_ssim=float(a.get("ssim", np.nan)), linear_ssim=float(b.get("ssim", np.nan))))
    pdf = pd.DataFrame(rows)
    return pdf


def win_tie_loss(pdf, thr=0.05):
    d = pdf["delta_psnr"].astype(float)
    wins = int((d > thr).sum()); losses = int((d < -thr).sum()); ties = int(len(d) - wins - losses)
    return dict(wins=wins, ties=ties, losses=losses,
                median_delta=float(d.median()) if len(d) else float("nan"),
                worst_delta=float(d.min()) if len(d) else float("nan"),
                p05_delta=float(np.percentile(d, 5)) if len(d) else float("nan"),
                mean_delta=float(d.mean()) if len(d) else float("nan"))
