"""
ir1b1_structure_loss.py
=======================
Differentiable structural-similarity loss for IR1B1 (§18-19 of implement_audit_IR1B.md).

    L_structure = 1 - SSIM(x_pred, x_gt)

It reproduces the **validated Phase A SSIM definition** (variant `A_fullslice_current`,
`skimage.metrics.structural_similarity`) as closely as is technically possible in torch:

  * 2-D, applied slice-wise along z (skimage is applied per slice);
  * uniform (box) 7x7 window, `gaussian_weights=False`;
  * K1 = 0.01, K2 = 0.03, `data_range = 1.0` fixed  ->  C1 = 1e-4, C2 = 9e-4;
  * **sample** covariance (`use_sample_covariance=True`), i.e. the NP/(NP-1) correction
    skimage applies with NP = win_size**2;
  * *valid* (unpadded) windows only — this is exactly skimage's behaviour of cropping
    (win_size-1)//2 = 3 px from each border before averaging the SSIM map, so the mean is
    taken only over window centres whose full 7x7 support lies inside the tensor;
  * the mean of the local SSIM map per slice, then the unweighted mean over valid slices.

Scope rules (§19):
  * computed on the FINAL COMPOSED reconstruction (after the anchor add and the clip);
  * only on MISSING slices — acquired slices are excluded;
  * slices whose ground truth holds fewer than `MIN_SLICE_FG_VOXELS` brain voxels
    ("empty-background-only" slices) are skipped transparently;
  * the images are **never** multiplied by a binary brain mask — that would create the
    artificial zero boundaries §7.4 warns about (and Phase A measured: variant D).

`ssim_valid_slice_count` is returned so training can log the number of valid SSIM slices
per batch.

Numerical validity is asserted by `verify_against_skimage`, which is run as a preflight
gate: if the torch implementation cannot match skimage to the declared tolerance, Phase B
must STOP and be classified BLOCKED rather than silently substituting another loss.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

WIN = 7
PAD = (WIN - 1) // 2
K1, K2 = 0.01, 0.03
DATA_RANGE = 1.0
C1 = (K1 * DATA_RANGE) ** 2
C2 = (K2 * DATA_RANGE) ** 2
NP = WIN * WIN
COV_NORM = NP / (NP - 1.0)          # skimage use_sample_covariance=True
MIN_SLICE_FG_VOXELS = 50


def _box(x):
    """Uniform 7x7 mean over the last two dims, VALID windows only. x: [N,1,H,W]."""
    k = x.new_full((1, 1, WIN, WIN), 1.0 / NP)
    return F.conv2d(x, k)


def ssim_map(pred, gt):
    """Local SSIM map for a batch of single-channel 2-D slices [N,1,H,W].
    Returns [N,1,H-2*PAD,W-2*PAD]."""
    ux, uy = _box(pred), _box(gt)
    uxx, uyy, uxy = _box(pred * pred), _box(gt * gt), _box(pred * gt)
    vx = COV_NORM * (uxx - ux * ux)
    vy = COV_NORM * (uyy - uy * uy)
    vxy = COV_NORM * (uxy - ux * uy)
    a1 = 2 * ux * uy + C1
    a2 = 2 * vxy + C2
    b1 = ux * ux + uy * uy + C1
    b2 = vx + vy + C2
    return (a1 * a2) / (b1 * b2)


def ssim_slices(pred, gt):
    """Mean SSIM per slice for [N,1,H,W]; returns [N]."""
    return ssim_map(pred, gt).mean(dim=(1, 2, 3))


def structure_loss(x_pred, gt, missing_mask, brain_mask=None,
                   min_fg=MIN_SLICE_FG_VOXELS):
    """L_structure = 1 - mean SSIM over VALID missing slices of a [B,1,D,H,W] batch.

    `missing_mask` is the [B,1,D,H,W] missing indicator (constant per slice).
    `brain_mask` (optional, [B,1,D,H,W]) is used ONLY to decide which slices are valid —
    it is never multiplied into the images.

    Returns (loss, n_valid_slices).
    """
    B, _, Dd, H, W = x_pred.shape
    p = x_pred.reshape(B * Dd, 1, H, W)
    g = gt.reshape(B * Dd, 1, H, W)
    miss = missing_mask.reshape(B * Dd, -1).amax(dim=1) > 0.5
    if brain_mask is not None:
        fg = (brain_mask.reshape(B * Dd, -1) > 0.5).sum(dim=1)
    else:
        fg = (g.reshape(B * Dd, -1) > 0).sum(dim=1)
    valid = miss & (fg >= min_fg)
    n = int(valid.sum())
    if n == 0 or H < WIN or W < WIN:
        return x_pred.sum() * 0.0, 0
    s = ssim_slices(p[valid], g[valid])
    return (1.0 - s.mean()), n


# --------------------------------------------------------------------------- #
#  numerical validation against the Phase A reference implementation
# --------------------------------------------------------------------------- #
def verify_against_skimage(n=8, size=64, seed=0, tol=1e-5):
    """Assert the torch SSIM equals skimage's `structural_similarity(..., data_range=1.0)`
    on random and structured inputs. Returns a dict of measured discrepancies."""
    from skimage.metrics import structural_similarity as sk
    rng = np.random.RandomState(seed)
    diffs, cases = [], []
    for i in range(n):
        if i % 4 == 0:
            g = rng.rand(size, size).astype(np.float32)
            p = np.clip(g + 0.05 * rng.randn(size, size).astype(np.float32), 0, 1)
        elif i % 4 == 1:
            yy, xx = np.mgrid[0:size, 0:size]
            g = np.clip(0.5 + 0.4 * np.sin(xx / 5.0) * np.cos(yy / 7.0), 0, 1).astype(np.float32)
            p = np.clip(g * 0.9 + 0.02, 0, 1).astype(np.float32)
        elif i % 4 == 2:                      # large flat-zero background (the BraTS case)
            yy, xx = np.mgrid[0:size, 0:size]
            r = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
            g = ((r < size / 3) * (0.5 + 0.3 * np.sin(r / 3))).astype(np.float32)
            p = np.clip(g + 0.01 * (r >= size / 3), 0, 1).astype(np.float32)
        else:
            g = rng.rand(size, size).astype(np.float32)
            p = g.copy()
        ref = float(sk(g, p, data_range=DATA_RANGE))
        got = float(ssim_slices(torch.from_numpy(p)[None, None].double(),
                                torch.from_numpy(g)[None, None].double())[0])
        diffs.append(abs(ref - got))
        cases.append(dict(case=i, skimage=ref, torch=got, abs_diff=abs(ref - got)))
    mx = float(max(diffs))
    return dict(max_abs_diff=mx, mean_abs_diff=float(np.mean(diffs)), tol=tol,
                passed=bool(mx <= tol), n_cases=n, cases=cases,
                definition=dict(win=WIN, K1=K1, K2=K2, data_range=DATA_RANGE,
                                window="uniform/box", covariance="sample (NP/(NP-1))",
                                padding="valid windows only (== skimage's %d px crop)" % PAD,
                                reduction="mean of local map per slice, then mean over slices"))
