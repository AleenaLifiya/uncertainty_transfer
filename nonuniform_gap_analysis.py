"""
nonuniform_gap_analysis.py
--------------------------
Per-missing-slice local-gap conditioning descriptors + gap/position binning +
slice-level metrics for the NU1 ablation (`implement_nonuniform.md` §7, §10).

`slice_conditioning` records, for one missing slice under an explicit retained set,
exactly the quantities §7 requires — including the UNCLAMPED local gap, the value
actually passed to the network (local_gap / STRIDE_NORM), and OOD flags (gap>8,
gap∉{2,4,8}). No clamping is applied.
"""
from __future__ import annotations

import numpy as np

STRIDE_NORM = 8.0
TRAIN_GAPS = (2, 4, 8)


def slice_conditioning(ret, d, nominal_R):
    """Descriptor for one missing slice d given explicit retained set `ret`."""
    ret = sorted(int(r) for r in ret)
    left = max(r for r in ret if r < d)
    right = min(r for r in ret if r > d)
    gap = right - left
    alpha = (d - left) / float(gap)
    dist_nearest = min(d - left, right - d)
    in_range = gap in TRAIN_GAPS
    return dict(
        d=int(d), left_idx=int(left), right_idx=int(right),
        local_gap=int(gap), alpha=float(alpha),
        dist_to_nearest_acquired=int(dist_nearest),
        nominal_R=int(nominal_R),
        local_gap_in_training_range=bool(in_range),
        local_gap_gt8=bool(gap > 8),
        stride_map_value_passed=float(gap / STRIDE_NORM),  # UNCLAMPED, fed to net
        out_of_distribution=bool(not in_range),            # gap not in {2,4,8}
    )


def gap_bin(gap: int) -> str:
    if gap == 2:
        return "gap=2"
    if 3 <= gap <= 4:
        return "gap=3-4"
    if 5 <= gap <= 8:
        return "gap=5-8"
    if 9 <= gap <= 12:
        return "gap=9-12"
    if 13 <= gap <= 16:
        return "gap=13-16"
    return "gap>16"


GAP_BIN_ORDER = ["gap=2", "gap=3-4", "gap=5-8", "gap=9-12", "gap=13-16", "gap>16"]


def position_bin(alpha: float) -> str:
    if alpha < 1.0 / 3:
        return "near_left"
    if alpha > 2.0 / 3:
        return "near_right"
    return "central"


# --------------------------------------------------------------------------- #
#  slice-level metrics (foreground-masked), aggregated per slice
# --------------------------------------------------------------------------- #
try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None


def slice_metrics(recon_hw_d, gt_hw_d, d, peak=1.0):
    """Slice-level masked MSE / MAE / SSIM for one missing slice index d."""
    p = recon_hw_d[:, :, d].astype(np.float32)
    g = gt_hw_d[:, :, d].astype(np.float32)
    m = g > 0.0
    diff2 = (p - g) ** 2
    mse = float(diff2[m].mean()) if m.any() else float(diff2.mean())
    mae = float(np.abs(p - g)[m].mean()) if m.any() else float(np.abs(p - g).mean())
    if _ssim is not None and g.max() > 0:
        ss = float(_ssim(g, p, data_range=peak))
    else:
        ss = float("nan")
    psnr = 10.0 * np.log10((peak ** 2) / (mse + 1e-12))
    return dict(slice_mse=mse, slice_mae=mae, slice_ssim=ss, slice_psnr=psnr)


def tumor_in_slice(seg_labelmap_hw_d, d) -> bool:
    """Whether an SSA segmentation slice contains any tumour label (>0). Used ONLY
    for retrospective gap analysis — never for mask generation or selection."""
    return bool((seg_labelmap_hw_d[:, :, d] > 0).any())
