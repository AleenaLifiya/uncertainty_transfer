"""
fidelity_metrics.py
-------------------
Extended reconstruction metrics for the frozen-VAE full experiment
(xdiff_control_fullVAE.md, "Evaluation metrics").

Everything here operates in the shared percentile-[0,1] metric space (the same
`prep["vol01"]` space used by every method), so metrics are comparable across
non-learned baselines, legacy models and the redesigned models.

Rules enforced (from the spec):
  * Metrics are computed ONLY on reconstructed missing slices.
  * The primary region is  missing slices  INTERSECT  foreground brain mask.
  * Squared error is aggregated over the selected voxels BEFORE converting to
    PSNR (never average slice-level PSNR and call it volume PSNR).
  * One row per subject; the notebook aggregates to mean / std / median / CI.

Metrics returned by `volume_metrics_full`:
  psnr_masked, psnr_unmasked, nmse, mae, ssim (masked axial),
  ms_ssim (masked axial), grad_mae (Sobel-gradient MAE), laplacian_mae,
  lpips (secondary natural-image feature metric, non-clinical), n_missing_slices.

LPIPS is optional: pass a cached net via `lpips_fn` (see `get_lpips`) or leave
None to skip it (column filled with NaN). LPIPS is explicitly labelled
non-clinical in every report.
"""

import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:  # pragma: no cover
    _ssim = None


# --------------------------------------------------------------------------
#  low-level 2D operators (numpy, deterministic)
# --------------------------------------------------------------------------
def _sobel_grad_mag(img):
    """Sobel gradient magnitude of a 2D [0,1] image (reflect padding)."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    p = np.pad(img.astype(np.float32), 1, mode="reflect")
    gx = (
        kx[0, 0] * p[:-2, :-2] + kx[0, 1] * p[:-2, 1:-1] + kx[0, 2] * p[:-2, 2:]
        + kx[1, 0] * p[1:-1, :-2] + kx[1, 2] * p[1:-1, 2:]
        + kx[2, 0] * p[2:, :-2] + kx[2, 1] * p[2:, 1:-1] + kx[2, 2] * p[2:, 2:]
    )
    gy = (
        ky[0, 0] * p[:-2, :-2] + ky[0, 2] * p[:-2, 2:]
        + ky[1, 0] * p[1:-1, :-2] + ky[1, 2] * p[1:-1, 2:]
        + ky[2, 0] * p[2:, :-2] + ky[2, 1] * p[2:, 1:-1] + ky[2, 2] * p[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def _laplacian(img):
    """4-neighbour Laplacian of a 2D [0,1] image (reflect padding)."""
    p = np.pad(img.astype(np.float32), 1, mode="reflect")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1])


def _ms_ssim(gt, pred, data_range=1.0, weights=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
    """Multi-scale SSIM for a 2D image pair.

    Downsamples by 2x2 average pooling per scale; combines per-scale SSIM with the
    standard Wang et al. weights. Falls back gracefully when the image becomes too
    small for the SSIM window (returns the product over the scales that fit).
    """
    if _ssim is None:
        return float("nan")
    vals = []
    g, p = gt.astype(np.float32), pred.astype(np.float32)
    for i, w in enumerate(weights):
        if min(g.shape) < 7:
            break
        s = _ssim(g, p, data_range=data_range)
        vals.append((max(s, 1e-6), w))
        # 2x2 average pool
        h, wd = (g.shape[0] // 2) * 2, (g.shape[1] // 2) * 2
        if h < 2 or wd < 2:
            break
        g = g[:h, :wd].reshape(h // 2, 2, wd // 2, 2).mean((1, 3))
        p = p[:h, :wd].reshape(h // 2, 2, wd // 2, 2).mean((1, 3))
    if not vals:
        return float("nan")
    wsum = sum(w for _, w in vals)
    prod = 1.0
    for s, w in vals:
        prod *= s ** (w / wsum)
    return float(prod)


# --------------------------------------------------------------------------
#  optional LPIPS (cached; CPU or CUDA). Non-clinical feature metric.
# --------------------------------------------------------------------------
_LPIPS_CACHE = {}


def get_lpips(device="cuda", net="alex"):
    """Return a cached LPIPS callable  f(gt01_2d, pred01_2d) -> float, or None."""
    key = (device, net)
    if key in _LPIPS_CACHE:
        return _LPIPS_CACHE[key]
    try:
        import torch
        import lpips as _lpips_mod
        model = _lpips_mod.LPIPS(net=net, verbose=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def _fn(gt2d, pred2d):
            with torch.no_grad():
                def _t(a):
                    a = np.clip(a, 0, 1).astype(np.float32) * 2 - 1
                    t = torch.from_numpy(a)[None, None].repeat(1, 3, 1, 1).to(device)
                    return t
                return float(model(_t(gt2d), _t(pred2d)).item())
        _LPIPS_CACHE[key] = _fn
        return _fn
    except Exception as e:  # pragma: no cover
        print(f"[fidelity_metrics] LPIPS unavailable ({e}); lpips column -> NaN")
        _LPIPS_CACHE[key] = None
        return None


# --------------------------------------------------------------------------
#  full per-subject metrics over the MISSING slices only
# --------------------------------------------------------------------------
def volume_metrics_full(recon_hw_d, gt_hw_d, missing_indices, peak=1.0,
                        lpips_fn=None, max_lpips_slices=None):
    """Per-subject metrics over the missing slices, masked to foreground.

    recon_hw_d, gt_hw_d : unpadded [0,1] volumes (H, W, D) in the shared metric
    space. Returns a dict of scalars (one subject row).
    """
    md = np.array(missing_indices, dtype=int)
    pred = recon_hw_d[:, :, md].astype(np.float32)
    gt = gt_hw_d[:, :, md].astype(np.float32)
    mask = gt > 0.0  # per-volume foreground brain mask

    diff2 = (pred - gt) ** 2
    # ---- PSNR: aggregate MSE over selected voxels BEFORE dB ----
    mse_m = diff2[mask].mean() if mask.any() else diff2.mean()
    psnr_masked = 10.0 * np.log10((peak ** 2) / (mse_m + 1e-12))
    mse_u = diff2.mean()
    psnr_unmasked = 10.0 * np.log10((peak ** 2) / (mse_u + 1e-12))
    # ---- NMSE / MAE (masked) ----
    denom = (gt[mask] ** 2).sum() if mask.any() else (gt ** 2).sum()
    nmse = float(diff2[mask].sum() / (denom + 1e-12)) if mask.any() \
        else float(diff2.sum() / (denom + 1e-12))
    mae = float(np.abs(pred - gt)[mask].mean()) if mask.any() \
        else float(np.abs(pred - gt).mean())

    # ---- per-slice masked metrics: SSIM, MS-SSIM, gradient/Laplacian MAE, LPIPS ----
    ssims, msssims, gmae, lmae, lp = [], [], [], [], []
    n_slices = pred.shape[2]
    lp_order = list(range(n_slices))
    if max_lpips_slices is not None and n_slices > max_lpips_slices:
        rng = np.random.RandomState(0)
        lp_order = sorted(rng.choice(n_slices, size=max_lpips_slices, replace=False).tolist())
    for k in range(n_slices):
        g = gt[:, :, k]
        p = pred[:, :, k]
        m2 = g > 0
        if g.max() <= 0:
            continue
        if _ssim is not None:
            ssims.append(_ssim(g, p, data_range=peak))
            msssims.append(_ms_ssim(g, p, data_range=peak))
        # gradient / Laplacian MAE inside the brain mask
        gg, pg = _sobel_grad_mag(g), _sobel_grad_mag(p)
        gl, pl = _laplacian(g), _laplacian(p)
        if m2.any():
            gmae.append(float(np.abs(gg - pg)[m2].mean()))
            lmae.append(float(np.abs(gl - pl)[m2].mean()))
        if lpips_fn is not None and k in lp_order:
            lp.append(lpips_fn(g, p))

    def _m(x):
        return float(np.mean(x)) if len(x) else float("nan")

    return {
        "psnr_masked": float(psnr_masked),
        "psnr_unmasked": float(psnr_unmasked),
        "nmse": nmse,
        "mae": mae,
        "ssim": _m(ssims),
        "ms_ssim": _m(msssims),
        "grad_mae": _m(gmae),
        "laplacian_mae": _m(lmae),
        "lpips": _m(lp),
        "n_missing_slices": int(len(md)),
    }
