"""
nu1_corrected_eval.py
=====================
Correction layer for the NU1 non-uniform slice-retention robustness experiment
(experiment_id NU1_CORRECTED). INFERENCE-ONLY. Nothing here retrains a model.

This module implements the fairness / metric corrections demanded by
`mixed_mask_training_nonuniform_sampling.md` §1-16 WITHOUT modifying the original
NU1 run or its completed modules. It reuses `nonuniform_masks`,
`nonuniform_gap_analysis` and `nonuniform_method_adapter` byte-for-byte and only
adds:

  A. one IMMUTABLE matched-evaluation index list per (R, mask) that EVERY method
     scores (Mode A) — subject-independent by construction, so the SHA-256 of the
     evaluated indices is identical across methods and subjects (Problem A);
  B. support-aware masked metrics that mark rows with insufficient foreground as
     UNDEFINED (NaN) instead of capping PSNR at 120 dB (Problems B, C);
  C. an L2 matched reconstruction that fully recurses only the intervals that
     contain evaluated indices (genuine recursive propagation) and scores exactly
     the evaluated indices (Problems A, E);
  D. absolute + relative robustness (ΔPSNR, MSE ratio, %MSE increase, worst-case,
     5th-percentile) (Problem F);
  E. gap-bin summaries with per-bin sample counts and low-sample warnings;
  F. SSA tumour-exposure descriptors computed ONLY after reconstruction (Problem G).

Evaluated-index selection is a deterministic function of the retained set (seeded
by its SHA-256) — it never sees image intensities or segmentation labels.
"""
from __future__ import annotations

import hashlib
import numpy as np

import nonuniform_masks as NM
import nonuniform_gap_analysis as GA
import nonuniform_method_adapter as A

# --------------------------------------------------------------------------- #
#  declared thresholds (saved into the run manifest) — Problem C
# --------------------------------------------------------------------------- #
MIN_SLICE_FG_VOXELS = 50      # a slice with < this many brain voxels is "empty"
MIN_TOTAL_FG_VOXELS = 1000    # a row with < this evaluated brain voxels is UNDEFINED
LOW_SAMPLE_GAP_BIN = 5        # gap bins with < this many samples get low_sample_warning
GLI_EVAL_BUDGET = 12          # matched evaluated missing slices per (subject,R,mask)
PEAK = 1.0

SUPPORT_RATIONALE = (
    f"Masked PSNR/MSE/NMSE/MAE/SSIM for a (subject,R,mask,method) row are defined "
    f"only if the evaluated slices contain at least MIN_TOTAL_FG_VOXELS="
    f"{MIN_TOTAL_FG_VOXELS} foreground (brain, GT>0) voxels; a slice counts as "
    f"non-empty at MIN_SLICE_FG_VOXELS={MIN_SLICE_FG_VOXELS}. Rows below the "
    f"threshold are recorded as NaN (undefined) — never as a large constant PSNR — "
    f"excluded from every mean, and counted in metric_exclusion_log.csv. This "
    f"directly fixes the original L2 U5 = 47.74 dB artifact, whose 120 dB member "
    f"came from evaluating 8 empty superior-end slices."
)


# --------------------------------------------------------------------------- #
#  A. matched evaluated-index selection (image-independent, deterministic)
# --------------------------------------------------------------------------- #
def _seed_from_ret(ret) -> int:
    h = hashlib.sha256(np.asarray(sorted(int(r) for r in ret), dtype="<i8").tobytes()).hexdigest()
    return int(h[:8], 16)


def select_evaluated_indices(ret, D, budget=GLI_EVAL_BUDGET):
    """Deterministic stratified subset of the MISSING slices, spanning local-gap
    bins so the matched evaluation is not concentrated in one gap regime. Seeded by
    the retained set's own hash => identical for every method and every subject that
    shares the mask. budget<=0 or >= n_missing => ALL missing slices."""
    miss = A.missing_indices(ret, D)
    if not budget or budget >= len(miss):
        return sorted(int(m) for m in miss)
    seed = _seed_from_ret(ret)
    rng = np.random.RandomState(seed)
    buckets = {}
    for d in miss:
        l, r = A.neighbours(ret, d)
        buckets.setdefault(r - l, []).append(d)
    for g in buckets:
        buckets[g] = list(rng.permutation(buckets[g]))
    order = sorted(buckets)                        # ascending local gap
    picked, i = [], 0
    while len(picked) < budget:
        progressed = False
        for g in order:
            if i < len(buckets[g]):
                picked.append(buckets[g][i]); progressed = True
                if len(picked) >= budget:
                    break
        if not progressed:
            break
        i += 1
    return sorted(int(x) for x in picked)


def eval_indices_hash(idx) -> str:
    return hashlib.sha256(np.asarray(sorted(int(i) for i in idx), dtype="<i8").tobytes()).hexdigest()


def build_matched_manifest(D, strides, budget=GLI_EVAL_BUDGET, specs=None):
    """One immutable row per (R, mask) with retained/missing/evaluated indices and
    their SHA-256 hashes. Subject-independent (masks + selection are image-free)."""
    man, _ = NM.build_manifest(D, strides, specs=specs)
    rows = []
    for m in man:
        ret = m["retained_indices"]
        miss = m["missing_indices"]
        ev = select_evaluated_indices(ret, D, budget)
        rows.append(dict(
            mask_id=m["mask_id"], mask_family=m["mask_family"], severity=m["severity"],
            mask_seed=int(m["seed"]), nominal_R=int(m["nominal_R"]), D=int(D),
            retained_count=len(ret), missing_count=len(miss), evaluated_missing_count=len(ev),
            max_gap=int(m["max_gap"]), longest_missing_run=int(m["longest_missing_run"]),
            retained_indices=[int(x) for x in ret],
            missing_indices=[int(x) for x in miss],
            evaluated_missing_indices=[int(x) for x in ev],
            retained_sha256=NM.retained_hash(ret),
            missing_sha256=eval_indices_hash(miss),
            evaluated_missing_sha256=eval_indices_hash(ev),
            primary=bool(m["primary"]),
        ))
    return rows


# --------------------------------------------------------------------------- #
#  B. support-aware masked metrics (aggregate MSE over voxels BEFORE dB)
# --------------------------------------------------------------------------- #
try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:                                  # pragma: no cover
    _ssim = None


def foreground_support(gt_hw_d, evaluated_indices):
    """Foreground voxel accounting over the evaluated slices (GT>0)."""
    md = np.asarray(sorted(int(i) for i in evaluated_indices), dtype=int)
    gt = gt_hw_d[:, :, md].astype(np.float32)
    fg = gt > 0.0
    per_slice = fg.reshape(-1, fg.shape[2]).sum(0).astype(int) if fg.size else np.array([], int)
    nonempty = int((per_slice >= MIN_SLICE_FG_VOXELS).sum())
    total_fg = int(fg.sum())
    return dict(
        n_evaluated_slices=int(md.size),
        evaluated_foreground_voxel_count=total_fg,
        n_nonempty_evaluated_slices=nonempty,
        min_foreground_voxels_in_slice=int(per_slice.min()) if per_slice.size else 0,
        support_ok=bool(total_fg >= MIN_TOTAL_FG_VOXELS and nonempty >= 1),
    )


def masked_metrics(recon_hw_d, gt_hw_d, evaluated_indices, lpips_fn=None,
                   max_lpips_slices=4, peak=PEAK, light=False):
    """Support-aware masked metrics on EXACTLY the evaluated slices. When support is
    insufficient every masked metric is NaN (undefined) — never a large constant.
    PSNR aggregates squared error over all evaluated foreground voxels before dB."""
    md = np.asarray(sorted(int(i) for i in evaluated_indices), dtype=int)
    sup = foreground_support(gt_hw_d, md)
    out = dict(sup)
    nan_fields = ["psnr_masked", "psnr_unmasked", "mse_masked", "nmse", "mae",
                  "ssim", "ms_ssim", "grad_mae", "laplacian_mae", "lpips"]
    if not sup["support_ok"] or md.size == 0:
        out.update({k: float("nan") for k in nan_fields})
        out["metric_defined"] = False
        return out

    pred = recon_hw_d[:, :, md].astype(np.float32)
    gt = gt_hw_d[:, :, md].astype(np.float32)
    mask = gt > 0.0
    diff2 = (pred - gt) ** 2
    mse_m = float(diff2[mask].mean())              # aggregate over voxels, then dB
    out["mse_masked"] = mse_m
    out["psnr_masked"] = float(10.0 * np.log10((peak ** 2) / (mse_m + 1e-12)))
    mse_u = float(diff2.mean())
    out["psnr_unmasked"] = float(10.0 * np.log10((peak ** 2) / (mse_u + 1e-12)))
    denom = float((gt[mask] ** 2).sum())
    out["nmse"] = float(diff2[mask].sum() / (denom + 1e-12))
    out["mae"] = float(np.abs(pred - gt)[mask].mean())

    if light:                                      # Mode-B descriptive: PSNR/MSE only
        out.update({k: float("nan") for k in ("ssim", "ms_ssim", "grad_mae", "laplacian_mae", "lpips")})
        out["metric_defined"] = True
        return out

    ssims, msssims, gmae, lmae, lp = [], [], [], [], []
    from ldm.models.diffusion.fidelity_metrics import _sobel_grad_mag, _laplacian, _ms_ssim
    n = pred.shape[2]
    lp_order = list(range(n))
    if max_lpips_slices is not None and n > max_lpips_slices:
        rng = np.random.RandomState(0)
        lp_order = sorted(rng.choice(n, size=max_lpips_slices, replace=False).tolist())
    for k in range(n):
        g = gt[:, :, k]; p = pred[:, :, k]; m2 = g > 0
        if int(m2.sum()) < MIN_SLICE_FG_VOXELS:
            continue                               # skip empty slices in per-slice stats
        if _ssim is not None:
            ssims.append(float(_ssim(g, p, data_range=peak)))
            msssims.append(float(_ms_ssim(g, p, data_range=peak)))
        gg, pg = _sobel_grad_mag(g), _sobel_grad_mag(p)
        gl, pl = _laplacian(g), _laplacian(p)
        gmae.append(float(np.abs(gg - pg)[m2].mean()))
        lmae.append(float(np.abs(gl - pl)[m2].mean()))
        if lpips_fn is not None and k in lp_order:
            lp.append(float(lpips_fn(g, p)))

    def _m(x): return float(np.mean(x)) if len(x) else float("nan")
    out.update(ssim=_m(ssims), ms_ssim=_m(msssims), grad_mae=_m(gmae),
               laplacian_mae=_m(lmae), lpips=_m(lp), metric_defined=True)
    return out


# --------------------------------------------------------------------------- #
#  C. L2 matched reconstruction — full recursion of touched intervals only
# --------------------------------------------------------------------------- #
def reconstruct_legacy_l2_matched(model, prep, ret, sampler_cfg, evaluated_indices,
                                  seed=0, native_hw=None):
    """L2 recursive bisection that GUARANTEES every evaluated index is generated:
    fully recurse each interval [ret[i],ret[i+1]] that contains >=1 evaluated index
    (no budget truncation inside it), so recursion depth / condition provenance are
    faithful. Scores exactly `evaluated_indices`. Reuses the audited legacy sampler.

    Returns (recon01, gen_rows, scored)."""
    import torch
    from ldm.models.diffusion import fidelity_legacy as FL
    prep = FL.prepare_legacy(prep)
    D = int(prep["depth"]); vol = prep["vol01"]
    native_hw = native_hw or vol.shape[:2]
    ret = sorted(int(r) for r in ret)
    recon = np.array(vol, dtype=np.float32, copy=True)
    known = {r: vol[:, :, r].copy() for r in ret}
    provenance = {r: "acquired" for r in ret}
    gen_rows, order = [], [0]
    ev = set(int(e) for e in evaluated_indices)

    intervals = [(ret[i], ret[i + 1], i) for i in range(len(ret) - 1)]
    touched = [t for t in intervals if any(t[0] < d < t[1] for d in ev)]

    def fill(left, right, interval_idx, level):
        if right - left <= 1:
            return
        d = (left + right) // 2
        g256 = FL._sample_legacy_slice(model, known[left], known[right], d, D,
                                       sampler_cfg, use_control=True, seed=seed + d)
        g240 = A._resize_240(g256, native_hw)
        slc = FL._remap_to_metric(g240, d, left, right, prep)
        known[d] = slc; recon[:, :, d] = slc; provenance[d] = "generated"
        gen_rows.append(dict(d=int(d), interval_idx=int(interval_idx),
                             recursion_level=int(level), gen_order=int(order[0]),
                             left_idx=int(left), right_idx=int(right),
                             left_provenance=provenance[left],
                             right_provenance=provenance[right],
                             local_gap=int(right - left)))
        order[0] += 1
        fill(left, d, interval_idx, level + 1)
        fill(d, right, interval_idx, level + 1)

    with torch.no_grad():
        for (l, r, idx) in touched:
            fill(l, r, idx, 0)
    scored = sorted(d for d in ev if any(l < d < r for (l, r, _) in touched))
    return recon, gen_rows, scored


# --------------------------------------------------------------------------- #
#  D. robustness — absolute + relative (Problem F)
# --------------------------------------------------------------------------- #
def paired_robustness(gli_df):
    """For each (subject, nominal_R, method) pair every non-U0 mask against U0 with
    ABSOLUTE and RELATIVE measures. Uses only rows where BOTH masked metrics are
    defined (Problem C/F). Returns a long dataframe."""
    import pandas as pd
    df = gli_df[gli_df.get("metric_defined", True) == True].copy()
    u0 = df[df["mask_family"] == "U0"].set_index(["subject", "nominal_R", "method"])
    rows = []
    for _, r in df[df["mask_family"] != "U0"].iterrows():
        k = (r["subject"], r["nominal_R"], r["method"])
        if k not in u0.index:
            continue
        base = u0.loc[k]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        p_nu, p_u = float(r["psnr_masked"]), float(base["psnr_masked"])
        m_nu, m_u = float(r["mse_masked"]), float(base["mse_masked"])
        if not (np.isfinite(p_nu) and np.isfinite(p_u) and np.isfinite(m_nu) and np.isfinite(m_u)):
            continue
        rows.append(dict(
            subject=r["subject"], nominal_R=r["nominal_R"], method=r["method"],
            mask_id=r["mask_id"], mask_family=r["mask_family"], severity=r["severity"],
            psnr_uniform=p_u, psnr_nonuniform=p_nu, delta_psnr=p_nu - p_u,
            mse_uniform=m_u, mse_nonuniform=m_nu,
            mse_ratio=m_nu / (m_u + 1e-12),
            percent_mse_increase=100.0 * (m_nu - m_u) / (m_u + 1e-12),
            ssim_uniform=float(base.get("ssim", np.nan)), ssim_nonuniform=float(r.get("ssim", np.nan)),
        ))
    cols = ["subject", "nominal_R", "method", "mask_id", "mask_family", "severity",
            "psnr_uniform", "psnr_nonuniform", "delta_psnr", "mse_uniform", "mse_nonuniform",
            "mse_ratio", "percent_mse_increase", "ssim_uniform", "ssim_nonuniform"]
    return pd.DataFrame(rows, columns=cols)


def robustness_summary(gli_df, paired_df):
    """Per-method corrected robustness combining absolute performance, relative
    degradation, and worst-case behaviour (Problem F: never rank on ΔPSNR alone)."""
    import pandas as pd
    df = gli_df[gli_df.get("metric_defined", True) == True].copy()
    rows = []
    for method, g in df.groupby("method"):
        p = g["psnr_masked"].astype(float).dropna()
        pr = paired_df[paired_df["method"] == method]
        d = pr["delta_psnr"].astype(float)
        wins = ties = losses = 0
        if not pr.empty:
            wins = int((d > 0.05).sum()); losses = int((d < -0.05).sum())
            ties = int(len(d) - wins - losses)
        rows.append(dict(
            method=method,
            n_rows=int(len(p)),
            abs_psnr_mean=float(p.mean()) if len(p) else np.nan,
            abs_psnr_median=float(p.median()) if len(p) else np.nan,
            abs_psnr_p05=float(np.percentile(p, 5)) if len(p) else np.nan,
            abs_psnr_worst=float(p.min()) if len(p) else np.nan,
            abs_mse_mean=float(g["mse_masked"].astype(float).mean()),
            delta_psnr_mean=float(d.mean()) if len(d) else np.nan,
            delta_psnr_median=float(d.median()) if len(d) else np.nan,
            delta_psnr_worst=float(d.min()) if len(d) else np.nan,
            mse_ratio_mean=float(pr["mse_ratio"].mean()) if not pr.empty else np.nan,
            mse_ratio_worst=float(pr["mse_ratio"].max()) if not pr.empty else np.nan,
            percent_mse_increase_mean=float(pr["percent_mse_increase"].mean()) if not pr.empty else np.nan,
            paired_wins=wins, paired_ties=ties, paired_losses=losses,
        ))
    out = pd.DataFrame(rows).sort_values("abs_psnr_mean", ascending=False)
    return out


# --------------------------------------------------------------------------- #
#  E. gap-bin summaries with counts + low-sample warnings (Problem §9)
# --------------------------------------------------------------------------- #
def gap_bin_summary(slice_df):
    import pandas as pd
    if slice_df.empty:
        return pd.DataFrame()
    df = slice_df.copy()
    df["gap_bin"] = pd.Categorical(df["gap_bin"], categories=GA.GAP_BIN_ORDER, ordered=True)
    rows = []
    for (method, gb), g in df.groupby(["method", "gap_bin"], observed=True):
        valid = g[np.isfinite(g["slice_mse"])]
        rows.append(dict(
            method=method, gap_bin=str(gb),
            n_samples=int(len(g)),
            n_subjects=int(g["subject"].nunique()),
            n_mask_families=int(g["mask_family"].nunique()),
            slice_mse_mean=float(valid["slice_mse"].mean()) if len(valid) else np.nan,
            slice_mse_std=float(valid["slice_mse"].std()) if len(valid) else np.nan,
            slice_mse_median=float(valid["slice_mse"].median()) if len(valid) else np.nan,
            slice_mae_mean=float(valid["slice_mae"].mean()) if len(valid) else np.nan,
            slice_mae_iqr=float(valid["slice_mae"].quantile(0.75) - valid["slice_mae"].quantile(0.25)) if len(valid) else np.nan,
            slice_ssim_mean=float(valid["slice_ssim"].dropna().mean()) if len(valid) else np.nan,
            dist_to_nearest_mean=float(g["dist_to_nearest_acquired"].mean()),
            alpha_mean=float(g["alpha"].mean()),
            low_sample_warning=bool(len(g) < LOW_SAMPLE_GAP_BIN),
        ))
    out = pd.DataFrame(rows)
    out["gap_bin"] = pd.Categorical(out["gap_bin"], categories=GA.GAP_BIN_ORDER, ordered=True)
    return out.sort_values(["method", "gap_bin"])
