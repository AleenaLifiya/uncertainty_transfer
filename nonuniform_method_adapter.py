"""
nonuniform_method_adapter.py
----------------------------
Explicit-retained-index reimplementations of every reconstruction path, for the
NU1 non-uniform ablation. The ONLY change vs the completed pipelines is that the
regular `retained_indices(depth, R)` is replaced by an EXPLICIT `ret` list; all
neighbour lookup, conditioning, sampling and hard-copy reuse the audited helpers
byte-for-byte (see NONUNIFORM_SAMPLING_AUDIT.md). Nothing here re-derives model
math, and no completed file is modified.

Covered:
  interpolation (nearest / linear / cubic / bspline / sparse)   — non-learned
  L1 legacy X-Diffusion, L2 legacy recursive ControlNet         — legacy
  L3 / L4 / L5 independent diffusion                            — reconstruct_volume_nu
  L3B / L3C shared residual predictor (one r_pred, both comps)  — reconstruct_l3bc_nu
  SSA geometry restoration + observed-slice hard-copy check     — explicit ret

Determinism: every path is seeded; interpolation is data-only. `targets` lets the
diffusion paths generate a stratified subset of missing slices for a bounded smoke
budget while the metrics score exactly those slices.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch

PROJECT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(PROJECT, "X-Diffusion", "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

from ldm.data.brats_preprocessing_fidelity import unpad_hw
from ldm.data.bratsloader_controlnet_fidelity import build_conditioning, STRIDE_NORM
from ldm.models.diffusion.fidelity_sampling import (
    build_model_cond, encode_linear_latent, ddim_sample, _img_to_gray01,
)
from ldm.models.diffusion.fidelity_residual_l3b import ddim_sample_residual
from ldm.models.diffusion.fidelity_residual_l3c import compose_l3c_hard_anchor
from ldm.models.diffusion import fidelity_legacy as FL


# --------------------------------------------------------------------------- #
#  neighbour / triple helpers (explicit ret)
# --------------------------------------------------------------------------- #
def missing_indices(ret, D):
    rs = set(int(r) for r in ret)
    return [d for d in range(D) if d not in rs]


def neighbours(ret, d):
    """Actual left/right retained neighbours bracketing missing slice d."""
    left = max(r for r in ret if r < d)
    right = min(r for r in ret if r > d)
    return int(left), int(right)


def triples_for(ret, D, targets=None):
    """(left,right,d) for each missing (or target) slice using ACTUAL neighbours."""
    miss = missing_indices(ret, D) if targets is None else sorted(int(t) for t in targets)
    return [(*neighbours(ret, d), d) for d in miss]


def stratified_targets(ret, D, budget, seed=0):
    """Deterministically pick up to `budget` missing slices spanning local-gap bins,
    so the diffusion gap-analysis subset is not concentrated in one gap regime.
    Returns a sorted list of slice indices. budget<=0 or None => all missing."""
    miss = missing_indices(ret, D)
    if not budget or budget >= len(miss):
        return miss
    rng = np.random.RandomState(seed)
    # bin by local gap
    buckets = {}
    for d in miss:
        l, r = neighbours(ret, d)
        buckets.setdefault(r - l, []).append(d)
    for g in buckets:
        buckets[g] = list(rng.permutation(buckets[g]))
    order = sorted(buckets)                        # ascending gap
    picked = []
    i = 0
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


# --------------------------------------------------------------------------- #
#  Keys bicubic convolution kernel (the SR-literature "bicubic")
# --------------------------------------------------------------------------- #
def keys_kernel(t, a=-0.5):
    """Keys (1981) cubic convolution kernel, the kernel MATLAB `imresize`, PIL
    `BICUBIC` and `torch.nn.functional.interpolate(mode='bicubic')` all use.

        u(t) = (a+2)|t|^3 - (a+3)|t|^2 + 1              |t| <= 1
               a|t|^3 - 5a|t|^2 + 8a|t| - 4a            1 < |t| < 2
               0                                        otherwise

    It is *interpolating* (u(0)=1, u(±1)=u(±2)=0), so a sample at an acquired
    position is reproduced exactly, and it is a partition of unity, so a constant
    volume is preserved. This is a LOCAL 4-tap kernel and is therefore a different
    interpolant from the global spline used by ``kind="cubic"``.
    """
    t = np.abs(np.asarray(t, dtype=np.float64))
    t2, t3 = t * t, t * t * t
    w = np.where(t <= 1.0,
                 (a + 2.0) * t3 - (a + 3.0) * t2 + 1.0,
                 np.where(t < 2.0,
                          a * t3 - 5.0 * a * t2 + 8.0 * a * t - 4.0 * a,
                          0.0))
    return w


def lr_coordinate(ret, z):
    """Map a through-plane index z to fractional LR-sample coordinate p(z).

        p(z) = j + (z - ret[j]) / (ret[j+1] - ret[j])   for ret[j] <= z <= ret[j+1]

    This is the slice-selection geometry the project already uses for ArSSR. On a
    uniform mask (ret = 0, R, 2R, ...) it reduces exactly to p = z / R, so the result
    is textbook bicubic upsampling along z; on the non-uniform families U1--U6 it is
    the natural generalisation, which is what lets bicubic be scored on the same grid
    as nearest / linear / cubic instead of being marked N/A.
    """
    ret = np.asarray(ret, dtype=np.float64)
    j = int(np.searchsorted(ret, z, side="right") - 1)
    j = max(0, min(j, len(ret) - 2))
    span = ret[j + 1] - ret[j]
    return j + (float(z) - ret[j]) / (span if span > 0 else 1.0)


def bicubic_z(vol_hw_d, ret, miss, a=-0.5):
    """Fill `miss` slices of an (H, W, D) volume with Keys bicubic along z.

    Taps are taken over the RETAINED stack (replicate-padded at both ends), so the
    kernel never sees a slice that was not acquired.
    """
    ret = sorted(int(r) for r in ret)
    K = len(ret)
    lr = vol_hw_d[:, :, np.asarray(ret, dtype=int)]          # (H, W, K)
    out = np.array(vol_hw_d, dtype=np.float32, copy=True)
    for z in miss:
        p = lr_coordinate(ret, z)
        base = int(np.floor(p))
        t = p - base
        idx = np.clip([base - 1, base, base + 1, base + 2], 0, K - 1)
        w = keys_kernel(np.array([t + 1.0, t, 1.0 - t, 2.0 - t]), a)
        w = w / w.sum()                                       # exact partition of unity
        acc = np.zeros(vol_hw_d.shape[:2], dtype=np.float64)
        for wi, ii in zip(w, idx):
            acc += wi * lr[:, :, ii]
        out[:, :, z] = np.clip(acc, 0.0, 1.0).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
#  non-learned interpolation (explicit ret; real physical positions)
# --------------------------------------------------------------------------- #
def interpolate_nu(prep, ret, kind="linear", targets=None):
    """Nearest / linear / cubic / bicubic / bspline / sparse over the ACTUAL retained z.

    - linear:  alpha=(d-left)/(right-left); (1-a)*x(left)+a*x(right)
    - nearest: closer retained slice; exact ties -> lower index (left)
    - bicubic: Keys a=-0.5 cubic convolution kernel over the retained stack, in
      LR-sample coordinates p(z) (see bicubic_z). Local 4-tap, not a global spline.
    - cubic / bspline: fit on the actual non-uniform (xs=ret) coordinates;
      deterministic fallback to linear when < min_pts retained (documented).
    - sparse:  missing slices left as zero (metrics only where meaningful)
    Returns (recon01_hw_d, scored_missing, ret).
    """
    D = int(prep["depth"])
    vol = prep["vol01"]
    ret = sorted(int(r) for r in ret)
    recon = np.array(vol, dtype=np.float32, copy=True)
    miss_all = missing_indices(ret, D)
    scored = miss_all if targets is None else sorted(int(t) for t in targets)

    if kind == "bicubic":
        return bicubic_z(vol, ret, miss_all), scored, ret

    if kind in ("cubic", "bspline"):
        xs = np.asarray(ret)
        ys = vol[:, :, xs]                          # (H,W,len(ret))
        H, W, _ = ys.shape
        min_pts = 4
        if len(ret) < min_pts:
            kind_eff = "linear_fallback"
        else:
            kind_eff = kind
        if kind_eff == "cubic":
            from scipy.interpolate import interp1d
            f = interp1d(xs, ys.reshape(H * W, -1), kind="cubic", axis=1,
                         bounds_error=False, fill_value="extrapolate")
            for d in miss_all:
                recon[:, :, d] = np.clip(f(d).reshape(H, W), 0.0, 1.0)
        elif kind_eff == "bspline":
            from scipy.interpolate import make_interp_spline
            k = min(3, len(ret) - 1)
            spl = make_interp_spline(xs, ys.reshape(H * W, -1), k=k, axis=1)
            for d in miss_all:
                recon[:, :, d] = np.clip(spl(d).reshape(H, W), 0.0, 1.0)
        else:  # linear fallback
            for d in miss_all:
                l, r = neighbours(ret, d)
                a = (d - l) / (r - l)
                recon[:, :, d] = (1 - a) * vol[:, :, l] + a * vol[:, :, r]
        return recon, scored, ret

    for d in miss_all:
        l, r = neighbours(ret, d)
        if kind == "sparse":
            recon[:, :, d] = 0.0
        elif kind == "nearest":
            src = l if (d - l) <= (r - d) else r     # exact tie -> lower index (left)
            recon[:, :, d] = vol[:, :, src]
        else:                                        # linear
            a = (d - l) / (r - l)
            recon[:, :, d] = (1 - a) * vol[:, :, l] + a * vol[:, :, r]
    return recon, scored, ret


# --------------------------------------------------------------------------- #
#  L3 / L4 / L5 independent diffusion (explicit ret)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct_volume_nu(model, prep, ret, sampler_cfg, batch_size=8, seed=0,
                          method="interp", targets=None):
    """Independent per-missing-slice diffusion (mirrors fidelity_sampling.
    reconstruct_volume) with an EXPLICIT retained set. Retained slices hard-copied
    from GT; only `targets` generated & scored. build_conditioning(debug=False) so
    arbitrary local gaps are accepted (local-gap conditioning, no clamp)."""
    D = int(prep["depth"]); pad = prep["pad"]; target = prep["target"]
    vol = prep["vol01"]
    ret = sorted(int(r) for r in ret)
    recon = np.array(vol, dtype=np.float32, copy=True)
    triples = triples_for(ret, D, targets)
    scored = [d for (_, _, d) in triples]
    for i in range(0, len(triples), batch_size):
        chunk = triples[i:i + batch_size]
        ic, ctrl, Ts = [], [], []
        for (left, right, d) in chunk:
            c = build_conditioning(prep, left, right, d, target=target, debug=False)
            ic.append(c["image_cond"]); ctrl.append(c["control"]); Ts.append(c["T"])
        ic = torch.stack(ic); ctrl = torch.stack(ctrl); Ts = torch.stack(Ts)
        cond = build_model_cond(model, ic, Ts, ctrl, uncond=False)
        uc = build_model_cond(model, ic, Ts, ctrl, uncond=True)
        z0 = encode_linear_latent(model, ic) if method == "interp" else None
        img = ddim_sample(
            model, cond, uc, z0_interp=z0,
            ddim_steps=sampler_cfg["ddim_steps"],
            interp_strength=sampler_cfg.get("interp_strength", 0.25) if method == "interp" else 1.0,
            control_scale=sampler_cfg.get("control_scale", 1.0),
            guidance_scale=sampler_cfg.get("guidance_scale", 1.0),
            seed=seed + i)
        gray = _img_to_gray01(img).cpu().numpy()
        for j, (left, right, d) in enumerate(chunk):
            recon[:, :, d] = unpad_hw(gray[j], pad)
    return recon, scored, ret


# --------------------------------------------------------------------------- #
#  L3B / L3C shared residual predictor (one r_pred -> both compositions)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct_l3bc_nu(model, prep, ret, sampler_cfg, batch_size=8, seed=0, targets=None):
    """Run the L3B smoke residual predictor ONCE and compose BOTH L3B and L3C from
    the SAME r_pred (isolates composition per §5). Returns (recon_l3b, recon_l3c,
    scored, ret). Retained slices hard-copied from GT in both."""
    D = int(prep["depth"]); pad = prep["pad"]; target = prep["target"]
    vol = prep["vol01"]
    ret = sorted(int(r) for r in ret)
    recon_l3b = np.array(vol, dtype=np.float32, copy=True)
    recon_l3c = np.array(vol, dtype=np.float32, copy=True)
    triples = triples_for(ret, D, targets)
    scored = [d for (_, _, d) in triples]
    for i in range(0, len(triples), batch_size):
        chunk = triples[i:i + batch_size]
        ic, ctrl, Ts = [], [], []
        for (left, right, d) in chunk:
            c = build_conditioning(prep, left, right, d, target=target, debug=False)
            ic.append(c["image_cond"]); ctrl.append(c["control"]); Ts.append(c["T"])
        ic = torch.stack(ic); ctrl = torch.stack(ctrl); Ts = torch.stack(Ts)
        cond = build_model_cond(model, ic, Ts, ctrl, uncond=False)
        uc = build_model_cond(model, ic, Ts, ctrl, uncond=True)
        z_interp = encode_linear_latent(model, ic)
        _, r_pred = ddim_sample_residual(
            model, cond, uc, z_interp,
            ddim_steps=sampler_cfg["ddim_steps"],
            resid_init=sampler_cfg.get("resid_init", "zero"),
            resid_strength=sampler_cfg.get("resid_strength", 0.4),
            control_scale=sampler_cfg.get("control_scale", 1.0),
            guidance_scale=sampler_cfg.get("guidance_scale", 1.0),
            seed=seed + i)
        out = compose_l3c_hard_anchor(model, ic, z_interp, r_pred)
        l3b = out["l3b"].cpu().numpy(); l3c = out["l3c"].cpu().numpy()
        for j, (left, right, d) in enumerate(chunk):
            recon_l3b[:, :, d] = unpad_hw(l3b[j], pad)
            recon_l3c[:, :, d] = unpad_hw(l3c[j], pad)
    return recon_l3b, recon_l3c, scored, ret


# --------------------------------------------------------------------------- #
#  L1 legacy X-Diffusion (checkpoint-compatible, explicit ret)
# --------------------------------------------------------------------------- #
def _resize_240(g256, native_hw=(240, 240)):
    t = torch.from_numpy(g256)[None, None].float()
    t = torch.nn.functional.interpolate(t, size=native_hw, mode="bilinear",
                                        align_corners=False)
    return t[0, 0].numpy()


@torch.no_grad()
def reconstruct_legacy_l1_nu(model, prep, ret, sampler_cfg, seed=0, targets=None,
                             native_hw=None):
    """L1 legacy independent recon with explicit ret. anchor = NEAREST acquired
    neighbour, hint = right neighbour (checkpoint-honest; L1 never saw a gap
    channel). Output remapped to metric space from the two acquired neighbours."""
    prep = FL.prepare_legacy(prep)
    D = int(prep["depth"]); vol = prep["vol01"]
    native_hw = native_hw or vol.shape[:2]
    ret = sorted(int(r) for r in ret)
    recon = np.array(vol, dtype=np.float32, copy=True)
    miss = missing_indices(ret, D) if targets is None else sorted(int(t) for t in targets)
    for d in miss:
        left, right = neighbours(ret, d)
        anchor_idx = left if (d - left) <= (right - d) else right
        g256 = FL._sample_legacy_slice(model, vol[:, :, anchor_idx], vol[:, :, right],
                                       d, D, sampler_cfg, use_control=False, seed=seed + d)
        g240 = _resize_240(g256, native_hw)
        recon[:, :, d] = FL._remap_to_metric(g240, d, left, right, prep)
    return recon, miss, ret


# --------------------------------------------------------------------------- #
#  L2 legacy recursive ControlNet (explicit ret; per-interval recursion +
#  full instrumentation: generation order, recursion level, condition provenance)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct_legacy_l2_nu(model, prep, ret, sampler_cfg, seed=0,
                             gen_budget=None, native_hw=None, only_contains=None):
    """L2 recursive bisection over NON-UNIFORM intervals between real retained
    slices. Each interval [ret[i], ret[i+1]] is filled independently by midpoint
    bisection d=(l+r)//2 using the CURRENT known neighbours (which may be
    previously-generated slices -> genuine recursive error propagation).

    Instrumentation (one row per generated slice): interval index, recursion level,
    global generation order, and whether the left/right condition was `acquired`
    or previously `generated`. To bound the smoke budget, intervals are processed
    LONGEST-FIRST and generation stops once `gen_budget` slices are produced, but an
    interval already started is finished (recursion stays complete within it).

    Returns (recon01, gen_rows, ret) where gen_rows is a list of dicts.
    """
    prep = FL.prepare_legacy(prep)
    D = int(prep["depth"]); vol = prep["vol01"]
    native_hw = native_hw or vol.shape[:2]
    ret = sorted(int(r) for r in ret)
    recon = np.array(vol, dtype=np.float32, copy=True)
    known = {r: vol[:, :, r].copy() for r in ret}
    provenance = {r: "acquired" for r in ret}      # index -> acquired/generated
    gen_rows = []
    order = [0]

    intervals = [(ret[i], ret[i + 1], i) for i in range(len(ret) - 1)]
    if only_contains is not None:
        # restrict to the single interval bracketing `only_contains` (fast panels;
        # recursion within that interval is unchanged / faithful)
        intervals = [t for t in intervals if t[0] < only_contains < t[1]]
    # longest interval first (deepest recursion / most propagation), stable by left idx
    intervals.sort(key=lambda t: (-(t[1] - t[0]), t[0]))

    def budget_left():
        return gen_budget is None or len(gen_rows) < gen_budget

    def fill(left, right, interval_idx, level):
        if right - left <= 1 or not budget_left():
            return
        d = (left + right) // 2
        g256 = FL._sample_legacy_slice(model, known[left], known[right], d, D,
                                       sampler_cfg, use_control=True, seed=seed + d)
        g240 = _resize_240(g256, native_hw)
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

    for (l, r, idx) in intervals:
        if not budget_left():
            break
        fill(l, r, idx, 0)
    scored = sorted(row["d"] for row in gen_rows)
    return recon, gen_rows, ret, scored


# --------------------------------------------------------------------------- #
#  SSA geometry restoration with EXPLICIT retained indices (no leakage)
# --------------------------------------------------------------------------- #
def restore_to_original_nu(paths, recon_cropped01, original_t2f_path, ret_cropped,
                           prep, output_path, hardcopy=True):
    """Non-uniform analogue of restoration.restore_to_original. Inverts
    norm+pad+crop identically, but hard-copies EXACTLY the explicit retained slices
    (ret_cropped + depth_start) rather than the regular retained_indices(depth,R).
    This prevents GT leakage / mis-hard-copy under non-uniform masks."""
    import nibabel as nib
    recon = np.asarray(recon_cropped01, dtype=np.float32)
    top, bottom, left, right = prep["pad"]; tgt = prep["target"]
    if recon.shape[0] == tgt and recon.shape[1] == tgt:
        recon = recon[top:tgt - bottom, left:tgt - right, :]
    img = nib.load(original_t2f_path)
    raw = np.asarray(img.dataobj, dtype=np.float32)
    orig_shape = tuple(int(x) for x in raw.shape)
    Hc, Wc, Dc = recon.shape
    assert (Hc, Wc) == orig_shape[:2], f"in-plane {(Hc,Wc)} != {orig_shape[:2]}"
    assert Dc == prep["depth"], f"cropped depth {Dc} != {prep['depth']}"
    lo, hi = float(prep["lo"]), float(prep["hi"])
    fg = recon > 0.0
    inv = np.where(fg, recon * (hi - lo) + lo, 0.0).astype(np.float32)
    ds = int(prep["depth_start"])
    out = np.zeros(orig_shape, dtype=np.float32)
    for z in range(orig_shape[2]):
        if not (ds <= z < ds + Dc):
            out[:, :, z] = raw[:, :, z]
    out[:, :, ds:ds + Dc] = inv
    obs_orig = [int(c) + ds for c in sorted(ret_cropped)]
    if hardcopy:
        for oz in obs_orig:
            out[:, :, oz] = raw[:, :, oz]
    hdr = img.header.copy(); hdr.set_data_dtype(np.float32)
    new = nib.Nifti1Image(out, img.affine, hdr)
    new.set_qform(img.affine, code=int(img.header["qform_code"]))
    new.set_sform(img.affine, code=int(img.header["sform_code"]))
    tmp = str(output_path) + ".tmp.nii.gz"
    nib.save(new, tmp); os.replace(tmp, output_path)
    return dict(observed_original=obs_orig, output_path=str(output_path), lo=lo, hi=hi)


def check_observed_hardcopy_nu(original_t2f_path, restored_path, ret_cropped, prep):
    """Verify the explicit non-uniform observed slices are bit-identical after
    restoration (analogue of restoration.check_observed_hardcopy)."""
    import nibabel as nib
    raw = np.asarray(nib.load(original_t2f_path).dataobj, dtype=np.float32)
    rarr = np.asarray(nib.load(restored_path).dataobj, dtype=np.float32)
    ds = int(prep["depth_start"])
    obs = [int(c) + ds for c in sorted(ret_cropped)]
    diffs = [float(np.abs(rarr[:, :, oz] - raw[:, :, oz]).max()) for oz in obs]
    maxabs = max(diffs) if diffs else 0.0
    return dict(observed_original=obs, max_abs_diff=maxabs, passed=bool(maxabs <= 1e-4))
