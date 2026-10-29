"""
ir1_audit_common.py
===================
Shared, INFERENCE-ONLY helpers for the IR1 SSIM audit (experiment_id IR1_SSIM_AUDIT) and
the post-hoc beta sweep. Nothing here trains, and nothing here writes into the original
IR1 run directory.

Provides:
  * fixed loading of the original IR1 run (config, checkpoints, eval grid, subjects);
  * a single inference pass per (subject, R, family) that returns x_linear, the bounded
    residual field (already multiplied by the missing mask) and the composed IR1 output,
    so every downstream diagnostic (beta sweep, residual stats, SSIM variants) reuses ONE
    forward pass and can never drift between methods;
  * SSIM definition variants A/B/C/D (§7.4) with the project-standard A as authoritative;
  * hashing utilities for the matched-evaluation manifest.
"""
from __future__ import annotations
import os, sys, json, math, time, hashlib
import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(PROJECT, "X-Diffusion", "scripts")
for p in (SCRIPTS, PROJECT):
    if p not in sys.path:
        sys.path.insert(0, p)

PEAK = 1.0
IR1_RUN = os.path.join(PROJECT, "runs", "brats_ir1_image_residual_smoke_20260728_121036")


# --------------------------------------------------------------------------- #
#  hashing
# --------------------------------------------------------------------------- #
def sha_arr(a) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha_idx(idx) -> str:
    return hashlib.sha256(np.asarray(sorted(int(i) for i in idx), dtype="<i8").tobytes()).hexdigest()


def sha_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_state_dict(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode())
        v = sd[k]
        try:
            h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
        except Exception:
            h.update(str(v).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  SSIM definitions (§7)
# --------------------------------------------------------------------------- #
from skimage.metrics import structural_similarity as _sk_ssim

SSIM_WIN = 7          # skimage default for 2-D float input
SSIM_PAD = (SSIM_WIN - 1) // 2
SSIM_K1, SSIM_K2 = 0.01, 0.03
MIN_SLICE_FG_VOXELS = 50


def ssim_A_fullslice(g, p, data_range=PEAK):
    """A — CURRENT PROJECT-STANDARD definition: skimage structural_similarity on the whole
    slice, uniform 7x7 window, K1=0.01, K2=0.03, sample covariance, fixed data_range."""
    return float(_sk_ssim(g, p, data_range=data_range))


def brain_bbox(mask2d):
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def ssim_B_bbox(g, p, data_range=PEAK, pad=4):
    """B — SSIM on the brain bounding-box crop (diagnostic). Crop is derived from the
    GROUND TRUTH brain mask only, so it is identical for every method."""
    bb = brain_bbox(g > 0)
    if bb is None:
        return float("nan")
    y0, y1, x0, x1 = bb
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(g.shape[0], y1 + pad); x1 = min(g.shape[1], x1 + pad)
    gc, pc = g[y0:y1, x0:x1], p[y0:y1, x0:x1]
    if min(gc.shape) < SSIM_WIN:
        return float("nan")
    return float(_sk_ssim(gc, pc, data_range=data_range))


def ssim_C_valid_map(g, p, data_range=PEAK):
    """C — local SSIM map averaged ONLY over window centres whose whole 7x7 support lies
    inside the brain (diagnostic). No image is multiplied by a mask, so no artificial
    zero boundary is created."""
    from scipy.ndimage import minimum_filter
    _, S = _sk_ssim(g, p, data_range=data_range, full=True)
    brain = (g > 0).astype(np.float32)
    supported = minimum_filter(brain, size=SSIM_WIN, mode="constant", cval=0.0) > 0.5
    valid = np.zeros_like(supported)
    valid[SSIM_PAD:-SSIM_PAD or None, SSIM_PAD:-SSIM_PAD or None] = True
    sel = supported & valid
    if sel.sum() < 50:
        return float("nan")
    return float(S[sel].mean())


def ssim_D_premasked(g, p, data_range=PEAK):
    """D — multiply BOTH images by the binary brain mask, then full-slice SSIM
    (diagnostic; demonstrates the artificial zero-boundary effect warned about in §7.4)."""
    m = (g > 0).astype(np.float32)
    return float(_sk_ssim(g * m, p * m, data_range=data_range))


SSIM_VARIANTS = {
    "A_fullslice_current": ssim_A_fullslice,
    "B_brain_bbox": ssim_B_bbox,
    "C_valid_window_centres": ssim_C_valid_map,
    "D_premasked_fullslice": ssim_D_premasked,
}


def ssim_bundle(pred_hw_d, gt_hw_d, evaluated_indices, variants=("A_fullslice_current",)):
    """Per-slice + mean SSIM for the requested definitions over exactly the evaluated
    slices, skipping slices with < MIN_SLICE_FG_VOXELS brain voxels (project rule)."""
    md = sorted(int(i) for i in evaluated_indices)
    per = {v: [] for v in variants}
    used = []
    for z in md:
        g = gt_hw_d[:, :, z].astype(np.float32)
        p = pred_hw_d[:, :, z].astype(np.float32)
        if int((g > 0).sum()) < MIN_SLICE_FG_VOXELS:
            continue
        used.append(z)
        for v in variants:
            per[v].append(SSIM_VARIANTS[v](g, p))
    out = {v: (float(np.nanmean(per[v])) if per[v] else float("nan")) for v in variants}
    return out, per, used


# --------------------------------------------------------------------------- #
#  extra structural metrics
# --------------------------------------------------------------------------- #
def grad_mag(a):
    from ldm.models.diffusion.fidelity_metrics import _sobel_grad_mag
    return _sobel_grad_mag(a)


def laplacian(a):
    from ldm.models.diffusion.fidelity_metrics import _laplacian
    return _laplacian(a)


def slice_metrics(pred_hw_d, gt_hw_d, z):
    """Per-slice PSNR/MSE/MAE (masked + unmasked) + grad/laplacian MAE."""
    g = gt_hw_d[:, :, z].astype(np.float32)
    p = pred_hw_d[:, :, z].astype(np.float32)
    m = g > 0
    d2 = (p - g) ** 2
    out = dict(n_fg=int(m.sum()), mse_unmasked=float(d2.mean()))
    if m.sum() >= MIN_SLICE_FG_VOXELS:
        mse_m = float(d2[m].mean())
        out.update(mse_masked=mse_m,
                   psnr_masked=float(10 * math.log10(PEAK ** 2 / (mse_m + 1e-12))),
                   mae_masked=float(np.abs(p - g)[m].mean()),
                   grad_mae=float(np.abs(grad_mag(g) - grad_mag(p))[m].mean()),
                   laplacian_mae=float(np.abs(laplacian(g) - laplacian(p))[m].mean()))
    else:
        out.update(mse_masked=float("nan"), psnr_masked=float("nan"), mae_masked=float("nan"),
                   grad_mae=float("nan"), laplacian_mae=float("nan"))
    out["psnr_unmasked"] = float(10 * math.log10(PEAK ** 2 / (out["mse_unmasked"] + 1e-12)))
    return out


def volume_metrics(pred_hw_d, gt_hw_d, evaluated_indices, with_ssim=True,
                   variants=("A_fullslice_current",)):
    """Volume-level bundle matching nu1_corrected_eval.masked_metrics for PSNR/MSE/MAE
    (aggregate squared error over evaluated foreground voxels BEFORE dB) plus the SSIM
    variants and the slice-averaged structural metrics."""
    md = np.asarray(sorted(int(i) for i in evaluated_indices), dtype=int)
    pred = pred_hw_d[:, :, md].astype(np.float32)
    gt = gt_hw_d[:, :, md].astype(np.float32)
    mask = gt > 0
    d2 = (pred - gt) ** 2
    n_fg = int(mask.sum())
    out = dict(n_evaluated_slices=int(md.size), evaluated_foreground_voxel_count=n_fg)
    if n_fg < 1000:
        out.update({k: float("nan") for k in
                    ("psnr_masked", "psnr_unmasked", "mse_masked", "mse_unmasked", "nmse", "mae")})
        out["metric_defined"] = False
        return out
    mse_m = float(d2[mask].mean()); mse_u = float(d2.mean())
    out.update(mse_masked=mse_m, mse_unmasked=mse_u,
               psnr_masked=float(10 * np.log10(PEAK ** 2 / (mse_m + 1e-12))),
               psnr_unmasked=float(10 * np.log10(PEAK ** 2 / (mse_u + 1e-12))),
               nmse=float(d2[mask].sum() / (float((gt[mask] ** 2).sum()) + 1e-12)),
               mae=float(np.abs(pred - gt)[mask].mean()), metric_defined=True)
    if with_ssim:
        sb, per, used = ssim_bundle(pred_hw_d, gt_hw_d, md, variants=variants)
        for k, v in sb.items():
            out[f"ssim_{k}"] = v
        out["ssim"] = sb.get("A_fullslice_current", float("nan"))
        out["n_ssim_slices"] = len(used)
        gm, lm = [], []
        for z in used:
            g = gt_hw_d[:, :, z].astype(np.float32); p = pred_hw_d[:, :, z].astype(np.float32)
            m2 = g > 0
            gm.append(float(np.abs(grad_mag(g) - grad_mag(p))[m2].mean()))
            lm.append(float(np.abs(laplacian(g) - laplacian(p))[m2].mean()))
        out["grad_mae"] = float(np.mean(gm)) if gm else float("nan")
        out["laplacian_mae"] = float(np.mean(lm)) if lm else float("nan")
    return out


# --------------------------------------------------------------------------- #
#  original-run loading (read-only)
# --------------------------------------------------------------------------- #
def load_cfg(path=None):
    from omegaconf import OmegaConf
    return OmegaConf.load(path or os.path.join(PROJECT, "configs", "ir1_smoke.yaml"))


def eval_grid(cfg, D):
    """Reproduce EXACTLY the original IR1 GLI evaluation grid (stage_gli)."""
    import image_residual as IR
    fam_seed = json.load(open(os.path.join(IR1_RUN, "validation_matrix.json")))["family_seed"]
    grid = []
    for R in list(cfg.gli_eval_strides):
        for fam in list(cfg.gli_eval_families):
            if fam == "U0":
                ret = IR.masks.uniform_mask(D, R)
            else:
                fs = fam_seed.get(fam) or (int(cfg.gli_eval_family_seed) + hash(fam) % 1000)
                ret = IR.masks.nonuniform_mask(D, R, fam, int(fs))
            grid.append((int(R), fam, [int(x) for x in ret]))
    return grid, fam_seed


def eval_preps(cfg, D):
    """Load the 3 historical GLI validation subjects with the identical normalization."""
    from ldm.models.diffusion import fidelity_runtime as FR
    L3_RUN = os.path.join(PROJECT, str(cfg.l3_run))
    VAL_SPLIT = os.path.join(L3_RUN, "splits", "brats2023_val60.txt")
    lines = [l.strip() for l in open(VAL_SPLIT) if l.strip()]

    def subject_of(p): return os.path.basename(os.path.dirname(p))

    files = []
    for w in list(cfg.gli_eval_subjects):
        hit = [v for v in lines if subject_of(os.path.join(PROJECT, v)) == w
               or os.path.basename(v).startswith(w)]
        assert hit, f"{w} not found in val split"
        files.append(os.path.join(PROJECT, hit[0]))
    preps = FR.prepare_volumes(files, cfg.p_lo, cfg.p_hi, D)
    for i, p in enumerate(preps):
        p["subject"] = subject_of(files[i])
        p["src_file"] = files[i]
    return preps, files


def load_ckpt_model(name, device):
    """Load an ORIGINAL IR1 checkpoint read-only. name in {step0, best, last}."""
    import torch, image_residual as IR
    fn = dict(step0="step0_ir1_linear_identity.pt",
              best="best_ir1_macro_psnr.pt",
              last="last_ir1.pt")[name]
    path = os.path.join(IR1_RUN, "checkpoints", fn)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = IR.model.build_ir1(ck["residual_scale"], arch=ck.get("arch")).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck, path


# --------------------------------------------------------------------------- #
#  one forward pass per cell -> everything downstream reuses it
# --------------------------------------------------------------------------- #
def infer_cell(model, vol_hw_d, ret, D, device, roi=(64, 128, 128), overlap=0.25,
               mode="gaussian", sw_batch=2, direct=False):
    """Return dict with x_linear, delta (missing-masked bounded residual), x_ir1
    (= clip(x_lin+delta) with acquired hard-copy), x_sparse and raw network output stats.
    `direct=True` runs the whole volume through the net in one window (no tiling)."""
    import torch
    from monai.inferers import sliding_window_inference
    import image_residual as IR

    model.eval()
    x6, x_lin, known_hw_d, missing_hw_d = IR.inputs.build_input_channels(vol_hw_d, ret, D)
    x6_t = torch.from_numpy(x6)[None].to(device)
    torch.cuda.synchronize() if device == "cuda" else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        if direct:
            # a MONAI UNet with 3 stride-2 levels needs every spatial dim divisible by 8;
            # D=153 is not, so pad by replication and crop the raw output back.
            import torch.nn.functional as _F
            pads = []
            for n in (x6_t.shape[-1], x6_t.shape[-2], x6_t.shape[-3]):
                pads += [0, (-n) % 8]
            padded = _F.pad(x6_t, pads, mode="replicate") if any(pads) else x6_t
            raw = model.raw(padded)
            raw = raw[..., :x6_t.shape[-3], :x6_t.shape[-2], :x6_t.shape[-1]].contiguous()
        else:
            raw = sliding_window_inference(x6_t, roi, sw_batch, model.raw,
                                           overlap=overlap, mode=mode)
        delta = model.residual_scale * torch.tanh(raw)
    dt = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else float("nan")

    miss_t = torch.from_numpy(np.moveaxis(missing_hw_d, 2, 0))[None, None].to(device)
    delta_m = (miss_t * delta)[0, 0].cpu().numpy()                     # [D,H,W]
    raw_np = raw[0, 0].cpu().numpy()
    del raw, delta, miss_t, x6_t
    if device == "cuda":
        import torch as _t; _t.cuda.empty_cache()

    delta_hw_d = np.moveaxis(delta_m, 0, 2).astype(np.float32)          # (H,W,D)
    x_sparse = IR.inputs.sparse_volume(vol_hw_d, ret, D)
    known = np.zeros(D, bool)
    for r in ret:
        known[int(r)] = True
    x_ir1 = compose(x_lin, delta_hw_d, vol_hw_d, known, beta=1.0)
    return dict(x_linear=x_lin.astype(np.float32), delta=delta_hw_d, x_ir1=x_ir1,
                x_sparse=x_sparse.astype(np.float32), known=known,
                raw_hw_d=np.moveaxis(raw_np, 0, 2).astype(np.float32),
                runtime_s=dt, peak_gb=peak_gb,
                residual_scale=float(model.residual_scale.item()))


def compose(x_lin, delta_hw_d, vol_hw_d, known_bool, beta=1.0):
    """x_beta = clip(x_linear + beta * missing_mask * delta, 0, 1), then acquired
    hard-copy. `delta_hw_d` already carries the missing mask (zero on acquired slices),
    so the hard-copy is exact and the clip happens exactly once."""
    x = np.clip(x_lin + beta * delta_hw_d, 0.0, 1.0).astype(np.float32)
    x[:, :, known_bool] = vol_hw_d[:, :, known_bool]
    return x
