"""
fidelity_sampling.py
--------------------
Fidelity-oriented sampling + independent per-slice volume reconstruction + 3D
metrics for through-plane sparse-slice reconstruction (Phases 4, 7, 8, 9).

Key properties:
  * Interpolation-initialised DDIM (encode linear interp, add noise at a
    configurable start strength, denoise from there) OR pure-noise DDIM ablation.
  * eta=0 deterministic sampling with fixed seeds.
  * classifier-free guidance scale (default 1.0), configurable control scale.
  * INDEPENDENT per-missing-slice generation: no generated slice is ever used as
    a control for another (removes autoregressive error propagation, Phase 4).
  * Retained real slices are hard-copied back into the reconstruction.
  * 3D PSNR aggregates MSE over evaluated voxels BEFORE converting to dB.
"""

import numpy as np
import torch

from ldm.data.brats_preprocessing_fidelity import unpad_hw
from ldm.data.bratsloader_controlnet_fidelity import build_conditioning, STRIDES


# --------------------------------------------------------------------------
#  conditioning helpers (mirror LatentDiffusion.get_input WITHOUT the target)
# --------------------------------------------------------------------------
def _hwc_to_bchw(t):
    return t.permute(0, 3, 1, 2).contiguous().float()


@torch.no_grad()
def build_model_cond(model, image_cond_bhwc, T_b, control_bhwc, uncond=False):
    """Build the hybrid conditioning dict for the (Control)LDM from batch tensors.

    Replicates ddpm.LatentDiffusion.get_input's conditioning branch exactly:
      c_crossattn = cc_projection([CLIP(linear) | T])
      c_concat    = encode_first_stage(linear).mode()      (NOT scale_factor'd)
      c_control   = hint                                    (ControlNet only)
    """
    device = model.device
    xc = _hwc_to_bchw(image_cond_bhwc).to(device)
    T = T_b.to(device).float()
    if uncond:
        clip_emb = model.get_learned_conditioning([""]).repeat(xc.shape[0], 1, 1)
    else:
        clip_emb = model.get_learned_conditioning(xc)
    cc = model.cc_projection(torch.cat([clip_emb, T[:, None, :]], dim=-1))
    cond = {"c_crossattn": [cc]}
    concat = model.encode_first_stage(xc).mode()
    if uncond:
        concat = torch.zeros_like(concat)
    cond["c_concat"] = [concat]
    if control_bhwc is not None and hasattr(model, "control_model"):
        ctrl = _hwc_to_bchw(control_bhwc).to(device)
        cond["c_control"] = [ctrl]
    return cond


@torch.no_grad()
def encode_linear_latent(model, image_cond_bhwc):
    """z0 for the linear-interpolation image, scaled like the training target z."""
    xc = _hwc_to_bchw(image_cond_bhwc).to(model.device)
    return model.scale_factor * model.encode_first_stage(xc).mode()


# --------------------------------------------------------------------------
#  DDIM sampler (manual, eta=0), interpolation- or pure-noise-initialised
# --------------------------------------------------------------------------
@torch.no_grad()
def ddim_sample(model, cond, uc_cond, z0_interp=None, ddim_steps=50,
                interp_strength=0.25, control_scale=1.0, guidance_scale=1.0,
                seed=0, device=None):
    """Return decoded predicted images in [-1,1] (b, 3, H, W).

    If z0_interp is given and interp_strength<1, start from the noised linear
    latent at t_start = interp_strength * T; else start from pure noise.
    """
    device = device or model.device
    acp = model.alphas_cumprod.to(device)
    ntime = model.num_timesteps
    ddim_ts = np.linspace(0, ntime - 1, ddim_steps, dtype=int)
    ddim_ts = np.unique(ddim_ts)

    if hasattr(model, "control_scales"):
        model.control_scales = [control_scale] * len(model.control_scales)

    b = cond["c_crossattn"][0].shape[0]
    g = torch.Generator(device=device).manual_seed(int(seed))

    if z0_interp is not None and interp_strength < 1.0:
        start_i = int(round(interp_strength * (len(ddim_ts) - 1)))
        start_i = max(1, min(len(ddim_ts) - 1, start_i))
        t_start = int(ddim_ts[start_i])
        noise = torch.randn(z0_interp.shape, generator=g, device=device)
        a_ts = acp[t_start]
        x = a_ts.sqrt() * z0_interp + (1 - a_ts).sqrt() * noise
        ts_used = ddim_ts[: start_i + 1]
    else:
        shape = (b, model.channels, model.image_size, model.image_size)
        x = torch.randn(shape, generator=g, device=device)
        ts_used = ddim_ts

    def _eps(x_in, t_int):
        t = torch.full((b,), int(t_int), device=device, dtype=torch.long)
        e_c = model.apply_model(x_in, t, cond)
        if guidance_scale != 1.0:
            e_u = model.apply_model(x_in, t, uc_cond)
            return e_u + guidance_scale * (e_c - e_u)
        return e_c

    for i in reversed(range(len(ts_used))):
        t_int = int(ts_used[i])
        eps = _eps(x, t_int)
        a_t = acp[t_int]
        x0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
        if i > 0:
            t_prev = int(ts_used[i - 1])
            a_prev = acp[t_prev]
        else:
            a_prev = torch.tensor(1.0, device=device)
        x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps  # eta=0

    img = model.decode_first_stage(x)
    return img.clamp(-1, 1)


def _img_to_gray01(img_bchw):
    """(b,3,H,W) in [-1,1] -> (b,H,W) gray in [0,1]."""
    g = img_bchw.mean(dim=1)
    return ((g + 1.0) / 2.0).clamp(0, 1)


# --------------------------------------------------------------------------
#  independent per-slice volume reconstruction
# --------------------------------------------------------------------------
def retained_indices(depth, R):
    idx = list(range(0, depth, R))
    if idx[-1] != depth - 1:
        idx.append(depth - 1)
    return idx


@torch.no_grad()
def reconstruct_volume(model, prep, R, sampler_cfg, batch_size=16, seed=0,
                       method="interp", max_missing=None, subsample_seed=0):
    """Reconstruct missing slices at stride R, INDEPENDENTLY.

    Returns (recon_hw_d in [0,1] unpadded, generated_indices, retained_indices).
    Every retained slice is hard-copied from ground truth. Only generated_indices
    are actually sampled (and should be used for metrics); if max_missing is set,
    a random subset of that size is generated (fast validation).
    """
    depth = prep["depth"]
    ret = retained_indices(depth, R)
    ret_set = set(ret)
    pad = prep["pad"]
    target = prep["target"]
    vol = prep["vol01"]  # (H,W,D) UNPADDED [0,1] ground truth (metric space)

    # Reconstruction lives in UNPADDED metric space. Retained slices are the
    # hard-copied GT; missing slices are filled by the model (unpadded from 256).
    recon = np.array(vol, dtype=np.float32, copy=True)

    missing = [d for d in range(depth) if d not in ret_set]
    if max_missing is not None and max_missing < len(missing):
        rng = np.random.RandomState(subsample_seed)
        missing = sorted(rng.choice(missing, size=max_missing, replace=False).tolist())
    triples = []
    for d in missing:
        left = max([r for r in ret if r < d])
        right = min([r for r in ret if r > d])
        triples.append((left, right, d))

    for i in range(0, len(triples), batch_size):
        chunk = triples[i:i + batch_size]
        ic, ctrl, Ts = [], [], []
        for (left, right, d) in chunk:
            # model works on padded target (256); gap (right-left) == R
            c = build_conditioning(prep, left, right, d, target=target, debug=False)
            ic.append(c["image_cond"]); ctrl.append(c["control"]); Ts.append(c["T"])
        ic = torch.stack(ic); ctrl = torch.stack(ctrl); Ts = torch.stack(Ts)
        cond = build_model_cond(model, ic, Ts, ctrl, uncond=False)
        uc = build_model_cond(model, ic, Ts, ctrl, uncond=True)
        z0 = None
        if method == "interp":
            z0 = encode_linear_latent(model, ic)
        img = ddim_sample(
            model, cond, uc, z0_interp=z0,
            ddim_steps=sampler_cfg["ddim_steps"],
            interp_strength=sampler_cfg["interp_strength"] if method == "interp" else 1.0,
            control_scale=sampler_cfg.get("control_scale", 1.0),
            guidance_scale=sampler_cfg.get("guidance_scale", 1.0),
            seed=seed + i,
        )
        gray = _img_to_gray01(img).cpu().numpy()  # (b, 256, 256)
        for j, (left, right, d) in enumerate(chunk):
            recon[:, :, d] = unpad_hw(gray[j], pad)  # 256 -> 240 metric space

    return recon, missing, ret


@torch.no_grad()
def reconstruct_indices(model, prep, R, indices, sampler_cfg, seed=0, method="interp"):
    """Generate ONLY the given missing `indices` (bracketed by retained neighbours).

    Returns {d: unpadded [0,1] slice}. Used for fast qualitative panels without
    reconstructing the whole volume.
    """
    ret = retained_indices(prep["depth"], R)
    pad = prep["pad"]; target = prep["target"]
    ic, ctrl, Ts, ds = [], [], [], []
    for d in indices:
        left = max([r for r in ret if r < d]); right = min([r for r in ret if r > d])
        c = build_conditioning(prep, left, right, d, target=target, debug=False)
        ic.append(c["image_cond"]); ctrl.append(c["control"]); Ts.append(c["T"]); ds.append(d)
    ic = torch.stack(ic); ctrl = torch.stack(ctrl); Ts = torch.stack(Ts)
    cond = build_model_cond(model, ic, Ts, ctrl, uncond=False)
    uc = build_model_cond(model, ic, Ts, ctrl, uncond=True)
    z0 = encode_linear_latent(model, ic) if method == "interp" else None
    img = ddim_sample(model, cond, uc, z0_interp=z0,
                      ddim_steps=sampler_cfg["ddim_steps"],
                      interp_strength=sampler_cfg["interp_strength"] if method == "interp" else 1.0,
                      control_scale=sampler_cfg.get("control_scale", 1.0),
                      guidance_scale=sampler_cfg.get("guidance_scale", 1.0), seed=seed)
    gray = _img_to_gray01(img).cpu().numpy()
    return {d: unpad_hw(gray[j], pad) for j, d in enumerate(ds)}


# --------------------------------------------------------------------------
#  metrics (per subject). MSE aggregated over voxels BEFORE dB conversion.
# --------------------------------------------------------------------------
try:
    from skimage.metrics import structural_similarity as _ssim
except Exception:
    _ssim = None


def volume_metrics(recon_hw_d, gt_hw_d, missing_indices, peak=1.0):
    """Per-subject metrics over the MISSING slices only.

    recon_hw_d, gt_hw_d : unpadded [0,1] volumes (H,W,D).
    Returns a dict of scalars.
    """
    md = np.array(missing_indices, dtype=int)
    pred = recon_hw_d[:, :, md]
    gt = gt_hw_d[:, :, md]
    mask = gt > 0.0  # per-volume foreground

    diff2 = (pred - gt) ** 2
    # masked (foreground) 3D PSNR — aggregate MSE first
    mse_m = diff2[mask].mean() if mask.any() else diff2.mean()
    psnr_masked = 10.0 * np.log10((peak ** 2) / (mse_m + 1e-12))
    # unmasked 3D PSNR
    mse_u = diff2.mean()
    psnr_unmasked = 10.0 * np.log10((peak ** 2) / (mse_u + 1e-12))
    # NMSE (masked)
    denom = (gt[mask] ** 2).sum() if mask.any() else (gt ** 2).sum()
    nmse = float(diff2[mask].sum() / (denom + 1e-12)) if mask.any() else float(diff2.sum() / (denom + 1e-12))
    # MAE (masked)
    mae = float(np.abs(pred - gt)[mask].mean()) if mask.any() else float(np.abs(pred - gt).mean())
    # axial SSIM: mean per missing slice
    ssims = []
    if _ssim is not None:
        for k in range(pred.shape[2]):
            g = gt[:, :, k]; p = pred[:, :, k]
            if g.max() > 0:
                ssims.append(_ssim(g, p, data_range=peak))
    ssim = float(np.mean(ssims)) if ssims else float("nan")

    return {
        "psnr_masked": float(psnr_masked),
        "psnr_unmasked": float(psnr_unmasked),
        "nmse": nmse,
        "mae": mae,
        "ssim": ssim,
        "n_missing_slices": int(len(md)),
    }


# --------------------------------------------------------------------------
#  interpolation baselines (need no model)
# --------------------------------------------------------------------------
def interpolate_volume(prep, R, kind="linear"):
    """Fill missing slices by nearest/linear interpolation of retained slices."""
    # Interpolation baselines operate directly in UNPADDED metric space.
    depth = prep["depth"]
    vol = prep["vol01"]
    ret = retained_indices(depth, R)
    ret_set = set(ret)
    recon = np.array(vol, dtype=np.float32, copy=True)
    missing = [d for d in range(depth) if d not in ret_set]
    for d in missing:
        left = max([r for r in ret if r < d])
        right = min([r for r in ret if r > d])
        if kind == "nearest":
            src = left if (d - left) <= (right - d) else right
            recon[:, :, d] = vol[:, :, src]
        else:  # linear
            a = (d - left) / (right - left)
            recon[:, :, d] = (1 - a) * vol[:, :, left] + a * vol[:, :, right]
    return recon, missing, ret


def cubic_interpolate_volume(prep, R):
    """Cubic (spline) interpolation along the through-plane axis."""
    from scipy.interpolate import interp1d
    depth = prep["depth"]
    vol = prep["vol01"]
    ret = retained_indices(depth, R)
    ret_set = set(ret)
    recon = np.array(vol, dtype=np.float32, copy=True)
    missing = [d for d in range(depth) if d not in ret_set]
    kind = "cubic" if len(ret) >= 4 else "linear"
    xs = np.array(ret)
    ys = vol[:, :, xs]  # (H,W,len(ret))
    H, W, _ = ys.shape
    f = interp1d(xs, ys.reshape(H * W, -1), kind=kind, axis=1,
                 bounds_error=False, fill_value="extrapolate")
    for d in missing:
        recon[:, :, d] = np.clip(f(d).reshape(H, W), 0.0, 1.0)
    return recon, missing, ret


