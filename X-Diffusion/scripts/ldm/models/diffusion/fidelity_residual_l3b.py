"""
fidelity_residual_l3b.py
------------------------
L3B — Interpolation-Anchored Residual Diffusion.

A NON-DESTRUCTIVE branch of L3 (the BraTS-adapted X-Diffusion backbone). It does
NOT modify L3/L4/L5 code. It reuses the exact shared frozen protocol:

  * same preprocessing / axis / crop / pad (brats_preprocessing_fidelity)
  * same conditioning tensors, train==inference (bratsloader build_conditioning)
  * same fixed retained-slice degradation + hard-copy (fidelity_sampling)
  * same model-independent PSNR validation metric (fidelity_runtime / volume_metrics)
  * same frozen Stable-Diffusion VAE (verified by hash before/after each stage)

The ONE scientific change vs L3 (see L3B_RESIDUAL_DIFFUSION_AUDIT.md):

  L3   diffusion target = eps on the FULL target-slice latent  z_tgt
       reconstruction   = decode(denoise(...))
  L3B  diffusion target = eps on the RESIDUAL latent  r = z_tgt - z_interp
       reconstruction   = decode( z_interp + r_pred )  -> clip -> hard-copy

Linear interpolation therefore becomes a DETERMINISTIC first-stage reconstruction
and diffusion learns only the correction. Where interpolation is already correct
the residual is ~0, so the model cannot drift far from the strong interp anchor.

`z_interp` here is the CLEAN interpolation latent  scale_factor * encode(linear).mode()
— identical to `fidelity_sampling.encode_linear_latent`, so training and sampling
use the same anchor. The interp latent is ALSO the c_concat conditioning fed to the
U-Net (in_channels = 4 residual + 4 interp = 8), so the network sees the anchor it
is correcting.
"""

import numpy as np
import torch

from ldm.models.diffusion.ddpm import LatentDiffusion, DiagonalGaussianDistribution  # noqa: F401
from ldm.data.brats_preprocessing_fidelity import unpad_hw
from ldm.data.bratsloader_controlnet_fidelity import build_conditioning
from ldm.models.diffusion.fidelity_sampling import (
    retained_indices, _img_to_gray01, build_model_cond, encode_linear_latent,
)


# ==========================================================================
#  Model: residual-target Latent Diffusion
# ==========================================================================
class LatentDiffusionResidual(LatentDiffusion):
    """LatentDiffusion whose diffusion x_start is the residual latent.

    Overriding ONLY get_input keeps forward / p_losses / q_sample / apply_model /
    the eps parameterization and the whole X-Diffusion hybrid conditioning
    (c_concat interp latent + CLIP/pose crossattn) exactly as in L3. The single
    difference is that the quantity being noised & predicted is
        r = scale_factor * ( encode(target).mode() - encode(linear).mode() )
    a clean, deterministic residual (mode, no VAE-posterior sampling noise).
    """

    def _raw_bchw(self, batch, key):
        """Base-DDPM style raw tensor fetch: (b,H,W,C)->(b,C,H,W) float on device."""
        x = batch[key]
        if x.dim() == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).contiguous().float().to(self.device)
        return x

    @torch.no_grad()
    def _clean_latent(self, x_bchw):
        return self.scale_factor * self.encode_first_stage(x_bchw).mode()

    def get_input(self, batch, k, **kwargs):
        out = super().get_input(batch, k, **kwargs)
        cond = out[1]
        # Clean, deterministic residual anchor (mode) for BOTH target and interp.
        x_tgt = self._raw_bchw(batch, self.first_stage_key)
        x_int = self._raw_bchw(batch, self.cond_stage_key)  # image_cond == linear
        z_tgt = self._clean_latent(x_tgt)
        z_int = self._clean_latent(x_int)
        r = (z_tgt - z_int).detach()
        out = [r, cond] + list(out[2:])
        return out


# ==========================================================================
#  Residual DDIM sampler  (eta=0, deterministic)
# ==========================================================================
@torch.no_grad()
def ddim_sample_residual(model, cond, uc_cond, z_interp, ddim_steps=50,
                         resid_init="zero", resid_strength=0.4,
                         control_scale=1.0, guidance_scale=1.0, seed=0, device=None):
    """Sample the RESIDUAL latent, then compose reconstruction = decode(z_interp + r).

    resid_init:
      "zero"  -> start from the zero-residual noised at t_start = resid_strength*T
                 (x_start = 0), denoise the last substeps. The interpolation is the
                 anchor added back, so a weak/failed correction collapses to interp.
      "noise" -> full pure-noise DDIM over the residual (standard diffusion).
    Returns decoded composed images in [-1,1] (b,3,H,W).
    """
    device = device or model.device
    acp = model.alphas_cumprod.to(device)
    ntime = model.num_timesteps
    ddim_ts = np.unique(np.linspace(0, ntime - 1, ddim_steps, dtype=int))

    if hasattr(model, "control_scales"):
        model.control_scales = [control_scale] * len(model.control_scales)

    b = cond["c_crossattn"][0].shape[0]
    g = torch.Generator(device=device).manual_seed(int(seed))
    shape = (b, model.channels, model.image_size, model.image_size)

    if resid_init == "zero" and resid_strength < 1.0:
        start_i = int(round(resid_strength * (len(ddim_ts) - 1)))
        start_i = max(1, min(len(ddim_ts) - 1, start_i))
        t_start = int(ddim_ts[start_i])
        noise = torch.randn(shape, generator=g, device=device)
        a_ts = acp[t_start]
        x = (1 - a_ts).sqrt() * noise           # x_start (residual) == 0
        ts_used = ddim_ts[: start_i + 1]
    else:
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
        r0 = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()        # predicted residual
        a_prev = acp[int(ts_used[i - 1])] if i > 0 else torch.tensor(1.0, device=device)
        x = a_prev.sqrt() * r0 + (1 - a_prev).sqrt() * eps    # eta=0
    r_pred = x

    z_recon = z_interp + r_pred                                # anchor + correction
    img = model.decode_first_stage(z_recon)
    return img.clamp(-1, 1), r_pred


# ==========================================================================
#  Independent per-slice volume reconstruction (residual composition)
# ==========================================================================
@torch.no_grad()
def reconstruct_volume_residual(model, prep, R, sampler_cfg, batch_size=16, seed=0,
                                max_missing=None, subsample_seed=0):
    """L3B reconstruction. reconstruction = interp + residual, retained hard-copied.

    Mirrors fidelity_sampling.reconstruct_volume exactly (same retained_indices,
    same unpad metric space, same hard-copy) — only the per-slice generator differs.
    Returns (recon_hw_d in [0,1] unpadded, generated_indices, retained_indices).
    """
    depth = prep["depth"]
    ret = retained_indices(depth, R)
    ret_set = set(ret)
    pad = prep["pad"]
    target = prep["target"]
    vol = prep["vol01"]
    recon = np.array(vol, dtype=np.float32, copy=True)   # retained slices hard-copied

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
            c = build_conditioning(prep, left, right, d, target=target, debug=False)
            ic.append(c["image_cond"]); ctrl.append(c["control"]); Ts.append(c["T"])
        ic = torch.stack(ic); ctrl = torch.stack(ctrl); Ts = torch.stack(Ts)
        cond = build_model_cond(model, ic, Ts, ctrl, uncond=False)
        uc = build_model_cond(model, ic, Ts, ctrl, uncond=True)
        z_interp = encode_linear_latent(model, ic)           # scale*encode(linear).mode()
        img, _ = ddim_sample_residual(
            model, cond, uc, z_interp,
            ddim_steps=sampler_cfg["ddim_steps"],
            resid_init=sampler_cfg.get("resid_init", "zero"),
            resid_strength=sampler_cfg.get("resid_strength", 0.4),
            control_scale=sampler_cfg.get("control_scale", 1.0),
            guidance_scale=sampler_cfg.get("guidance_scale", 1.0),
            seed=seed + i,
        )
        gray = _img_to_gray01(img).cpu().numpy()
        for j, (left, right, d) in enumerate(chunk):
            recon[:, :, d] = unpad_hw(gray[j], pad)
    return recon, missing, ret
