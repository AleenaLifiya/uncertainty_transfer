"""
fidelity_residual_l3c.py
------------------------
L3C — Hard Image-Space Interpolation Anchor with Learned Diffusion Correction.

A NON-DESTRUCTIVE branch of L3B. It does NOT modify/rename L3, L3B, L4 or L5. It
reuses the L3B residual predictor (`LatentDiffusionResidual`), the L3B residual
DDIM sampler (`ddim_sample_residual`), the identical frozen SD VAE, conditioning,
noise schedule, residual scaling and warm-start. See `L3C_HARD_ANCHOR_AUDIT.md`.

The ONE scientific change vs L3B is the final OUTPUT COMPOSITION:

  L3B  x = decode(z_interp + r_pred)                                  (VAE round-trip
                                                                       of interp when r=0)
  L3C  x = x_linear + decode(z_interp + r_pred) − decode(z_interp)    (DIRECT interp
                                                                       when r=0, exact)

where `x_linear` is the DIRECT image-space linear interpolation the model was
conditioned on (gray01 of image_cond), NOT its VAE round-trip. The frozen VAE can
therefore no longer degrade the interpolation anchor: the model contributes ONLY the
decoder-space difference on the MISSING slices. When the predicted residual is zero
the correction is exactly zero and L3C returns direct linear interpolation.

Guarantee (proven by construction, tested in preflight):
    r_pred == 0  ⇒  delta_image == 0  ⇒  x_pred == x_linear (direct interp)
This is a FLOOR on the anchor, not a proof of improvement — a nonzero learned
correction may still reduce quality.
"""

import numpy as np
import torch

from ldm.data.brats_preprocessing_fidelity import unpad_hw
from ldm.data.bratsloader_controlnet_fidelity import build_conditioning
from ldm.models.diffusion.fidelity_sampling import (
    retained_indices, _img_to_gray01, _hwc_to_bchw, build_model_cond,
    encode_linear_latent,
)
from ldm.models.diffusion.fidelity_residual_l3b import (
    LatentDiffusionResidual, ddim_sample_residual,
)


# ==========================================================================
#  Model: identical training objective to L3B, tagged distinctly for L3C
# ==========================================================================
class LatentDiffusionHardAnchor(LatentDiffusionResidual):
    """L3C training model.

    Training is byte-for-byte the L3B residual objective (get_input returns the
    residual latent r = scale*(enc(target).mode − enc(interp).mode); eps on r). The
    L3C difference is purely at reconstruction/inference time (`compose_l3c_hard_anchor`
    / `reconstruct_volume_l3c`), so no method is overridden here. The distinct class +
    attribute exist so `l3c_is_not_l3b_composition` / `l3c_is_not_l3` can assert
    identity, and so the config target is unambiguous.
    """

    composition = "hard_image_anchor"   # vs L3B's implicit "latent_residual"


# ==========================================================================
#  Core composition helper (independently unit-testable)
# ==========================================================================
@torch.no_grad()
def compose_l3c_hard_anchor(model, image_cond_bhwc, z_interp, residual_latent):
    """Compose the L3C hard-anchored reconstruction for a batch of slices.

    All returned images are gray01 (b, H, W) in [0,1], in the model's padded space
    (unpadding + volume hard-copy are handled by `reconstruct_volume_l3c`).

    Parameters
    ----------
    model            : the (frozen-VAE) diffusion model providing decode_first_stage.
    image_cond_bhwc  : (b,H,W,3) in [-1,1] — the DIRECT linear interpolation the model
                       is conditioned on. gray01(image_cond) == x_linear exactly.
    z_interp         : (b,4,h,w) scaled interp latent = scale*encode(linear).mode().
    residual_latent  : (b,4,h,w) predicted latent correction r_pred (may be 0).

    Returns
    -------
    dict of gray01 (b,H,W) tensors:
      linear     : x_linear                          (direct interp)
      interp_vae : decode(z_interp)                  (VAE round-trip of interp)
      l3b        : decode(z_interp + r_pred)         (L3B composition)
      delta      : l3b − interp_vae                  (learned image-space correction)
      l3c        : clip(linear + delta, 0, 1)        (L3C composition)
    """
    device = model.device
    x_linear = _img_to_gray01(_hwc_to_bchw(image_cond_bhwc).to(device))       # (b,H,W)
    x_l3b = _img_to_gray01(model.decode_first_stage(z_interp + residual_latent).clamp(-1, 1))
    x_interp_vae = _img_to_gray01(model.decode_first_stage(z_interp).clamp(-1, 1))
    delta = x_l3b - x_interp_vae
    x_l3c = (x_linear + delta).clamp(0, 1)
    return {"linear": x_linear, "interp_vae": x_interp_vae, "l3b": x_l3b,
            "delta": delta, "l3c": x_l3c}


# ==========================================================================
#  Per-slice volume reconstruction with diagnostic component volumes
# ==========================================================================
@torch.no_grad()
def _residual_for_chunk(model, ic, cond, uc, z_interp, sampler_cfg, seed,
                        residual_source, z_target=None):
    """Produce the predicted latent residual r_pred for a chunk of slices.

    residual_source:
      "sample" -> run the L3B residual DDIM sampler (trained/warm-start weights).
      "zero"   -> r_pred = 0 (zero-residual identity test / L3C floor).
      "oracle" -> r_pred = z_target − z_interp (exact latent residual; needs z_target).
    Returns r_pred (b,4,h,w).
    """
    if residual_source == "zero":
        return torch.zeros_like(z_interp)
    if residual_source == "oracle":
        assert z_target is not None, "oracle residual needs z_target"
        return (z_target - z_interp)
    # "sample": reuse the exact L3B residual sampler; take its r_pred output.
    _, r_pred = ddim_sample_residual(
        model, cond, uc, z_interp,
        ddim_steps=sampler_cfg["ddim_steps"],
        resid_init=sampler_cfg.get("resid_init", "zero"),
        resid_strength=sampler_cfg.get("resid_strength", 0.4),
        control_scale=sampler_cfg.get("control_scale", 1.0),
        guidance_scale=sampler_cfg.get("guidance_scale", 1.0),
        seed=seed,
    )
    return r_pred


@torch.no_grad()
def reconstruct_volume_l3c(model, prep, R, sampler_cfg, batch_size=16, seed=0,
                           max_missing=None, subsample_seed=0,
                           residual_source="sample", return_components=False):
    """L3C reconstruction: x = x_linear + decode(z_interp+r) − decode(z_interp).

    Mirrors `reconstruct_volume_residual` exactly (same retained_indices, same
    unpad metric space, same hard-copy) — only the composition differs. In one pass
    it produces, for the SAME predicted residual, ALL diagnostic component volumes so
    the composition-only and oracle diagnostics are a single deterministic call.

    Returns
    -------
    (recon_l3c, missing, ret)               if return_components is False
    (recon_l3c, missing, ret, components)   if return_components is True
        components: dict of (H,W,D) unpadded [0,1] volumes with retained slices
        hard-copied from GT: "l3c", "l3b", "interp_vae", "linear", plus a raw
        "delta" map (missing-slice correction, 0 on retained).
    """
    depth = prep["depth"]
    ret = retained_indices(depth, R)
    ret_set = set(ret)
    pad = prep["pad"]
    target = prep["target"]
    vol = prep["vol01"]                                   # (H,W,D) unpadded [0,1] GT
    device = model.device

    # Retained slices are hard-copied GT (the sparse input) for every component.
    recon_l3c = np.array(vol, dtype=np.float32, copy=True)
    comp = {k: np.array(vol, dtype=np.float32, copy=True)
            for k in ("l3b", "interp_vae", "linear")} if return_components else None
    delta_map = np.zeros_like(vol) if return_components else None

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
        ic, ctrl, Ts, tgt = [], [], [], []
        for (left, right, d) in chunk:
            c = build_conditioning(prep, left, right, d, target=target, debug=False)
            ic.append(c["image_cond"]); ctrl.append(c["control"]); Ts.append(c["T"])
            tgt.append(c["image_target"])
        ic = torch.stack(ic); ctrl = torch.stack(ctrl); Ts = torch.stack(Ts)
        cond = build_model_cond(model, ic, Ts, ctrl, uncond=False)
        uc = build_model_cond(model, ic, Ts, ctrl, uncond=True)
        z_interp = encode_linear_latent(model, ic)        # scale*encode(linear).mode()
        z_target = None
        if residual_source == "oracle":
            z_target = encode_linear_latent(model, torch.stack(tgt))  # scale*enc(target).mode()
        r_pred = _residual_for_chunk(model, ic, cond, uc, z_interp, sampler_cfg,
                                     seed + i, residual_source, z_target=z_target)
        out = compose_l3c_hard_anchor(model, ic, z_interp, r_pred)
        l3c = out["l3c"].cpu().numpy()
        for j, (left, right, d) in enumerate(chunk):
            recon_l3c[:, :, d] = unpad_hw(l3c[j], pad)
        if return_components:
            for name in ("l3b", "interp_vae", "linear"):
                arr = out[name].cpu().numpy()
                for j, (left, right, d) in enumerate(chunk):
                    comp[name][:, :, d] = unpad_hw(arr[j], pad)
            dl = out["delta"].cpu().numpy()
            for j, (left, right, d) in enumerate(chunk):
                delta_map[:, :, d] = unpad_hw(dl[j], pad)

    if return_components:
        comp["l3c"] = recon_l3c
        comp["delta"] = delta_map
        return recon_l3c, missing, ret, comp
    return recon_l3c, missing, ret
