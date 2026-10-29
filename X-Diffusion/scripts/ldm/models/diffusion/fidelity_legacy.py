"""
fidelity_legacy.py
------------------
Checkpoint-compatible reconstruction for the LEGACY X-Diffusion models
(xdiff_control_fullVAE.md, "Important fairness rule for legacy models"):

  L1  Legacy X-Diffusion WITHOUT ControlNet
  L2  Legacy recursive ControlNet
      = old X-Diffusion checkpoint
      + original single-neighbour ControlNet (3-channel image hint)
      + recursive / bisection inference (autoregressive).

Why a separate module?  The legacy checkpoints were trained with a DIFFERENT
input representation than the redesigned fidelity pipeline, and the fairness rule
forbids silently feeding an old checkpoint an input it never saw. Documented
legacy conditioning (from ldm/data/bratsloader_cn.py + inference.py):

  * normalisation : PER-SLICE min-max to [0,1], uint8-quantised (via PIL);
  * geometry      : RESIZE 240 -> 256 (bilinear), NOT symmetric padding;
  * image_cond    : an acquired anchor slice (CLIP + c_concat);
  * control (L2)  : a single NEIGHBOUR slice, repeated to 3 channels;
  * pose T        : [ax0, ax1, ax2, depth_norm(d)] into cc_projection (772->768);
  * sampler       : pure-noise DDIM (eta=0) with classifier-free guidance.

Consequence (a real finding, not hidden): per-slice min-max destroys absolute
intensity scale, so a legacy output slice is only defined up to its own [min,max].
To evaluate every method with the SAME metric code in the SAME percentile-[0,1]
metric space, each legacy output slice is remapped into metric space using the
intensity range LINEARLY INTERPOLATED from its two acquired neighbours (uses only
acquired data — never the target slice — so there is no leakage). This scale
ambiguity is exactly what the redesigned volume-level normalisation removes.
"""

import numpy as np
import torch
import torchvision
from torchvision import transforms
from einops import rearrange
from PIL import Image

from ldm.data.brats_preprocessing_fidelity import (
    load_raw_volume, central_depth_crop, unpad_hw, THROUGH_PLANE_AXIS,
)
from ldm.models.diffusion.fidelity_sampling import (
    build_model_cond, ddim_sample, retained_indices,
)

_LEGACY_TFORM = transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: rearrange(x * 2.0 - 1.0, "c h w -> h w c")),
])


def _slice_to_rgb_pil(slice2d):
    """Replicate bratsloader_cn._slice_to_rgb_pil: PER-SLICE min-max -> uint8 RGB."""
    s = slice2d.astype(np.float32)
    s = (s - s.min()) / (s.max() - s.min() + 1e-5)
    s = np.repeat(s[..., None], 3, axis=2)
    return Image.fromarray(np.uint8(s * 255.0))


def _legacy_input_hwc(slice2d_metric):
    """Metric-space 2D slice -> legacy model input tensor (H,W,3) in [-1,1] @256."""
    return _LEGACY_TFORM(_slice_to_rgb_pil(slice2d_metric)).float()


def prepare_legacy(prep):
    """Attach per-slice metric-space [min,max] ranges to a fidelity prep.

    Uses prep['vol01'] (the shared percentile metric space) so that remapping a
    legacy per-slice-normalised output back to metric space is consistent with
    the ground truth every other method is scored against.
    """
    vol = prep["vol01"]
    smin = vol.min(axis=(0, 1))  # (D,)
    smax = vol.max(axis=(0, 1))
    prep = dict(prep)
    prep["slice_min"] = smin.astype(np.float32)
    prep["slice_max"] = smax.astype(np.float32)
    return prep


def _remap_to_metric(gray01_240, d, left, right, prep):
    """Map a legacy per-slice-normalised [0,1] slice into metric space using the
    intensity range linearly interpolated from the two acquired neighbours."""
    smin, smax = prep["slice_min"], prep["slice_max"]
    a = (d - left) / float(right - left)
    lo = (1 - a) * smin[left] + a * smin[right]
    hi = (1 - a) * smax[left] + a * smax[right]
    return np.clip(gray01_240 * (hi - lo) + lo, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def _sample_legacy_slice(model, anchor_metric_2d, hint_metric_2d, d, depth,
                         sampler_cfg, use_control, seed):
    """One legacy slice generation. Returns 240x240 gray in per-slice [0,1]."""
    device = model.device
    ic = _legacy_input_hwc(anchor_metric_2d)[None].to(device)      # (1,256,256,3)
    ctrl = None
    if use_control:
        ctrl = _legacy_input_hwc(hint_metric_2d)[None].to(device)  # (1,256,256,3)
    depth_norm = 2.0 * (d / max(1, depth - 1)) - 1.0
    T = torch.tensor([[0.0, 0.0, 1.0, float(depth_norm)]], dtype=torch.float32)
    cond = build_model_cond(model, ic, T, ctrl, uncond=False)
    uc = build_model_cond(model, ic, T, ctrl, uncond=True)
    img = ddim_sample(
        model, cond, uc, z0_interp=None,          # pure-noise legacy sampler
        ddim_steps=sampler_cfg["ddim_steps"], interp_strength=1.0,
        control_scale=sampler_cfg.get("control_scale", 1.0),
        guidance_scale=sampler_cfg.get("guidance_scale", 1.0), seed=seed)
    # (1,3,256,256) [-1,1] -> gray [0,1] @256 (caller resizes to native in-plane)
    g = ((img.mean(1) + 1) / 2).clamp(0, 1)  # (1,256,256)
    return g[0].cpu().numpy()


@torch.no_grad()
def reconstruct_volume_legacy(model, prep, R, sampler_cfg, use_control,
                              recursive=False, seed=0, native_hw=(240, 240),
                              max_missing=None, subsample_seed=0):
    """Legacy reconstruction in the shared metric space.

    use_control=False, recursive=False -> L1 (independent, no ControlNet).
    use_control=True,  recursive=True  -> L2 (single-neighbour ControlNet,
                                             recursive/bisection, autoregressive).
    Retained slices are hard-copied from GT; metrics use `generated` indices.
    """
    prep = prepare_legacy(prep)
    depth = prep["depth"]
    pad = prep["pad"]
    Hn, Wn = native_hw
    vol = prep["vol01"]                     # metric-space GT (H,W,D)
    ret = retained_indices(depth, R)
    ret_set = set(ret)
    recon = np.array(vol, dtype=np.float32, copy=True)

    def _resize_240(g256):
        t = torch.from_numpy(g256)[None, None].float()
        t = torch.nn.functional.interpolate(t, size=(Hn, Wn), mode="bilinear",
                                             align_corners=False)
        return t[0, 0].numpy()

    if not recursive:
        # ---- L1 / independent: every missing slice from acquired neighbours ----
        missing = [d for d in range(depth) if d not in ret_set]
        if max_missing is not None and max_missing < len(missing):
            rng = np.random.RandomState(subsample_seed)
            missing = sorted(rng.choice(missing, size=max_missing, replace=False).tolist())
        gen = []
        for d in missing:
            left = max([r for r in ret if r < d])
            right = min([r for r in ret if r > d])
            # anchor = nearest acquired neighbour; hint = the other neighbour
            anchor_idx = left if (d - left) <= (right - d) else right
            g256 = _sample_legacy_slice(model, vol[:, :, anchor_idx], vol[:, :, right],
                                        d, depth, sampler_cfg, use_control, seed + d)
            g240 = _resize_240(g256)
            recon[:, :, d] = _remap_to_metric(g240, d, left, right, prep)
            gen.append(d)
        return recon, sorted(gen), ret

    # ---- L2 / recursive bisection (autoregressive; generated slices reused) ----
    known = {r: vol[:, :, r].copy() for r in ret}  # metric-space known slices
    generated = []

    def fill(left, right):
        if right - left <= 1:
            return
        d = (left + right) // 2
        # anchor & hint are the CURRENT known neighbours (may be generated)
        g256 = _sample_legacy_slice(model, known[left], known[right],
                                    d, depth, sampler_cfg, use_control, seed + d)
        g240 = _resize_240(g256)
        slc = _remap_to_metric(g240, d, left, right, prep)
        known[d] = slc
        recon[:, :, d] = slc
        generated.append(d)
        fill(left, d)
        fill(d, right)

    for i in range(len(ret) - 1):
        fill(ret[i], ret[i + 1])
    return recon, sorted(generated), ret
