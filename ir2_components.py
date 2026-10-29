"""
ir2_components.py
=================
IR2 --- the fixes derived from the IR1B1_E100 failure analysis.

Nothing in `image_residual/` or `ir1b1_*` is modified: those modules back frozen runs whose
reproducibility is evidence. IR2 re-implements the four pieces that had to change.

WHAT CHANGED, AND WHY
---------------------
1. BACKGROUND FIDELITY  (fixes the out-of-brain haze)
   The IR1/IR1B1 objective computes charbonnier and grad-MAE on `missing AND brain` only,
   so the background is constrained by nothing but `0.001 * mean|delta|`. With
   residual_scale = 0.585, flooding the background with mid-grey costs ~6e-4 of loss --
   effectively free, which is precisely what the R=16/U4 panels show.
   IR2 adds an explicit background term. Ground truth is 0 outside the brain, so this
   drives the haze to zero directly instead of relying on the [0,1] clip to absorb it
   (measured clip_fraction in E100 was 0.46-0.66, i.e. the clip was doing this job by
   accident).

2. FULL-DEPTH PATCHES  (fixes unconditional generation, and R=16)
   E100 trained on 64-slice crops. The U4/R=16 gap is 142 slices, so a long gap could never
   be seen bracketed by real data, and 34.6% of U4/R=16 crops contained ZERO acquired
   slices -- the model was being trained to invent brain content from no evidence, and
   squared error rewards predicting the dataset mean. That grey blob IS the learned mean.
   IR2 trains on the FULL through-plane extent, so every patch contains every acquired
   slice and an unconditioned crop cannot occur by construction.

3. LOG-SCALED POSITIONAL CHANNELS  (fixes R=16 conditioning)
   `LOCAL_GAP_NORM = 16.0` means a 142-slice gap enters the network as 8.875 on a channel
   whose trained range is ~1.0 -- far out of distribution exactly where the model fails.
   IR2 uses log1p scaling, which keeps every realisable gap inside [0, 1].

4. R=16 OVERSAMPLING + CAPACITY
   Stride sampling is weighted toward the sparse end, and the U-Net is widened.

NOT changed (deliberately): the hard anchor, the bounded tanh residual (E100's
frac_tanh_saturated was ~0, so the bound is not binding), the SSIM structural term, the
checkpoint guardrail, and seed handling.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from image_residual import masks as M, inputs as IN, losses as LOSS
import ir1b1_structure_loss as SL

MULT = 8                                   # 2 ** len(strides) for the U-Net
GAP_LOG_NORM = float(np.log1p(160.0))      # every realisable gap maps into [0, 1]
NEAR_LOG_NORM = float(np.log1p(80.0))


# --------------------------------------------------------------------------- #
#  positional channels, log-scaled
# --------------------------------------------------------------------------- #
def profiles_log(ret, D):
    """Same positional scalars as `masks.positional_profiles`, but with the gap and
    nearest-distance channels log-compressed so extreme sparsity stays in range."""
    p = M.positional_profiles(ret, D)
    return dict(known=p["known"], alpha=p["alpha"],
                gap_n=(np.log1p(p["local_gap"]) / GAP_LOG_NORM).astype(np.float32),
                near_n=(np.log1p(p["nearest"]) / NEAR_LOG_NORM).astype(np.float32))


def build_channels_ir2(vol_hw_d, ret, D, x_linear=None):
    """IR2 six-channel input over a whole volume (inference path)."""
    H, W, Dv = vol_hw_d.shape
    assert Dv == D
    if x_linear is None:
        x_linear = IN.linear_interp_volume(vol_hw_d, ret, D)
    x_sparse = IN.sparse_volume(vol_hw_d, ret, D)
    pr = profiles_log(ret, D)

    def bc(v): return np.broadcast_to(v[:, None, None], (D, H, W)).astype(np.float32)
    def cd(a): return np.moveaxis(a, 2, 0).astype(np.float32)

    x6 = np.stack([cd(x_sparse), cd(x_linear), bc(pr["known"]), bc(pr["alpha"]),
                   bc(pr["gap_n"]), bc(pr["near_n"])], 0).astype(np.float32)
    known = np.zeros(D, bool)
    for r in ret:
        known[int(r)] = True
    missing_hw_d = np.broadcast_to((~known)[None, None, :], (H, W, D)).astype(np.float32)
    return x6, x_linear.astype(np.float32), missing_hw_d


def build_patch_ir2(vol_hw_d, x_lin_hw_d, pr, D, y0, x0, hp, wp):
    """FULL-DEPTH patch: the entire through-plane extent, cropped only in H and W."""
    sl = (slice(y0, y0 + hp), slice(x0, x0 + wp), slice(0, D))
    gt = vol_hw_d[sl]
    lin = x_lin_hw_d[sl]
    known = pr["known"]
    sparse = np.where(known[None, None, :] > 0.5, gt, 0.0).astype(np.float32)

    def cd(a): return np.moveaxis(a, 2, 0).astype(np.float32)
    def bc(v): return np.broadcast_to(v[:, None, None], (D, hp, wp)).astype(np.float32)

    x6 = np.stack([cd(sparse), cd(lin), bc(known), bc(pr["alpha"]),
                   bc(pr["gap_n"]), bc(pr["near_n"])], 0)
    gt_c = cd(gt)[None]
    missing = bc(1.0 - known)[None]
    brain = (gt_c > 0.0).astype(np.float32)
    lin_c = cd(lin)[None]
    return x6, gt_c, lin_c, missing, brain


# --------------------------------------------------------------------------- #
#  stride-weighted mask sampling
# --------------------------------------------------------------------------- #
def sample_mask_weighted(D, rng, strides, weights, uniform_prob=0.5):
    """`masks.sample_training_mask` with a non-uniform prior over R, so the sparse end
    gets more training signal than 1/len(strides)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    R = int(rng.choice(np.asarray(strides), p=w))
    if rng.rand() < uniform_prob:
        return M.uniform_mask(D, R), dict(mask_family="U0", R=R, seed=0)
    fam = str(rng.choice(M.NONUNIFORM_FAMILIES))
    seed = int(rng.randint(0, 1_000_000))
    return M.nonuniform_mask(D, R, fam, seed), dict(mask_family=fam, R=R, seed=seed)


# --------------------------------------------------------------------------- #
#  padded forward (differentiable; mirrors the inference padding exactly)
# --------------------------------------------------------------------------- #
def forward_padded(model, x6, x_linear, missing, clip=True):
    """Whole-patch forward with replicate padding to a multiple of 8, cropped back.

    Training must pad the same way inference does, or the network sees a different
    spatial extent at test time -- which matters here because norm='instance' computes
    statistics over the extent it is given.
    """
    pads = []
    for n in (x6.shape[-1], x6.shape[-2], x6.shape[-3]):
        pads += [0, (-n) % MULT]
    padded = F.pad(x6, pads, mode="replicate") if any(pads) else x6
    raw = model.raw(padded)
    raw = raw[..., :x6.shape[-3], :x6.shape[-2], :x6.shape[-1]].contiguous()
    d = model.residual_scale * torch.tanh(raw)
    x = x_linear + missing * d
    if clip:
        x = torch.clamp(x, 0.0, 1.0)
    return x, d


@torch.inference_mode()
def reconstruct_ir2(model, vol_hw_d, ret, D, device):
    """IR2 whole-volume inference. Same composition and hard anchor as the corrected IR1
    path, but with the IR2 (log-scaled) input channels."""
    model.eval()
    x6, x_lin, missing_hw_d = build_channels_ir2(vol_hw_d, ret, D)
    x6_t = torch.from_numpy(x6)[None].to(device)
    pads = []
    for n in (x6_t.shape[-1], x6_t.shape[-2], x6_t.shape[-3]):
        pads += [0, (-n) % MULT]
    padded = F.pad(x6_t, pads, mode="replicate") if any(pads) else x6_t
    raw = model.raw(padded)[..., :D, :vol_hw_d.shape[0], :vol_hw_d.shape[1]]
    delta = (model.residual_scale * torch.tanh(raw))[0, 0].float().cpu().numpy()
    delta_hw_d = np.moveaxis(delta, 0, 2).astype(np.float32)
    miss = np.moveaxis(missing_hw_d, 2, 2)          # already (H,W,D)
    delta_hw_d = delta_hw_d * miss
    del raw, x6_t, padded
    if device == "cuda":
        torch.cuda.empty_cache()
    known = np.zeros(D, bool)
    for r in ret:
        known[int(r)] = True
    x = np.clip(x_lin + delta_hw_d, 0.0, 1.0).astype(np.float32)
    x[:, :, known] = vol_hw_d[:, :, known]          # exact hard copy
    return x, x_lin.astype(np.float32), delta_hw_d, dict(inference="ir2_full_volume")


# --------------------------------------------------------------------------- #
#  loss
# --------------------------------------------------------------------------- #
def ir2_loss(x_pred, delta, gt, missing_mask, brain_mask, lambda_structure, w_background):
    """IR1B1's objective plus an explicit BACKGROUND fidelity term.

        total = charbonnier(missing AND brain)
              + w_background * charbonnier(missing AND NOT brain)     <-- NEW
              + lambda_structure * (1 - SSIM(full slice))
              + 0.05 * grad_mae(missing AND brain)
              + 0.001 * mean|delta|(missing)

    The background term is what makes "leave the background alone" an objective rather
    than an accident of the [0,1] clip.
    """
    region_brain = missing_mask * brain_mask
    region_bg = missing_mask * (1.0 - brain_mask)
    ch = LOSS.charbonnier(x_pred, gt, region_brain)
    ch_bg = LOSS.charbonnier(x_pred, gt, region_bg)
    gm = LOSS.grad_mae(x_pred, gt, region_brain)
    rm = LOSS.residual_magnitude(delta, missing_mask)
    with torch.cuda.amp.autocast(enabled=False):
        st, n_valid = SL.structure_loss(x_pred.float(), gt.float(), missing_mask.float(),
                                        brain_mask.float())
    total = (ch + w_background * ch_bg + lambda_structure * st
             + LOSS.GRAD_W * gm + LOSS.RESID_W * rm)
    terms = dict(charbonnier=float(ch.detach()), charbonnier_bg=float(ch_bg.detach()),
                 structure=float(st.detach()), grad_mae=float(gm.detach()),
                 residual_magnitude=float(rm.detach()),
                 lambda_structure=float(lambda_structure),
                 w_background=float(w_background),
                 ssim_valid_slices=int(n_valid), total=float(total.detach()))
    return total, terms


# --------------------------------------------------------------------------- #
#  diagnostic: how much energy lands outside the brain
# --------------------------------------------------------------------------- #
def background_energy(x_pred, gt, missing_mask):
    """Mean predicted intensity on missing background voxels. This is the haze, measured.
    Linear interpolation and the ground truth both sit near 0 here."""
    bg = missing_mask * (gt <= 0.0).float()
    s = bg.sum()
    return float(((x_pred * bg).sum() / (s + 1e-8)).detach())
