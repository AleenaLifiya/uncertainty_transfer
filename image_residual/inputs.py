"""
image_residual/inputs.py
------------------------
Builds the 6-channel image-space input tensor for IR1 from a volume and an
acquisition mask (§14). All positional channels are generated from the mask ONLY.

Channel order (fixed):
  0 sparse volume        (zeros on missing slices)
  1 linear interpolation (real physical retained neighbours)
  2 known-slice mask     (1 acquired, 0 missing)
  3 alpha within local gap
  4 normalized local gap
  5 normalized distance to nearest retained slice

Internal layout [C, D, H, W] (torch [B,C,D,H,W]). x is in [0,1].
"""
from __future__ import annotations
import numpy as np

from . import masks as M


def linear_interp_volume(vol_hw_d, ret, D):
    """Direct image-space linear interpolation over the ACTUAL retained physical z.
    Retained slices copied from vol; missing slices = (1-a)*x(left)+a*x(right)."""
    ret = sorted(int(r) for r in ret)
    out = np.array(vol_hw_d, dtype=np.float32, copy=True)
    for d in M.missing_of(ret, D):
        l, r = M.neighbours(ret, d)
        a = (d - l) / (r - l)
        out[:, :, d] = (1 - a) * vol_hw_d[:, :, l] + a * vol_hw_d[:, :, r]
    return out


def sparse_volume(vol_hw_d, ret, D):
    """Acquired slices kept, missing slices zeroed."""
    ret = set(int(r) for r in ret)
    out = np.zeros_like(vol_hw_d, dtype=np.float32)
    for z in range(D):
        if z in ret:
            out[:, :, z] = vol_hw_d[:, :, z]
    return out


def build_input_channels(vol_hw_d, ret, D, x_linear=None):
    """Return (x6 [6,D,H,W] float32, x_linear_hw_d, known_mask_hw_d, missing_mask_hw_d).
    vol_hw_d is (H,W,D) in [0,1]. Positional channels are broadcast per-slice scalars."""
    H, W, Dv = vol_hw_d.shape
    assert Dv == D
    if x_linear is None:
        x_linear = linear_interp_volume(vol_hw_d, ret, D)
    x_sparse = sparse_volume(vol_hw_d, ret, D)
    prof = M.positional_profiles(ret, D)
    known = prof["known"]                                   # [D]
    alpha = prof["alpha"]
    gap_n = prof["local_gap"] / M.LOCAL_GAP_NORM
    near_n = prof["nearest"] / M.NEAREST_NORM

    def bc(vec):                                            # [D] -> [D,H,W]
        return np.broadcast_to(vec[:, None, None], (D, H, W)).astype(np.float32)

    # to [C,D,H,W]
    def to_cdhw(hw_d):
        return np.moveaxis(hw_d, 2, 0).astype(np.float32)   # (H,W,D)->(D,H,W)

    x6 = np.stack([
        to_cdhw(x_sparse),
        to_cdhw(x_linear),
        bc(known),
        bc(alpha),
        bc(gap_n),
        bc(near_n),
    ], axis=0).astype(np.float32)                           # [6,D,H,W]

    # masks returned in (H,W,D) to match vol / x_linear convention
    known_hw_d = np.broadcast_to(known[None, None, :], (H, W, D)).astype(np.float32)
    missing_hw_d = (1.0 - known_hw_d).astype(np.float32)
    return x6, x_linear, known_hw_d, missing_hw_d


def clamp_stats(ret, D):
    """Report whether any positional channel exceeds [0,1] after normalization (i.e.
    would be clamped by a naive normalizer). IR1 does NOT clamp; this is diagnostic."""
    prof = M.positional_profiles(ret, D)
    gap_n = prof["local_gap"] / M.LOCAL_GAP_NORM
    near_n = prof["nearest"] / M.NEAREST_NORM
    return dict(max_local_gap_norm=float(gap_n.max()), max_nearest_norm=float(near_n.max()),
                local_gap_exceeds_one=bool((gap_n > 1.0).any()),
                nearest_exceeds_one=bool((near_n > 1.0).any()),
                clamped=False, note="IR1 does not clamp positional channels; instance norm handles scale")
