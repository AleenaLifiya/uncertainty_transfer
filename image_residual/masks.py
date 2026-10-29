"""
image_residual/masks.py
-----------------------
Acquisition-mask sampling for IR1 (hard_anchored_image_residual_unet). Masks are a
PURE function of (D, family, R, seed) — they never see image intensities or tumour /
segmentation labels (§13, §17). Uniform masks reuse the existing validated convention
(nonuniform_masks.uniform_retained == fidelity_sampling.retained_indices). Non-uniform
masks reuse the corrected NU1 U1-U6 generators with exact retained-count matching.

Positional channels (alpha / local-gap / nearest-distance / known) are derived here
from the retained set only, so the training and evaluation code share one definition.
"""
from __future__ import annotations
import os, sys
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

import nonuniform_masks as NM

# normalization constants for positional channels (documented; not clipped)
LOCAL_GAP_NORM = 16.0     # R=16 uniform gap -> 1.0; longer non-uniform gaps -> >1
NEAREST_NORM = 8.0        # distance to nearest acquired slice

NONUNIFORM_FAMILIES = ["U1", "U2", "U3", "U4", "U5", "U6"]


def uniform_mask(D: int, R: int):
    """Existing validated uniform retained set (== fidelity_sampling.retained_indices)."""
    return NM.uniform_retained(D, R)


def nonuniform_mask(D: int, R: int, family: str, seed: int):
    """Corrected NU1 U1-U6 generator with exact retained-count matching (K_R)."""
    if family == "U1":
        return NM.mask_U1(D, R, seed)
    if family == "U2":
        return NM.mask_U2(D, R, seed)
    if family == "U3":
        return NM.mask_U3(D, R, seed)
    if family == "U4":
        sev = "burst_mild" if (seed % 2 == 0) else "burst_strong"
        return NM.mask_U4(D, R, sev, seed)
    if family == "U5":
        return NM.mask_U5(D, R, seed)
    if family == "U6":
        return NM.mask_U6(D, R, seed)
    raise ValueError(f"unknown non-uniform family {family!r}")


def sample_training_mask(D: int, rng: np.random.RandomState, strides=(2, 4, 8, 16),
                         uniform_prob=0.5):
    """Draw one mask for a training step (§17): 50% uniform, 50% non-uniform; within
    uniform sample R equally; within non-uniform sample family approximately equally.
    Returns (ret, meta) where meta records family/R/seed for the realized-distribution
    log. Deterministic given rng state."""
    R = int(rng.choice(strides))
    if rng.rand() < uniform_prob:
        ret = uniform_mask(D, R)
        fam = "U0"
        seed = 0
    else:
        fam = str(rng.choice(NONUNIFORM_FAMILIES))
        seed = int(rng.randint(0, 1_000_000))
        ret = nonuniform_mask(D, R, fam, seed)
    return ret, dict(mask_family=fam, R=int(R), seed=int(seed))


def missing_of(ret, D):
    rs = set(int(r) for r in ret)
    return [d for d in range(D) if d not in rs]


def neighbours(ret, d):
    left = max(r for r in ret if r < d)
    right = min(r for r in ret if r > d)
    return int(left), int(right)


def positional_profiles(ret, D):
    """Per-slice positional scalars from the retained set only (image-free).
    Returns dict of length-D float arrays:
      known    : 1 on acquired slices, 0 on missing
      alpha    : (d-left)/(right-left) on missing; 0 on acquired
      local_gap: (right-left) on missing; 0 on acquired    (raw)
      nearest  : min(d-left, right-d) on missing; 0 on acquired  (raw)
    """
    ret = sorted(int(r) for r in ret)
    known = np.zeros(D, np.float32)
    alpha = np.zeros(D, np.float32)
    gap = np.zeros(D, np.float32)
    nearest = np.zeros(D, np.float32)
    rs = set(ret)
    for z in range(D):
        if z in rs:
            known[z] = 1.0
            continue
        l, r = neighbours(ret, z)
        alpha[z] = (z - l) / (r - l)
        gap[z] = (r - l)
        nearest[z] = min(z - l, r - z)
    return dict(known=known, alpha=alpha, local_gap=gap, nearest=nearest)


def realized_distribution(metas):
    """Summarize a list of step metas into the required training-distribution table."""
    import pandas as pd
    rows = []
    for m in metas:
        rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby(["mask_family", "R"]).size().reset_index(name="n_samples")
    return agg
