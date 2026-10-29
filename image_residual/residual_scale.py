"""
image_residual/residual_scale.py
--------------------------------
Fixed TRAINING-ONLY residual scale for IR1 (§16). NEVER uses validation/test/SSA data
and never computes a per-subject scale.

    residual_gt = x_gt - x_linear        (image space)
    scale = percentile_{99.5}( |residual_gt| )   over
            training subjects  ∩  missing slices  ∩  brain foreground (x_gt>0)

The scale is a single scalar used as  delta_pred = scale * tanh(raw).
"""
from __future__ import annotations
import os, sys, json, hashlib
import numpy as np

from . import inputs as IN
from . import masks as M


def _subject_hash(files):
    h = hashlib.sha256()
    for f in sorted(os.path.basename(os.path.dirname(f)) for f in files):
        h.update(f.encode())
    return h.hexdigest()


def compute_residual_scale(preps, files, D, strides=(2, 4, 8, 16), percentile=99.5,
                           seed=42, masks_per_subject=4, out_json=None):
    """Aggregate |x_gt - x_linear| over the training subjects, missing slices and brain
    foreground, then take the given percentile. Deterministic given (seed, preps).

    A deterministic set of `masks_per_subject` masks (mix of uniform + non-uniform,
    R sampled over `strides`) is drawn per subject so the scale reflects the mixed-mask
    training distribution (§17) rather than one stride."""
    rng = np.random.RandomState(seed)
    abs_res = []
    n_vox = 0
    for prep in preps:
        gt = prep["vol01"]                                 # (H,W,D) in [0,1]
        for _ in range(masks_per_subject):
            ret, meta = M.sample_training_mask(D, rng, strides=strides)
            x_lin = IN.linear_interp_volume(gt, ret, D)
            miss = M.missing_of(ret, D)
            for d in miss:
                g = gt[:, :, d]
                fg = g > 0.0
                if fg.sum() == 0:
                    continue
                res = np.abs(g[fg] - x_lin[:, :, d][fg])
                abs_res.append(res.astype(np.float32))
                n_vox += int(fg.sum())
    allres = np.concatenate(abs_res) if abs_res else np.array([0.0], np.float32)
    scale = float(np.percentile(allres, percentile))
    rec = dict(
        residual_scale=scale, percentile=float(percentile),
        n_training_subjects=len(preps),
        subject_manifest_hash=_subject_hash(files),
        strides=list(strides), masks_per_subject=int(masks_per_subject),
        n_foreground_voxels=int(n_vox), seed=int(seed),
        mask_generation="uniform (fidelity convention) + NU1 U1-U6, exact-count matched; image-free",
        region="training subjects ∩ missing slices ∩ brain foreground (x_gt>0)",
        mean_abs_residual=float(allres.mean()), max_abs_residual=float(allres.max()),
        note="training-only; NOT per-subject; NOT from val/test/SSA",
    )
    if out_json:
        json.dump(rec, open(out_json, "w"), indent=2)
    return scale, rec
