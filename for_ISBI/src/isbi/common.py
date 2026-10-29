"""for_ISBI core protocol: paths, patient roles, sampling masks, gap geometry.

Nothing here reads reconstruction results. Masks depend only on (patient id, pattern, R,
realisation) -- never on intensities, tumour location, errors or uncertainty.
"""
from __future__ import annotations

from project_paths import root_path

import hashlib
import json
import os
import sys

import numpy as np

ROOT = root_path('')
ISBI = os.path.join(ROOT, "for_ISBI")
for _p in (ROOT, os.path.join(ROOT, "X-Diffusion", "scripts"),
           os.path.join(ROOT, "IR3C_more_comparisons")):
    if _p not in sys.path:
        sys.path.append(_p)

D = 153                       # protocol depth (central crop of 155), 1 mm isotropic
SPACING_MM = 1.0              # BraTS-GLI 2023 header zooms (1,1,1) -- verified in audit
STRIDES = (4, 8)              # x4 primary, x8 confirmation
PATTERNS = ("uniform", "jittered", "clustered")
N_REAL = 3                    # realisations of each stochastic pattern
ROLE_SEED = 20261016
VAL_DIR = "brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
SPLIT_DIR = os.path.join(ROOT, "runs/brats_frozen_vae_full_20260716_122401/splits")


def _h(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- roles
def patient_of(scan: str) -> str:
    """BraTS-GLI-PPPPP-TTT -> BraTS-GLI-PPPPP (timepoints of one patient share PPPPP)."""
    return scan.rsplit("-", 1)[0]


def build_roles(n_sel=12, n_fit=30, n_cal=30):
    """Patient-level roles inside the GLI ValidationData pool (never used for training).

    * every scan of a patient goes to ONE role; one scan per patient (lowest timepoint) is used
    * patients whose scans were in the source checkpoints' 6-scan selection set (val60 lines
      1-6) are forced into SEL (they influenced the source checkpoints)
    * remaining patients: seeded permutation -> SEL fill, FIT, CAL, then TEST order
      (TEST size is fixed after the GPU budget pilot, before any test evaluation).
    """
    scans = sorted(os.listdir(os.path.join(ROOT, VAL_DIR)))
    pats = {}
    for s in scans:
        pats.setdefault(patient_of(s), []).append(s)
    val6 = [l.strip() for l in open(os.path.join(SPLIT_DIR, "brats2023_val60.txt")) if l.strip()][:6]
    forced = sorted({patient_of(os.path.basename(os.path.dirname(p))) for p in val6})
    rest = sorted(p for p in pats if p not in forced)
    rng = np.random.RandomState(ROLE_SEED)
    rest = [rest[i] for i in rng.permutation(len(rest))]
    k = n_sel - len(forced)
    roles = dict(SEL=forced + rest[:k], FIT=rest[k:k + n_fit], CAL=rest[k + n_fit:k + n_fit + n_cal],
                 TEST_ORDER=rest[k + n_fit + n_cal:])
    scan_of = {p: sorted(v)[0] for p, v in pats.items()}
    return roles, scan_of, pats


def scan_path(scan):
    return os.path.join(ROOT, VAL_DIR, scan, f"{scan}-t2f.nii.gz")


def load_roles():
    return json.load(open(os.path.join(ISBI, "configs", "roles.json")))


# --------------------------------------------------------------------------- masks
def uniform_mask(R, D=D):
    m = list(range(0, D, R))
    assert m[-1] == D - 1, "anchor D-1 must be on the uniform grid"
    return m


def jittered_mask(R, seed, D=D):
    """Uniform grid with each INTERIOR point moved by an integer offset in [-J, J],
    J = R//2 - 1 (x4: +-1, x8: +-3). Anchors 0 and D-1 fixed; order is preserved because
    2J < R, so K is identical to uniform."""
    g = np.asarray(uniform_mask(R, D))
    J = R // 2 - 1
    rng = np.random.RandomState(seed)
    off = rng.randint(-J, J + 1, size=len(g))
    off[0] = off[-1] = 0
    m = g + off
    assert np.all(np.diff(m) > 0)
    return [int(x) for x in m]


def clustered_mask(R, seed, D=D):
    """Same K as uniform; the K-2 interior slices are grouped into clusters of 2-3 ADJACENT
    slices whose starts are spread evenly (+-1 jitter) over the volume, so the missing runs
    between clusters are ~2-3x longer than the uniform gap (x4: ~6-8 mm, x8: ~14-17 mm).
    Never removes a contiguous majority of the volume."""
    K = len(uniform_mask(R, D))
    M = K - 2
    rng = np.random.RandomState(seed)
    sizes = []
    while sum(sizes) < M:
        sizes.append(int(rng.choice([2, 3])))
    sizes[-1] -= sum(sizes) - M
    if sizes[-1] < 1:
        sizes = sizes[:-1]
        sizes[-1] += M - sum(sizes)
    nc = len(sizes)
    span = (D - 1) / (nc + 1)
    out = [0]
    for i, s in enumerate(sizes):
        c = int(round((i + 1) * span)) + int(rng.randint(-1, 2))
        start = max(c - s // 2, out[-1] + 2)
        out.extend(range(start, start + s))
    out.append(D - 1)
    out = sorted(set(out))
    assert len(out) == K and out[-2] < D - 1, (R, seed, len(out), K)
    return [int(x) for x in out]


def make_mask(pattern, R, patient, rep, D=D):
    if pattern == "uniform":
        return uniform_mask(R, D)
    seed = _h("isbi-mask", patient, pattern, R, rep)
    return (jittered_mask if pattern == "jittered" else clustered_mask)(R, seed, D)


def conditions(test=True):
    """(pattern, R, rep) grid. Uniform has a single (deterministic) realisation."""
    out = []
    for R in STRIDES:
        out.append(("uniform", R, 0))
        if test:
            for pat in ("jittered", "clustered"):
                out += [(pat, R, r) for r in range(N_REAL)]
    return out


def cond_key(pattern, R, rep):
    return f"x{R}_{pattern}_r{rep}"


# --------------------------------------------------------------------------- geometry
def gap_geometry(ret, D=D):
    """Per-slice geometry (mm) for every z: nearest acquired distance, bracketing gap
    length, relative position t in the gap, missing-run length. Acquired slices: 0/0/0/0."""
    ret = np.asarray(sorted(ret))
    z = np.arange(D)
    j = np.clip(np.searchsorted(ret, z, side="right") - 1, 0, len(ret) - 2)
    left, right = ret[j], ret[j + 1]
    acq = np.isin(z, ret)
    gap = (right - left).astype(float)
    t = (z - left) / np.maximum(gap, 1)
    near = np.minimum(z - left, right - z).astype(float)
    run = gap - 1
    for a in (gap, t, near, run):
        a[acq] = 0
    return dict(left=left, right=right, acquired=acq, gap_mm=gap * SPACING_MM,
                near_mm=near * SPACING_MM, t=t, run_len=run)


def mask_stats(ret, D=D):
    g = gap_geometry(ret, D)
    miss = ~g["acquired"]
    d = np.diff(sorted(ret))
    runs = d[d > 1] - 1
    return dict(K=len(ret), n_missing=int(miss.sum()), max_gap_mm=float(d.max() * SPACING_MM),
                mean_run=float(runs.mean()) if len(runs) else 0.0, max_run=int(runs.max()) if len(runs) else 0,
                mean_near_mm=float(g["near_mm"][miss].mean()), max_near_mm=float(g["near_mm"][miss].max()),
                mean_gap_mm_missing=float(g["gap_mm"][miss].mean()))
