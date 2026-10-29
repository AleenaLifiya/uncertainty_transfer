"""
nonuniform_masks.py
-------------------
Deterministic non-uniform through-plane slice-retention mask generators for the
NU1 ablation (`implement_nonuniform.md` §2-4). NO training, NO model, NO image or
segmentation data ever enters mask generation — masks are a pure function of
(depth D, family, severity, seed). This is the whole point: masks must be
independent of intensities, tumour position and segmentation labels.

Fair retention-rate matching: for nominal R∈{2,4,8} the exact retained count is
    K_R = len(uniform_retained(D, R))            # 153 -> 77 / 39 / 20
Every family at level R retains EXACTLY K_R slices, always including the two
boundary anchors z=0 and z=D-1.

Families (§3):
  U0 regular uniform control          — reproduces retained_indices(D,R) exactly
  U1 mildly jittered uniform          — jitter = max(1, floor(R/3))
  U2 strongly jittered uniform        — jitter = max(1, floor(R/2))
  U3 random exact-count               — seeds 101, 202
  U4 clustered / burst                — burst_mild, burst_strong
  U5 centre-dense variable-density    — fixed triangular pdf on normalized z
  U6 mixed run-length                 — seeded alternating retained/missing runs

This module is import-light (numpy + hashlib only) so the unit tests can run it
without importing torch / X-Diffusion.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  fair retention matching
# --------------------------------------------------------------------------- #
def uniform_retained(D: int, R: int) -> List[int]:
    """Byte-for-byte reimplementation of fidelity_sampling.retained_indices.

    idx = 0, R, 2R, ...  then force-append D-1 if not already present.
    A unit test asserts this equals the real fidelity_sampling.retained_indices.
    """
    idx = list(range(0, D, R))
    if idx[-1] != D - 1:
        idx.append(D - 1)
    return idx


def K_R(D: int, R: int) -> int:
    """Exact retained-slice count for nominal level R (the fairness target)."""
    return len(uniform_retained(D, R))


# --------------------------------------------------------------------------- #
#  exact-count enforcement (keeps anchors, deterministic)
# --------------------------------------------------------------------------- #
def _enforce_exact_count(candidate, D: int, K: int, rng: np.random.RandomState) -> List[int]:
    """Return a sorted unique length-K index list in [0,D-1] that always keeps the
    two boundary anchors {0, D-1}. Excess interior indices are removed and deficits
    filled deterministically via `rng` (never touching the anchors)."""
    anchors = {0, D - 1}
    s = set(int(i) for i in candidate) | anchors
    s = {i for i in s if 0 <= i <= D - 1}
    interior = sorted(s - anchors)
    if len(s) > K:
        n_remove = len(s) - K
        # remove interior indices uniformly at random (deterministic given rng)
        drop = rng.choice(len(interior), size=n_remove, replace=False)
        drop_set = {interior[j] for j in drop}
        s = s - drop_set
    elif len(s) < K:
        n_add = K - len(s)
        pool = [i for i in range(1, D - 1) if i not in s]
        add = rng.choice(len(pool), size=n_add, replace=False)
        s = s | {pool[j] for j in add}
    out = sorted(s)
    assert len(out) == K, f"exact-count enforcement failed: {len(out)} != {K}"
    assert out[0] == 0 and out[-1] == D - 1, "anchors lost during enforcement"
    return out


# --------------------------------------------------------------------------- #
#  families
# --------------------------------------------------------------------------- #
def mask_U0(D: int, R: int) -> List[int]:
    """Regular uniform control — must equal retained_indices(D,R)."""
    return uniform_retained(D, R)


def _jittered(D: int, R: int, max_jitter: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    base = uniform_retained(D, R)
    out = {0, D - 1}
    for u in base[1:-1]:
        j = int(rng.randint(-max_jitter, max_jitter + 1))
        v = int(np.clip(u + j, 1, D - 2))
        out.add(v)
    return _enforce_exact_count(sorted(out), D, K_R(D, R), rng)


def mask_U1(D: int, R: int, seed: int = 42) -> List[int]:
    """Mildly jittered uniform (§3 U1): max_jitter = max(1, floor(R/3))."""
    return _jittered(D, R, max(1, R // 3), seed)


def mask_U2(D: int, R: int, seed: int = 42) -> List[int]:
    """Strongly jittered uniform (§3 U2): max_jitter = max(1, floor(R/2))."""
    return _jittered(D, R, max(1, R // 2), seed)


def mask_U3(D: int, R: int, seed: int) -> List[int]:
    """Random exact-count sampling (§3 U3). Anchors fixed; interior uniform w/o
    replacement. Fixed mask seeds 101 and 202."""
    rng = np.random.RandomState(seed)
    K = K_R(D, R)
    interior_pool = list(range(1, D - 1))
    pick = rng.choice(len(interior_pool), size=K - 2, replace=False)
    out = {0, D - 1} | {interior_pool[j] for j in pick}
    return _enforce_exact_count(sorted(out), D, K, rng)


def mask_U4(D: int, R: int, severity: str, seed: int = 42) -> List[int]:
    """Clustered / burst sampling (§3 U4). Groups of consecutive retained slices
    separated by larger missing gaps. Two deterministic severities:
      burst_mild   — many small clusters (short-ish long gaps)
      burst_strong — fewer larger clusters (longer missing gaps)
    Cluster placement uses ONLY normalized z (never anatomy)."""
    rng = np.random.RandomState(seed)
    K = K_R(D, R)
    cluster_size = 3 if severity == "burst_mild" else 6
    n_clusters = max(2, int(round(K / cluster_size)))
    # evenly spaced cluster start positions across the volume; ends anchored
    starts = np.linspace(0, D - cluster_size, n_clusters)
    starts = np.clip(np.round(starts).astype(int), 0, D - cluster_size)
    out = {0, D - 1}
    for s0 in starts:
        for k in range(cluster_size):
            out.add(int(np.clip(s0 + k, 0, D - 1)))
    return _enforce_exact_count(sorted(out), D, K, rng)


def _centre_weights(D: int) -> np.ndarray:
    """Fixed triangular density on normalized interior z (peak at centre)."""
    zs = np.arange(1, D - 1, dtype=np.float64)
    zc = (D - 1) / 2.0
    w = 1.0 - np.abs(zs - zc) / zc          # triangular, 1 at centre -> 0 at ends
    w = np.clip(w, 1e-3, None)
    return w / w.sum()


def mask_U5(D: int, R: int, seed: int = 505) -> List[int]:
    """Centre-dense variable-density sampling (§3 U5). More retained slices near
    the centre, fewer near superior/inferior ends; anchors retained. Uses a fixed
    triangular pdf of normalized z only."""
    rng = np.random.RandomState(seed)
    K = K_R(D, R)
    interior_pool = np.arange(1, D - 1)
    w = _centre_weights(D)
    pick = rng.choice(len(interior_pool), size=K - 2, replace=False, p=w)
    out = {0, D - 1} | {int(interior_pool[j]) for j in pick}
    return _enforce_exact_count(sorted(out), D, K, rng)


def mask_U6(D: int, R: int, seed: int = 606) -> List[int]:
    """Mixed run-length sampling (§3 U6). Seeded alternating runs of retained and
    missing slices producing isolated slices, short blocks, short gaps and
    occasional long gaps. Exact final K enforced. Run-length distributions:
      retained-run length ~ choice([1,1,1,2,3])
      missing-run  length ~ choice([1,2,3,4,8,12])  (occasional long gap)
    """
    rng = np.random.RandomState(seed)
    K = K_R(D, R)
    ret_lens = np.array([1, 1, 1, 2, 3])
    gap_lens = np.array([1, 2, 3, 4, 8, 12])
    out = {0, D - 1}
    z = 1
    retained_phase = True
    while z < D - 1:
        if retained_phase:
            L = int(rng.choice(ret_lens))
            for _ in range(L):
                if z < D - 1:
                    out.add(z); z += 1
        else:
            L = int(rng.choice(gap_lens))
            z += L
        retained_phase = not retained_phase
    return _enforce_exact_count(sorted(out), D, K, rng)


# --------------------------------------------------------------------------- #
#  mask sweep specification (families x severities x seeds) — frozen
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaskSpec:
    family: str          # U0..U6
    severity: str        # regular/jitter/random/burst_mild/... (descriptive)
    seed: int
    primary: bool        # the single fixed seed used for the SSA mini-screen


def default_mask_specs() -> List[MaskSpec]:
    """The frozen NU1 mask sweep (independent of R). One row per (family,severity,
    seed). U3 has two seeds; U4 has two severities. `primary` marks the single
    configuration per family used in the R=4 SSA mini-screen."""
    return [
        MaskSpec("U0", "regular",      0,   True),
        MaskSpec("U1", "jitter_mild",  42,  True),
        MaskSpec("U2", "jitter_strong",42,  True),
        MaskSpec("U3", "random",       101, True),
        MaskSpec("U3", "random",       202, False),
        MaskSpec("U4", "burst_mild",   42,  True),
        MaskSpec("U4", "burst_strong", 42,  False),
        MaskSpec("U5", "centre_dense", 505, True),
        MaskSpec("U6", "mixed_runlen", 606, True),
    ]


def generate_mask(spec: MaskSpec, D: int, R: int) -> List[int]:
    """Dispatch a MaskSpec to its family generator. Deterministic in (spec, D, R)."""
    fam = spec.family
    if fam == "U0":
        return mask_U0(D, R)
    if fam == "U1":
        return mask_U1(D, R, spec.seed)
    if fam == "U2":
        return mask_U2(D, R, spec.seed)
    if fam == "U3":
        return mask_U3(D, R, spec.seed)
    if fam == "U4":
        return mask_U4(D, R, spec.severity, spec.seed)
    if fam == "U5":
        return mask_U5(D, R, spec.seed)
    if fam == "U6":
        return mask_U6(D, R, spec.seed)
    raise ValueError(f"unknown family {fam!r}")


def mask_id(spec: MaskSpec, R: int) -> str:
    return f"{spec.family}_R{R}_{spec.severity}_s{spec.seed}"


# --------------------------------------------------------------------------- #
#  gap statistics + hashing
# --------------------------------------------------------------------------- #
def retained_hash(ret) -> str:
    """SHA-256 of the retained-index array (int64 little-endian bytes)."""
    arr = np.asarray(sorted(int(i) for i in ret), dtype="<i8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _retained_runs(ret_sorted: List[int]) -> int:
    """Number of maximal consecutive-integer runs in the retained set."""
    if not ret_sorted:
        return 0
    runs = 1
    for a, b in zip(ret_sorted[:-1], ret_sorted[1:]):
        if b != a + 1:
            runs += 1
    return runs


def gap_statistics(ret, D: int) -> Dict:
    """Spacing statistics between consecutive retained slices.

    `gaps` are the spacings right-left between consecutive retained indices
    (gap==1 means adjacent retained slices; gap==k means k-1 missing between them).
    longest_missing_run = max(gap) - 1.
    """
    r = sorted(int(i) for i in ret)
    diffs = np.diff(np.asarray(r, dtype=int))
    diffs_f = diffs.astype(np.float64)
    missing = [z for z in range(D) if z not in set(r)]
    if len(diffs) == 0:
        stats = dict(min_gap=0, max_gap=0, mean_gap=0.0, median_gap=0.0,
                     std_gap=0.0, cv_gap=0.0, longest_missing_run=0)
    else:
        mean = float(diffs_f.mean())
        stats = dict(
            min_gap=int(diffs.min()),
            max_gap=int(diffs.max()),
            mean_gap=mean,
            median_gap=float(np.median(diffs_f)),
            std_gap=float(diffs_f.std()),
            cv_gap=float(diffs_f.std() / mean) if mean > 0 else 0.0,
            longest_missing_run=int(diffs.max()) - 1,
        )
    stats.update(
        retained_count=len(r),
        retained_fraction=len(r) / float(D),
        n_retained_runs=_retained_runs(r),
        n_missing=len(missing),
    )
    return stats


def build_manifest(D: int, strides, specs=None) -> Tuple[List[Dict], List[Dict]]:
    """Build the immutable mask manifest (one row per mask_id) and a parallel gap
    statistics table. Returns (manifest_rows, gap_rows). Deterministic."""
    specs = specs or default_mask_specs()
    manifest, gaps = [], []
    for R in strides:
        for spec in specs:
            ret = generate_mask(spec, D, R)
            mid = mask_id(spec, R)
            g = gap_statistics(ret, D)
            missing = [z for z in range(D) if z not in set(ret)]
            row = dict(
                mask_id=mid, mask_family=spec.family, severity=spec.severity,
                nominal_R=int(R), D=int(D), seed=int(spec.seed),
                primary=bool(spec.primary),
                retained_count=g["retained_count"],
                retained_fraction=round(g["retained_fraction"], 6),
                min_gap=g["min_gap"], max_gap=g["max_gap"],
                mean_gap=round(g["mean_gap"], 4), median_gap=round(g["median_gap"], 4),
                std_gap=round(g["std_gap"], 4), cv_gap=round(g["cv_gap"], 4),
                longest_missing_run=g["longest_missing_run"],
                n_retained_runs=g["n_retained_runs"],
                retained_sha256=retained_hash(ret),
                retained_indices=ret, missing_indices=missing,
            )
            manifest.append(row)
            gaps.append(dict(mask_id=mid, mask_family=spec.family, severity=spec.severity,
                             nominal_R=int(R), **{k: g[k] for k in (
                                 "min_gap", "max_gap", "mean_gap", "median_gap",
                                 "std_gap", "cv_gap", "longest_missing_run",
                                 "n_retained_runs", "retained_count",
                                 "retained_fraction", "n_missing")}))
    return manifest, gaps


def verify_manifest_row(row: Dict, D: int) -> bool:
    """Fail-loud check that a manifest row is internally consistent and reproduces
    its own hash and expected retained count for its nominal R."""
    ret = row["retained_indices"]
    assert ret == sorted(set(ret)), "not sorted/unique"
    assert ret[0] == 0 and ret[-1] == D - 1, "anchors missing"
    assert all(0 <= i <= D - 1 for i in ret), "index out of range"
    assert len(ret) == K_R(D, row["nominal_R"]), "retained count != K_R"
    assert row["retained_sha256"] == retained_hash(ret), "hash mismatch"
    return True


if __name__ == "__main__":
    D = 153
    man, gaps = build_manifest(D, (2, 4, 8))
    for r in man:
        verify_manifest_row(r, D)
    print(f"{len(man)} masks, all rows verified. K: "
          + ", ".join(f"R{R}={K_R(D,R)}" for R in (2, 4, 8)))
    for r in man:
        if r["nominal_R"] == 4:
            print(f"  {r['mask_id']:28s} K={r['retained_count']} "
                  f"maxgap={r['max_gap']:2d} cv={r['cv_gap']:.2f} "
                  f"runs={r['n_retained_runs']}")
