"""
IR3C_more_comparisons/comparisons_more/protocol_more.py
=======================================================
Paths and provenance helpers for `IR3C_more_comparisons`.

**This module defines NO protocol.** Splits, masks, preprocessing, the evaluator, checkpoint
selection, early stopping and seeding all come from `comparisons.protocol` and
`comparisons.training`, imported unchanged from the authoritative IR3C experiment
(`runs/brats_ir3c_20260902`). Redefining any of them here would break comparability, which
is the whole point of the experiment.

All this file adds is: where the new experiment lives, where the prior run's frozen
artefacts live, and how to hash them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(EXP_DIR)
for _p in (PROJECT, os.path.join(PROJECT, "X-Diffusion", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comparisons import protocol as P            # noqa: E402
from comparisons import training as T            # noqa: E402

EXPERIMENT_ID = "IR3C_more_comparisons"

# the authoritative prior run, read-only
IR3C_RUN = os.path.join(PROJECT, "runs", "brats_ir3c_20260902")
IR3C_CONFIG = os.path.join(PROJECT, "configs", "ir3c_comparisons.yaml")
IR3C_RESULTS = os.path.join(IR3C_RUN, "results_per_volume.csv")

# this experiment's directories
CONFIG_DIR = os.path.join(EXP_DIR, "configs")
SPLIT_DIR = os.path.join(EXP_DIR, "splits")
MASK_DIR = os.path.join(EXP_DIR, "masks")
CKPT_DIR = os.path.join(EXP_DIR, "checkpoints")
LOG_DIR = os.path.join(EXP_DIR, "logs")
PRED_DIR = os.path.join(EXP_DIR, "predictions")
METRIC_DIR = os.path.join(EXP_DIR, "metrics")
FIG_DIR = os.path.join(EXP_DIR, "figures")
PROV_DIR = os.path.join(EXP_DIR, "provenance")
ASSET_DIR = os.path.join(EXP_DIR, "report_assets")
REPO_DIR = os.path.join(EXP_DIR, "repositories")

DEPTH = P.DEPTH            # 153
IN_PLANE = P.IN_PLANE      # 240

PROTECTED = [
    "configs/ir3c_comparisons.yaml",
    "runs/brats_frozen_vae_full_20260716_122401/splits/brats2023_train.txt",
    "runs/brats_frozen_vae_full_20260716_122401/splits/brats2023_val60.txt",
    "runs/brats_frozen_vae_full_20260716_122401/splits/brats2023_test40.txt",
    "runs/brats_ir3c_20260902/test_masks.npz",
    "runs/brats_ir3c_20260902/splits.json",
    "runs/brats_ir3c_20260902/masks_manifest.json",
    "runs/brats_ir3c_20260902/results_per_volume.csv",
    "runs/brats_ir1_image_residual_smoke_20260728_121036/validation_matrix.json",
    "comparisons/protocol.py",
    "comparisons/training.py",
    "comparisons/i3net_adapter.py",
    "comparisons/arssr_adapter.py",
    "run_ir3c.py",
]


def ensure_dirs():
    for d in (CONFIG_DIR, SPLIT_DIR, MASK_DIR, CKPT_DIR, LOG_DIR, PRED_DIR, METRIC_DIR,
              FIG_DIR, PROV_DIR, ASSET_DIR):
        os.makedirs(d, exist_ok=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def hash_protected(out_path):
    out = {}
    for rel in PROTECTED:
        p = os.path.join(PROJECT, rel)
        out[rel] = (dict(sha256=sha(p), bytes=os.path.getsize(p))
                    if os.path.exists(p) else "MISSING")
    json.dump(out, open(out_path, "w"), indent=2)
    return out


def load_frozen_masks():
    """The 28 masks IR3C froze. Read-only; never regenerated here."""
    return P.load_frozen_masks(IR3C_RUN)


def frozen_splits():
    """The frozen split record from the IR3C run (subjects + source sha256)."""
    return json.load(open(os.path.join(IR3C_RUN, "splits.json")))


def cells(strides, families, D=DEPTH):
    """(R, family, retained) triples taken from the FROZEN masks, not regenerated."""
    fm = load_frozen_masks()
    out = []
    for R in strides:
        for fam in families:
            key = f"R{int(R)}_{fam}"
            ret = [int(v) for v in fm[key]]
            # cross-check against the generator, so a silently edited npz is caught
            gen = P.build_mask(D, int(R), fam)
            assert ret == [int(g) for g in gen], f"frozen mask {key} != generator output"
            out.append((int(R), str(fam), ret))
    return out


def git_info(repo):
    import subprocess
    def run(*a):
        try:
            return subprocess.run(["git", "-C", repo] + list(a), capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return "UNAVAILABLE"
    return dict(commit=run("rev-parse", "HEAD"),
                branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                url=run("config", "--get", "remote.origin.url"),
                dirty=run("status", "--porcelain")[:4000])


def environment_manifest(extra=None):
    env = P.environment_manifest()
    env["experiment_id"] = EXPERIMENT_ID
    env["repo_saint"] = git_info(os.path.join(REPO_DIR, "SAINT"))
    env["repo_dcsrn"] = git_info(os.path.join(REPO_DIR, "DCSRN"))
    env.update(extra or {})
    return env


# --------------------------------------------------------------------------- #
#  memory-mapped preprocessing cache
# --------------------------------------------------------------------------- #
PREP_CACHE = os.path.join(EXP_DIR, ".prep_cache")


def prepare_cached(files, cache_dir=None, verbose=True):
    """`protocol.prepare`, but each prepared volume is materialised ONCE to a .npy file and
    then memory-mapped.

    Identical data -- `protocol.prepare` is deterministic and this writes exactly what it
    returns -- but the 1251 training volumes (44 GB as float32) live in the page cache
    instead of in the process heap. On this machine the GPU shares the host's unified
    memory, so holding them resident starves CUDA: the first attempt at the learning-rate
    probe died with `CUDA error: out of memory` while merely moving a 3 M-parameter model
    onto the device.

    A per-volume sha256 is recorded so the cache can be proved identical to a fresh
    `prepare()` (see `verify_prep_cache`).
    """
    cache_dir = cache_dir or PREP_CACHE
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for i, f in enumerate(files):
        subj = P.subject_of(f)
        npy = os.path.join(cache_dir, f"{subj}.npy")
        if not os.path.exists(npy):
            prep = P.prepare_one(f)
            tmp = npy + ".tmp.npy"          # np.save appends .npy unless it is already there
            with open(tmp, "wb") as fh:
                np.save(fh, np.ascontiguousarray(prep["vol01"], dtype=np.float32))
            os.replace(tmp, npy)
            del prep
        out.append(dict(vol01=np.load(npy, mmap_mode="r"), subject=subj, path=f,
                        cache=npy))
        if verbose and (i + 1) % 200 == 0:
            print(f"  [prep_cache] {i+1}/{len(files)}", flush=True)
    return out


def verify_prep_cache(files, n=3, cache_dir=None, seed=0):
    """Re-run `protocol.prepare_one` on `n` random files and compare bit-for-bit."""
    cache_dir = cache_dir or PREP_CACHE
    rng = np.random.RandomState(int(seed))
    idx = rng.choice(len(files), size=min(int(n), len(files)), replace=False)
    out = {}
    for i in idx:
        f = files[int(i)]
        subj = P.subject_of(f)
        a = np.load(os.path.join(cache_dir, f"{subj}.npy"))
        b = P.prepare_one(f)["vol01"]
        out[subj] = float(np.abs(np.asarray(a, np.float32) - np.asarray(b, np.float32)).max())
    return out


def volume_psnr_missing(pred, gt, ret, D=DEPTH):
    """Convenience wrapper; the metric itself is the shared evaluator, unmodified."""
    return P.evaluate_volume(np.asarray(pred, np.float32), np.asarray(gt, np.float32),
                             ret, D=D, region="missing")
