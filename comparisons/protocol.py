"""
comparisons/protocol.py
=======================
THE SINGLE SOURCE OF TRUTH for IR3C.

Splits, frozen acquisition masks, preprocessing, the shared evaluator, checkpoint
selection and seeding all live here. Every method -- IR3 (read-only), I3Net, ArSSR and the
classical interpolators -- goes through these functions, so no definition can drift between
methods (ir3_comparisons.md §8).

Nothing in this module trains. Nothing in this module is method-specific.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(PROJECT, "X-Diffusion", "scripts")
for _p in (SCRIPTS, PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import image_residual as IR                       # noqa: E402
import ir1_audit_common as AC                     # noqa: E402
import nonuniform_masks as NM                     # noqa: E402
from ldm.models.diffusion import fidelity_runtime as FR  # noqa: E402

# --------------------------------------------------------------------------- #
#  fixed protocol constants (all inherited from IR3; none is new)
# --------------------------------------------------------------------------- #
DEPTH = 153
IN_PLANE = 240
P_LO, P_HI = 1.0, 99.5
PEAK = 1.0
AUTH_SSIM = "A_fullslice_current"
SSIM_VARIANTS = (AUTH_SSIM, "B_brain_bbox")

IR3_RUN = os.path.join(PROJECT, "runs", "brats_ir3_standalone_20260830_182349")
IR1_RUN = os.path.join(PROJECT, "runs", "brats_ir1_image_residual_smoke_20260728_121036")
L3_RUN = os.path.join(PROJECT, "runs", "brats_frozen_vae_full_20260716_122401")

SPLIT_FILES = dict(
    train=os.path.join(L3_RUN, "splits", "brats2023_train.txt"),
    val=os.path.join(L3_RUN, "splits", "brats2023_val60.txt"),
    test=os.path.join(L3_RUN, "splits", "brats2023_test40.txt"),
)
N_TRAIN, N_VAL, N_TEST = 1251, 6, 88

FAMILIES = ["U0", "U1", "U2", "U3", "U4", "U5", "U6"]
STRIDES_IN_DIST = [2, 4, 8]
STRIDES_TEST = [2, 4, 8, 16]

# The metric column order used in EVERY comparable table (§9 Rule 2).
METRIC_ORDER = ["psnr_masked", "psnr_unmasked", "ssim", "ssim_B_brain_bbox",
                "nmse", "mae", "bg_haze"]
METRIC_LABELS = {
    "psnr_masked": "PSNR-brain (dB) ↑",
    "psnr_unmasked": "PSNR-full (dB) ↑",
    "ssim": "SSIM ↑",
    "ssim_B_brain_bbox": "SSIM-bbox ↑",
    "nmse": "NMSE ↓",
    "mae": "MAE ↓",
    "bg_haze": "bg energy ↓",
}
METRIC_HIGHER_IS_BETTER = {"psnr_masked": True, "psnr_unmasked": True, "ssim": True,
                           "ssim_B_brain_bbox": True, "nmse": False, "mae": False,
                           "bg_haze": False}


# --------------------------------------------------------------------------- #
#  seeding (§12)
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    """Seed python, numpy, torch CPU and torch CUDA. Returns the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return int(seed)


def worker_init_fn(worker_id: int):
    """Deterministic dataloader workers (§12)."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


# --------------------------------------------------------------------------- #
#  splits (§6)
# --------------------------------------------------------------------------- #
def subject_of(path): return os.path.basename(os.path.dirname(path))


def _read_split(path, n):
    lines = [l.strip() for l in open(path) if l.strip()]
    return [os.path.join(PROJECT, l) for l in lines[:int(n)]]


def load_splits():
    """The IR3 splits, verbatim. Subject-level, disjoint (asserted)."""
    out = dict(train=_read_split(SPLIT_FILES["train"], N_TRAIN),
               val=_read_split(SPLIT_FILES["val"], N_VAL),
               test=_read_split(SPLIT_FILES["test"], N_TEST))
    sets = {k: set(subject_of(p) for p in v) for k, v in out.items()}
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"
    return out


def freeze_splits(run_dir):
    """Write splits.json so the experiment reproduces without regenerating anything."""
    sp = load_splits()
    payload = {k: dict(n=len(v),
                       subjects=[subject_of(p) for p in v],
                       files=[os.path.relpath(p, PROJECT) for p in v],
                       source_file=os.path.relpath(SPLIT_FILES[k], PROJECT),
                       source_sha256=AC.sha_file(SPLIT_FILES[k]))
               for k, v in sp.items()}
    payload["_note"] = ("Subject-level split inherited verbatim from IR3. "
                        "The first N lines of each source file are taken, as IR3 did.")
    path = os.path.join(run_dir, "splits.json")
    json.dump(payload, open(path, "w"), indent=2)
    return path, sp


# --------------------------------------------------------------------------- #
#  frozen acquisition masks (§6)
# --------------------------------------------------------------------------- #
def family_seeds():
    """The fixed per-family mask seeds IR3 used. Read-only."""
    return json.load(open(os.path.join(IR1_RUN, "validation_matrix.json")))["family_seed"]


def build_mask(D, R, family, fs=None):
    fs = fs or family_seeds()
    if family == "U0":
        return [int(x) for x in NM.uniform_retained(D, R)]
    return [int(x) for x in IR.masks.nonuniform_mask(D, R, family, int(fs[family]))]


def eval_grid(strides=None, families=None, D=DEPTH):
    """The frozen (R, family, retained) evaluation grid, identical for every method."""
    fs = family_seeds()
    return [(int(R), fam, build_mask(D, R, fam, fs))
            for R in (strides or STRIDES_TEST) for fam in (families or FAMILIES)], fs


def freeze_masks(run_dir, D=DEPTH):
    """Persist every evaluation mask to test_masks.npz + a human-readable manifest."""
    grid, fs = eval_grid()
    arrays, rows = {}, []
    for R, fam, ret in grid:
        key = f"R{R}_{fam}"
        arrays[key] = np.asarray(ret, dtype=np.int32)
        gaps = np.diff(np.asarray(ret))
        rows.append(dict(key=key, R=int(R), family=fam, D=int(D), K=len(ret),
                         seed=(0 if fam == "U0" else int(fs[fam])),
                         max_gap=int(gaps.max()), min_gap=int(gaps.min()),
                         exactly_uniform=bool(len(set(gaps.tolist())) == 1),
                         sha256=AC.sha_idx(ret)))
    np.savez_compressed(os.path.join(run_dir, "test_masks.npz"), **arrays)
    json.dump(dict(depth=int(D), family_seed=fs, n_masks=len(rows), masks=rows,
                   note="One mask per (R, family); shared by every subject and every "
                        "method. Pure function of (D, family, R, seed) -- never sees "
                        "image intensity or segmentation labels."),
              open(os.path.join(run_dir, "masks_manifest.json"), "w"), indent=2)
    return rows


def load_frozen_masks(run_dir):
    z = np.load(os.path.join(run_dir, "test_masks.npz"))
    return {k: [int(v) for v in z[k]] for k in z.files}


def missing_of(ret, D=DEPTH):
    return IR.masks.missing_of(ret, D)


# --------------------------------------------------------------------------- #
#  preprocessing (§A.3) -- identical for every method
# --------------------------------------------------------------------------- #
def prepare(files):
    """Project-standard preprocessing: (240,240,153) float32 in [0,1]."""
    preps = FR.prepare_volumes(list(files), P_LO, P_HI, DEPTH)
    for p, f in zip(preps, files):
        p["subject"] = subject_of(f)
        p["path"] = f
    return preps


def prepare_one(path):
    return prepare([path])[0]


# --------------------------------------------------------------------------- #
#  in-plane zero-padding adapter (I3Net needs 256x256; §D row 21)
# --------------------------------------------------------------------------- #
def pad_to(vol_hw_d, target=256):
    """Symmetric zero-pad the in-plane axes of an (H,W,D) volume. Returns (padded, pad)."""
    H, W, D = vol_hw_d.shape
    assert H <= target and W <= target, f"refuse to resize {vol_hw_d.shape}"
    ph, pw = target - H, target - W
    top, left = ph // 2, pw // 2
    pad = (top, ph - top, left, pw - left)
    out = np.pad(vol_hw_d, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)),
                 mode="constant", constant_values=0.0)
    return out.astype(np.float32), pad


def unpad_from(vol_hw_d, pad):
    """Exact inverse of pad_to."""
    t, b, l, r = pad
    H, W, _ = vol_hw_d.shape
    return vol_hw_d[t:H - b, l:W - r]


# --------------------------------------------------------------------------- #
#  the ONE evaluator (§8) -- every method's prediction enters here
# --------------------------------------------------------------------------- #
def evaluate_volume(pred_hw_d, gt_hw_d, ret, D=DEPTH, region="missing"):
    """Score one reconstruction.

    Parameters
    ----------
    pred_hw_d, gt_hw_d : (240, 240, 153) float32 in [0,1]
    ret                : retained (acquired) slice indices
    region             : "missing"  -> score ONLY the reconstructed slices (primary)
                         "whole"    -> score every slice of the volume

    All metric definitions come from ir1_audit_common, unchanged, so they are bit-identical
    to the ones IR3 was scored with.
    """
    assert pred_hw_d.shape == gt_hw_d.shape == (IN_PLANE, IN_PLANE, D), \
        f"shape {pred_hw_d.shape} != {(IN_PLANE, IN_PLANE, D)}"
    miss = missing_of(ret, D)
    idx = miss if region == "missing" else list(range(D))
    m = AC.volume_metrics(np.asarray(pred_hw_d, np.float32), np.asarray(gt_hw_d, np.float32),
                          idx, with_ssim=True, variants=SSIM_VARIANTS)
    md = np.asarray(sorted(int(i) for i in idx), dtype=int)
    bg = gt_hw_d[:, :, md] <= 0.0
    m["bg_haze"] = float(pred_hw_d[:, :, md][bg].mean()) if bg.sum() else float("nan")
    m["region"] = region
    m["n_scored_slices"] = int(md.size)
    return m


def interpolate_baseline(gt_hw_d, ret, kind, D=DEPTH):
    """nearest / linear / cubic along z, from the acquired slices only.

    Identical call path to the one IR3 used (`nonuniform_method_adapter.interpolate_nu`
    for nearest/cubic, `image_residual.inputs.linear_interp_volume` for linear), so the
    stored IR3 baseline numbers are reproduced bit-for-bit.
    """
    import nonuniform_method_adapter as A
    if kind == "linear":
        return IR.inputs.linear_interp_volume(gt_hw_d, ret, D).astype(np.float32)
    return A.interpolate_nu(dict(vol01=gt_hw_d, depth=int(D)), ret, kind)[0].astype(np.float32)


# --------------------------------------------------------------------------- #
#  leakage guard (§7) -- assert a model input cannot depend on missing slices
# --------------------------------------------------------------------------- #
def leakage_probe(gt_hw_d, ret, D=DEPTH, seed=0):
    """Return a copy of the volume with every MISSING slice replaced by noise.

    Any adapter whose input tensor is bit-identical on the original and the probe provably
    cannot see the ground truth it is asked to reconstruct.
    """
    rng = np.random.RandomState(seed)
    out = np.array(gt_hw_d, dtype=np.float32, copy=True)
    for z in missing_of(ret, D):
        out[:, :, z] = rng.rand(*gt_hw_d.shape[:2]).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
#  shared checkpoint selection (§4, §13) -- identical rule for every method
# --------------------------------------------------------------------------- #
def select_checkpoint(per_ckpt, slack=0.01):
    """Best checkpoint by macro brain-masked PSNR on the VALIDATION matrix, subject to the
    IR3 SSIM guardrail (macro_ssim >= linear_macro_ssim - slack). Test data is never read.

    `per_ckpt` rows need: step, epoch, path, macro_psnr, macro_ssim, linear_macro_ssim.
    """
    import pandas as pd
    df = pd.DataFrame(per_ckpt).copy()
    df["ssim_threshold"] = df["linear_macro_ssim"] - float(slack)
    df["eligible"] = df["macro_ssim"] >= df["ssim_threshold"]
    if (df.step == 0).any():
        df.loc[df.step == 0, "eligible"] = True
    elig = df[df.eligible]
    if len(elig) == 0:
        elig = df
    sel = elig.sort_values(["macro_psnr", "step"], ascending=[False, True]).iloc[0]
    df["selected"] = df.step == sel.step
    return df, dict(selected_step=int(sel.step), selected_epoch=int(sel.epoch),
                    selected_path=str(sel.path),
                    selected_macro_psnr=float(sel.macro_psnr),
                    selected_macro_ssim=float(sel.macro_ssim),
                    n_candidates=int(len(df)), n_eligible=int(len(elig)),
                    criterion="max macro brain-masked PSNR on missing slices, "
                              f"guardrail macro_ssim >= linear_macro_ssim - {slack}")


class EarlyStopper:
    """Shared early-stopping rule (§4): stop after `patience` validations without an
    improvement in the checkpoint metric. Identical for every method."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = -np.inf
        self.bad = 0
        self.best_step = 0

    def update(self, value: float, step: int) -> bool:
        """Returns True if training should stop."""
        if np.isfinite(value) and value > self.best + self.min_delta:
            self.best, self.bad, self.best_step = float(value), 0, int(step)
        else:
            self.bad += 1
        return self.bad >= self.patience


# --------------------------------------------------------------------------- #
#  provenance (§11)
# --------------------------------------------------------------------------- #
def _git(repo, *args):
    try:
        return subprocess.run(["git", "-C", repo] + list(args), capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:
        return "UNAVAILABLE"


def environment_manifest():
    import platform
    import pandas
    import skimage
    import scipy
    import monai
    import nibabel
    env = dict(
        python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, torch_cuda=torch.version.cuda,
        cuda_available=bool(torch.cuda.is_available()),
        gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
        gpu_capability=(str(torch.cuda.get_device_capability(0))
                        if torch.cuda.is_available() else "n/a"),
        numpy=np.__version__, pandas=pandas.__version__, skimage=skimage.__version__,
        scipy=scipy.__version__, monai=monai.__version__, nibabel=nibabel.__version__,
        conda_env=os.environ.get("CONDA_DEFAULT_ENV", "?"),
    )
    env["repo_main_commit"] = _git(PROJECT, "rev-parse", "HEAD") or "NOT A GIT REPOSITORY"
    env["repo_i3net_commit"] = _git(os.path.join(PROJECT, "I3Net"), "rev-parse", "HEAD")
    env["repo_i3net_dirty"] = _git(os.path.join(PROJECT, "I3Net"), "status", "--porcelain")[:2000]
    env["repo_arssr_commit"] = _git(os.path.join(PROJECT, "ArSSR"), "rev-parse", "HEAD")
    env["repo_arssr_dirty"] = _git(os.path.join(PROJECT, "ArSSR"), "status", "--porcelain")[:2000]
    return env


def sha_file(path):
    return AC.sha_file(path)
