"""
ir3c_ixi/protocol_ixi.py
========================
THE SINGLE SOURCE OF TRUTH FOR `IR3C_IXItiny_08Sept`.

It is a thin IXI-geometry adapter around the IR3C protocol. Everything that defines a
*result* -- preprocessing recipe, mask generator, mask family seeds, metric
implementations, interpolation baselines (including bicubic), checkpoint-selection rule,
seeding -- is IMPORTED from the IR3C modules and used unchanged. The only things defined
here are the IXI-specific *interface* facts that the IR3C modules take as parameters
anyway:

    depth 153 -> 83            (IXI-tiny volumes have 83 slices along the S-I axis)
    in-plane 240 -> 44 x 55    (IXI-tiny volumes are 44 x 55 in the axial plane)
    through-plane axis 2 -> 0  (the S-I axis is array axis 0 in the IXI-tiny grid)
    BraTS subject list -> the IXI-tiny subject list, split stratified by site

Nothing under `comparisons/`, `image_residual/`, `ir2_*`, `ir1_audit_common.py`,
`nonuniform_method_adapter.py`, `ir3c_oai/`, `IXI_tiny/` or any `runs/brats_*` /
`runs/IR3C_OAI_08Sept` directory is written to or modified.

WHY axis 0. IR3C decimates BraTS array axis 2, which the BraTS affine (LPS) shows is the
SUPERIOR-INFERIOR axis: IR3C reconstructs missing AXIAL slices. The IXI-tiny affine (SRA)
puts the superior-inferior axis at array position 0, so axis 0 is the same ANATOMICAL
reconstruction direction, and `prepare_volume(axis=0)` moves it to position 2 exactly as
the BraTS call does with `axis=2` (a no-op there). It is also the axis with 83 samples,
the most of the three, and the one along which the anatomy changes least per sample
(normalised adjacent-slice difference 0.126, against 0.282 and 0.240) -- i.e. the axis
that is densely enough sampled for decimation to be a meaningful experiment. The choice is
made from cohort geometry alone and does not look at any model score.

The ONE function that is re-stated rather than imported is `evaluate_volume`: the IR3C
version (`comparisons.protocol.evaluate_volume`) begins with

    assert pred.shape == gt.shape == (240, 240, D)

which is a BraTS in-plane assertion, not a metric definition. The body below is the IR3C
body verbatim with that single assertion relaxed to "pred and gt agree and are 3-D"; the
metrics themselves still come from `ir1_audit_common.volume_metrics` with the identical
arguments and the identical SSIM variants, so IR3C, IR3C_OAI_08Sept and
IR3C_IXItiny_08Sept numbers are produced by bit-identical metric code.
"""
from __future__ import annotations

from project_paths import root_path

import glob
import json
import os
import sys

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

from comparisons import protocol as P              # noqa: E402  IR3C source of truth
import ir1_audit_common as AC                      # noqa: E402  metric implementations
import image_residual as IR                        # noqa: E402
import nonuniform_method_adapter as A              # noqa: E402  bicubic lives here

RUN_ID = "IR3C_IXItiny_08Sept"
DAY = "08Sept"

# --------------------------------------------------------------------------- #
#  IXI-tiny geometry (verified in the dataset audit, not assumed)
# --------------------------------------------------------------------------- #
IXI_ROOT = root_path('IXI_tiny')     # READ-ONLY source data
IXI_GLOB = os.path.join(IXI_ROOT, "image", "*_image.nii.gz")
RAW_SHAPE = (83, 44, 55)          # every one of the 566 volumes; no variation
SPACING_MM = (2.18, 4.13, 3.95)   # every one of the 566 volumes; anisotropic
AXCODES = "SRA"                   # one identical affine across the whole cohort
THROUGH_PLANE_AXIS = 0            # = S = superior-inferior = axial slices (see module doc)
DEPTH = 83                        # slices along the through-plane axis; no depth crop
IN_PLANE_H = 44                   # array axis 1 of the raw grid = R-L
IN_PLANE_W = 55                   # array axis 2 of the raw grid = A-P
THROUGH_PLANE_SPACING_MM = 2.18
SITES = ("Guys", "HH", "IOP")

# inherited from IR3C unchanged
P_LO, P_HI = P.P_LO, P.P_HI                       # 1.0, 99.5
PEAK = P.PEAK                                     # 1.0
AUTH_SSIM = P.AUTH_SSIM                           # "A_fullslice_current"
SSIM_VARIANTS = P.SSIM_VARIANTS
METRIC_ORDER = P.METRIC_ORDER
METRIC_LABELS = P.METRIC_LABELS
METRIC_HIGHER_IS_BETTER = P.METRIC_HIGHER_IS_BETTER
FAMILIES = P.FAMILIES                             # U0..U6
STRIDES_IN_DIST = P.STRIDES_IN_DIST               # [2, 4, 8]
STRIDES_TEST = P.STRIDES_TEST                     # [2, 4, 8, 16]  (16 = OOD)

# the IR3C source model
IR3_RUN = P.IR3_RUN
RUN_DIR = os.path.join(PROJECT, "runs", RUN_ID)

seed_everything = P.seed_everything
worker_init_fn = P.worker_init_fn
select_checkpoint = P.select_checkpoint
EarlyStopper = P.EarlyStopper
environment_manifest = P.environment_manifest
sha_file = P.sha_file


# --------------------------------------------------------------------------- #
#  subjects and the subject-level, site-stratified split
# --------------------------------------------------------------------------- #
def subject_of(path):
    """`IXI002-Guys-0828_image.nii.gz` -> `IXI002-Guys-0828`."""
    return os.path.basename(path)[: -len("_image.nii.gz")]


def site_of(path_or_id):
    """Acquisition site from the IXI naming convention: IXI###-<SITE>-####."""
    sid = path_or_id if "/" not in str(path_or_id) else subject_of(path_or_id)
    parts = str(sid).split("-")
    return parts[1] if len(parts) > 1 else "UNKNOWN"


def all_subjects():
    """Every IXI-tiny image volume, sorted. One volume per subject (verified in audit)."""
    files = sorted(glob.glob(IXI_GLOB))
    if not files:
        raise FileNotFoundError(f"no IXI-tiny volumes under {IXI_ROOT}")
    return files


def _largest_remainder(counts, total):
    """Proportional allocation of `total` items over `counts`, largest-remainder rule."""
    n = float(sum(counts.values()))
    exact = {k: total * v / n for k, v in counts.items()}
    base = {k: int(np.floor(v)) for k, v in exact.items()}
    left = int(total - sum(base.values()))
    order = sorted(counts, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:left]:
        base[k] += 1
    return base


def make_splits(n_train, n_val, n_test, seed=42, stratify=True):
    """Deterministic SUBJECT-LEVEL split of the IXI-tiny cohort, stratified by site.

    One subject == one image file == one volume, so a subject-level split is also a
    volume-level split and no slice of a subject can appear in two splits. Within each
    site the permutation is a fixed-seed `RandomState(seed).permutation` over the sorted
    file list, so the split is reproducible from (file list, seed) alone. Site quotas are
    proportional to the cohort composition (largest-remainder rule), which keeps the
    smallest site (IOP, 74 subjects) represented in TEST.
    """
    files = all_subjects()
    need = int(n_train) + int(n_val) + int(n_test)
    if need > len(files):
        raise ValueError(f"asked for {need} subjects, only {len(files)} exist")
    by_site = {}
    for f in files:
        by_site.setdefault(site_of(f), []).append(f)
    counts = {k: len(v) for k, v in by_site.items()}
    if not stratify:
        by_site = {"ALL": files}
        counts = {"ALL": len(files)}
    q_test = _largest_remainder(counts, int(n_test))
    q_val = _largest_remainder(counts, int(n_val))
    out = dict(test=[], val=[], train=[])
    rng = np.random.RandomState(int(seed))
    for s in sorted(by_site):
        fl = sorted(by_site[s])
        order = rng.permutation(len(fl))
        take = [fl[i] for i in order]
        nt, nv = int(q_test[s]), int(q_val[s])
        out["test"] += take[:nt]
        out["val"] += take[nt:nt + nv]
        out["train"] += take[nt + nv:]
    # TRAIN is trimmed (never re-drawn) if the config asks for fewer than all remaining.
    out["train"] = sorted(out["train"])[: int(n_train)]
    out["val"] = sorted(out["val"])
    out["test"] = sorted(out["test"])
    sets = {k: set(subject_of(p) for p in v) for k, v in out.items()}
    assert not (sets["train"] & sets["val"]), "train/val overlap"
    assert not (sets["train"] & sets["test"]), "train/test overlap"
    assert not (sets["val"] & sets["test"]), "val/test overlap"
    assert sum(len(v) for v in sets.values()) == need, \
        f"split sizes {[len(v) for v in sets.values()]} != {need}"
    return out


def freeze_splits(run_dir, n_train, n_val, n_test, seed=42, stratify=True):
    """Write `ixi_splits_IR3C_IXItiny_08Sept.json` with the exact IDs (§7)."""
    sp = make_splits(n_train, n_val, n_test, seed, stratify)
    files = all_subjects()
    payload = {}
    for k, v in sp.items():
        sites = {}
        for p in v:
            sites[site_of(p)] = sites.get(site_of(p), 0) + 1
        payload[k] = dict(n=len(v), sites=dict(sorted(sites.items())),
                          subjects=[subject_of(p) for p in v],
                          files=[os.path.relpath(p, PROJECT) for p in v])
    cohort_sites = {}
    for p in files:
        cohort_sites[site_of(p)] = cohort_sites.get(site_of(p), 0) + 1
    payload["_meta"] = dict(
        run_id=RUN_ID, dataset_root=IXI_ROOT, n_available=len(files),
        cohort_sites=dict(sorted(cohort_sites.items())),
        seed=int(seed), unit="subject (== one image file == one volume)",
        stratified_by=("acquisition site (Guys / HH / IOP), proportional quotas by the "
                       "largest-remainder rule" if stratify else "none"),
        strategy=("per site: sorted file list -> numpy RandomState(seed).permutation -> "
                  "first quota = test, next quota = val, remainder = train"),
        disjointness_verified=True,
        frozen_before_any_test_metric=True)
    path = os.path.join(run_dir, f"ixi_splits_{RUN_ID}.json")
    json.dump(payload, open(path, "w"), indent=2)
    return path, sp


def load_splits(run_dir):
    p = json.load(open(os.path.join(run_dir, f"ixi_splits_{RUN_ID}.json")))
    return {k: [os.path.join(PROJECT, f) for f in p[k]["files"]]
            for k in ("train", "val", "test")}


# --------------------------------------------------------------------------- #
#  masks -- IR3C generator, IR3C family seeds, IXI depth
# --------------------------------------------------------------------------- #
def family_seeds():
    """The frozen IR3C/IR3 per-family mask seeds, read-only. Reused verbatim so the
    IXI mask families are the same deterministic construction as the BraTS ones."""
    return P.family_seeds()


def build_mask(R, family, fs=None, D=DEPTH):
    """`comparisons.protocol.build_mask` at IXI depth. No new mask code."""
    return P.build_mask(int(D), int(R), family, fs)


def eval_grid(strides=None, families=None, D=DEPTH):
    """The frozen (R, family, retained) grid, identical for every method."""
    fs = family_seeds()
    return ([(int(R), fam, build_mask(R, fam, fs, D))
             for R in (strides or STRIDES_TEST) for fam in (families or FAMILIES)], fs)


def freeze_masks(run_dir, split_name, strides, families=None, D=DEPTH):
    """Persist the evaluation masks for one split to
    `ixi_{split}_masks_IR3C_IXItiny_08Sept.npz` plus a human-readable manifest (§11)."""
    grid, fs = eval_grid(strides, families, D)
    arrays, rows = {}, []
    for R, fam, ret in grid:
        key = f"R{R}_{fam}"
        arrays[key] = np.asarray(ret, dtype=np.int32)
        gaps = np.diff(np.asarray(ret))
        rows.append(dict(key=key, R=int(R), family=fam, D=int(D), K=len(ret),
                         seed=(0 if fam == "U0" else int(fs[fam])),
                         max_gap=int(gaps.max()), min_gap=int(gaps.min()),
                         max_gap_mm=round(float(gaps.max()) * THROUGH_PLANE_SPACING_MM, 2),
                         exactly_uniform=bool(len(set(gaps.tolist())) == 1),
                         sha256=AC.sha_idx(ret)))
    npz = os.path.join(run_dir, f"ixi_{split_name}_masks_{RUN_ID}.npz")
    np.savez_compressed(npz, **arrays)
    json.dump(dict(run_id=RUN_ID, split=split_name, depth=int(D), family_seed=fs,
                   n_masks=len(rows), masks=rows,
                   through_plane_spacing_mm=THROUGH_PLANE_SPACING_MM,
                   note=("One mask per (R, family); shared by EVERY subject and EVERY "
                         "method (IR3C-IXI, IR3C zero-shot, nearest, linear, cubic, "
                         "bicubic). Pure function of (D, family, R, seed); never sees "
                         "image intensity. Frozen before any test metric is computed.")),
              open(os.path.join(run_dir, f"ixi_{split_name}_masks_manifest_{RUN_ID}.json"),
                   "w"), indent=2)
    return npz, rows


def load_frozen_masks(run_dir, split_name):
    z = np.load(os.path.join(run_dir, f"ixi_{split_name}_masks_{RUN_ID}.npz"))
    return {k: [int(v) for v in z[k]] for k in z.files}


def missing_of(ret, D=DEPTH):
    return IR.masks.missing_of(ret, D)


# --------------------------------------------------------------------------- #
#  preprocessing -- the IR3C recipe, IXI axis/depth
# --------------------------------------------------------------------------- #
def prepare(files, D=DEPTH):
    """IR3C preprocessing (`brats_preprocessing_fidelity.prepare_volume`) called with the
    IXI through-plane axis and depth. The FUNCTION is unchanged: robust 1.0/99.5 percentile
    window over the non-zero foreground, clipped once and mapped once to [0,1], central
    depth crop (a no-op at depth == 83), through-plane axis moved to position 2.

    No resampling, no reorientation and no in-plane resizing is applied: the array is used
    on its native grid, so anatomical proportions are untouched.
    """
    from ldm.data.brats_preprocessing_fidelity import prepare_volume
    preps = []
    for f in files:
        p = prepare_volume(f, p_lo=P_LO, p_hi=P_HI, axis=THROUGH_PLANE_AXIS, depth=int(D),
                           target=256)
        p["subject"] = subject_of(f)
        p["site"] = site_of(f)
        p["path"] = f
        preps.append(p)
    return preps


def prepare_one(path, D=DEPTH):
    return prepare([path], D)[0]


# --------------------------------------------------------------------------- #
#  the ONE evaluator (§22) -- IR3C body, in-plane assertion relaxed
# --------------------------------------------------------------------------- #
def evaluate_volume(pred_hw_d, gt_hw_d, ret, D=DEPTH, region="missing"):
    """Score one reconstruction. Identical to `comparisons.protocol.evaluate_volume`
    except that the (240, 240, D) shape assertion is replaced by a shape-agreement check.

    region : "missing" -> score ONLY the reconstructed slices (primary, §23)
             "whole"   -> score every slice of the volume (secondary, own table)
    """
    pred_hw_d = np.asarray(pred_hw_d, np.float32)
    gt_hw_d = np.asarray(gt_hw_d, np.float32)
    assert pred_hw_d.shape == gt_hw_d.shape and pred_hw_d.ndim == 3, \
        f"shape mismatch {pred_hw_d.shape} vs {gt_hw_d.shape}"
    assert pred_hw_d.shape[2] == int(D), f"depth {pred_hw_d.shape[2]} != {D}"
    miss = missing_of(ret, D)
    idx = miss if region == "missing" else list(range(int(D)))
    m = AC.volume_metrics(pred_hw_d, gt_hw_d, idx, with_ssim=True, variants=SSIM_VARIANTS)
    md = np.asarray(sorted(int(i) for i in idx), dtype=int)
    bg = gt_hw_d[:, :, md] <= 0.0
    m["bg_haze"] = float(pred_hw_d[:, :, md][bg].mean()) if bg.sum() else float("nan")
    m["region"] = region
    m["n_scored_slices"] = int(md.size)
    return m


# --------------------------------------------------------------------------- #
#  classical baselines -- REUSED, not reimplemented (§19, §20)
# --------------------------------------------------------------------------- #
def interpolate_baseline(gt_hw_d, ret, kind, D=DEPTH):
    """Delegates straight to `comparisons.protocol.interpolate_baseline`, which is the
    IR3C implementation:

        linear   -> image_residual.inputs.linear_interp_volume
        nearest  -> nonuniform_method_adapter.interpolate_nu(..., "nearest")
        cubic    -> nonuniform_method_adapter.interpolate_nu(..., "cubic")   (global spline)
        bicubic  -> nonuniform_method_adapter.interpolate_nu(..., "bicubic")
                    -> nonuniform_method_adapter.bicubic_z(vol, ret, miss, a=-0.5)

    Every one of these is depth- and in-plane-agnostic, so the IXI call needs no new code
    and no changed parameter: only `D` differs, and `D` was already an argument.
    """
    return P.interpolate_baseline(np.asarray(gt_hw_d, np.float32), ret, kind, int(D))


BASELINES = ["nearest", "linear", "cubic", "bicubic"]


def leakage_probe(gt_hw_d, ret, D=DEPTH, seed=0):
    return P.leakage_probe(gt_hw_d, ret, int(D), seed)


# --------------------------------------------------------------------------- #
#  the IR3C source model  (READ-ONLY; never written, never overwritten)
# --------------------------------------------------------------------------- #
def load_ir3c_source_model(device="cuda", step_cap=None):
    """Load the IR3C reconstruction model exactly as `run_ir3c.py:_load_ir3` and
    `ir3c_oai.protocol_oai.load_ir3c_source_model` do.

    `step_cap=None` reproduces IR3C's `IR3_full` selection: the checkpoint chosen by the
    SHARED IR3C rule (max macro brain-masked validation PSNR on the BraTS validation
    matrix, subject to the SSIM guardrail) over ALL of IR3's checkpoints. That is the
    checkpoint IR3C reports as its reconstruction model, and it is selected on BraTS
    validation data only -- no IXI data of any kind is involved (§6).
    """
    import pandas as pd
    import torch
    per = pd.read_csv(os.path.join(IR3_RUN, "validation_metrics_per_checkpoint.csv"))
    if step_cap is not None:
        per = per[per.step <= int(step_cap)]
    _, sel = P.select_checkpoint(per.to_dict("records"))
    ckpt_path = os.path.join(IR3_RUN, sel["selected_path"])
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IR.model.build_ir1(ck["residual_scale"], arch=ck.get("arch")).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    sel["selected_abs_path"] = ckpt_path
    sel["selected_sha256"] = AC.sha_file(ckpt_path)
    sel["source_run"] = os.path.relpath(IR3_RUN, PROJECT)
    return model, sel, ck


def reconstruct(model, gt_hw_d, ret, D=DEPTH, device="cuda"):
    """IR3C inference, unchanged: `ir2_components.reconstruct_ir2`.

    Bounded tanh residual on the linear interpolant, masked to missing slices, clipped to
    [0,1], acquired slices hard-copied bit-exactly. The forward pass replicate-pads the
    volume to a multiple of 8 and crops back, which is what makes the 44 x 55 x 83 IXI
    grid run through a network trained at 240 x 240 x 153 with no architectural change.
    """
    import ir2_components as C2
    return C2.reconstruct_ir2(model, np.asarray(gt_hw_d, np.float32), ret, int(D), device)[0]
