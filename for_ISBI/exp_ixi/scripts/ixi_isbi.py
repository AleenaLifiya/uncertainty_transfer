"""IXI-tiny port of the for_ISBI protocol (second dataset; added 2026-10-23 at the user's request). SAINT only.

Data      <repository>/IXI_tiny (READ-ONLY), T1, 83x44x55, through-plane axis 0 (S-I, 2.18 mm; the FINEST axis).
          Prepared with the IR3C IXI recipe (ir3c_ixi.protocol_ixi.prepare: 1.0/99.5 percentile window -> [0,1], axis 0 ->
          position 2) with a central depth crop to D = 81 so that uniform x4 (K=21) and x8 (K=11) grids contain both end
          slices exactly (83 is not 1 mod 4). Images are NOT skull-stripped: uncertainty and PSNR are scored inside the
          provided brain mask (IXI_tiny/label), moved/cropped identically.
Model     the IXI-tiny SAINT of IR3C_experiments_V2 stage 2 (BraTS SAINT fine-tuned on IXI-tiny train, uniform masks,
          selected on IXI val): stage2_finetune/IXI/checkpoints/SAINT_rfn/SAINT_rfn_step000654.pt (full MSR + RFN).
          Marginals are computed exactly as ir3c_v2.baselines_v2.saint_reconstruct_v2 (IXI voxel spacing in the meta-kernel,
          separate sagittal/coronal stacks for the non-square 44x55 grid), stopped before the RFN.
Roles     from the frozen IXI split (train 436 / val 30 / test 100): SAINT was trained on train and selected on val.
          SEL = val[0:10]; FIT = val[10:30] + test[0:10]; CAL = test[10:40]; TEST = test[40:100] (TEST only original test).
Geometry  gap / nearest distance in mm = slices x 2.18.
"""
from __future__ import annotations

from project_paths import root_path

import copy
import json
import os
import sys

import numpy as np
import torch

ROOT = root_path('')
for p_ in (ROOT, os.path.join(ROOT, "IR3C_experiments_V2"), os.path.join(ROOT, "IR3C_more_comparisons")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)
from isbi import common as C          # noqa: E402
from isbi import models as M          # noqa: E402
from ir3c_ixi import protocol_ixi as PX   # noqa: E402

D = 81
SP = 2.18
EXP = os.path.join(C.ISBI, "exp_ixi")
SPLITS = os.path.join(ROOT, "runs", "IR3C_IXItiny_08Sept", "ixi_splits_IR3C_IXItiny_08Sept.json")
SAINT_CKPT = os.path.join(ROOT, "IR3C_experiments_V2", "stage2_finetune", "IXI", "checkpoints", "SAINT_rfn",
                          "SAINT_rfn_step000654.pt")
DEV = "cuda"


def _abs(f):
    return f if f.startswith("/") else os.path.join(ROOT, f)


def split_files():
    sp = json.load(open(SPLITS))
    return {k: [_abs(f) for f in sp[k]["files"]] for k in ("train", "val", "test")}


def roles():
    s = split_files()
    sid = lambda fs: [PX.subject_of(f) for f in fs]  # noqa: E731
    r = dict(SEL=sid(s["val"][:10]), FIT=sid(s["val"][10:30] + s["test"][:10]), CAL=sid(s["test"][10:40]),
             TEST=sid(s["test"][40:100]), TRAIN=sid(s["train"]))
    allr = [p for k in ("SEL", "FIT", "CAL", "TEST") for p in r[k]]
    assert len(set(allr)) == len(allr) and not (set(allr) & set(r["TRAIN"]))
    return r


def image_path(subject):
    return os.path.join(PX.IXI_ROOT, "image", f"{subject}_image.nii.gz")


def load(subject):
    """(vol01 (44,55,81) float32 in [0,1], brain mask (44,55,81) bool)."""
    import nibabel as nib
    p = PX.prepare_one(image_path(subject), D=D)
    v = p["vol01"].astype(np.float32)
    lab = nib.load(os.path.join(PX.IXI_ROOT, "label", f"{subject}_label.nii.gz")).get_fdata() > 0
    lab = np.moveaxis(lab, PX.THROUGH_PLANE_AXIS, 2)
    ds = int(p["depth_start"])
    lab = lab[:, :, ds:ds + D]
    assert lab.shape == v.shape, (lab.shape, v.shape)
    return v, lab


def make_mask(pattern, R, subject, rep):
    return C.make_mask(pattern, R, "IXI|" + subject, rep, D)


def geometry(ret):
    g = C.gap_geometry(ret, D)
    g["gap_mm"] = g["gap_mm"] * SP
    g["near_mm"] = g["near_mm"] * SP
    return g


def linear(gt, ret):
    """Linear interpolation along z between the acquired slices (uniform physical spacing)."""
    ret = np.asarray(sorted(ret))
    z = np.arange(D)
    j = np.clip(np.searchsorted(ret, z, side="right") - 1, 0, len(ret) - 2)
    l, r = ret[j], ret[j + 1]
    a = ((z - l) / (r - l)).astype(np.float32)
    return ((1 - a) * gt[:, :, l] + a * gt[:, :, r]).astype(np.float32)


def adapter_input(gt, ret, R):
    """SAINT adapter: uniform grid values = linear interpolation of the acquired slices (identity if uniform)."""
    return linear(gt, ret), C.uniform_mask(R, D)


def copy_back(x, gt, ret):
    x = np.array(x, np.float32, copy=True)
    x[:, :, ret] = gt[:, :, ret]
    return x


# ------------------------------------------------------------------ SAINT
def load_saint():
    from comparisons_more import saint_adapter as SA
    m, _ = SA.build_model(DEV, stage=2)
    m.load_state_dict(torch.load(SAINT_CKPT, map_location="cpu", weights_only=False)["state_dict"])
    m.eval()
    m.stage = 2
    return m


@torch.no_grad()
def saint_marginals(model, vin, G, R):
    """Stage-1 (MSR) marginals on the uniform grid G, exactly as baselines_v2.saint_reconstruct_v2 up to the RFN."""
    from comparisons_more import saint_adapter as SA
    from ir3c_v2 import baselines_v2 as BV
    G = sorted(int(g) for g in G)
    assert G[0] == 0 and G[-1] == D - 1 and set(np.diff(G).tolist()) == {R}
    sag_np, cor_np = BV.saint_view_stacks(vin, G)
    sag, cor = torch.from_numpy(sag_np).to(DEV), torch.from_numpy(cor_np).to(DEV)
    dist = torch.from_numpy(SA.get_dist(R, spacing=BV.SPACING["IXI"], n_slots=None)).unsqueeze(0).to(DEV)
    model.eval()
    weights = None
    for i in range(R):
        wi = model.GW(dist[:, [i], :, :]).view(1, 1, 64, 3, 3)
        weights = wi if weights is None else torch.cat((weights, wi), 1)
    cs = torch.clamp(model.marginal_inference(sag, weights, chunk=16), 0.0, 1.0)
    cc = torch.clamp(model.marginal_inference(cor, weights, chunk=16), 0.0, 1.0).permute(1, 0, 2)
    assert cs.shape[2] >= D and cc.shape[2] >= D, (cs.shape, cc.shape)   # K*R phase slots; keep the first D (as V2)
    return cs[:, :, :D].float().cpu().numpy(), cc[:, :, :D].float().cpu().numpy()


@torch.no_grad()
def saint_fuse(model, sag, cor):
    """Deterministic RFN, one axial slice at a time (as saint_reconstruct_v2)."""
    vol = torch.stack([torch.from_numpy(sag), torch.from_numpy(cor)]).permute(3, 0, 1, 2).to(DEV)
    ref = [model.RFN(vol[[i]])[0] for i in range(vol.shape[0])]
    return torch.clamp(torch.cat(ref).permute(1, 2, 0), 0.0, 1.0).float().cpu().numpy()


def build_prob(model, ckpt=None):
    prfn = M.ProbRFN(copy.deepcopy(model.RFN)).to(DEV)
    if ckpt:
        prfn.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    prfn.eval()
    return prfn


def psnr_brain(x, gt, brain, ret):
    miss = np.setdiff1d(np.arange(D), ret)
    m = np.zeros_like(brain)
    m[:, :, miss] = brain[:, :, miss]
    return float(10 * np.log10(1.0 / np.mean((x[m] - gt[m]) ** 2)))
