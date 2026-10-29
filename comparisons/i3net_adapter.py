"""
comparisons/i3net_adapter.py
============================
DATASET/INTERFACE ADAPTATION ONLY. The I3Net method (architecture, the hard copy of
acquired slices, the DCT window attention, the cross-view blocks) is imported from
`I3Net/` at commit 30be7ba and is NOT modified.

What this file does, and why (ir3_comparisons.md §2, §17):

  * I3Net is hard-constrained to 256x256 in-plane. Verified empirically: IntraSliceBranch
    runs a Conv1d over `win_num_sqrt**2 = 256` tokens taken from windows of side
    `h // win_num_sqrt`, so `(h/16)**2 = 256` forces `h = 256`; independently CrossViewBlock
    does `Rearrange('b c h w -> b h c w'); PixelShuffle(2); Conv2d(64, ...)` which forces
    `h/4 = 64`. A 240x240 forward raises a channel-count RuntimeError.
    -> BraTS 240x240 is SYMMETRICALLY ZERO-PADDED to 256x256 and unpadded before scoring.
       BraTS background is already exactly 0 after the project normalisation, so the pad
       adds no information, and no scored voxel changes position.

  * I3Net has one model per integer scale (the repo ships Colon_x2/x4/x6). We therefore
    train one specialist per R in {2,4,8}, which is the method's own design.

  * I3Net assumes REGULARLY spaced slices (`lr = gt[..., ::upscale]` throughout). The
    non-uniform families U1..U6 have irregular gaps and cannot be served without replacing
    the method. I3Net is scored on U0 only; see APPLICABLE_FAMILIES.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys

import numpy as np
import torch

from . import protocol as P

I3NET_DIR = os.path.join(P.PROJECT, "I3Net")

APPLICABLE_FAMILIES = ["U0"]          # fixed uniform scale -> uniform masks only
APPLICABLE_STRIDES = [2, 4, 8]        # R=16 uniform mask on D=153 is NOT exactly uniform
LR_SLICE_PATCH = 4                    # I3Net default (config.py --lr_slice_patch)


@contextlib.contextmanager
def _in_repo():
    """select_model() opens 'opt/i3net.json' by RELATIVE path; run it from the repo root."""
    cwd = os.getcwd()
    added = I3NET_DIR not in sys.path
    if added:
        sys.path.insert(0, I3NET_DIR)
    os.chdir(I3NET_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)


def build_args(R, lr_slice_patch=LR_SLICE_PATCH):
    a = argparse.Namespace(model="i3net", upscale=int(R), lr_slice_patch=int(lr_slice_patch))
    a.hr_slice_patch = a.upscale * (a.lr_slice_patch - 1) + 1
    return a


def build_model(R, lr_slice_patch=LR_SLICE_PATCH, device="cuda"):
    """The unmodified upstream I3Net at scale R."""
    with _in_repo():
        from select_model import select_model
        args = build_args(R, lr_slice_patch)
        model = select_model(args)
    return model.to(device), args


def is_applicable(R, family):
    return (int(R) in APPLICABLE_STRIDES) and (str(family) in APPLICABLE_FAMILIES)


# --------------------------------------------------------------------------- #
#  training samples
# --------------------------------------------------------------------------- #
def sample_run(vol_hw_d_256, hr_slice_patch, rng):
    """One random run of `hr_slice_patch` consecutive slices -> (256, 256, hr_slice_patch).

    This is upstream `data.trainSet` semantics (`main.py` then forms the LR input as
    `gt[..., ::upscale]`). No augmentation: IR3 uses none, so giving I3Net flips would be
    an uncontrolled advantage (audit §D row 20).
    """
    D = vol_hw_d_256.shape[2]
    assert D >= hr_slice_patch
    s = int(rng.randint(0, D - hr_slice_patch + 1))
    return np.ascontiguousarray(vol_hw_d_256[:, :, s:s + hr_slice_patch], dtype=np.float32)


class RunSampler:
    """Draws (B, 256, 256, hr_slice_patch) ground-truth runs from prepared volumes.

    The LR input is formed by the caller as `gt[..., ::R]`, exactly as upstream does, so
    the model can only ever see slices on the R-grid.
    """

    def __init__(self, preps, R, lr_slice_patch=LR_SLICE_PATCH, seed=42, cache_cap=64):
        self.preps = preps
        self.R = int(R)
        self.hr = int(R) * (int(lr_slice_patch) - 1) + 1
        self.rng = np.random.RandomState(int(seed))
        self._cache, self._cap = {}, int(cache_cap)

    def _padded(self, i):
        if i not in self._cache:
            if len(self._cache) >= self._cap:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = P.pad_to(self.preps[i]["vol01"], 256)[0]
        return self._cache[i]

    def batch(self, idxs, device="cuda"):
        runs = [sample_run(self._padded(int(i)), self.hr, self.rng) for i in idxs]
        gt = torch.from_numpy(np.stack(runs, 0)).to(device)     # (B,256,256,hr)
        lr = gt[..., ::self.R].contiguous()                      # (B,256,256,n_lr)
        return lr, gt


# --------------------------------------------------------------------------- #
#  whole-volume inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_lr_stack(gt_hw_d, ret):
    """The ONLY thing the model is allowed to see: the acquired slices, zero-padded to 256.

    Returned tensor is provably a function of `gt_hw_d[:, :, ret]` alone, which is what the
    leakage preflight gate asserts.
    """
    padded, pad = P.pad_to(np.asarray(gt_hw_d, np.float32), 256)
    lr = np.ascontiguousarray(padded[:, :, sorted(int(r) for r in ret)], dtype=np.float32)
    return lr, pad


@torch.no_grad()
def reconstruct(model, gt_hw_d, ret, R, D=P.DEPTH, device="cuda",
                lr_slice_patch=LR_SLICE_PATCH, window_batch=4):
    """Upstream sliding-window reconstruction, extended to cover EVERY missing slice.

    Upstream `test.py` discards the first and last `upscale` slices of the volume; we cannot,
    because the shared evaluator scores every missing slice. Windows s = 0 .. K-n_lr cover
    HR indices 0 .. (K-1)*R inclusive, i.e. the whole volume; border slices simply receive
    fewer overlapping votes. No extra information is used.
    """
    R = int(R)
    ret = sorted(int(r) for r in ret)
    K = len(ret)
    gaps = np.diff(np.asarray(ret))
    assert len(set(gaps.tolist())) == 1 and int(gaps[0]) == R, \
        f"I3Net requires an exactly-uniform mask of stride {R}; got gaps {sorted(set(gaps.tolist()))}"
    assert K >= lr_slice_patch, f"K={K} < lr_slice_patch={lr_slice_patch}"

    lr_np, pad = build_lr_stack(gt_hw_d, ret)
    lr = torch.from_numpy(lr_np).to(device)                     # (256,256,K)
    span = (lr_slice_patch - 1) * R + 1

    acc = torch.zeros(256, 256, D, device=device, dtype=torch.float32)
    cnt = torch.zeros(D, device=device, dtype=torch.float32)
    model.eval()

    starts = list(range(0, K - lr_slice_patch + 1))
    for b0 in range(0, len(starts), window_batch):
        chunk = starts[b0:b0 + window_batch]
        x = torch.stack([lr[:, :, s:s + lr_slice_patch] for s in chunk], 0)   # (b,256,256,n_lr)
        y = model(x).float()                                                  # (b,256,256,span)
        y = torch.clamp(y, 0.0, 1.0)
        for j, s in enumerate(chunk):
            z0 = ret[s]
            acc[:, :, z0:z0 + span] += y[j]
            cnt[z0:z0 + span] += 1.0
    assert bool((cnt > 0).all()), "some slices were never covered by a window"
    out = (acc / cnt[None, None, :]).cpu().numpy().astype(np.float32)
    out = P.unpad_from(out, pad)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
