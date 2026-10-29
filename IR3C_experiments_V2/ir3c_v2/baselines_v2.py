"""
ir3c_v2/baselines_v2.py
=======================
Geometry-general wrappers so I3Net, ArSSR, DCSRN and SAINT can be fine-tuned and evaluated
on OAIZIB (384x384x160) and IXI-tiny (44x55x83) under the SAME protocol the IR3C transfer
runs used.

No method is re-implemented and no original file is edited. Every network, loss, optimiser
and inference routine comes from the unmodified adapters; this module only supplies the
target geometry where the BraTS adapters hard-coded it, and copies (never edits) the two
routines that were written against a fixed 240x240x153 grid.

Per-model adaptation, and why each is necessary
-----------------------------------------------

I3Net   is architecturally locked to a 256x256 in-plane extent (IntraSliceBranch forces
        (h/16)^2 == 256). IXI 44x55 is symmetrically ZERO-PADDED to 256x256, exactly as
        BraTS 240x240 is. OAIZIB 384x384 is LARGER than 256, so it is covered by an
        in-plane sliding window of 256x256 tiles at stride 128 (origins {0,128} on each
        axis), averaged by overlap count -- the same accumulate-and-divide the method
        already uses along z. Training draws a random 256x256 in-plane crop, which is what
        upstream's own patch training does.

ArSSR   is fully convolutional plus a coordinate decoder, so it is geometry-general. The
        training slab sampler already uses `b = min(base, H, W)`; only its mask draw was
        pinned to the BraTS depth, which is overridden here. Inference z-tiling is reduced
        on OAIZIB purely for memory (a 384x384 tile is 2.56x a BraTS tile); tiling with a
        margin is the adapter's own documented mechanism and does not change the method.

DCSRN   is fully convolutional. Its 64^3 upstream patch does not fit IXI's 44x55 in-plane
        extent, so IXI uses 32^3 -- the largest cube that fits -- and OAIZIB keeps 64^3.
        The inference mode (whole-volume vs sliding 64^3) is chosen on VALIDATION, exactly
        as the BraTS run chose it.

SAINT   reads its in-plane extent from the input tensor (already patched upstream-side for
        BraTS) and is otherwise geometry-general. Its meta-kernel takes the voxel SPACING,
        which is dataset-specific and is supplied here from each dataset's own header
        values instead of BraTS's 1 mm isotropic.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch

from .v2paths import MORE_DIR, PROJECT                      # noqa: F401 (sys.path bootstrap)

from comparisons import protocol as P                        # noqa: E402
from comparisons import i3net_adapter as I3                  # noqa: E402
from comparisons import arssr_adapter as AR                  # noqa: E402
import ir2_components as C                                   # noqa: E402

if MORE_DIR not in sys.path:
    sys.path.insert(0, MORE_DIR)
from comparisons_more import saint_adapter as SA             # noqa: E402
from comparisons_more import dcsrn_adapter as DC             # noqa: E402

I3NET_EXTENT = 256


# ===================================================== the short-final-gap problem
#
# The frozen U0 masks on the TARGET datasets are NOT exactly uniform. They are
#
#       ret = [0, R, 2R, ..., R*m, D-1]
#
# because D-1 is not divisible by R (IXI D=83, OAIZIB D=160), so the acquisition ends with
# one SHORT final gap. BraTS did not have this: D-1 = 152 is divisible by 2, 4 and 8, so
# every BraTS U0 mask is exactly uniform and the three uniform-only methods (I3Net, DCSRN,
# SAINT) never met the case.
#
# Those three methods ASSERT an exactly-uniform stride -- it is baked into the architecture
# (I3Net's fixed `upscale`, SAINT's phase interleaving, DCSRN's fixed-ratio decimation), not
# into the adapter. The mask cannot be changed: it is the frozen object every method,
# including IR3C, is evaluated on, and regenerating it would break the comparison.
#
# Resolution, applied identically to all three and disclosed in every table:
#
#   * the model reconstructs the UNIFORM PREFIX  [0 .. R*m], which is an exactly-uniform
#     stride-R acquisition and is what the method was designed for;
#   * the few missing slices in the short final gap  (R*m, D-1)  are filled by LINEAR
#     interpolation between the two ACQUIRED slices that bracket them, R*m and D-1, so no
#     unacquired voxel is ever read;
#   * `tail_missing_slices` and `tail_fraction_of_missing` are recorded per cell.
#
# The affected fraction is small (IXI: 1 of 41/61/71 missing slices; OAIZIB: 0-6 of
# 79/119/139) and it is identical for the three methods, so it cannot reorder them. IR3C
# and the classical baselines reconstruct the tail natively and are unaffected.


def uniform_prefix(ret, R, D):
    """Split a frozen U0 mask into its exactly-uniform prefix and the short-gap tail.

    Returns (ret_uniform, z_max_uniform, tail_missing) where `tail_missing` lists the
    MISSING slice indices above `z_max_uniform`.
    """
    R = int(R)
    ret = sorted(int(r) for r in ret)
    keep = [ret[0]]
    for z in ret[1:]:
        if z - keep[-1] == R:
            keep.append(z)
        else:
            break
    zmax = keep[-1]
    acq = set(ret)
    tail = [z for z in range(zmax + 1, int(D)) if z not in acq]
    return keep, zmax, tail


def fill_short_tail(out, gt_hw_d, ret, R, D):
    """Linear interpolation across the short final gap, from ACQUIRED slices only."""
    keep, zmax, tail = uniform_prefix(ret, R, D)
    if not tail:
        return out, dict(tail_missing_slices=0, tail_fraction_of_missing=0.0)
    acq = sorted(int(r) for r in ret)
    hi = min(z for z in acq if z > zmax)
    lo = zmax
    a = np.asarray(gt_hw_d, np.float32)[:, :, lo]
    b = np.asarray(gt_hw_d, np.float32)[:, :, hi]
    for z in tail:
        w = (z - lo) / float(hi - lo)
        out[:, :, z] = (1.0 - w) * a + w * b
    for z in acq:
        if z > zmax:
            out[:, :, z] = np.asarray(gt_hw_d, np.float32)[:, :, z]
    n_miss = int(D) - len(acq)
    return out, dict(tail_missing_slices=len(tail),
                     tail_fraction_of_missing=len(tail) / max(1, n_miss))

# voxel spacing (through-plane, in-plane, in-plane) as measured in each dataset's audit
SPACING = {
    "OAIZIB": (0.7, 0.3646, 0.3646),
    "IXI": (2.18, 4.13, 3.95),
    "BraTS": (1.0, 1.0, 1.0),
}

# DCSRN training patch: the largest upstream-shaped cube that fits the target grid
DCSRN_PATCH = {"OAIZIB": 64, "IXI": 32, "BraTS": 64}

# ArSSR inference tiling: reduced on OAIZIB for memory only (see module docstring)
ARSSR_INFERENCE = {
    # z_margin is BraTS's 8, restored: it is the encoder receptive-field guard and changing
    # it changes the result. z_chunk and slice_batch are pure batching and change nothing.
    # Measured on one OAIZIB volume: (4,4,1) 27.1 s / 10.1 GB; (8,8,8) 25.9 s / 20.0 GB;
    # (16,8,8) 18.6 s / 33.9 GB. (8,8,8) is chosen: faster than the previous setting AND
    # faithful to the BraTS margin, without the 34 GB peak.
    "OAIZIB": dict(z_chunk=8, z_margin=8, slice_batch=8),
    "IXI": dict(z_chunk=16, z_margin=8, slice_batch=8),
    "BraTS": dict(z_chunk=32, z_margin=8, slice_batch=12),
}


# ===================================================================== I3Net
def _pad_or_tile(H, W, extent=I3NET_EXTENT):
    """In-plane coverage plan: symmetric pad when the volume is smaller than `extent`,
    an overlapping tile grid when it is larger."""
    if H <= extent and W <= extent:
        return "pad", [(0, 0)]
    stride = extent // 2
    ys = sorted(set(list(range(0, max(1, H - extent + 1), stride)) + [max(0, H - extent)]))
    xs = sorted(set(list(range(0, max(1, W - extent + 1), stride)) + [max(0, W - extent)]))
    return "tile", [(y, x) for y in ys for x in xs]


class I3NetRunSamplerV2:
    """(B, 256, 256, hr_slice_patch) runs from volumes of ANY in-plane extent.

    Smaller than 256: symmetric zero pad (the BraTS behaviour, unchanged).
    Larger than 256 : a random 256x256 in-plane crop, which is upstream's own patch-training
                      granularity. The LR input is still formed by the caller as
                      `gt[..., ::R]`, so the model can only ever see slices on the R grid.
    """

    def __init__(self, preps, R, lr_slice_patch=I3.LR_SLICE_PATCH, seed=42, cache_cap=32,
                 extent=I3NET_EXTENT):
        self.preps = preps
        self.R = int(R)
        self.extent = int(extent)
        self.hr = int(R) * (int(lr_slice_patch) - 1) + 1
        self.rng = np.random.RandomState(int(seed))
        self._cache, self._cap = {}, int(cache_cap)

    def _vol(self, i):
        if i not in self._cache:
            if len(self._cache) >= self._cap:
                self._cache.pop(next(iter(self._cache)))
            v = np.array(self.preps[i]["vol01"], dtype=np.float32, copy=True)
            H, W, _ = v.shape
            if H <= self.extent and W <= self.extent:
                v = P.pad_to(v, self.extent)[0]
            self._cache[i] = v
        return self._cache[i]

    def _run(self, i):
        v = self._vol(int(i))
        H, W, D = v.shape
        e = self.extent
        y0 = 0 if H <= e else int(self.rng.randint(0, H - e + 1))
        x0 = 0 if W <= e else int(self.rng.randint(0, W - e + 1))
        s = int(self.rng.randint(0, max(1, D - self.hr + 1)))
        return np.ascontiguousarray(v[y0:y0 + e, x0:x0 + e, s:s + self.hr], dtype=np.float32)

    def batch(self, idxs, device="cuda"):
        gt = torch.from_numpy(np.stack([self._run(int(i)) for i in idxs], 0)).to(device)
        lr = gt[..., ::self.R].contiguous()
        return lr, gt


def i3net_lr_stack_v2(gt_hw_d, ret, extent=I3NET_EXTENT):
    """Everything I3Net is allowed to see, across every in-plane tile: the acquired slices
    only. Used by the leakage gate; `P.pad_to` refuses extents above 256, so the BraTS
    builder cannot be reused on OAIZIB."""
    gt = np.asarray(gt_hw_d, np.float32)
    H, W, _ = gt.shape
    ret = sorted(int(r) for r in ret)
    mode, origins = _pad_or_tile(H, W, extent)
    parts = []
    for (y0, x0) in origins:
        tile = (P.pad_to(gt, extent)[0] if mode == "pad"
                else gt[y0:y0 + extent, x0:x0 + extent])
        parts.append(np.ascontiguousarray(tile[:, :, ret], np.float32).ravel())
    return np.concatenate(parts)


@torch.no_grad()
def i3net_reconstruct_v2(model, gt_hw_d, ret, R, D, device="cuda",
                         lr_slice_patch=I3.LR_SLICE_PATCH, window_batch=4,
                         extent=I3NET_EXTENT, return_stats=False):
    """`comparisons.i3net_adapter.reconstruct`, COPIED (not edited) and extended with the
    in-plane tiling needed when the volume is wider than the architecture's 256 extent.
    The z-window logic, the uniform-mask precondition and the overlap averaging are
    unchanged."""
    R = int(R)
    full_ret = sorted(int(r) for r in ret)
    ret, zmax, tail = uniform_prefix(full_ret, R, D)
    K = len(ret)
    gaps = np.diff(np.asarray(ret))
    assert len(set(gaps.tolist())) == 1 and int(gaps[0]) == R, \
        f"I3Net requires an exactly-uniform mask of stride {R}; got {sorted(set(gaps.tolist()))}"
    assert K >= lr_slice_patch, f"K={K} < lr_slice_patch={lr_slice_patch}"

    gt = np.asarray(gt_hw_d, np.float32)
    H, W, _ = gt.shape
    mode, origins = _pad_or_tile(H, W, extent)
    span = (lr_slice_patch - 1) * R + 1
    acc = torch.zeros(H, W, D, device=device, dtype=torch.float32)
    cnt = torch.zeros(H, W, D, device=device, dtype=torch.float32)
    model.eval()
    starts = list(range(0, K - lr_slice_patch + 1))

    for (y0, x0) in origins:
        if mode == "pad":
            padded, pad = P.pad_to(gt, extent)
            tile = padded[:, :, ret]
            ty0 = tx0 = 0
        else:
            pad = None
            tile = gt[y0:y0 + extent, x0:x0 + extent][:, :, ret]
            ty0, tx0 = y0, x0
        lr = torch.from_numpy(np.ascontiguousarray(tile, np.float32)).to(device)
        tacc = torch.zeros(extent, extent, D, device=device, dtype=torch.float32)
        tcnt = torch.zeros(D, device=device, dtype=torch.float32)
        for b0 in range(0, len(starts), window_batch):
            chunk = starts[b0:b0 + window_batch]
            x = torch.stack([lr[:, :, s:s + lr_slice_patch] for s in chunk], 0)
            y = torch.clamp(model(x).float(), 0.0, 1.0)
            for j, s in enumerate(chunk):
                z0 = ret[s]
                tacc[:, :, z0:z0 + span] += y[j]
                tcnt[z0:z0 + span] += 1.0
        covered = tcnt > 0
        assert bool(covered[:zmax + 1].all()), "the uniform prefix was not fully covered"
        tout = tacc / torch.clamp(tcnt, min=1.0)[None, None, :]
        if mode == "pad":
            t, b, l, r = pad
            tout = tout[t:extent - b, l:extent - r]
            acc += tout
            cnt += 1.0
        else:
            acc[ty0:ty0 + extent, tx0:tx0 + extent] += tout
            cnt[ty0:ty0 + extent, tx0:tx0 + extent] += 1.0
        del lr, tacc
    out = (acc / torch.clamp(cnt, min=1.0)).cpu().numpy().astype(np.float32)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out, tail_stats = fill_short_tail(out, gt_hw_d, full_ret, R, D)
    return (np.clip(out, 0.0, 1.0).astype(np.float32), tail_stats) if return_stats \
        else np.clip(out, 0.0, 1.0).astype(np.float32)


# ===================================================================== ArSSR
class ArSSRSlabSamplerV2(AR.SlabSampler):
    """`arssr_adapter.SlabSampler` with the mask draw taken at the TARGET depth instead of
    the BraTS depth. Everything else -- crop-origin logic, coordinate construction,
    sample_size, the R distribution -- is inherited unchanged."""

    def __init__(self, preps, D, **kw):
        self.D = int(D)
        super().__init__(preps, **kw)

    def _mask(self):
        return C.sample_mask_weighted(self.D, self.rng, self.strides, list(self.w))

    def draw(self, i):
        vol = self.preps[int(i)]["vol01"]
        H, W, D = vol.shape
        ret, meta = self._mask()
        ret = sorted(int(r) for r in ret)
        K = len(ret)
        j = int(self.rng.randint(0, K - self.n_lr + 1))
        zs = ret[j:j + self.n_lr]
        z_lo, z_hi = zs[0], zs[-1]
        b = min(self.base, H, W)
        y0, x0 = self._crop_origin(vol, zs, H, W, b)
        lr = np.ascontiguousarray(
            np.moveaxis(vol[y0:y0 + b, x0:x0 + b, zs], 2, 0), dtype=np.float32)
        hr = np.ascontiguousarray(
            np.moveaxis(vol[y0:y0 + b, x0:x0 + b, z_lo:z_hi + 1], 2, 0), dtype=np.float32)
        p = AR.lr_positions(ret, D)[z_lo:z_hi + 1] - j
        cz = AR.norm_coord(p, self.n_lr)
        cy = AR.norm_coord(np.arange(b), b)
        cx = AR.norm_coord(np.arange(b), b)
        gz, gy, gx = np.meshgrid(cz, cy, cx, indexing="ij")
        xyz = np.stack([gz, gy, gx], -1).reshape(-1, 3).astype(np.float32)
        vals = hr.reshape(-1, 1).astype(np.float32)
        n = xyz.shape[0]
        idx = self.rng.choice(n, self.sample_size, replace=n < self.sample_size)
        return lr, xyz[idx], vals[idx], meta


def arssr_reconstruct_v2(model, gt_hw_d, ret, D, device="cuda", dataset="BraTS"):
    inf = ARSSR_INFERENCE.get(dataset, ARSSR_INFERENCE["BraTS"])
    return AR.reconstruct(model, gt_hw_d, ret, D, device, inf["z_chunk"], inf["z_margin"],
                          inf["slice_batch"])


# ===================================================================== DCSRN
def dcsrn_patch_sampler(preps, R, D, dataset, seed=42, mixed=False, uniform_prob=0.5):
    """DCSRN training patches.

    `mixed=False` -> `PatchSampler`: the acquisition mask is the fixed uniform U0 at stride
    R, so the model only ever sees uniform sampling and is evaluable on U0 alone.

    `mixed=True`  -> `MixedMaskPatchSampler`: the mask is redrawn per sample from the
    PROTOCOL's own training distribution (`uniform_prob` uniform U0, the rest spread over
    U1-U6 with a fresh seed each draw), with the stride pinned to this specialist's R. Only
    the FAMILY is mixed, never the stride -- one fixed-scale model per R, as the protocol
    requires.

    BOTH are required. DCSRN is fully convolutional and, unlike I3Net and SAINT, does NOT
    need an exactly uniform stride; the BraTS comparison trains and reports both variants
    (`DCSRN` on U0, `DCSRN_mixed` on U0-U6). Shipping only the U0 one silently removes DCSRN
    from the family-agnostic comparison it is entitled to be in.
    """
    patch = DCSRN_PATCH.get(dataset, 64)
    if mixed:
        return DC.MixedMaskPatchSampler(preps, R, D=D, patch=patch, seed=seed,
                                        uniform_prob=float(uniform_prob))
    return DC.PatchSampler(preps, R, D=D, patch=patch, seed=seed)


# DCSRN's whole-volume forward materialises a dense 4-layer DenseNet activation stack over
# the ENTIRE volume. Measured on BraTS (240x240x153 = 8.8 M voxels) it peaked at 18.7 GB, so
# it scales to roughly 2.1 GB per million voxels. An OAIZIB volume is 23.6 M voxels -> ~50 GB,
# which on this unified-memory machine gets the process SIGKILLed by the OS OOM killer. That
# cannot be caught with try/except, so it is PREDICTED and refused instead.
DCSRN_WHOLE_VOLUME_GB_PER_MVOXEL = 2.13
DCSRN_WHOLE_VOLUME_BUDGET_GB = float(os.environ.get("V2_DCSRN_WV_BUDGET_GB", "40"))


def dcsrn_whole_volume_is_affordable(shape, budget_gb=None):
    """(affordable, estimated_gb) for a whole-volume DCSRN forward on `shape`."""
    mvox = float(np.prod(shape)) / 1e6
    est = mvox * DCSRN_WHOLE_VOLUME_GB_PER_MVOXEL
    budget = DCSRN_WHOLE_VOLUME_BUDGET_GB if budget_gb is None else float(budget_gb)
    return bool(est <= budget), est


def dcsrn_reconstruct_v2(model, gt_hw_d, ret, R, D, device="cuda", mode="tiled64",
                         dataset="BraTS", return_stats=False):
    """DCSRN builds its LR volume by z-decimation at the mask positions followed by z-only
    linear interpolation back onto the HR grid, so the short final gap is already handled
    correctly by `build_lr_volume`; no prefix split is needed. The tail statistics are
    reported as zero for symmetry with the other two uniform-only methods."""
    if mode == "whole_volume":
        ok, est = dcsrn_whole_volume_is_affordable(np.asarray(gt_hw_d).shape)
        if not ok:
            raise MemoryError(
                f"DCSRN whole-volume inference on {np.asarray(gt_hw_d).shape} would need "
                f"about {est:.1f} GB (budget {DCSRN_WHOLE_VOLUME_BUDGET_GB:.0f} GB). "
                f"Refused before allocation: an OS OOM kill cannot be caught. Use the "
                f"tiled mode, which is upstream's own inference granularity.")
        out = DC.reconstruct_wholevolume(model, gt_hw_d, ret, R, D, device)
    else:
        roi = min(DCSRN_PATCH.get(dataset, 64), int(D),
                  int(gt_hw_d.shape[0]), int(gt_hw_d.shape[1]))
        out = DC.reconstruct_tiled(model, gt_hw_d, ret, R, D, device, roi=roi)
    stats = dict(tail_missing_slices=0, tail_fraction_of_missing=0.0)
    return (out, stats) if return_stats else out


# ===================================================================== SAINT
class SAINTSlabSamplerV2(SA.SlabSampler):
    """`saint_adapter.SlabSampler` with the dataset's own voxel spacing fed to the
    meta-kernel. The slab construction, phase interleaving and crop are inherited."""

    def __init__(self, preps, strides, weights, spacing, patch_size=10, seed=42,
                 cache_cap=48):
        super().__init__(preps, strides, weights, patch_size=patch_size, seed=seed,
                         cache_cap=cache_cap)
        self.spacing = tuple(float(x) for x in spacing)

    def batch(self, idxs, device="cuda"):
        R = self.draw_factor()
        # ONE view per batch. A sagittal slab is (3, W, D) and a coronal one is (3, H, D),
        # so on a NON-SQUARE in-plane grid (IXI-tiny is 44x55) mixing the two views inside a
        # batch cannot be stacked. The upstream sampler mixed them freely because BraTS and
        # OAIZIB are square. Drawing the view once per batch keeps the same 50/50 marginal
        # over views, the same slab construction and the same number of samples; only the
        # grouping changes.
        view = "sag" if self.rng.rand() < 0.5 else "cor"
        lrs, hrs = [], []
        for i in idxs:
            slab = self._slab(int(i), view)
            D = slab.shape[2]
            if D % R:
                slab = slab[..., D % R:]
            lr = slab[..., ::R]
            hr = SA.get_hr_phases(slab[1], R, n_slots=None)
            K = lr.shape[-1]
            patch = min(self.patch, K)
            c = int(self.rng.randint(0, K - patch + 1))
            lrs.append(lr[..., c:c + patch])
            hrs.append(hr[..., c:c + patch])
        lr_t = torch.from_numpy(np.stack(lrs, 0)).to(device)
        hr_t = torch.from_numpy(np.stack(hrs, 0)).to(device)
        d = torch.from_numpy(SA.get_dist(R, spacing=self.spacing, n_slots=None)).to(device)
        dist_t = d.unsqueeze(0).expand(len(idxs), -1, -1, -1).contiguous()
        return lr_t, hr_t, dist_t, R


def saint_view_stacks(gt_hw_d, ret):
    """The two marginal-view LR stacks, built SEPARATELY so a non-square in-plane extent
    works.

    `saint_adapter.build_lr_stack` packs both views into one (2, H, 3, W, K) array, which
    silently assumes H == W. BraTS (240x240) and OAIZIB (384x384) are square; IXI-tiny is
    44x55 and the pack raises. The two views are independent inputs to
    `marginal_inference`, so building them separately is mathematically identical and
    removes the assumption.

    Returns (sag, cor): (H, 3, W, K) and (W, 3, H, K), a provable function of
    `gt_hw_d[:, :, ret]` alone.
    """
    v = np.asarray(gt_hw_d, dtype=np.float32)
    ret = sorted(int(r) for r in ret)
    outs = []
    for view in (v, np.transpose(v, (1, 0, 2))):
        n, L, _ = view.shape
        lo = view[:, :, ret]                                 # (n, L, K)
        o = np.zeros((3, n, L, len(ret)), dtype=np.float32)
        o[1] = lo
        o[0, 1:] = lo[:-1]
        o[2, :-1] = lo[1:]
        outs.append(np.ascontiguousarray(o.transpose(1, 0, 2, 3)))   # (n, 3, L, K)
    return outs[0], outs[1]


@torch.no_grad()
def saint_reconstruct_v2(model, gt_hw_d, ret, R, D, device="cuda", stage=2, chunk=16,
                         dataset="BraTS", return_stats=False):
    """`MSR_RDN.test` reproduced on the target geometry.

    Two changes from `saint_adapter.reconstruct`, both required and both disclosed:
      * `get_dist` uses the DATASET's voxel spacing, not BraTS's 1 mm isotropic (the
        meta-kernel input is physical);
      * the two marginal views are built and run separately, so a non-square in-plane
        extent works (see `saint_view_stacks`).
    The network calls, the phase interleaving, the clamping and the RFN fusion are the
    upstream ones, unchanged.
    """
    sp = SPACING.get(dataset, SPACING["BraTS"])
    D = int(D)
    R = int(R)
    full_ret = sorted(int(r) for r in ret)
    ret_u, zmax, tail = uniform_prefix(full_ret, R, D)
    gaps = np.diff(np.asarray(ret_u))
    assert len(set(gaps.tolist())) == 1 and int(gaps[0]) == R, \
        f"SAINT requires an exactly-uniform mask of stride {R}; got {sorted(set(gaps.tolist()))}"
    assert ret_u[0] == 0, "SAINT phase 0 is the acquired slice; the mask must start at 0"
    H, W, _ = gt_hw_d.shape

    sag_np, cor_np = saint_view_stacks(gt_hw_d, ret_u)
    sag = torch.from_numpy(sag_np).to(device)
    cor = torch.from_numpy(cor_np).to(device)
    dist = torch.from_numpy(SA.get_dist(R, spacing=sp, n_slots=None)).unsqueeze(0).to(device)

    model.eval()
    model.stage = int(stage)
    weights = None
    for i in range(R):
        wi = model.GW(dist[:, [i], :, :]).view(1, 1, 64, 3, 3)
        weights = wi if weights is None else torch.cat((weights, wi), 1)

    comb_sag = torch.clamp(model.marginal_inference(sag, weights, chunk=chunk), 0.0, 1.0)
    comb_cor = torch.clamp(model.marginal_inference(cor, weights, chunk=chunk),
                           0.0, 1.0).permute(1, 0, 2)
    if int(stage) == 1:
        fused = 0.5 * (comb_sag + comb_cor)
    else:
        vol = torch.stack([comb_sag, comb_cor]).permute(3, 0, 1, 2)      # (D', 2, H, W)
        ref = [model.RFN(vol[[i]])[0] for i in range(vol.shape[0])]
        fused = torch.clamp(torch.cat(ref).permute(1, 2, 0), 0.0, 1.0)

    vol = fused.float().cpu().numpy()
    assert vol.shape[0] == H and vol.shape[1] == W, f"in-plane changed: {vol.shape[:2]}"
    zcov = zmax + 1
    out = np.zeros((H, W, D), dtype=np.float32)
    n = min(zcov, vol.shape[2])
    out[:, :, :n] = vol[:, :, :n]
    out, tail_stats = fill_short_tail(out, gt_hw_d, full_ret, R, D)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    return (out, tail_stats) if return_stats else out


# ===================================================================== applicability
APPLICABLE = {
    "I3Net": dict(families=["U0"], strides=[2, 4, 8], per_scale=True),
    "ArSSR": dict(families=["U0", "U1", "U2", "U3", "U4", "U5", "U6"],
                  strides=[2, 4, 8, 16], per_scale=False),
    "DCSRN": dict(families=["U0"], strides=[2, 4, 8], per_scale=True),
    # The mixed-mask DCSRN sees U0-U6 during training and is evaluable on all of them.
    "DCSRNmix": dict(families=["U0", "U1", "U2", "U3", "U4", "U5", "U6"],
                     strides=[2, 4, 8], per_scale=True),
    "SAINT": dict(families=["U0"], strides=[2, 4, 8], per_scale=False),
}


def applicable(model, R, family):
    a = APPLICABLE[model]
    return int(R) in a["strides"] and str(family) in a["families"]
