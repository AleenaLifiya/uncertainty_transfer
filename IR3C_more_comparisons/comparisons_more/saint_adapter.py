"""
IR3C_more_comparisons/comparisons_more/saint_adapter.py
=======================================================
DATASET / INTERFACE ADAPTATION ONLY for SAINT (Peng et al., CVPR 2020).

The method itself -- the RDN backbone, the distance-conditioned meta-kernel generator
(`GenWeights`) that *is* the "spatially aware interpolation", the marginal sagittal/coronal
super-resolution, and the second-stage Residual Fusion Network (`FUSE_RDN`) -- is imported
from the official repository

    https://github.com/cpeng93/SAINT   @ a4a1b8f9c7b74a5c5d00b8b08132ff15a1e65802

and is NOT reimplemented here. The only edits to that repository are the four recorded in
`provenance/SAINT/code_changes.patch` (unused `scipy.misc` import; the hard-coded MSD-CT
in-plane size 512; the hard-coded [0,4000] CT output window; and the hard-coded 6 phase
slots / int32 target dtype). Every edit keeps its upstream default.

Geometry
--------
The shared protocol stores volumes as `(H, W, D) = (240, 240, 153)` with the through-plane
axis LAST -- which is exactly SAINT's own layout (`srdata._load_file`, test branch:
`sag = hr`, `cor = np.transpose(sag, (1,0,2))`, super-resolution along the last axis).
No transpose or resize is therefore needed, and `verify_orientation()` proves it on a
synthetic separable phantom before any metric is computed.

  * sagittal view : `sag[x]` is the image `(W, D)`; 3 channels are x-1, x, x+1
  * coronal view  : `cor[y]` is the image `(H, D)`; 3 channels are y-1, y, y+1
  * `marginal_inference` returns `(X, Y, D_hr)` for sag and `(Y, X, D_hr)` for cor, the
    latter permuted back to `(X, Y, D_hr)` by upstream `MSR_RDN.test`.
  * stage 2 (`FUSE_RDN`) fuses the two marginal volumes slice-by-slice in the AXIAL plane.

Applicability
-------------
SAINT reconstructs the `factor - 1` slices between two ACQUIRED slices by producing the
`factor` interleaved phases of a regularly decimated stack (`data[..., i::factor]`). It
therefore requires an exactly-uniform mask, like I3Net -- family `U0`, `R in {2,4,8}`.
Unlike I3Net it needs only ONE model for all three factors: the meta-network conditions the
3x3 reconstruction kernel on the physical distance between the query position and the kernel
centre, so the factor enters as data, not as architecture.
"""
from __future__ import annotations

import contextlib
import os
import sys
import types

import numpy as np
import torch

from . import protocol_more as PM
from comparisons import protocol as P

SAINT_DIR = os.path.join(PM.EXP_DIR, "repositories", "SAINT")

APPLICABLE_FAMILIES = ["U0"]
APPLICABLE_STRIDES = [2, 4, 8]

# upstream `--RDNconfig C --n_colors 3 --G0 64 --RDNkSize 3`
RDN_CONFIG = "C"
G0 = 64
RDN_KSIZE = 3
N_COLORS = 3
SPACING = (1.0, 1.0, 1.0)          # BraTS-GLI 2023 is 1 mm isotropic; HR spacing


# SAINT ships TOP-LEVEL packages called `model`, `data`, `dataloader`, `utility`, `option`
# and `loss`. `ArSSR/model.py` -- already on this project's sys.path, and imported by
# `comparisons.arssr_adapter` -- claims the same `model` name, so whichever repository is
# imported first shadows the other for the rest of the process. The qualitative stage loads
# both, so the collision has to be handled rather than avoided by ordering.
_CLASH = ("model", "data", "dataloader", "utility", "option", "loss")
_SAINT_MODULES = {}          # SAINT's modules, parked here between uses


def _clashing(mods):
    return [k for k in list(mods)
            if k in _CLASH or any(k.startswith(c + ".") for c in _CLASH)]


@contextlib.contextmanager
def _in_repo():
    """Import SAINT's top-level packages with the colliding names temporarily evicted.

    On entry: park whatever currently owns those names, restore SAINT's own modules from a
    previous call (so repeated use is cheap), and put SAINT_DIR first on sys.path.
    On exit: park SAINT's modules again and give the names back to their previous owners.
    The result is order-independent -- neither repository can shadow the other.
    """
    cwd = os.getcwd()
    saved_path = list(sys.path)
    parked = {k: sys.modules.pop(k) for k in _clashing(sys.modules)}
    sys.modules.update(_SAINT_MODULES)
    sys.path.insert(0, SAINT_DIR)
    os.chdir(SAINT_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)
        for k in _clashing(sys.modules):
            _SAINT_MODULES[k] = sys.modules.pop(k)
        sys.modules.update(parked)
        sys.path[:] = saved_path


def _stub_missing():
    """The repo's `data/__init__.py` imports torch.utils.data only; `data.srdata` imports
    `data.common`, which imports `skimage.color` and `torchvision`. Both are present in the
    xdiff environment, so nothing is stubbed. Kept as an explicit no-op so the import path
    is documented rather than accidental."""
    return


def build_args(stage=2):
    a = types.SimpleNamespace()
    a.scale = [4]                  # only FUSE_RDN reads scale[0], and it ignores the value
    a.G0 = G0
    a.RDNkSize = RDN_KSIZE
    a.RDNconfig = RDN_CONFIG
    a.stage = int(stage)
    a.n_colors = N_COLORS
    return a


def build_model(device="cuda", stage=2):
    """The unmodified upstream SAINT network (MSR + RFN)."""
    _stub_missing()
    with _in_repo():
        from model import meta_multi
        args = build_args(stage)
        model = meta_multi.make_model(args)
    return model.to(device), args


def _srdata():
    with _in_repo():
        from data import srdata
    return srdata


def get_dist(factor, spacing=SPACING, n_slots=None):
    """Upstream `SRData.get_dist`, called unbound (it uses nothing from `self`).

    Returns (factor, 3, 3): the physical distance from each of the `factor` query phases to
    every tap of the 3x3 kernel. This is SAINT's spatial-awareness input.
    """
    S = _srdata()
    d = S.SRData.get_dist(None, list(spacing), int(factor), n_slots=n_slots)[0]
    return np.ascontiguousarray(d, dtype=np.float32)


def get_hr_phases(img_hw, factor, n_slots=None):
    """Upstream `SRData.get_hr`, called unbound: the `factor` interleaved phases."""
    S = _srdata()
    return S.SRData.get_hr(None, np.asarray(img_hw, np.float32), int(factor),
                           n_slots=n_slots, dtype="float32")


def is_applicable(R, family):
    return (int(R) in APPLICABLE_STRIDES) and (str(family) in APPLICABLE_FAMILIES)


def n_parameters(model, stage=None):
    if stage == 1:
        return sum(p.numel() for n, p in model.named_parameters() if not n.startswith("RFN."))
    if stage == 2:
        return sum(p.numel() for n, p in model.named_parameters() if n.startswith("RFN."))
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
#  stage-1 (MSR) training samples
# --------------------------------------------------------------------------- #
class SlabSampler:
    """Draws upstream-shaped SAINT training pairs from prepared BraTS volumes.

    One sample, exactly as `srdata._load_file` builds it for `TRAIN_SLICES`:

        slab  : (3, L, D')   three ADJACENT parallel lines of a sagittal or coronal view
        lr    : slab[..., ::R]                       -> (3, L, K)
        hr    : get_hr(slab[1], R)                   -> (R, L, K)   interleaved phases
        patch : random crop of `patch_size` LR columns along the through-plane axis

    Differences from upstream, all recorded in `adaptations.md`:
      * `gen_diff_thickness` (random z-decimation of the source volume) is NOT applied. The
        IR3C protocol fixes the HR grid at D=153 / 1 mm and forbids augmentation; the scale
        variety it provides is supplied instead by drawing R from the protocol's own
        `train_strides` with `train_stride_weights`.
      * ONE factor per batch, with `n_slots = factor`, instead of a fixed 6 phase slots.
        Upstream's 6 slots exist only so that x4 and x6 samples can share a batch; with our
        {2,4,8} they would make 6 of 8 targets identically zero at R=2.
    """

    def __init__(self, preps, strides, weights, patch_size=10, seed=42, cache_cap=48):
        self.preps = preps
        self.strides = [int(s) for s in strides]
        w = np.asarray([float(x) for x in weights], dtype=np.float64)
        self.weights = w / w.sum()
        self.patch = int(patch_size)
        self.rng = np.random.RandomState(int(seed))
        self._cache, self._cap = {}, int(cache_cap)

    def _vol(self, i):
        if i not in self._cache:
            if len(self._cache) >= self._cap:
                self._cache.pop(next(iter(self._cache)))
            # explicit copy: prepared volumes may be memory-mapped (protocol_more.prepare_cached)
            self._cache[i] = np.array(self.preps[i]["vol01"], dtype=np.float32, copy=True)
        return self._cache[i]

    def draw_factor(self):
        return int(self.rng.choice(self.strides, p=self.weights))

    def _slab(self, i, view):
        """(3, L, D): three adjacent lines of one marginal view, upstream channel order."""
        v = self._vol(i)
        H, W, D = v.shape
        if view == "sag":                      # sag[x] is (W, D); neighbours along x
            n, L = H, W
            get = lambda k: v[k]               # noqa: E731
        else:                                  # cor[y] is (H, D); neighbours along y
            n, L = W, H
            get = lambda k: v[:, k]            # noqa: E731
        j = int(self.rng.randint(0, n))
        out = np.zeros((3, L, D), dtype=np.float32)
        out[1] = get(j)
        if j > 0:
            out[0] = get(j - 1)
        if j < n - 1:
            out[2] = get(j + 1)
        return out

    def batch(self, idxs, device="cuda"):
        R = self.draw_factor()
        lrs, hrs = [], []
        for i in idxs:
            slab = self._slab(int(i), "sag" if self.rng.rand() < 0.5 else "cor")
            D = slab.shape[2]
            if D % R:                                   # upstream trims from the front
                slab = slab[..., D % R:]
            lr = slab[..., ::R]                         # (3, L, K)
            hr = get_hr_phases(slab[1], R, n_slots=None)  # (R, L, K)
            K = lr.shape[-1]
            assert self.patch <= K, f"patch {self.patch} > K {K} at R={R}"
            c = int(self.rng.randint(0, K - self.patch + 1))
            lrs.append(lr[..., c:c + self.patch])
            hrs.append(hr[..., c:c + self.patch])
        lr_t = torch.from_numpy(np.stack(lrs, 0)).to(device)
        hr_t = torch.from_numpy(np.stack(hrs, 0)).to(device)
        d = torch.from_numpy(get_dist(R, n_slots=None)).to(device)
        dist_t = d.unsqueeze(0).expand(len(idxs), -1, -1, -1).contiguous()
        return lr_t, hr_t, dist_t, R


# --------------------------------------------------------------------------- #
#  whole-volume inference
# --------------------------------------------------------------------------- #
def build_lr_stack(gt_hw_d, ret):
    """The ONLY thing SAINT is allowed to see: the acquired slices, in upstream test layout.

    Returns (2, H, 3, W, K) -- [sag, cor] x [prev, cur, next] -- a provable function of
    `gt_hw_d[:, :, ret]` alone, which is what the leakage gate asserts.
    """
    v = np.asarray(gt_hw_d, dtype=np.float32)
    ret = sorted(int(r) for r in ret)
    sag = v
    cor = np.transpose(v, (1, 0, 2))
    H, W, _ = v.shape
    out = np.zeros((2, 3, H, W, len(ret)), dtype=np.float32)
    for k, view in enumerate((sag, cor)):
        lo = view[:, :, ret]
        out[k, 1] = lo
        out[k, 0, 1:] = lo[:-1]
        out[k, 2, :-1] = lo[1:]
    return np.ascontiguousarray(out.transpose(0, 2, 1, 3, 4))       # (2, H, 3, W, K)


@torch.no_grad()
def reconstruct(model, gt_hw_d, ret, R, D=None, device="cuda", stage=2, chunk=16,
                return_marginals=False):
    """Upstream `MSR_RDN.test`, on the shared protocol's tensors.

    Preconditions asserted (the §5-style geometry gates):
      * the mask is exactly uniform with gap R and starts at 0, so `vol[..., ret]` IS the
        regularly decimated stack SAINT's phase interleaving assumes;
      * the two in-plane dimensions are never touched;
      * the output is trimmed -- never resized -- to the D of the reference.
    """
    D = int(D if D is not None else gt_hw_d.shape[2])
    R = int(R)
    ret = sorted(int(r) for r in ret)
    gaps = np.diff(np.asarray(ret))
    assert len(set(gaps.tolist())) == 1 and int(gaps[0]) == R, \
        f"SAINT requires an exactly-uniform mask of stride {R}; got gaps {sorted(set(gaps.tolist()))}"
    assert ret[0] == 0, f"SAINT phase 0 is the acquired slice; mask must start at 0, got {ret[0]}"
    H, W, _ = gt_hw_d.shape

    inp = torch.from_numpy(build_lr_stack(gt_hw_d, ret)).unsqueeze(0).to(device)  # (1,2,H,3,W,K)
    dist = torch.from_numpy(get_dist(R, n_slots=None)).unsqueeze(0).to(device)    # (1,R,3,3)

    model.eval()
    model.stage = int(stage)
    out = model.test(inp, dist, R, vmin=0.0, vmax=1.0, round_output=False, chunk=chunk)
    if stage == 1:
        sag, cor = out
        fused = 0.5 * (sag[0] + cor[0])          # the README's "average" stage-1 estimate
        sagn, corn = sag[0].float().cpu().numpy(), cor[0].float().cpu().numpy()
    else:
        ref, sag, cor = out
        fused = ref[0]
        sagn, corn = sag[0].float().cpu().numpy(), cor[0].float().cpu().numpy()

    vol = fused.float().cpu().numpy()
    assert vol.shape[0] == H and vol.shape[1] == W, \
        f"in-plane changed: {vol.shape[:2]} != {(H, W)}"
    assert vol.shape[2] >= D, f"through-plane {vol.shape[2]} < D {D}"
    vol = np.ascontiguousarray(vol[:, :, :D], dtype=np.float32)
    vol = np.clip(vol, 0.0, 1.0)
    if return_marginals:
        return vol, sagn[:, :, :D], corn[:, :, :D]
    return vol


@torch.no_grad()
def marginals(model, gt_hw_d, ret, R, D=None, device="cuda", chunk=16):
    """The two stage-1 marginal volumes, which are the RFN's training input."""
    _, sag, cor = reconstruct(model, gt_hw_d, ret, R, D, device, stage=1, chunk=chunk,
                              return_marginals=True)
    return sag, cor


@torch.no_grad()
def fuse(model, sag, cor, device="cuda", slab=16):
    """Stage 2 alone: apply the frozen-architecture RFN to cached marginal volumes."""
    s = torch.from_numpy(np.asarray(sag, np.float32)).to(device)
    c = torch.from_numpy(np.asarray(cor, np.float32)).to(device)
    vol = torch.stack([s, c]).permute(3, 0, 1, 2)          # (D, 2, H, W)
    outs = []
    for i in range(0, vol.shape[0], slab):
        outs.append(model.RFN(vol[i:i + slab]))
    ref = torch.cat(outs)[:, 0].permute(1, 2, 0)           # (H, W, D)
    return torch.clamp(ref, 0.0, 1.0).float().cpu().numpy()


# --------------------------------------------------------------------------- #
#  orientation / geometry verification (prompt §6)
# --------------------------------------------------------------------------- #
def verify_orientation(D=153, H=240, W=240, R=4):
    """Prove the sag/cor/axial bookkeeping on a separable phantom with three distinct axes.

    `f(x,y,z) = a(x) + b(y) + c(z)` with mutually incommensurate frequencies: any transpose,
    axis swap or view mix-up changes the reconstructed array, so an exact round trip of the
    ACQUIRED slices through `build_lr_stack` + the phase interleaving is a real test.
    """
    x = np.arange(H)[:, None, None] / H
    y = np.arange(W)[None, :, None] / W
    z = np.arange(D)[None, None, :] / D
    vol = (0.2 * np.sin(2 * np.pi * 3 * x) + 0.3 * np.sin(2 * np.pi * 7 * y)
           + 0.4 * np.sin(2 * np.pi * 11 * z) + 0.5).astype(np.float32)
    ret = list(range(0, D, R))
    st = build_lr_stack(vol, ret)                     # (2, H, 3, W, K)
    out = {}
    # sagittal centre channel must be vol[x][:, ret]
    out["sag_centre_exact"] = float(np.abs(st[0][:, 1] - vol[:, :, ret]).max())
    # coronal centre channel must be transpose(vol,(1,0,2))[y][:, ret]
    cor = np.transpose(vol, (1, 0, 2))
    out["cor_centre_exact"] = float(np.abs(st[1][:, 1] - cor[:, :, ret]).max())
    # neighbour channels are the adjacent lines, not a shift along z
    out["sag_prev_exact"] = float(np.abs(st[0][1:, 0] - vol[:-1][:, :, ret]).max())
    out["sag_next_exact"] = float(np.abs(st[0][:-1, 2] - vol[1:][:, :, ret]).max())
    # phase interleaving reproduces the source exactly for a decimated line.
    # Upstream requires the z extent to be divisible by the factor and trims from the FRONT
    # (`srdata._load_file`: `hr = hr[..., hr.shape[-1] % factor:]`); the same trim is used
    # here and in SlabSampler, so the check exercises the code path training actually uses.
    line = vol[0]
    if line.shape[-1] % R:
        line = line[..., line.shape[-1] % R:]
    ph = get_hr_phases(line, R, n_slots=None)         # (R, W, K)
    K = ph.shape[-1]
    rec = np.zeros_like(line)
    for k in range(R):
        rec[:, k::R] = ph[k]
    out["phase_interleave_exact"] = float(np.abs(rec - line).max())
    out["in_plane_unchanged"] = bool(st.shape[1] == H and st.shape[3] == W)
    out["only_z_reduced"] = bool(st.shape[4] == len(ret) and len(ret) < D)
    return out
