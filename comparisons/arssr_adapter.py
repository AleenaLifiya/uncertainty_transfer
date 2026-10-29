"""
comparisons/arssr_adapter.py
============================
DATASET/INTERFACE ADAPTATION ONLY. The ArSSR method (RDN 3D encoder, trilinear feature
lookup, coordinate MLP decoder, L1 objective) is imported from `ArSSR/` at commit 7e10fe0
and is NOT modified.

Two adaptations are made, both to the QUERY COORDINATES only -- no weight, layer or loss is
touched (ir3_comparisons.md §2, §17):

  D1. THROUGH-PLANE ONLY.
      Upstream trains isotropic SR (`nd.zoom(1/s, order=3)` in all three axes). This task
      decimates z only and leaves the in-plane grid untouched, so LR and HR share the same
      in-plane grid and only the z coordinate is densified. (This conversion already existed
      locally in `ArSSR/*_tp.py`; it is re-derived here against the BraTS protocol.)

  D2. SLICE-SELECTION COORDINATE GEOMETRY, AND NON-UNIFORM SAMPLING.
      Upstream builds queries with `make_coord(hr_shape)`, i.e. cell centres of the HR grid.
      That is exact when the LR volume is a cell-centre RESAMPLE of the HR volume, which is
      what `nd.zoom` produces. Our degradation is slice SELECTION: LR slice j IS HR slice
      ret[j]. Under selection the correct LR-grid position of HR slice z is

          p(z) = j + (z - ret[j]) / (ret[j+1] - ret[j]),      ret[j] <= z <= ret[j+1]

      which is monotone, piecewise linear, reduces to z/R for a uniform stride, and places
      every acquired slice exactly on its own LR sample. Because it is expressed in LR-index
      units it also handles the NON-UNIFORM families U1..U6 with no further change -- the
      natural extension of "arbitrary scale" to "arbitrary sampling".
      `verify_coord_geometry()` asserts p(ret[j]) == j exactly.

Acquired slices are NOT hard-copied: upstream re-predicts every voxel and we keep that.
Metrics are computed on missing slices only, so this cannot affect any reported number
(asserted by a preflight gate).
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
import torch

from . import protocol as P

ARSSR_DIR = os.path.join(P.PROJECT, "ArSSR")

# native ArSSR defaults (train.py / the local through-plane adaptation)
ENCODER_NAME = "RDN"
FEATURE_DIM = 128
DECODER_DEPTH = 8          # ArSSR() halves this internally: decoder_depth=int(8/2)
DECODER_WIDTH = 256
N_LR = 4                   # LR slices per training slab
BASE = 48                  # in-plane training patch side
SAMPLE_SIZE = 8000         # coordinates supervised per slab
APPLICABLE_FAMILIES = P.FAMILIES      # coordinate-based -> every family
APPLICABLE_STRIDES = P.STRIDES_TEST   # arbitrary scale -> every stride


@contextlib.contextmanager
def _in_repo():
    cwd = os.getcwd()
    if ARSSR_DIR not in sys.path:
        sys.path.insert(0, ARSSR_DIR)
    os.chdir(ARSSR_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)


def build_model(device="cuda", encoder_name=ENCODER_NAME, feature_dim=FEATURE_DIM,
                decoder_depth=DECODER_DEPTH, decoder_width=DECODER_WIDTH):
    """The unmodified upstream ArSSR network."""
    with _in_repo():
        import model as arssr_model
        net = arssr_model.ArSSR(encoder_name=encoder_name, feature_dim=feature_dim,
                                decoder_depth=int(decoder_depth / 2),
                                decoder_width=decoder_width)
    return net.to(device)


def is_applicable(R, family):
    return True


# --------------------------------------------------------------------------- #
#  D2: coordinate geometry
# --------------------------------------------------------------------------- #
def lr_positions(ret, D=P.DEPTH):
    """Continuous LR-index position p(z) of every HR slice z in 0..D-1.

    p(ret[j]) == j exactly; between acquired slices p is linear in z. Because the mask
    protocol always pins the anchors {0, D-1}, every z is bracketed and no extrapolation
    is ever needed.
    """
    ret = np.asarray(sorted(int(r) for r in ret), dtype=np.float64)
    z = np.arange(int(D), dtype=np.float64)
    return np.interp(z, ret, np.arange(len(ret), dtype=np.float64)).astype(np.float32)


def norm_coord(pos, n):
    """Index position -> grid_sample(align_corners=False) normalised coordinate."""
    return (2.0 * (np.asarray(pos, np.float64) + 0.5) / float(n) - 1.0).astype(np.float32)


def verify_coord_geometry(D=P.DEPTH, R=4):
    """Preflight: acquired slices must land exactly on their own LR sample."""
    from . import protocol as _P
    out = {}
    for fam in _P.FAMILIES:
        ret = _P.build_mask(D, R, fam)
        p = lr_positions(ret, D)
        err = max(abs(float(p[int(r)]) - j) for j, r in enumerate(sorted(ret)))
        mono = bool(np.all(np.diff(p) >= -1e-6))
        out[fam] = dict(max_anchor_error=err, monotone=mono, K=len(ret))
    return out


# --------------------------------------------------------------------------- #
#  training samples
# --------------------------------------------------------------------------- #
class SlabSampler:
    """Anisotropic slabs: n_lr acquired slices in, the HR slices they bracket supervised.

    The mask distribution is IDENTICAL to the one IR3 trained on (audit §F): R drawn from
    {2,4,8} with weights [0.30,0.35,0.35], 50 % uniform / 50 % one of U1..U6.
    """

    def __init__(self, preps, strides=(2, 4, 8), stride_weights=(0.30, 0.35, 0.35),
                 n_lr=N_LR, base=BASE, sample_size=SAMPLE_SIZE, seed=42, is_train=True):
        import image_residual as IR
        self._IR = IR
        self.preps = preps
        self.strides = list(strides)
        self.w = np.asarray(stride_weights, np.float64) / np.sum(stride_weights)
        self.n_lr, self.base, self.sample_size = int(n_lr), int(base), int(sample_size)
        self.rng = np.random.RandomState(int(seed))
        self.is_train = bool(is_train)

    def _mask(self):
        import ir2_components as C
        return C.sample_mask_weighted(P.DEPTH, self.rng, self.strides, list(self.w))

    def _crop_origin(self, vol, zs, H, W, b):
        """Choose an in-plane crop that OVERLAPS the anatomy, using only the LR input.

        Why this exists. Upstream ArSSR trains on a released set of pre-extracted HR brain
        patches, so essentially every training sample contains tissue. Sampling a 48x48 crop
        uniformly over the BraTS 240x240 field of view does not: measured on this data,
        **29.8 % of such crops contain zero brain voxels** and the median crop is only 8.3 %
        tissue, because the brain fills about a sixth of the slice. The other two methods do
        not have this problem -- I3Net is fed the full 256x256 extent (0-3 % empty samples,
        mean 15-16 % tissue) and IR3 trains on full-extent patches. Leaving it in place would
        have handicapped ArSSR through an adapter choice of ours rather than through anything
        about the method, which is exactly the kind of unintended disadvantage the fairness
        audit is supposed to catch.

        The bounding box is computed from the `n_lr` ACQUIRED slices that form the LR input
        and from nothing else, so it cannot leak any missing slice. The crop origin is drawn
        uniformly over the range that keeps the crop overlapping that box, so margin crops
        still contain plenty of background and the model still learns to leave it alone. If
        the slab is entirely background (volume ends), the draw falls back to uniform.
        """
        fg = (vol[:, :, zs] > 0).any(axis=2)
        ys, xs = np.where(fg)
        if ys.size == 0:
            return (int(self.rng.randint(0, H - b + 1)), int(self.rng.randint(0, W - b + 1)))
        lo_y, hi_y = int(ys.min()) - b + 1, int(ys.max())
        lo_x, hi_x = int(xs.min()) - b + 1, int(xs.max())
        lo_y, hi_y = max(0, lo_y), min(H - b, max(0, hi_y))
        lo_x, hi_x = max(0, lo_x), min(W - b, max(0, hi_x))
        return (int(self.rng.randint(lo_y, hi_y + 1)), int(self.rng.randint(lo_x, hi_x + 1)))

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
            np.moveaxis(vol[y0:y0 + b, x0:x0 + b, zs], 2, 0), dtype=np.float32)  # (n_lr,b,b)
        hr = np.ascontiguousarray(
            np.moveaxis(vol[y0:y0 + b, x0:x0 + b, z_lo:z_hi + 1], 2, 0), dtype=np.float32)

        p = lr_positions(ret, D)[z_lo:z_hi + 1] - j          # positions inside this slab
        cz = norm_coord(p, self.n_lr)
        cy = norm_coord(np.arange(b), b)
        cx = norm_coord(np.arange(b), b)
        gz, gy, gx = np.meshgrid(cz, cy, cx, indexing="ij")
        xyz = np.stack([gz, gy, gx], -1).reshape(-1, 3).astype(np.float32)
        vals = hr.reshape(-1, 1).astype(np.float32)

        n = xyz.shape[0]
        idx = self.rng.choice(n, self.sample_size, replace=n < self.sample_size)
        return lr, xyz[idx], vals[idx], meta

    def batch(self, idxs, device="cuda"):
        L, C_, V, M = [], [], [], []
        for i in idxs:
            l, c, v, m = self.draw(i)
            L.append(l); C_.append(c); V.append(v); M.append(m)
        t = lambda a: torch.from_numpy(np.stack(a)).to(device)
        return t(L).unsqueeze(1).float(), t(C_).float(), t(V).float(), M


# --------------------------------------------------------------------------- #
#  whole-volume inference
# --------------------------------------------------------------------------- #
def _query(net, feature_map, xyz):
    """ArSSR.forward's decoder half, copied verbatim so behaviour cannot drift."""
    fv = torch.nn.functional.grid_sample(
        feature_map, xyz.flip(-1).unsqueeze(1).unsqueeze(1),
        mode="bilinear", align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
    cat = torch.cat([fv, xyz], dim=-1)
    N, K = xyz.shape[:2]
    return net.decoder(cat.view(N * K, -1)).view(N, K, -1)


@torch.no_grad()
def build_lr_stack(gt_hw_d, ret):
    """The ONLY thing the model sees: the acquired slices, as (K, H, W)."""
    ret = sorted(int(r) for r in ret)
    return np.ascontiguousarray(np.moveaxis(np.asarray(gt_hw_d, np.float32)[:, :, ret], 2, 0))


@torch.no_grad()
def reconstruct(model, gt_hw_d, ret, D=P.DEPTH, device="cuda", z_chunk=16, z_margin=8,
                slice_batch=4):
    """Whole-volume reconstruction, encoded in z-tiles with margin.

    The RDN encoder is fully convolutional, so tiling along the LR z axis with a margin of
    `z_margin` acquired slices reproduces the untiled feature map on the interior to within
    the encoder's receptive field. Queries for HR slice z are taken only from the tile whose
    interior contains p(z).
    """
    ret = sorted(int(r) for r in ret)
    K = len(ret)
    H, W = gt_hw_d.shape[:2]
    lr = torch.from_numpy(build_lr_stack(gt_hw_d, ret)).to(device)[None, None]  # (1,1,K,H,W)
    p = lr_positions(ret, D)
    cy = torch.from_numpy(norm_coord(np.arange(H), H)).to(device)
    cx = torch.from_numpy(norm_coord(np.arange(W), W)).to(device)
    gy, gx = torch.meshgrid(cy, cx, indexing="ij")
    plane = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)                  # (H*W, 2)

    out = np.zeros((H, W, D), dtype=np.float32)
    filled = np.zeros(D, dtype=bool)
    model.eval()

    for a in range(0, K, z_chunk):
        b = min(K, a + z_chunk)
        a2, b2 = max(0, a - z_margin), min(K, b + z_margin)
        feat = model.encoder(lr[:, :, a2:b2])                                  # (1,C,nz,H,W)
        nz = b2 - a2
        zs = [z for z in range(D) if (a <= p[z] < b) or (b == K and p[z] >= a)]
        zs = [z for z in zs if not filled[z]]
        for i0 in range(0, len(zs), slice_batch):
            sub = zs[i0:i0 + slice_batch]
            cz = torch.from_numpy(norm_coord(p[sub] - a2, nz)).to(device)      # (n,)
            xyz = torch.cat([cz[:, None, None].expand(len(sub), plane.shape[0], 1),
                             plane[None].expand(len(sub), -1, -1)], -1)        # (n,H*W,3)
            pred = _query(model, feat.expand(len(sub), -1, -1, -1, -1), xyz)   # (n,H*W,1)
            v = pred.view(len(sub), H, W).float().cpu().numpy()
            for k, z in enumerate(sub):
                out[:, :, z] = v[k]
                filled[z] = True
        del feat
        if device == "cuda":
            torch.cuda.empty_cache()
    assert filled.all(), f"{int((~filled).sum())} slices unfilled"
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def verify_query_pipeline(gt_hw_d, ret, D=P.DEPTH, device="cpu"):
    """Model-free end-to-end test of the query pipeline (ir3_comparisons.md §14.6).

    Runs the EXACT tensor path the model uses -- the same coordinate construction, the same
    `xyz.flip(-1)` axis order and the same `grid_sample(align_corners=False)` -- but on the
    single-channel LR stack instead of a feature map. Trilinear sampling of the acquired
    slices at our coordinates must then reproduce direct linear interpolation along z. If the
    coordinate order were transposed, or the z mapping shifted by even a fraction of a slice,
    this would not hold.

    Returns max |grid_sample(LR, xyz) - linear_interp(gt, ret)| over the interior.
    """
    import image_residual as IR
    lr = torch.from_numpy(build_lr_stack(gt_hw_d, ret)).to(device)[None, None].float()
    K = lr.shape[2]
    H, W = gt_hw_d.shape[:2]
    p = lr_positions(ret, D)
    cy = torch.from_numpy(norm_coord(np.arange(H), H)).to(device)
    cx = torch.from_numpy(norm_coord(np.arange(W), W)).to(device)
    gy, gx = torch.meshgrid(cy, cx, indexing="ij")
    plane = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)
    ref = IR.inputs.linear_interp_volume(gt_hw_d, ret, D)
    worst = 0.0
    for z in range(D):
        cz = torch.full((1,), float(norm_coord(np.array([p[z]]), K)[0]), device=device)
        xyz = torch.cat([cz[:, None, None].expand(1, plane.shape[0], 1),
                         plane[None]], -1)
        v = torch.nn.functional.grid_sample(
            lr, xyz.flip(-1).unsqueeze(1).unsqueeze(1), mode="bilinear",
            align_corners=False)[0, 0, 0, 0, :].reshape(H, W).cpu().numpy()
        worst = max(worst, float(np.abs(v - ref[:, :, z]).max()))
    return worst


@torch.no_grad()
def verify_query_pipeline_upstream_convention(gt_hw_d, ret, D=P.DEPTH, device="cpu"):
    """The same test as `verify_query_pipeline`, but with UPSTREAM's cell-centre coordinate
    convention `cz = 2*(z+0.5)/D - 1` (what `make_coord(hr_shape)` produces).

    Reports how far that convention is from correct under slice-selection decimation. It is
    exact for zoom-resampled LR, which is what upstream generates, and wrong here -- which is
    the entire justification for deviation D2.
    """
    import image_residual as IR
    lr = torch.from_numpy(build_lr_stack(gt_hw_d, ret)).to(device)[None, None].float()
    H, W = gt_hw_d.shape[:2]
    cy = torch.from_numpy(norm_coord(np.arange(H), H)).to(device)
    cx = torch.from_numpy(norm_coord(np.arange(W), W)).to(device)
    gy, gx = torch.meshgrid(cy, cx, indexing="ij")
    plane = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)
    ref = IR.inputs.linear_interp_volume(gt_hw_d, ret, D)
    up = 2.0 * (np.arange(D) + 0.5) / D - 1.0
    worst = 0.0
    for z in range(D):
        xyz = torch.cat([torch.full((1, plane.shape[0], 1), float(up[z]), device=device),
                         plane[None]], -1)
        v = torch.nn.functional.grid_sample(
            lr, xyz.flip(-1).unsqueeze(1).unsqueeze(1), mode="bilinear",
            align_corners=False)[0, 0, 0, 0, :].reshape(H, W).cpu().numpy()
        worst = max(worst, float(np.abs(v - ref[:, :, z]).max()))
    return worst
