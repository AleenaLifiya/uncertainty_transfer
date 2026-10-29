"""
IR3C_more_comparisons/comparisons_more/dcsrn_adapter.py
=======================================================
DCSRN -- "Brain MRI Super Resolution Using 3D Deep Densely Connected Neural Networks"
(Chen et al., ISBI 2018), official implementation
https://github.com/YunzeMan/DCSRN @ 72558e272d647aae61e2ab07f3c09221be6a7e2f.

Two things live in this file, and the distinction matters:

  1. `tf_graph()` builds the **official TensorFlow graph**, unmodified except for the two
     recorded TF1 -> TF2 compatibility edits (`provenance/DCSRN/code_changes.patch`). It runs
     on this machine and is the reference implementation.

  2. `DCSRNNet` is a **line-by-line PyTorch transcription of that same graph**, used for
     training only, because the TensorFlow wheel available for this machine has no GPU
     backend (see `provenance/DCSRN/adaptations.md` for the full blocker report: aarch64 +
     Python 3.10, `tf.test.is_built_with_cuda() == False`, 11.8 s per training step on CPU
     = 41 h per model per scale; TF 1.x is not installable at all; Docker is not available).
     The transcription is **not asserted to be equivalent, it is tested**: `verify_port()`
     transplants the TensorFlow variables into the PyTorch module and compares outputs on
     random volumes, and `verify_training_equivalence()` runs both optimisers from the same
     initialisation on the same batches and compares the loss trajectories. Both reports are
     saved to `provenance/DCSRN/`.

Architecture (`tf_dcsrn/dcsrn.py:create_conv_net`, channels=1, layers=4, growth_rate=24,
filter_size=3), reproduced exactly:

    x  --conv3d(3^3, 1 -> 48)-->  h0
    for layer in 0..3:
        last = h                              # BEFORE the concat
        h    = concat([h] + convs, dim=C)     # -> (2+layer)*24 channels
        h    = conv3d(3^3, . -> 24)(elu(batch_norm(h)))
        convs.append(last)
    out = conv3d(3^3, 144 -> 1)(concat([h] + convs))

`batch_norm` is `tf.contrib.layers.batch_norm(h, 0)`: decay 0 and is_training=True, i.e.
normalise by the CURRENT batch moments over (N, D, H, W), trainable beta, no gamma,
eps 1e-3. There is no residual connection, no output activation, and no upsampling layer --
DCSRN maps a degraded volume to a restored volume **on the same grid**.

Through-plane-only degradation (prompt §5, mandatory)
----------------------------------------------------
The paper degrades by isotropic k-space truncation. That is a different reconstruction
problem from this benchmark and would invalidate the comparison, so the degradation is
replaced by the IR3C one and ONLY the slice axis is touched:

    HR (240, 240, 153)
      --keep the IR3C mask's acquired slices-->  (240, 240, K)      1 x 1 x R decimation
      --z-only linear interpolation back-->      (240, 240, 153)    1 x 1 x R interpolation
      --DCSRN-->                                 (240, 240, 153)

`assert_through_plane_only()` checks, for every (R, family) cell: in-plane H unchanged,
in-plane W unchanged, only the slice dimension reduced, the retained slice locations exactly
equal to the frozen IR3C mask, the output shape exactly equal to the HR reference, and the
in-plane content of every ACQUIRED slice bit-identical to the HR slice (i.e. nothing resized
or resampled in-plane anywhere in the pipeline).

Tensor order: the shared protocol uses numpy `(H, W, D)`. TensorFlow's graph is `NDHWC`
and PyTorch is `NCDHW`; `D` is the SLICE axis in both. The mapping is done once, in
`to_tensor` / `from_tensor`, and `assert_through_plane_only()` runs on the numpy side where
the axis meaning is unambiguous, so the `(R,1,1)` vs `(1,1,R)` trap the prompt warns about
cannot occur silently.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import protocol_more as PM
from comparisons import protocol as P

DCSRN_DIR = os.path.join(PM.EXP_DIR, "repositories", "DCSRN")

# upstream defaults: create_conv_net(channels=1, layers=4, growth_rate=24, filter_size=3)
CHANNELS = 1
LAYERS = 4
GROWTH_RATE = 24
FILTER_SIZE = 3
BN_EPS = 1e-3
PATCH = 64                      # NII2NPY/data_augment.py: 64x64x64 patches
BATCH = 2                       # dcsrn.Trainer(batch_size=2)
LR = 1e-5                       # dcsrn.Trainer._get_optimizer default

APPLICABLE_FAMILIES = ["U0"]
APPLICABLE_STRIDES = [2, 4, 8]


def is_applicable(R, family):
    return (int(R) in APPLICABLE_STRIDES) and (str(family) in APPLICABLE_FAMILIES)


# --------------------------------------------------------------------------- #
#  1. the official TensorFlow graph (reference implementation)
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _in_repo():
    cwd = os.getcwd()
    if DCSRN_DIR not in sys.path:
        sys.path.insert(0, DCSRN_DIR)
    os.chdir(DCSRN_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)


def tf_graph():
    """Build the official `DCSRN` graph. Returns (net, tf module). TF 2.x, v1 compat."""
    with _in_repo():
        from tf_dcsrn import dcsrn as _d
        import tensorflow.compat.v1 as tf
        net = _d.DCSRN(channels=CHANNELS)
    return net, tf


# --------------------------------------------------------------------------- #
#  2. the PyTorch transcription
# --------------------------------------------------------------------------- #
class _ContribBatchNorm(nn.Module):
    """`tf.contrib.layers.batch_norm(x, 0)`: batch moments over (N, D, H, W), trainable
    beta, gamma fixed at 1 (scale=False), eps 1e-3, never a moving average."""

    def __init__(self, c, eps=BN_EPS):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(c))
        self.eps = float(eps)

    def forward(self, x):
        dims = (0, 2, 3, 4)                       # every axis but the channel axis
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) + self.beta.view(1, -1, 1, 1, 1)


class DCSRNNet(nn.Module):
    """`tf_dcsrn.dcsrn.create_conv_net`, transcribed. Same weights, same order, same result.

    Input/output are `(N, 1, D, H, W)`; D is the slice axis.
    """

    def __init__(self, channels=CHANNELS, layers=LAYERS, growth_rate=GROWTH_RATE,
                 filter_size=FILTER_SIZE):
        super().__init__()
        self.layers = int(layers)
        g = int(growth_rate)
        k = int(filter_size)
        pad = (k - 1) // 2                       # TF 'SAME' with odd k and stride 1
        self.w0 = nn.Conv3d(channels, 2 * g, k, padding=pad, bias=False)
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(self.layers):
            in_features = (2 + layer) * g
            self.bns.append(_ContribBatchNorm(in_features))
            self.convs.append(nn.Conv3d(in_features, g, k, padding=pad, bias=False))
        self.w2 = nn.Conv3d((2 + self.layers) * g, channels, k, padding=pad, bias=False)
        self.reset_parameters(k, g)

    def reset_parameters(self, k=FILTER_SIZE, g=GROWTH_RATE):
        """`weight_variable(shape, stddev)` = `tf.truncated_normal(shape, stddev)`,
        `stddev = sqrt(2 / (filter_size**3 * growth_rate))`."""
        stddev = math.sqrt(2.0 / (k ** 3 * g))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=stddev, a=-2 * stddev, b=2 * stddev)
            elif isinstance(m, _ContribBatchNorm):
                nn.init.zeros_(m.beta)

    def forward(self, x):
        h = self.w0(x)
        convs = []
        for layer in range(self.layers):
            last = h
            if convs:
                h = torch.cat([h] + convs, dim=1)
            h = self.convs[layer](F.elu(self.bns[layer](h)))
            convs.append(last)
        h = torch.cat([h] + convs, dim=1)
        return self.w2(h)


def build_model(device="cuda", seed=None):
    if seed is not None:
        torch.manual_seed(int(seed))
    m = DCSRNNet().to(device)
    return m


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


def loss_fn(pred, target):
    """`cost = reduce_sum((logits - y)^2) / (N*D*H*W)`; with channels=1 this is the MSE."""
    n = pred.shape[0] * pred.shape[2] * pred.shape[3] * pred.shape[4]
    return torch.sum((pred - target) ** 2) / n


# --------------------------------------------------------------------------- #
#  tensor-order helpers -- the ONE place (H,W,D) <-> NCDHW happens
# --------------------------------------------------------------------------- #
def to_tensor(vol_hw_d, device="cuda"):
    """(H, W, D) numpy -> (1, 1, D, H, W) torch. D (the SLICE axis) becomes the depth dim."""
    a = np.ascontiguousarray(np.asarray(vol_hw_d, np.float32).transpose(2, 0, 1))
    return torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(device)


def from_tensor(t):
    """(1, 1, D, H, W) torch -> (H, W, D) numpy."""
    a = t[0, 0].detach().float().cpu().numpy()
    return np.ascontiguousarray(a.transpose(1, 2, 0))


def to_tensor_ndhwc(vol_hw_d):
    """(H, W, D) numpy -> (1, D, H, W, 1) numpy, the TensorFlow NDHWC layout."""
    a = np.asarray(vol_hw_d, np.float32).transpose(2, 0, 1)
    return a[None, ..., None]


# --------------------------------------------------------------------------- #
#  through-plane-only degradation
# --------------------------------------------------------------------------- #
def decimate(gt_hw_d, ret):
    """1 x 1 x R decimation: keep the acquired slices, touch nothing in-plane."""
    return np.ascontiguousarray(
        np.asarray(gt_hw_d, np.float32)[:, :, sorted(int(r) for r in ret)])


def build_lr_volume(gt_hw_d, ret, D=None):
    """1 x 1 x R interpolation back onto the HR grid, through the SHARED interpolator.

    `protocol.interpolate_baseline(..., "linear")` is the project's own z-only linear
    interpolation (`image_residual.inputs.linear_interp_volume`) -- the same function that
    produces the `linear` baseline row and IR3's anchor -- so DCSRN's input is a quantity the
    benchmark already reports, and no new resampling code enters the comparison.
    """
    D = int(D if D is not None else gt_hw_d.shape[2])
    return P.interpolate_baseline(np.asarray(gt_hw_d, np.float32), ret, "linear", D)


def assert_through_plane_only(gt_hw_d, ret, D=None, pred_hw_d=None):
    """The prompt §5 gates. Returns a dict of measured quantities; raises on violation."""
    D = int(D if D is not None else gt_hw_d.shape[2])
    gt = np.asarray(gt_hw_d, np.float32)
    H, W, Dg = gt.shape
    ret = sorted(int(r) for r in ret)
    lo = decimate(gt, ret)
    lr = build_lr_volume(gt, ret, D)

    out = {}
    out["hr_shape"] = (H, W, Dg)
    out["lr_decimated_shape"] = tuple(int(s) for s in lo.shape)
    out["lr_interpolated_shape"] = tuple(int(s) for s in lr.shape)
    out["in_plane_H_unchanged"] = bool(lo.shape[0] == H and lr.shape[0] == H)
    out["in_plane_W_unchanged"] = bool(lo.shape[1] == W and lr.shape[1] == W)
    out["only_slice_dim_reduced"] = bool(lo.shape[2] == len(ret) < Dg)
    out["interp_restores_D"] = bool(lr.shape[2] == D == Dg)
    # the acquired slices must survive the round trip bit-for-bit -- proves no in-plane
    # interpolation, no resize, no shift
    out["acquired_slices_bit_exact"] = float(np.abs(lr[:, :, ret] - gt[:, :, ret]).max())
    out["decimated_slices_bit_exact"] = float(np.abs(lo - gt[:, :, ret]).max())
    out["retained_indices_sha256"] = P.AC.sha_idx(ret) if hasattr(P, "AC") else ""
    if pred_hw_d is not None:
        pr = np.asarray(pred_hw_d, np.float32)
        out["pred_shape"] = tuple(int(s) for s in pr.shape)
        out["pred_matches_hr"] = bool(pr.shape == gt.shape)
        assert pr.shape == gt.shape, f"output {pr.shape} != HR {gt.shape}"
    assert out["in_plane_H_unchanged"], "in-plane height changed"
    assert out["in_plane_W_unchanged"], "in-plane width changed"
    assert out["only_slice_dim_reduced"], "slice dimension was not the only one reduced"
    assert out["interp_restores_D"], "interpolation did not restore the HR depth"
    assert out["acquired_slices_bit_exact"] == 0.0, "acquired slices were altered"
    assert out["decimated_slices_bit_exact"] == 0.0, "decimation altered slice content"
    return out


def build_lr_stack(gt_hw_d, ret, D=None):
    """The ONLY thing DCSRN sees. Provably a function of `gt_hw_d[:, :, ret]` alone."""
    return build_lr_volume(gt_hw_d, ret, D)


# --------------------------------------------------------------------------- #
#  training samples: 64^3 patches (NII2NPY/data_augment.py)
# --------------------------------------------------------------------------- #
class PatchSampler:
    """Random 64x64x64 (LR, HR) patch pairs, upstream's augmentation scheme.

    `data_augment.py` cuts 100 random 64^3 patches per subject; we draw them on the fly from
    the same prepared volumes every other method uses, at the fixed stride R this specialist
    is trained for. No flips or rotations (the IR3C protocol allows none).
    """

    def __init__(self, preps, R, D=None, patch=PATCH, seed=42, cache_cap=24):
        self.preps = preps
        self.R = int(R)
        self.D = int(D if D is not None else PM.DEPTH)
        self.patch = int(patch)
        self.rng = np.random.RandomState(int(seed))
        self.ret = P.build_mask(self.D, self.R, "U0")
        self._cache, self._cap = {}, int(cache_cap)

    def _pair(self, i):
        if i not in self._cache:
            if len(self._cache) >= self._cap:
                self._cache.pop(next(iter(self._cache)))
            # explicit copy: prepared volumes may be memory-mapped (protocol_more.prepare_cached)
            gt = np.array(self.preps[i]["vol01"], dtype=np.float32, copy=True)
            self._cache[i] = (build_lr_volume(gt, self.ret, self.D), gt)
        return self._cache[i]

    def batch(self, idxs, device="cuda"):
        lrs, hrs = [], []
        p = self.patch
        for i in idxs:
            lr, gt = self._pair(int(i))
            H, W, D = gt.shape
            y = int(self.rng.randint(0, H - p + 1))
            x = int(self.rng.randint(0, W - p + 1))
            z = int(self.rng.randint(0, D - p + 1))
            lrs.append(lr[y:y + p, x:x + p, z:z + p])
            hrs.append(gt[y:y + p, x:x + p, z:z + p])
        # (B, H, W, D) -> (B, 1, D, H, W)
        f = lambda a: torch.from_numpy(                                    # noqa: E731
            np.ascontiguousarray(np.stack(a, 0).transpose(0, 3, 1, 2))).unsqueeze(1).to(device)
        return f(lrs), f(hrs)


class MixedMaskPatchSampler(PatchSampler):
    """64^3 patches under the PROTOCOL's OWN training mask distribution.

    Identical to `PatchSampler` in every respect but one: the acquisition mask is redrawn
    per sample from `image_residual.masks.sample_training_mask`, with the stride pinned to
    this specialist's R and `uniform_prob` from the config. That is exactly the family
    exposure ArSSR and IR3 received during training -- ~50 % uniform `U0`, the rest spread
    over `U1`-`U6` with a fresh seed on every draw -- so a DCSRN trained this way can be
    compared with them on the non-uniform families like for like.

    Only the FAMILY is mixed, not the stride: the prompt requires one fixed-scale model per
    R, so each specialist keeps its own R. That difference from ArSSR/IR3 (which train one
    arbitrary-scale model across strides) is a property of fixed-scale methods and is
    reported, not hidden.

    The interpolant is built on the IN-PLANE CROP rather than the whole volume. Linear
    interpolation along z is pointwise in the in-plane axes, so cropping first is exactly
    equivalent -- asserted once per sampler in `_check_crop_equivalence` -- and ~5x cheaper,
    which matters because the mask changes every sample and cannot be cached.
    """

    def __init__(self, preps, R, D=None, patch=PATCH, seed=42, cache_cap=24,
                 uniform_prob=0.5):
        super().__init__(preps, R, D, patch, seed, cache_cap)
        self.uniform_prob = float(uniform_prob)
        self.mask_rng = np.random.RandomState(int(seed) + 7)
        self.realised = []
        self._checked = False

    def _gt(self, i):
        if i not in self._cache:
            if len(self._cache) >= self._cap:
                self._cache.pop(next(iter(self._cache)))
            self._cache[i] = np.array(self.preps[i]["vol01"], dtype=np.float32, copy=True)
        return self._cache[i]

    def _check_crop_equivalence(self, gt, ret, y0, x0, p):
        """Prove the in-plane crop shortcut changes nothing, once per sampler."""
        full = build_lr_volume(gt, ret, self.D)[y0:y0 + p, x0:x0 + p, :]
        crop = build_lr_volume(gt[y0:y0 + p, x0:x0 + p, :], ret, self.D)
        err = float(np.abs(full - crop).max())
        assert err == 0.0, f"in-plane crop changed the interpolant by {err}"
        self._checked = True
        return err

    def draw_mask(self):
        import image_residual as IR
        return IR.masks.sample_training_mask(self.D, self.mask_rng, strides=(self.R,),
                                             uniform_prob=self.uniform_prob)

    def batch(self, idxs, device="cuda"):
        lrs, hrs = [], []
        p = self.patch
        for i in idxs:
            gt = self._gt(int(i))
            H, W, D = gt.shape
            y = int(self.rng.randint(0, H - p + 1))
            x = int(self.rng.randint(0, W - p + 1))
            z = int(self.rng.randint(0, D - p + 1))
            ret, meta = self.draw_mask()
            if not self._checked:
                self._check_crop_equivalence(gt, ret, y, x, p)
            sub = np.ascontiguousarray(gt[y:y + p, x:x + p, :])
            lr = build_lr_volume(sub, ret, self.D)
            lrs.append(lr[:, :, z:z + p])
            hrs.append(sub[:, :, z:z + p])
            self.realised.append(meta)
        f = lambda a: torch.from_numpy(                                    # noqa: E731
            np.ascontiguousarray(np.stack(a, 0).transpose(0, 3, 1, 2))).unsqueeze(1).to(device)
        return f(lrs), f(hrs)

    def distribution(self):
        """The realised family distribution, for the training-distribution table."""
        import collections
        c = collections.Counter(m["mask_family"] for m in self.realised)
        n = max(1, sum(c.values()))
        return [dict(mask_family=k, n=v, fraction=v / n) for k, v in sorted(c.items())]


# --------------------------------------------------------------------------- #
#  whole-volume inference
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruct_wholevolume(model, gt_hw_d, ret, R=None, D=None, device="cuda"):
    """One forward pass over the WHOLE volume.

    NOTE: this is *not* what upstream does -- see `reconstruct_tiled`. It is kept because it
    is the obvious adaptation, it is what this experiment tried first, and reporting the
    comparison between the two is the honest way to record why the other one was chosen.
    """
    D = int(D if D is not None else gt_hw_d.shape[2])
    lr = build_lr_volume(gt_hw_d, ret, D)
    x = to_tensor(lr, device)
    model.eval()
    y = model(x)
    out = from_tensor(y)
    assert out.shape == np.asarray(gt_hw_d).shape, \
        f"output {out.shape} != HR {np.asarray(gt_hw_d).shape}"
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@torch.no_grad()
def reconstruct_tiled(model, gt_hw_d, ret, R=None, D=None, device="cuda", roi=PATCH,
                      sw_batch=4, overlap=0.25, mode="gaussian"):
    """Upstream's inference granularity: 64^3 patches, predicted in batches of 4.

    Why this matters, and why it is the FAITHFUL mode rather than a convenience:

      * `NII2NPY/data_augment.py` cuts the dataset into 64x64x64 patches, and
        `tf_dcsrn/image_util.py: MedicalImageDataProvider` yields those patches. Upstream's
        own evaluation, `dcsrn_test.py`, is
            `test_x, test_y = test_provider(4);  net.predict(path, test_x)`
        -- i.e. FOUR 64^3 patches, never a whole volume.
      * DCSRN's normalisation is `tf.contrib.layers.batch_norm(x, 0)`: decay 0 and
        is_training=True, so it normalises by the CURRENT BATCH's moments at test time too;
        the moving averages are overwritten every step and never read.

    Those two facts together mean the inference granularity is part of the method. A
    240x240x153 BraTS volume is ~84 % exact-zero background, so its moments are nothing like
    a brain-centred 64^3 training patch, and a whole-volume forward evaluates the network far
    outside the statistics it was trained under. Tiling restores them.

    The windows are reassembled by `monai.inferers.sliding_window_inference`, the same
    inferer `image_residual.inference.reconstruct_volume` uses for IR3, with `sw_batch_size=4`
    to match upstream's batch of four patches. The output is the full (H, W, D) volume, so
    the shared evaluator scores it exactly as it scores every other method.
    """
    from monai.inferers import sliding_window_inference
    D = int(D if D is not None else gt_hw_d.shape[2])
    lr = build_lr_volume(gt_hw_d, ret, D)
    x = to_tensor(lr, device)                                   # (1, 1, D, H, W)
    model.eval()
    y = sliding_window_inference(x, (int(roi),) * 3, int(sw_batch), model,
                                 overlap=float(overlap), mode=str(mode))
    out = from_tensor(y)
    assert out.shape == np.asarray(gt_hw_d).shape, \
        f"output {out.shape} != HR {np.asarray(gt_hw_d).shape}"
    return np.clip(out, 0.0, 1.0).astype(np.float32)


INFERENCE_MODES = {"whole_volume": reconstruct_wholevolume, "tiled64": reconstruct_tiled}


def reconstruct(model, gt_hw_d, ret, R=None, D=None, device="cuda", mode="tiled64"):
    """Dispatch. `tiled64` is upstream's granularity and is the default; the mode actually
    used for the reported results is chosen on the VALIDATION set and recorded in
    `metrics/dcsrn_inference_mode.json`."""
    return INFERENCE_MODES[str(mode)](model, gt_hw_d, ret, R, D, device)


# --------------------------------------------------------------------------- #
#  equivalence tests against the official TensorFlow graph
# --------------------------------------------------------------------------- #
def verify_port(shape=(1, 24, 32, 32, 1), seed=0, device="cpu", out_json=None):
    """Transplant the TensorFlow variables into `DCSRNNet` and compare outputs.

    Returns a dict with the maximum absolute and relative differences. This is the evidence
    that the PyTorch module used for training IS the official graph.

    The default device is CPU so that the comparison is float32-vs-float32 on the same
    backend: on GPU, cuDNN's conv3d algorithms and TF32 accumulation introduce a ~5e-4
    relative difference that has nothing to do with the transcription. Both devices are
    measured and reported.
    """
    net, tf = tf_graph()
    rng = np.random.RandomState(int(seed))
    x_ndhwc = rng.rand(*shape).astype(np.float32)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_out = sess.run(net.logits, feed_dict={net.x: x_ndhwc,
                                                 net.y: np.zeros_like(x_ndhwc)})
        vals = {v.name: sess.run(v) for v in tf.trainable_variables()}
        names = [v.name for v in tf.trainable_variables()]

    # graph creation order, read straight off the trainable-variable list
    conv_names = [n for n in names if n.startswith("Variable")]
    beta_names = [n for n in names if n.startswith("beta")]

    m = DCSRNNet().to(device).eval()
    def tf2torch(w):                       # (kd,kh,kw,in,out) -> (out,in,kd,kh,kw)
        return torch.from_numpy(np.ascontiguousarray(w.transpose(4, 3, 0, 1, 2)))
    with torch.no_grad():
        m.w0.weight.copy_(tf2torch(vals[conv_names[0]]).to(device))
        for i in range(LAYERS):
            m.convs[i].weight.copy_(tf2torch(vals[conv_names[1 + i]]).to(device))
            m.bns[i].beta.copy_(torch.from_numpy(vals[beta_names[i]]).to(device))
        m.w2.weight.copy_(tf2torch(vals[conv_names[-1]]).to(device))

        x_t = torch.from_numpy(np.ascontiguousarray(x_ndhwc.transpose(0, 4, 1, 2, 3))).to(device)
        t_out = m(x_t).float().cpu().numpy().transpose(0, 2, 3, 4, 1)

    d = np.abs(t_out - tf_out)
    scale = float(np.abs(tf_out).max()) or 1.0
    rep = dict(shape=list(shape), seed=int(seed),
               tf_n_params=int(sum(int(np.prod(v.shape)) for v in vals.values())),
               torch_n_params=int(n_parameters(m)),
               max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()),
               max_rel_diff=float(d.max() / scale),
               tf_output_absmax=scale,
               tf_variables=names,
               device=str(device),
               equivalent=bool(d.max() / scale < 1e-5))
    if out_json:
        json.dump(rep, open(out_json, "w"), indent=2)
    return rep


def verify_training_equivalence(steps=25, shape=(2, 32, 32, 32, 1), lr=1e-3, seed=0,
                                device="cpu", out_json=None):
    """Run the OFFICIAL TF optimiser and the PyTorch port from the SAME initialisation on
    the SAME batches and compare the loss trajectories step by step.

    A matching trajectory means the port reproduces not only the forward pass but the loss,
    the gradients and the parameter update as well.
    """
    net, tf = tf_graph()
    opt = tf.train.AdamOptimizer(learning_rate=float(lr)).minimize(net.cost)
    rng = np.random.RandomState(int(seed))
    xs = [rng.rand(*shape).astype(np.float32) for _ in range(int(steps))]
    ys = [rng.rand(*shape).astype(np.float32) for _ in range(int(steps))]

    tf_losses, vals, names = [], None, None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        vals = {v.name: sess.run(v) for v in tf.trainable_variables()}
        names = [v.name for v in tf.trainable_variables()]
        for i in range(int(steps)):
            _, c = sess.run((opt, net.cost), feed_dict={net.x: xs[i], net.y: ys[i]})
            tf_losses.append(float(c))

    conv_names = [n for n in names if n.startswith("Variable")]
    beta_names = [n for n in names if n.startswith("beta")]
    m = DCSRNNet().to(device)
    def tf2torch(w):
        return torch.from_numpy(np.ascontiguousarray(w.transpose(4, 3, 0, 1, 2)))
    with torch.no_grad():
        m.w0.weight.copy_(tf2torch(vals[conv_names[0]]).to(device))
        for i in range(LAYERS):
            m.convs[i].weight.copy_(tf2torch(vals[conv_names[1 + i]]).to(device))
            m.bns[i].beta.copy_(torch.from_numpy(vals[beta_names[i]]).to(device))
        m.w2.weight.copy_(tf2torch(vals[conv_names[-1]]).to(device))

    # tf.train.AdamOptimizer defaults: beta1 .9, beta2 .999, eps 1e-8 (same as torch.optim.Adam)
    o = torch.optim.Adam(m.parameters(), lr=float(lr), betas=(0.9, 0.999), eps=1e-8)
    m.train()
    pt_losses = []
    for i in range(int(steps)):
        xt = torch.from_numpy(np.ascontiguousarray(xs[i].transpose(0, 4, 1, 2, 3))).to(device)
        yt = torch.from_numpy(np.ascontiguousarray(ys[i].transpose(0, 4, 1, 2, 3))).to(device)
        o.zero_grad(set_to_none=True)
        loss = loss_fn(m(xt), yt)
        loss.backward()
        o.step()
        pt_losses.append(float(loss.detach()))

    a, b = np.asarray(tf_losses), np.asarray(pt_losses)
    rep = dict(steps=int(steps), shape=list(shape), lr=float(lr), seed=int(seed),
               tf_losses=[float(v) for v in a], torch_losses=[float(v) for v in b],
               max_abs_diff=float(np.abs(a - b).max()),
               max_rel_diff=float((np.abs(a - b) / np.maximum(np.abs(a), 1e-12)).max()),
               device=str(device),
               equivalent=bool((np.abs(a - b) / np.maximum(np.abs(a), 1e-12)).max() < 1e-4))
    if out_json:
        json.dump(rep, open(out_json, "w"), indent=2)
    return rep
