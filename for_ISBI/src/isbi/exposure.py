"""Training-exposure experiment helpers (added 2026-10-17 at the user's request; exploratory arm).

Variant a: training masks = uniform + jittered + clustered.  Variant b: uniform + jittered only (clustered unseen).
Irregular TRAINING masks use the same generators as the test masks (common.jittered_mask / clustered_mask) with
independent random seeds; training subjects (GLI train split) are patient-disjoint from every test role.
"""
from __future__ import annotations

import numpy as np
import torch

from . import common as C

VARIANT_PATTERNS = {"a": ("jittered", "clustered"), "b": ("jittered",)}


def train_mask(pattern, R, rng, D=C.D):
    seed = int(rng.randint(0, 2 ** 31 - 1))
    if pattern == "uniform":
        return C.uniform_mask(R, D)
    return (C.jittered_mask if pattern == "jittered" else C.clustered_mask)(R, seed, D)


def interp_slices(vol, ret, zs):
    """Linear interpolation in physical z of the ACQUIRED slices `ret` at positions zs -> (H, W, len(zs))."""
    ret = np.asarray(sorted(ret))
    out = np.empty(vol.shape[:2] + (len(zs),), np.float32)
    for k, z in enumerate(zs):
        j = np.searchsorted(ret, z, side="right") - 1
        j = min(max(j, 0), len(ret) - 2)
        l, r = ret[j], ret[j + 1]
        a = (z - l) / (r - l)
        out[:, :, k] = (1 - a) * vol[:, :, l] + a * vol[:, :, r]
    return out


class I3IrregularSampler:
    """I3Net training windows under a (possibly irregular) mask through the interpolation adapter.

    Window = 4 consecutive grid slices G[s..s+3] (G = uniform grid of stride R), input = adapter values at those
    positions (exact where acquired), target = GT slices G[s]..G[s]+3R. Supervision weight: 1 on positions that
    were NOT acquired and are NOT grid positions (grid positions are hard-copied from the input by upstream I3Net).
    With probability p_uniform the mask is uniform (= the original training distribution).
    """

    def __init__(self, preps, R, patterns, p_uniform=0.5, seed=42):
        from comparisons import protocol as P
        self.P = P
        self.vols = [P.pad_to(p["vol01"], 256)[0] for p in preps]
        self.R, self.pats, self.pu = int(R), list(patterns), float(p_uniform)
        self.G = C.uniform_mask(self.R)
        self.rng = np.random.RandomState(seed)
        self.S = 3 * self.R + 1

    def sample(self):
        v = self.vols[int(self.rng.randint(len(self.vols)))]
        pat = "uniform" if self.rng.rand() < self.pu else self.pats[int(self.rng.randint(len(self.pats)))]
        ret = train_mask(pat, self.R, self.rng)
        s = int(self.rng.randint(0, len(self.G) - 3))
        gz = self.G[s:s + 4]
        x = interp_slices(v, ret, gz)
        z0 = gz[0]
        y = np.ascontiguousarray(v[:, :, z0:z0 + self.S])
        w = np.ones(self.S, np.float32)
        w[::self.R] = 0
        acq = set(ret)
        for k in range(self.S):
            if z0 + k in acq:
                w[k] = 0
        return x, y, w

    def batch(self, n, device):
        xs, ys, ws = zip(*[self.sample() for _ in range(n)])
        t = lambda a: torch.from_numpy(np.stack(a)).to(device)  # noqa: E731
        return t(xs), t(ys), t(ws)[:, None, None, :].expand(n, 256, 256, self.S)
