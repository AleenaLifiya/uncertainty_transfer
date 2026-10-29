"""Probabilistic (heteroscedastic Laplace + MC-dropout) adaptations of I3Net and SAINT.

Both wrap the UNMODIFIED upstream modules (imported read-only through the project's adapters)
and add exactly three things, at the network's own output stage:
  * Dropout2d(p) at documented positions (trained during fine-tuning),
  * a positive scale head  b = exp(conv(h)) + B_MIN, initialised to zero weights and a
    constant bias (so b starts at B0 everywhere),
  * nothing else: the mean head IS the original output conv, so with dropout off the mean is
    bit-identical to the source checkpoint at initialisation (asserted in tests).

I3Net  (spatial, per-scale model, input = n_lr acquired slices of a 256x256 window stack)
  body/head/alignment unchanged -> res (64 ch)
  tail: [Dropout2d] -> conv3x3 -> ReLU -> [Dropout2d] -> { mean: original conv3x3 (64->S)
                                                           { scale: new conv3x3 (64->S), b = exp(.) + B_MIN }
  acquired slice positions (::R) of the window are hard-copied (upstream) -> b irrelevant there.
SAINT  (MSR marginal SR on sagittal + coronal lines, then axial 2-D RFN fusion)
  MSR unchanged and FROZEN (deterministic). RFN: SFENet -> 4 RDBs -> [Dropout2d on concat]
  -> GFF (+skip) -> [Dropout2d] -> { mean: original UPNet conv + mean(inputs) residual
                                     { scale: new conv3x3 (64->1) }
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

B_MIN = 1e-3            # lower bound of the Laplace scale (intensity units, [0,1] range)
B0 = 0.02               # initial scale (order of the source models' validation MAE)
P_DROP = 0.1            # modest channel dropout


def inv_softplus(y):
    return float(y + math.log(-math.expm1(-y)))


def _scale_head(cin, cout, hidden=32):
    """Scale head: conv3x3(cin->32) -> ReLU -> conv3x3(32->cout); b = B_MIN + exp(.).
    Last conv zero-initialised (b = B0 everywhere at start). History (superseded, recorded in
    logs): (1) single conv + softplus learned ~50x too slowly at b~0.02; (2) a single LINEAR conv
    on pre-activation RFN features cannot represent edge magnitude (b tracked intensity only,
    within-volume Spearman with |e| ~0 on SEL), hence the hidden ReLU layer."""
    last = nn.Conv2d(hidden, cout, 3, padding=1)
    nn.init.zeros_(last.weight)
    nn.init.constant_(last.bias, math.log(B0 - B_MIN))
    return nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.ReLU(inplace=False), last)


def to_scale(s):
    return torch.exp(s.float().clamp(-12, 4)) + B_MIN


def dropout_modules(m):
    """Trained MC-dropout layers: the Dropout2d of ProbRFN / ProbI3Net and the marker classes of native.py
    (MCDropout, MCDropout3d). Upstream modules such as I3Net's nn.Dropout(p=0) are never included."""
    return [x for x in m.modules() if isinstance(x, nn.Dropout2d) or type(x).__name__ in ("MCDropout", "MCDropout3d")]


def set_mc(model, on: bool):
    """eval() everything, then (optionally) re-enable ONLY the trained Dropout2d modules."""
    model.eval()
    for d in dropout_modules(model):
        d.train(on)


# ============================================================================ I3Net
class ProbI3Net(nn.Module):
    def __init__(self, base, p=P_DROP):
        super().__init__()
        self.base = base                       # upstream I3Net (weights from source ckpt)
        self.upscale = int(base.args.upscale)
        S = base.args.hr_slice_patch
        nf = base.args.n_feats
        self.drop_in = nn.Dropout2d(p)
        self.drop_mid = nn.Dropout2d(p)
        self.scale_head = _scale_head(nf, S)

    def features(self, x):
        """Verbatim copy of I3Net.forward up to the tail input."""
        b = self.base
        x = x.permute(0, 3, 1, 2).contiguous()
        x_head = b.head(x)
        res = x_head
        align = []
        res = b.alignment[0](res) + res
        align.append(res)
        for i, layer in enumerate(b.body):
            res = layer(res)
            if i in [3, 7]:
                res = b.alignment[i // 4 + 1](res) + res
                align.append(res)
        res = b.fuse_align(torch.cat(align, 1))
        res = res + x_head
        return res, x

    def tail(self, res, x):
        t = self.base.tail                     # [conv, ReLU, conv(mean)]
        h = t[1](t[0](self.drop_in(res)))
        h = self.drop_mid(h)
        mu = t[2](h)
        b = to_scale(self.scale_head(h))
        mu = mu.clone()
        mu[:, ::self.upscale] = x.to(mu.dtype)
        return mu.permute(0, 2, 3, 1).contiguous(), b.permute(0, 2, 3, 1).contiguous()

    def forward(self, x):
        res, xp = self.features(x)
        return self.tail(res, xp)


# ============================================================================ SAINT RFN
class ProbRFN(nn.Module):
    def __init__(self, rfn, p=P_DROP):
        super().__init__()
        self.rfn = rfn                         # upstream FUSE_RDN (weights from SAINT_rfn)
        G0 = rfn.SFENet2.out_channels
        self.drop_cat = nn.Dropout2d(p)
        self.drop_out = nn.Dropout2d(p)
        self.scale_head = _scale_head(G0, 1)

    def trunk(self, inp):
        """Deterministic part (no dropout before this point): SFENet + RDB stack."""
        r = self.rfn
        f1 = r.SFENet1(inp)
        x = r.SFENet2(f1)
        outs = []
        for i in range(r.D):
            x = r.RDBs[i](x)
            outs.append(x)
        return torch.cat(outs, 1), f1

    def forward(self, inp):
        return self.head(*self.trunk(inp), inp)

    def head(self, cat, f1, inp):
        r = self.rfn
        x = r.GFF(self.drop_cat(cat))
        x = x + f1
        x = self.drop_out(x)
        mu = r.UPNet(x) + inp.mean(1).unsqueeze(1)
        b = to_scale(self.scale_head(x))
        return mu, b


# ============================================================================ loss
def laplace_nll(y, mu, b, w=None):
    """Mean of log(2b) + |y-mu|/b over the weighted voxels (w = supervision mask)."""
    l = torch.log(2 * b) + (y - mu).abs() / b
    if w is None:
        return l.mean()
    return (l * w).sum() / w.sum().clamp_min(1)


# ============================================================================ option 3: geometry-aware scale head (SAINT)
class ProbRFNGeo(ProbRFN):
    """SAINT-PG: ProbRFN whose scale head also receives 3 constant-per-slice geometry maps of the TRUE acquisition
    (log(1+nearest distance mm), log(gap mm), position-in-gap 1-|2t-1|). Mean path and dropout are inherited
    unchanged (identical RNG consumption -> identical mu_t to SAINT-P for the same seed). The first conv of the
    scale head is widened by 3 input channels whose weights start at zero, so b starts identical to SAINT-P."""

    def __init__(self, prob_rfn: ProbRFN):
        nn.Module.__init__(self)
        self.rfn, self.drop_cat, self.drop_out = prob_rfn.rfn, prob_rfn.drop_cat, prob_rfn.drop_out
        old = prob_rfn.scale_head
        c0 = old[0]
        new0 = nn.Conv2d(c0.in_channels + 3, c0.out_channels, 3, padding=1)
        with torch.no_grad():
            new0.weight.zero_()
            new0.weight[:, :c0.in_channels] = c0.weight
            new0.bias.copy_(c0.bias)
        self.scale_head = nn.Sequential(new0, nn.ReLU(inplace=False), old[2])

    def head(self, cat, f1, inp, geo):
        r = self.rfn
        x = r.GFF(self.drop_cat(cat))
        x = x + f1
        x = self.drop_out(x)
        mu = r.UPNet(x) + inp.mean(1).unsqueeze(1)
        b = to_scale(self.scale_head(torch.cat([x, geo.to(x.dtype)], 1)))
        return mu, b


def geo_maps(ret, D, H, W, device):
    """(D, 3, H, W) geometry maps from the true acquired set (acquired slices get near=0)."""
    import numpy as _np
    from . import common as _C
    g = _C.gap_geometry(ret, D)
    v = _np.stack([_np.log1p(g["near_mm"]), _np.log(_np.maximum(g["gap_mm"], 1.0)), 1 - _np.abs(2 * g["t"] - 1)], 1)
    t = torch.from_numpy(v.astype(_np.float32)).to(device)
    return t[:, :, None, None].expand(D, 3, H, W)
