"""Preflight gates (GPU). Run: $PY tests/test_preflight.py  -> runs/preflight/preflight.json"""
import json
import os
import sys
import time

import numpy as np
import torch

from isbi import common as C
from isbi import recon as RC
from isbi import models as M

OUT = os.path.join(C.ISBI, "runs", "preflight")
os.makedirs(OUT, exist_ok=True)
res = {}
torch.backends.cudnn.deterministic = True

from comparisons import protocol as P                     # noqa: E402
from comparisons import i3net_adapter as I3                # noqa: E402

roles = C.load_roles()
scan = roles["scan_of"][roles["SEL"][0]]
gt = P.prepare_one(C.scan_path(scan))["vol01"]

# 1. ProbI3Net dropout-off == upstream I3Net inference (window-clamped path), both strides
for R in (4, 8):
    pm, _ = RC.build_prob_i3net(R)
    G = C.uniform_mask(R)
    ref = I3.reconstruct(pm.base, gt, G, R, C.D, RC.DEV, 4)
    mu, b = RC.i3net_mc(pm, gt, G, R, 0, 0, clamp_windows=True)
    got = np.clip(mu[0].cpu().numpy(), 0, 1)
    res[f"i3net_x{R}_dropout_off_maxabs"] = float(np.abs(got - ref).max())
    res[f"i3net_x{R}_init_scale_b0"] = float(b[0].mean())
    # MC passes differ only through trained dropout; dropout modules are the only train-mode ones
    M.set_mc(pm, True)
    res[f"i3net_x{R}_train_mode_modules"] = sorted({type(m).__name__ for m in pm.modules() if m.training})
    M.set_mc(pm, False)
    del pm

# 2. ProbRFN dropout-off == upstream SAINT stage-2 output; MSR in SAINT_rfn == cached SAINT_msr
net, prfn, chunk, sel = RC.build_prob_saint()
for R in (4, 8):
    G = C.uniform_mask(R)
    from comparisons_more import saint_adapter as SA
    ref = SA.reconstruct(net, gt, G, R, C.D, RC.DEV, stage=2, chunk=chunk)
    sag, cor = RC.saint_marginals(net, gt, G, R, chunk)
    mu, b = RC.saint_mc(prfn, sag, cor, 0, 0)
    got = np.clip(mu[0].cpu().numpy(), 0, 1)
    res[f"saint_x{R}_dropout_off_maxabs"] = float(np.abs(got - ref).max())
ck_msr = torch.load(os.path.join(C.ROOT, "IR3C_more_comparisons/checkpoints/SAINT/SAINT_msr/SAINT_msr_step011259.pt"),
                    map_location="cpu", weights_only=False)["state_dict"]
sd = net.state_dict()
res["saint_rfn_ckpt_msr_equals_msr_step11259"] = all(
    torch.equal(sd[k].cpu(), v) for k, v in ck_msr.items() if not k.startswith("RFN."))

# 3. leakage: adapter + network inputs depend only on acquired slices
ret = C.make_mask("clustered", 4, roles["SEL"][0], 0)
probe = P.leakage_probe(gt, ret, C.D, seed=1)
a, G = RC.adapter_input(gt, ret, 4)
b_, _ = RC.adapter_input(probe, ret, 4)
res["adapter_leak_maxabs_on_G"] = float(np.abs(a[:, :, G] - b_[:, :, G]).max())
res["linear_leak"] = float(np.abs(RC.linear(gt, ret) - RC.linear(probe, ret))[:, :, [z for z in range(C.D) if z not in ret]].max())
res["bicubic_leak"] = float(np.abs(RC.bicubic(gt, ret) - RC.bicubic(probe, ret))[:, :, [z for z in range(C.D) if z not in ret]].max())

# 4. mixture quantile vs sampling
g = torch.Generator().manual_seed(0)
mu = torch.randn(20, 2000, generator=g) * 0.05
bb = torch.rand(20, 2000, generator=g) * 0.05 + 0.005
q = RC.mixture_quantile(mu, bb, 0.95)
comp = torch.randint(0, 20, (20000, 2000), generator=g)
lap = torch.distributions.Laplace(mu.gather(0, comp), bb.gather(0, comp)).sample()
res["mixture_q95_empirical_cdf"] = float((lap <= q[None]).float().mean())

json.dump(res, open(os.path.join(OUT, "preflight.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
