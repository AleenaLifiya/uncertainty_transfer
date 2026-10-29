"""IXI-tiny evaluation (FIT / CAL: uniform x4, x8; TEST: x4, x8 x {uniform, jittered x3, clustered x3}).
Per (subject, condition): linear, SAINT_det (IXI SAINT), SAINT_P (MC T=20, only trained dropout active, seed =
hash('isbi-ixi-mc', subject, condition)); acquired slices copied back. UQ voxels: up to 40,000 voxels of the BRAIN MASK
in the originally missing slices, seeded per (subject, condition), identical for every method. Output files use the
same keys as runs/eval/<ROLE>/cases (BraTS), so exp_uniform_arssr/scripts/analysis_cpu.py and ranking_v3.py run on them
with ISBI_EV pointing here. Recon metrics: brain-mask PSNR on missing slices.
    python eval_ixi.py --role FIT|CAL|TEST
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import ixi_isbi as X
from isbi import common as C
from isbi import evalcore as E
from isbi import recon as RC

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True, choices=["FIT", "CAL", "TEST"])
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.backends.cudnn.deterministic = True
T, N_SUB = 20, 40000
OUT = os.path.join(X.EXP, "runs", "eval", a.role)
for d in ("cases", "rows"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
sel = json.load(open(os.path.join(X.EXP, "runs", "train", "saint_prob", "selected.json")))
net = X.load_saint()
prfn = X.build_prob(net, sel["selected_path"])
subjects = X.roles()[a.role][: a.limit or None]
conds = [("uniform", R, 0) for R in (4, 8)]
if a.role == "TEST":
    conds += [(p, R, r) for R in (4, 8) for p in ("jittered", "clustered") for r in range(3)]
t0 = time.time()
n = 0
for s in subjects:
    gt, brain = X.load(s)
    for pattern, R, rep in conds:
        key = f"{s}__{C.cond_key(pattern, R, rep)}"
        if os.path.exists(os.path.join(OUT, "rows", key + ".csv")):
            continue
        ret = X.make_mask(pattern, R, s, rep)
        geo = X.geometry(ret)
        miss = np.where(~geo["acquired"])[0]
        region = np.zeros(gt.shape, bool)
        region[:, :, miss] = brain[:, :, miss]
        cand = np.flatnonzero(region)
        rng = np.random.RandomState(C._h("isbi-ixi-subset", s, C.cond_key(pattern, R, rep)))
        pick = np.sort(rng.choice(cand, min(N_SUB, cand.size), replace=False))
        sub = np.unravel_index(pick, gt.shape)
        out = dict(scan=s, patient=s, pattern=pattern, R=R, rep=rep, ret=np.asarray(ret), idx=pick.astype(np.int64),
                   y=gt[sub].astype(np.float32), n_region=int(cand.size),
                   near_mm=geo["near_mm"][sub[2]].astype(np.float32), gap_mm=geo["gap_mm"][sub[2]].astype(np.float32))
        lin = X.linear(gt, ret)
        vin, G = X.adapter_input(gt, ret, R)
        sag, cor = X.saint_marginals(net, vin, G, R)
        recons = dict(linear=lin, SAINT_det=X.saint_fuse(net, sag, cor))
        seed = C._h("isbi-ixi-mc", s, C.cond_key(pattern, R, rep))
        mu, b = RC.saint_mc(prfn, sag, cor, T, seed)
        mean, epi, alea = RC.moments(mu, b)
        recons["SAINT_P"] = mean.cpu().numpy()
        iy, ix, iz = (torch.from_numpy(np.asarray(v_)).to(mu.device) for v_ in sub)
        out["SAINT_P__mu_t"] = mu[:, iy, ix, iz].cpu().numpy().astype(np.float32)
        out["SAINT_P__b_t"] = b[:, iy, ix, iz].cpu().numpy().astype(np.float32)
        out["SAINT_P__epi"] = epi[iy, ix, iz].cpu().numpy().astype(np.float32)
        out["SAINT_P__alea"] = alea[iy, ix, iz].cpu().numpy().astype(np.float32)
        rows = []
        for m, x in recons.items():
            x = X.copy_back(np.clip(x, 0, 1), gt, ret)
            out[f"{m}__x"] = x[sub].astype(np.float32)
            out[f"{m}__feat"] = E.features(x, lin, gt, ret, geo, sub)
            rows.append(dict(patient=s, pattern=pattern, R=R, rep=rep, method=m, psnr_brain=X.psnr_brain(x, gt, brain, ret),
                             mae_brain=float(np.abs(x - gt)[region].mean())))
        np.savez(os.path.join(OUT, "cases", key + ".tmp.npz"), **out)
        os.replace(os.path.join(OUT, "cases", key + ".tmp.npz"), os.path.join(OUT, "cases", key + ".npz"))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "rows", key + ".csv"), index=False)
        n += 1
    print(f"[{a.role}] {s} done ({n} cases, {time.time() - t0:.0f}s)", flush=True)
print(f"[{a.role}] complete", flush=True)
