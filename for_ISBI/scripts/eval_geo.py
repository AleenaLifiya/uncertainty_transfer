"""Option 3 evaluation: SAINT-PG on the saved CAL and TEST cases (same voxel subsets, same MC seeds).

    python scripts/eval_geo.py --role CAL|TEST
Writes runs/eval_geo/<ROLE>/<case>.npz with SAINT_PG__{mu_t,b_t,epi,alea} and a shared-mean check against
the stored SAINT_P__mu_t (identical mean path + seed -> identical up to GPU float noise).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from isbi import common as C
from isbi import recon as RC
from comparisons import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True)
ap.add_argument("--tag", default="saint_geo")
a = ap.parse_args()
torch.backends.cudnn.deterministic = True
proto = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))
gsel = json.load(open(os.path.join(C.ISBI, "runs", "train", a.tag, "selected.json")))
net, pg, chunk = RC.build_prob_saint_geo(proto["selected_checkpoints"]["saint_prob"], gsel["selected_path"])
OUT = os.path.join(C.ISBI, "runs", "eval_geo" if a.tag == "saint_geo" else f"eval_{a.tag}", a.role)
os.makedirs(OUT, exist_ok=True)
files = sorted(glob.glob(os.path.join(C.ISBI, "runs", "eval", a.role, "cases", "*.npz")))
t0, worst = time.time(), 0.0
for k, f in enumerate(files):
    o = os.path.join(OUT, os.path.basename(f))
    if os.path.exists(o):
        continue
    c = np.load(f)
    p, pat, R, rep = c["patient"].item(), c["pattern"].item(), int(c["R"]), int(c["rep"])
    gt = P.prepare_one(C.scan_path(c["scan"].item()))["vol01"]
    ret = [int(r) for r in c["ret"]]
    vin, G = RC.adapter_input(gt, ret, R)
    sag, cor = RC.saint_marginals(net, vin, G, R, chunk)
    mu, b = RC.saint_geo_mc(pg, sag, cor, ret, proto["T"], C._h("isbi-mc", p, C.cond_key(pat, R, rep)))
    mean, epi, alea = RC.moments(mu, b)
    sub = [torch.from_numpy(np.asarray(v)).to(mu.device) for v in np.unravel_index(c["idx"], gt.shape)]
    mt = mu[:, sub[0], sub[1], sub[2]].cpu().numpy()
    d = float(np.abs(mt - c["SAINT_P__mu_t"]).max())
    worst = max(worst, d)
    np.savez(o + ".tmp.npz", SAINT_PG__mu_t=mt, SAINT_PG__b_t=b[:, sub[0], sub[1], sub[2]].cpu().numpy(),
             SAINT_PG__epi=epi[sub[0], sub[1], sub[2]].cpu().numpy(), SAINT_PG__alea=alea[sub[0], sub[1], sub[2]].cpu().numpy(),
             mean_check_maxabs=d)
    os.replace(o + ".tmp.npz", o)
    if k % 20 == 0:
        print(f"[{a.role}] {k + 1}/{len(files)} {time.time() - t0:.0f}s  max|mu_t(PG)-mu_t(P)| so far {worst:.2e}", flush=True)
print(f"[{a.role}] complete; max |mu_t(PG) - mu_t(P)| = {worst:.3e}", flush=True)
