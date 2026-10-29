"""SAINT-P seed evaluation on BraTS (exp_seeds, 2026-10-24). For every saved case of runs/eval/<ROLE>/cases (same patients,
masks, UQ voxel subset), recompute SAINT-P with the seed's checkpoint exactly as isbi.evalcore.run_case does (adapter ->
frozen MSR marginals -> ProbRFN MC, T = 20, MC seed = hash('isbi-mc', patient, condition), copy-back, features with the
linear interpolant). Writes the base keys + SAINT_P__* to exp_seeds/runs/brats_s<seed>/eval/<ROLE>/cases.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import evalcore as E
from isbi import recon as RC
from comparisons import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--role", required=True)
a = ap.parse_args()
torch.backends.cudnn.deterministic = True
T = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))["T"]
SD = os.path.join(C.ISBI, "exp_seeds", "runs", f"brats_s{a.seed}")
sel = json.load(open(os.path.join(SD, "train", "saint_prob", "selected.json")))["selected_path"]
net, prfn, chunk, _ = RC.build_prob_saint(sel)
OUT = os.path.join(SD, "eval", a.role)
for d in ("cases", "rows"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
BASE = ["scan", "patient", "pattern", "R", "rep", "ret", "idx", "y", "n_region", "near_mm", "gap_mm"]
files = sorted(glob.glob(os.path.join(C.ISBI, "runs", "eval", a.role, "cases", "*.npz")))[: int(os.environ.get("ISBI_LIMIT", "100000"))]
t0, gtc = time.time(), {}
for k, f in enumerate(files):
    key = os.path.basename(f)[:-4]
    if os.path.exists(os.path.join(OUT, "rows", key + ".csv")):
        continue
    c = np.load(f)
    out = {b: c[b] for b in BASE}
    p, pat, R, rep = c["patient"].item(), c["pattern"].item(), int(c["R"]), int(c["rep"])
    if p not in gtc:
        gtc = {p: P.prepare_one(C.scan_path(c["scan"].item()))["vol01"]}
    gt = gtc[p]
    ret = [int(r) for r in c["ret"]]
    geo = C.gap_geometry(ret)
    sub = np.unravel_index(c["idx"], gt.shape)
    lin = RC.linear(gt, ret)
    vin, G = RC.adapter_input(gt, ret, R)
    sag, cor = RC.saint_marginals(net, vin, G, R, chunk)
    mu, b = RC.saint_mc(prfn, sag, cor, T, C._h("isbi-mc", p, C.cond_key(pat, R, rep)))
    mean, epi, alea = RC.moments(mu, b)
    iy, ix, iz = (torch.from_numpy(np.asarray(v)).to(mu.device) for v in sub)
    out["SAINT_P__mu_t"] = mu[:, iy, ix, iz].cpu().numpy().astype(np.float32)
    out["SAINT_P__b_t"] = b[:, iy, ix, iz].cpu().numpy().astype(np.float32)
    out["SAINT_P__epi"] = epi[iy, ix, iz].cpu().numpy().astype(np.float32)
    out["SAINT_P__alea"] = alea[iy, ix, iz].cpu().numpy().astype(np.float32)
    x = RC.copy_back(np.clip(mean.cpu().numpy(), 0, 1), gt, ret)
    out["SAINT_P__x"] = x[sub].astype(np.float32)
    out["SAINT_P__feat"] = E.features(x, lin, gt, ret, geo, sub)
    np.savez(os.path.join(OUT, "cases", key + ".tmp.npz"), **out)
    os.replace(os.path.join(OUT, "cases", key + ".tmp.npz"), os.path.join(OUT, "cases", key + ".npz"))
    ev = P.evaluate_volume(x, gt, ret, C.D, region="missing")
    pd.DataFrame([dict(patient=p, pattern=pat, R=R, rep=rep, method="SAINT_P", psnr=ev["psnr_masked"])]).to_csv(
        os.path.join(OUT, "rows", key + ".csv"), index=False)
    if k % 50 == 0:
        print(f"[s{a.seed} {a.role}] {k + 1}/{len(files)} {time.time() - t0:.0f}s", flush=True)
print(f"[s{a.seed} {a.role}] complete", flush=True)
