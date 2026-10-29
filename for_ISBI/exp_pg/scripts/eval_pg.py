"""Evaluate the PG models on the saved CAL/TEST cases (same voxel subsets, masks and MC seeds as P) -- exp_pg 2026-10-26.
For each case: PG__{mu_t,b_t,epi,alea} on the UQ voxels + mean_check = max |mu_t(PG) - mu_t(P)| against that seed's saved P
outputs (mean path, dropout and RNG consumption are shared, so this should be ~0).
    python eval_pg.py --arch saint|arssr --dataset brats|ixi --role CAL|TEST [--seeds 42,43,44]
Writes exp_pg/runs/<dataset>_<arch>_s<seed>/eval/<ROLE>/<case>.npz
"""

from project_paths import root_path
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, root_path('for_ISBI/exp_ixi/scripts'))
import pg as PG  # noqa: E402
from isbi import common as C  # noqa: E402
from isbi import models as M  # noqa: E402
from isbi import native as N  # noqa: E402
from isbi import recon as RC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["saint", "arssr"])
ap.add_argument("--dataset", required=True, choices=["brats", "ixi"])
ap.add_argument("--role", required=True, choices=["CAL", "TEST"])
ap.add_argument("--seeds", default="42,43,44")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
torch.backends.cudnn.deterministic = True
I = C.ISBI
T = 20
SEEDS = [int(s) for s in a.seeds.split(",")]
SP = 1.0 if a.dataset == "brats" else 2.18
BASE = os.path.join(I, "runs", "eval") if a.dataset == "brats" else os.path.join(I, "exp_ixi", "runs", "eval")
ES = os.path.join(I, "exp_seeds", "runs")


def p_dir(seed):
    """Where that seed's P outputs (mu_t) live."""
    if a.arch == "saint":
        if seed == 42:
            return os.path.join(BASE, a.role, "cases"), "SAINT_P"
        return os.path.join(ES, f"{a.dataset}_s{seed}", "eval", a.role, "cases"), "SAINT_P"
    if seed == 42:
        d = {"brats": os.path.join(I, "exp_uniform_arssr", "runs", "eval_native", "arssr"),
             "ixi": os.path.join(I, "exp_ixi", "runs", "eval_native", "arssr")}[a.dataset]
    else:
        d = os.path.join(ES, f"{a.dataset}_s{seed}", "eval_native", "arssr")
    return os.path.join(d, a.role, "cases"), "NAT"


def sel(seed):
    return json.load(open(os.path.join(I, "exp_pg", "runs", f"{a.dataset}_{a.arch}_s{seed}", "train", "selected.json")))


# ------------------------------------------------------------------ models
if a.dataset == "ixi":
    import ixi_isbi as X
if a.arch == "saint":
    if a.dataset == "brats":
        net, _, chunk, _ = RC.build_prob_saint(None)
    else:
        net = X.load_saint()
    PGM = {}
    for s in SEEDS:
        j = sel(s)
        prfn = M.ProbRFN(__import__("copy").deepcopy(net.RFN)).to("cuda") if a.dataset == "brats" else X.build_prob(net)
        m = M.ProbRFNGeo(prfn).to("cuda")
        m.load_state_dict(torch.load(j["selected_path"], map_location="cpu", weights_only=False)["state_dict"])
        m.eval()
        PGM[s] = m
else:
    from IR3C_LPIPS import brats_methods as BM
    if a.dataset == "brats":
        _, INF, _ = BM._load_arssr_slowlr()
    else:
        from ir3c_v2 import baselines_v2 as BV
        INF = BV.ARSSR_INFERENCE["IXI"]
    PGM = {}
    for s in SEEDS:
        b0, _, _ = BM._load_arssr_slowlr()
        m = PG.ArSSRGeo(N.ProbArSSR(b0)).to("cuda")
        m.load_state_dict(torch.load(sel(s)["selected_path"], map_location="cpu", weights_only=False)["state_dict"])
        m.eval()
        PGM[s] = m
OUT = {s: os.path.join(I, "exp_pg", "runs", f"{a.dataset}_{a.arch}_s{s}", "eval", a.role) for s in SEEDS}
for d in OUT.values():
    os.makedirs(d, exist_ok=True)
files = sorted(glob.glob(os.path.join(BASE, a.role, "cases", "*.npz")))[: a.limit or None]
t0, worst = time.time(), {s: 0.0 for s in SEEDS}
gtc = {}
for k, f in enumerate(files):
    key = os.path.basename(f)
    todo = [s for s in SEEDS if not os.path.exists(os.path.join(OUT[s], key))]
    if not todo:
        continue
    c = np.load(f)
    p, pat, R, rep = c["patient"].item(), c["pattern"].item(), int(c["R"]), int(c["rep"])
    if p not in gtc:
        if a.dataset == "brats":
            from comparisons import protocol as P
            gtc = {p: P.prepare_one(C.scan_path(c["scan"].item()))["vol01"]}
        else:
            gtc = {p: X.load(p)[0]}
    gt = gtc[p]
    ret = [int(r) for r in c["ret"]]
    D_ = gt.shape[2]
    seed_mc = C._h("isbi-mc" if a.dataset == "brats" else "isbi-ixi-mc", p, C.cond_key(pat, R, rep))
    sub = [torch.from_numpy(np.asarray(v)).to("cuda") for v in np.unravel_index(c["idx"], gt.shape)]
    if a.arch == "saint":
        if a.dataset == "brats":
            vin, G = RC.adapter_input(gt, ret, R)
            sag, cor = RC.saint_marginals(net, vin, G, R, chunk)
        else:
            vin, G = X.adapter_input(gt, ret, R)
            sag, cor = X.saint_marginals(net, vin, G, R)
    for s in todo:
        if a.arch == "saint":
            mu, b = PG.saint_geo_mc(PGM[s], sag, cor, ret, T, seed_mc, SP)
        else:
            mu, b = PG.arssr_geo_mc(PGM[s], gt, ret, T, seed_mc, D_, SP, **INF)
        mean, epi, alea = RC.moments(mu, b)
        mt = mu[:, sub[0], sub[1], sub[2]].cpu().numpy()
        pd_, pk = p_dir(s)
        ref = np.load(os.path.join(pd_, key))[f"{pk}__mu_t"]
        dd = float(np.abs(mt - ref).max())
        worst[s] = max(worst[s], dd)
        o = os.path.join(OUT[s], key)
        np.savez(o + ".tmp.npz", PG__mu_t=mt, PG__b_t=b[:, sub[0], sub[1], sub[2]].cpu().numpy(),
                 PG__epi=epi[sub[0], sub[1], sub[2]].cpu().numpy(), PG__alea=alea[sub[0], sub[1], sub[2]].cpu().numpy(),
                 mean_check_maxabs=dd)
        os.replace(o + ".tmp.npz", o)
    if k % 25 == 0:
        print(f"[{a.dataset} {a.arch} {a.role}] {k + 1}/{len(files)} {time.time() - t0:.0f}s mean-check {worst}", flush=True)
print(f"[{a.dataset} {a.arch} {a.role}] complete; max |mu_t(PG)-mu_t(P)| per seed {worst}", flush=True)
