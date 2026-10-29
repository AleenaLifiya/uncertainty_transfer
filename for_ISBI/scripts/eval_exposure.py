"""Evaluate an exposure-trained model on the saved FIT / CAL / TEST cases (same masks, voxel subsets, MC seeds).

    python scripts/eval_exposure.py --arch saint|i3net --variant a|b --role FIT|CAL|TEST
Writes runs/eval_exp/<ARCH>_exp<v>/<ROLE>/cases/<case>.npz (fields named EXP__*) and rows/<case>.csv (recon metrics).
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
ap.add_argument("--arch", required=True)
ap.add_argument("--variant", required=True)
ap.add_argument("--role", required=True)
a = ap.parse_args()
torch.backends.cudnn.deterministic = True
proto = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))
T = proto["T"]
tag = f"{a.arch}_exp{a.variant}"
sel = lambda n: json.load(open(os.path.join(C.ISBI, "runs", "train", n, "selected.json")))["selected_path"]  # noqa: E731
if a.arch == "saint":
    net, prfn, chunk, _ = RC.build_prob_saint(sel(tag))
else:
    i3 = {R: RC.build_prob_i3net(R, sel(f"{tag}_x{R}"))[0] for R in C.STRIDES}
OUT = os.path.join(C.ISBI, "runs", "eval_exp", tag, a.role)
for d in ("cases", "rows"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
files = sorted(glob.glob(os.path.join(C.ISBI, "runs", "eval", a.role, "cases", "*.npz")))
t0 = time.time()
for k, f in enumerate(files):
    key = os.path.basename(f)[:-4]
    if os.path.exists(os.path.join(OUT, "rows", key + ".csv")):
        continue
    c = np.load(f)
    p, pat, R, rep = c["patient"].item(), c["pattern"].item(), int(c["R"]), int(c["rep"])
    gt = P.prepare_one(C.scan_path(c["scan"].item()))["vol01"]
    ret = [int(r) for r in c["ret"]]
    seed = C._h("isbi-mc", p, C.cond_key(pat, R, rep))
    vin, G = RC.adapter_input(gt, ret, R)
    ts = time.time()
    if a.arch == "saint":
        sag, cor = RC.saint_marginals(net, vin, G, R, chunk)
        mu, b = RC.saint_mc(prfn, sag, cor, T, seed)
    else:
        mu, b = RC.i3net_mc(i3[R], vin, G, R, T, seed)
    torch.cuda.synchronize()
    sec = time.time() - ts
    mean, epi, alea = RC.moments(mu, b)
    sub = np.unravel_index(c["idx"], gt.shape)
    st = [torch.from_numpy(np.asarray(v)).to(mu.device) for v in sub]
    x = RC.copy_back(np.clip(mean.cpu().numpy(), 0, 1), gt, ret)
    lin = RC.linear(gt, ret)
    out = {"EXP__mu_t": mu[:, st[0], st[1], st[2]].cpu().numpy(), "EXP__b_t": b[:, st[0], st[1], st[2]].cpu().numpy(),
           "EXP__epi": epi[st[0], st[1], st[2]].cpu().numpy(), "EXP__alea": alea[st[0], st[1], st[2]].cpu().numpy(),
           "EXP__x": x[sub].astype(np.float32), "EXP__feat": E.features(x, lin, gt, ret, C.gap_geometry(ret), sub)}
    np.savez(os.path.join(OUT, "cases", key + ".tmp.npz"), **out)
    os.replace(os.path.join(OUT, "cases", key + ".tmp.npz"), os.path.join(OUT, "cases", key + ".npz"))
    ev = P.evaluate_volume(x, gt, ret, C.D, region="missing")
    pd.DataFrame([dict(patient=p, pattern=pat, R=R, rep=rep, method=tag, psnr=ev["psnr_masked"], ssim=ev["ssim"],
                       mae=ev["mae"], bg_haze=ev["bg_haze"], sec=sec)]).to_csv(os.path.join(OUT, "rows", key + ".csv"), index=False)
    if k % 25 == 0:
        print(f"[{tag} {a.role}] {k + 1}/{len(files)} {time.time() - t0:.0f}s psnr={ev['psnr_masked']:.2f}", flush=True)
print(f"[{tag} {a.role}] complete", flush=True)
