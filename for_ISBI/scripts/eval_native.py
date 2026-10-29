"""Step 3c: evaluate ArSSR-P / DCSRN-P on the saved FIT / CAL / TEST cases (same masks, voxel subsets, MC seeds).

    python scripts/eval_native.py --arch arssr|dcsrn --role FIT|CAL|TEST
No adapter: both models take the ORIGINALLY acquired slices at their true positions. Also records the extended
deterministic model's reconstruction (accuracy reference). Output runs/eval_native/<arch>/<ROLE>/{cases,rows}/.
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
from isbi import native as N
from isbi import recon as RC
from comparisons import protocol as P
from comparisons import arssr_adapter as AR
from comparisons_more import dcsrn_adapter as DC
from IR3C_LPIPS import brats_methods as BM

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True)
ap.add_argument("--role", required=True)
a = ap.parse_args()
torch.backends.cudnn.deterministic = True
T = json.load(open(os.path.join(C.ISBI, "configs", "protocol_frozen.json")))["T"]
TR = os.environ.get("ISBI_TRAIN_DIR", os.path.join(C.ISBI, "runs", "train"))
sel = lambda n: json.load(open(os.path.join(TR, n, "selected.json")))["selected_path"]  # noqa: E731
ld = lambda m, p: m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["state_dict"])  # noqa: E731
if a.arch == "arssr":
    det, inf, _ = BM._load_arssr_slowlr(); ld(det, sel("arssr_ext"))
    b2, _, _ = BM._load_arssr_slowlr(); pm = N.ProbArSSR(b2).cuda(); ld(pm, sel("arssr_prob")); pm.eval()
    run_det = lambda gt, ret, R: AR.reconstruct(det, gt, ret, C.D, "cuda", **inf)  # noqa: E731
    run_mc = lambda gt, ret, R, s: N.arssr_mc(pm, gt, ret, T, s, **inf)  # noqa: E731
else:
    nets, _, modes = BM._load_dcsrn(mixed=True)
    dets, pms = {}, {}
    for R in C.STRIDES:
        ld(nets[R], sel(f"dcsrn_ext_x{R}")); dets[R] = nets[R]
        n2, _, _ = BM._load_dcsrn(mixed=True)
        pms[R] = N.ProbDCSRN(n2[R]).cuda(); ld(pms[R], sel(f"dcsrn_prob_x{R}")); pms[R].eval()
    run_det = lambda gt, ret, R: DC.reconstruct(dets[R], gt, ret, R, C.D, "cuda", mode=modes[R])  # noqa: E731
    run_mc = lambda gt, ret, R, s: N.dcsrn_mc(pms[R], gt, ret, T, s, mode=modes[R])  # noqa: E731
OUT = os.path.join(C.ISBI, "runs", os.environ.get("ISBI_NATIVE_OUT", "eval_native"), a.arch, a.role)
for d in ("cases", "rows"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
files = sorted(glob.glob(os.path.join(C.ISBI, "runs", "eval", a.role, "cases", "*.npz")))[:int(os.environ.get("ISBI_LIMIT", "100000"))]
t0 = time.time()
for k, f in enumerate(files):
    key = os.path.basename(f)[:-4]
    if os.path.exists(os.path.join(OUT, "rows", key + ".csv")):
        continue
    c = np.load(f)
    p, pat, R, rep = c["patient"].item(), c["pattern"].item(), int(c["R"]), int(c["rep"])
    gt = P.prepare_one(C.scan_path(c["scan"].item()))["vol01"]
    ret = [int(r) for r in c["ret"]]
    ts = time.time()
    mu, b = run_mc(gt, ret, R, C._h("isbi-mc", p, C.cond_key(pat, R, rep)))
    torch.cuda.synchronize(); sec = time.time() - ts
    mean, epi, alea = RC.moments(mu, b)
    sub = np.unravel_index(c["idx"], gt.shape)
    st = [torch.from_numpy(np.asarray(v)).to(mu.device) for v in sub]
    x = RC.copy_back(np.clip(mean.cpu().numpy(), 0, 1), gt, ret)
    xd = RC.copy_back(np.clip(run_det(gt, ret, R), 0, 1), gt, ret)
    lin = RC.linear(gt, ret)
    out = {"NAT__mu_t": mu[:, st[0], st[1], st[2]].cpu().numpy(), "NAT__b_t": b[:, st[0], st[1], st[2]].cpu().numpy(),
           "NAT__epi": epi[st[0], st[1], st[2]].cpu().numpy(), "NAT__alea": alea[st[0], st[1], st[2]].cpu().numpy(),
           "NAT__x": x[sub].astype(np.float32), "NAT__feat": E.features(x, lin, gt, ret, C.gap_geometry(ret), sub)}
    np.savez(os.path.join(OUT, "cases", key + ".tmp.npz"), **out)
    os.replace(os.path.join(OUT, "cases", key + ".tmp.npz"), os.path.join(OUT, "cases", key + ".npz"))
    rows = []
    for m, v in ((f"{a.arch}_P", x), (f"{a.arch}_ext_det", xd)):
        ev = P.evaluate_volume(v, gt, ret, C.D, "missing")
        rows.append(dict(patient=p, pattern=pat, R=R, rep=rep, method=m, psnr=ev["psnr_masked"], ssim=ev["ssim"],
                         mae=ev["mae"], bg_haze=ev["bg_haze"], sec=sec if m.endswith("_P") else np.nan))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rows", key + ".csv"), index=False)
    if k % 25 == 0:
        print(f"[{a.arch} {a.role}] {k + 1}/{len(files)} {time.time() - t0:.0f}s P={rows[0]['psnr']:.2f} det={rows[1]['psnr']:.2f}", flush=True)
print(f"[{a.arch} {a.role}] complete", flush=True)
