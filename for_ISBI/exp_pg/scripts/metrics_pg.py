"""Metrics for the learned + geometry ('PG') arm, identical definitions to exp_seeds/scripts/metrics_seed.py ('learned' arm):
lambda_0.9 calibrated on CAL uniform (mean per-patient coverage = 0.9, per R); coverage / width / IS90 / AUSE-MAE / UCE /
ENCE on TEST, plus gap-bin coverage. The reconstruction (mean) is the seed's P mean (shared, verified by mean_check).
Writes exp_seeds/results/<dataset>_s<seed>_<SAINT_PG|ArSSR_PG>/{case,gapbin_case,summary}.csv with arm = 'learned_geo'.
    python metrics_pg.py --arch saint|arssr --dataset brats|ixi --seed 42|43|44
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import uq as U

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
torch.set_num_threads(a.threads)
I = C.ISBI
ALPHA = 0.1
BASE = os.path.join(I, "runs", "eval") if a.dataset == "brats" else os.path.join(I, "exp_ixi", "runs", "eval")
ES = os.path.join(I, "exp_seeds", "runs")
PGD = os.path.join(I, "exp_pg", "runs", f"{a.dataset}_{a.arch}_s{a.seed}", "eval")
NAME = {"saint": "SAINT_PG", "arssr": "ArSSR_PG"}[a.arch]
OUT = os.path.join(I, "exp_seeds", "results", f"{a.dataset}_s{a.seed}_{NAME}")
os.makedirs(OUT, exist_ok=True)
BINS = ([(2, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 16), (17, 22)] if a.dataset == "brats" else
        [(4, 5), (6, 7), (8, 9), (10, 14), (15, 20), (21, 30), (31, 45)])


def p_case(role, key):
    if a.arch == "saint":
        d = os.path.join(BASE, role, "cases") if a.seed == 42 else os.path.join(ES, f"{a.dataset}_s{a.seed}", "eval", role, "cases")
        return np.load(os.path.join(d, key))["SAINT_P__x"]
    d = ({"brats": os.path.join(I, "exp_uniform_arssr", "runs", "eval_native", "arssr"),
          "ixi": os.path.join(I, "exp_ixi", "runs", "eval_native", "arssr")}[a.dataset] if a.seed == 42 else
         os.path.join(ES, f"{a.dataset}_s{a.seed}", "eval_native", "arssr"))
    return np.load(os.path.join(d, role, "cases", key))["NAT__x"]


def load(role, key):
    z = np.load(os.path.join(BASE, role, "cases", key))
    g = np.load(os.path.join(PGD, role, key))
    return dict(patient=z["patient"].item(), pattern=z["pattern"].item(), R=int(z["R"]), rep=int(z["rep"]), y=z["y"],
                gap=z["gap_mm"], x=p_case(role, key), mu=g["PG__mu_t"], b=g["PG__b_t"], mc=float(g["mean_check_maxabs"]))


def keys(role):
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(PGD, role, "*.npz")))


def calib(vals, c=0.9):
    srt = [np.sort(v) for v in vals]
    f = lambda k: np.mean([np.searchsorted(v, k, side="right") / len(v) for v in srt])  # noqa: E731
    lo, hi = np.log(1e-4), np.log(1e4)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(np.exp(mid)) < c else (lo, mid)
    return float(np.exp(0.5 * (lo + hi)))


def lam_thr(c):
    y, mu, b = (torch.from_numpy(np.asarray(v, np.float64)) for v in (c["y"], c["mu"], c["b"]))

    def cov(lam):
        z = (y[None] - mu) / (lam * b)
        F = torch.where(z < 0, 0.5 * torch.exp(z.clamp(max=0)), 1 - 0.5 * torch.exp(-z.clamp(min=0))).mean(0)
        return (F >= ALPHA / 2) & (F <= 1 - ALPHA / 2)
    lo = torch.full(y.shape, np.log(1e-3), dtype=torch.float64)
    hi = torch.full(y.shape, np.log(1e3), dtype=torch.float64)
    c_lo, c_hi = cov(torch.exp(lo)), cov(torch.exp(hi))
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        cm = cov(torch.exp(mid))
        hi, lo = torch.where(cm, mid, hi), torch.where(cm, lo, mid)
    t = torch.exp(hi).numpy()
    t[c_lo.numpy()] = 0
    t[~c_hi.numpy()] = np.inf
    return t


def uce(e2, s2, M=20):
    o = np.argsort(s2, kind="stable")
    return float(sum(len(ix) / len(o) * abs(e2[ix].mean() - s2[ix].mean()) for ix in np.array_split(o, M)))


def ence(e2, s2, M=20):
    o = np.argsort(s2, kind="stable")
    return float(np.mean([abs(np.sqrt(s2[ix].mean()) - np.sqrt(e2[ix].mean())) / np.sqrt(s2[ix].mean())
                          for ix in np.array_split(o, M)]))


t0 = time.time()
cal = [load("CAL", k) for k in keys("CAL")]
LAM = {R: calib([lam_thr(c) for c in cal if c["R"] == R]) for R in (4, 8)}
worst = max(c["mc"] for c in cal)
del cal
rows, brow = [], []
for k in keys("TEST"):
    c = load("TEST", k)
    worst = max(worst, c["mc"])
    R, y, x = c["R"], c["y"], c["x"]
    err = y - x
    lam = LAM[R]
    mu, b = torch.from_numpy(c["mu"]), torch.from_numpy(c["b"])
    lo = np.clip(U.mix_quantile(mu, b, ALPHA / 2, lam).numpy(), 0, 1)
    hi = np.clip(U.mix_quantile(mu, b, 1 - ALPHA / 2, lam).numpy(), 0, 1)
    s2 = (c["mu"].var(0) + (2 * (lam * c["b"]) ** 2).mean(0)).astype(np.float64)
    w = hi - lo
    cv = (y >= lo) & (y <= hi)
    isc = w + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    base = dict(dataset=a.dataset, seed=a.seed, method=NAME, patient=c["patient"], pattern=c["pattern"], R=R, rep=c["rep"])
    rows.append(dict(base, arm="learned_geo", cov90=cv.mean(), width90=w.mean(), is90=isc.mean(), mae=np.abs(err).mean(),
                     ause_norm=U.ause(err, s2)[0] / np.abs(err).mean(), uce_e3=1e3 * uce(err ** 2, s2), ence=ence(err ** 2, s2)))
    for b0, b1 in BINS:
        sel = (c["gap"] >= b0) & (c["gap"] <= b1)
        if sel.sum() >= 100:
            brow.append(dict(base, arm="learned_geo", gap_bin=f"{b0}-{b1}", cov90=cv[sel].mean()))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case.csv"), index=False)
pd.DataFrame(brow).to_csv(os.path.join(OUT, "gapbin_case.csv"), index=False)
MET = ["cov90", "width90", "is90", "mae", "ause_norm", "uce_e3", "ence"]
pp = d.groupby(["arm", "R", "pattern", "patient"], as_index=False)[MET].mean()
rng = np.random.RandomState(0)
out = []
for (arm, R, pat), g in pp.groupby(["arm", "R", "pattern"]):
    r = dict(dataset=a.dataset, seed=a.seed, method=NAME, arm=arm, R=R, pattern=pat, n=len(g))
    for m in MET:
        v = g[m].values
        bs = v[rng.randint(0, len(v), (5000, len(v)))].mean(1)
        r[m], r[m + "_lo"], r[m + "_hi"] = v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    out.append(r)
pd.DataFrame(out).to_csv(os.path.join(OUT, "summary.csv"), index=False)
print(f"[{a.dataset} {a.arch} s{a.seed}] lambda {LAM}; max mean-check {worst:.2e}; done ({time.time() - t0:.0f}s)", flush=True)
