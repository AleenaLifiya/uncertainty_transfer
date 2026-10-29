"""Uncertainty metrics for one (dataset, seed, reconstruction) from SAVED evaluation outputs (CPU only; exp_seeds 2026-10-24).

Arms     learned (probabilistic models; lambda_0.9 calibrated on CAL uniform), post-hoc WITHOUT geometry, post-hoc WITH
         geometry (main-effects Laplace GLMs fitted on FIT uniform, kappa_0.9 on CAL uniform) -- identical to
         exp_uniform_arssr/scripts/analysis_cpu.py ('learned', 'ph_nogeo', 'ph_geo').
Variance each arm's CALIBRATED predictive distribution:
           learned   sigma^2 = Var_t(mu_t) + (1/T) sum_t 2 (lambda b_t)^2         (law of total variance, scaled mixture)
           post-hoc  sigma^2 = 2 (kappa b_hat / ln 10)^2                           (Laplace whose central 90 % interval is
                                                                                    exactly x +- kappa b_hat)
Metrics  per TEST case, on the fixed UQ voxel subset:
           cov90, width90 (clipped to [0,1]), IS90 (Gneiting & Raftery 2007),
           AUSE / MAE : area between the sparsification curve (remove voxels by decreasing sigma, MAE of the rest) and the
                        oracle curve, removal grid 0..0.98 (uq.ause, as in the study), normalised by the case MAE;
           UCE        : sum_m |B_m|/N |mean(e^2)_m - mean(sigma^2)_m|, M = 20 equal-mass bins of sigma^2 (Laves et al. 2020,
                        equal-width bins replaced by equal-mass bins because sigma^2 is heavy-tailed); reported x 1e3;
           ENCE       : (1/M) sum_m |RMV_m - RMSE_m| / RMV_m, M = 20 equal-mass bins of sigma (Levi et al. 2022);
         plus cov90 by bracketing-gap bin. Patient = unit (realisations averaged), mean over patients with patient-bootstrap
         95 % CI.
Directories (seed 42 = the study's own runs; 43/44 = exp_seeds):
  BraTS SAINT_P : runs/eval (42) | exp_seeds/runs/brats_s<seed>/eval
  BraTS ArSSR   : base runs/eval + NAT exp_uniform_arssr/runs/eval_native/arssr (42) | exp_seeds/runs/brats_s<seed>/eval_native/arssr
  IXI   SAINT_P : exp_ixi/runs/eval (42) | exp_seeds/runs/ixi_s<seed>/eval
  IXI   ArSSR   : base exp_ixi/runs/eval + NAT exp_ixi/runs/eval_native/arssr (42) | exp_seeds/runs/ixi_s<seed>/eval_native/arssr
  linear        : base dir of the dataset (seed-independent)
    python metrics_seed.py --dataset brats|ixi --seed 42|43|44 --method SAINT_P|ArSSR|linear
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
ap.add_argument("--dataset", required=True, choices=["brats", "ixi"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--method", required=True, choices=["SAINT_P", "ArSSR", "linear"])
ap.add_argument("--threads", type=int, default=6)
a = ap.parse_args()
torch.set_num_threads(a.threads)
I = C.ISBI
S = os.path.join(I, "exp_seeds", "runs")
BASE = {"brats": os.path.join(I, "runs", "eval"), "ixi": os.path.join(I, "exp_ixi", "runs", "eval")}[a.dataset]
NAT = None
if a.method == "SAINT_P":
    EV = BASE if a.seed == 42 else os.path.join(S, f"{a.dataset}_s{a.seed}", "eval")
elif a.method == "ArSSR":
    EV = BASE
    NAT = ({"brats": os.path.join(I, "exp_uniform_arssr", "runs", "eval_native", "arssr"),
            "ixi": os.path.join(I, "exp_ixi", "runs", "eval_native", "arssr")}[a.dataset] if a.seed == 42 else
           os.path.join(S, f"{a.dataset}_s{a.seed}", "eval_native", "arssr"))
else:
    EV = BASE
OUT = os.path.join(I, "exp_seeds", "results", f"{a.dataset}_s{a.seed}_{a.method}")
os.makedirs(OUT, exist_ok=True)
EPS, ALPHA, LN10 = U.EPS, 0.1, float(np.log(10.0))
BINS = ([(2, 3), (4, 4), (5, 6), (7, 8), (9, 12), (13, 16), (17, 22)] if a.dataset == "brats" else
        [(4, 5), (6, 7), (8, 9), (10, 14), (15, 20), (21, 30), (31, 45)])
PROB = a.method != "linear"


def keys(role):
    return sorted(os.path.basename(f) for f in glob.glob(os.path.join(EV, role, "cases", "*.npz")))


def load(role, k):
    z = np.load(os.path.join(EV, role, "cases", k))
    c = dict(patient=z["patient"].item(), pattern=z["pattern"].item(), R=int(z["R"]), rep=int(z["rep"]), y=z["y"],
             gap=z["gap_mm"], near=z["near_mm"])
    if a.method == "ArSSR":
        n = np.load(os.path.join(NAT, role, "cases", k))
        c.update(x=n["NAT__x"], F=n["NAT__feat"], mu=n["NAT__mu_t"], b=n["NAT__b_t"])
    elif a.method == "SAINT_P":
        c.update(x=z["SAINT_P__x"], F=z["SAINT_P__feat"], mu=z["SAINT_P__mu_t"], b=z["SAINT_P__b_t"])
    else:
        c.update(x=z["linear__x"], F=z["linear__feat"])
    return c


def phi(F, geo):
    cols = [F[:, 0], np.log(F[:, 1] + EPS), np.log(F[:, 2] + EPS), np.log(F[:, 3] + EPS)]
    if geo:
        cols += [np.log(F[:, 4]), np.log(F[:, 5]), F[:, 6]]
    return np.stack(cols, 1).astype(np.float64)


def fit_glm(X, e):
    m, s = X.mean(0), X.std(0) + 1e-8
    Xt = torch.from_numpy((X - m) / s)
    aa = torch.from_numpy(np.abs(e).astype(np.float64))
    th = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    c0 = torch.tensor([float(np.log(np.mean(np.abs(e)) + 1e-6))], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th, c0], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lb = c0 + Xt @ th
        loss = (lb + aa * torch.exp(-lb)).mean() + 1e-4 * (th ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    th_, c_ = th.detach().numpy(), float(c0.detach())
    return lambda X_: np.exp(c_ + ((X_ - m) / s) @ th_)


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
    v = [abs(np.sqrt(s2[ix].mean()) - np.sqrt(e2[ix].mean())) / np.sqrt(s2[ix].mean()) for ix in np.array_split(o, M)]
    return float(np.mean(v))


t0 = time.time()
fit = [load("FIT", k) for k in keys("FIT")]
F = np.concatenate([c["F"] for c in fit])
e = np.concatenate([c["y"] - c["x"] for c in fit])
G = {kind: (fit_glm(phi(F, geo), e), geo) for kind, geo in (("ph_nogeo", False), ("ph_geo", True))}
del fit
cal = [load("CAL", k) for k in keys("CAL")]
K, LAM = {}, {}
for R in (4, 8):
    cr = [c for c in cal if c["R"] == R]
    for kind, (g, geo) in G.items():
        K[(kind, R)] = calib([np.abs(c["y"] - c["x"]) / g(phi(c["F"], geo)) for c in cr])
    if PROB:
        LAM[R] = calib([lam_thr(c) for c in cr])
del cal
print(f"[{a.dataset} s{a.seed} {a.method}] kappa {K} lambda {LAM} ({time.time() - t0:.0f}s)", flush=True)

rows, brow = [], []
for k in keys("TEST"):
    c = load("TEST", k)
    R, y, x = c["R"], c["y"], c["x"]
    err = y - x
    e2, ae = err ** 2, np.abs(err)
    base = dict(dataset=a.dataset, seed=a.seed, method=a.method, patient=c["patient"], pattern=c["pattern"], R=R, rep=c["rep"])
    arms = {}
    for kind, (g, geo) in G.items():
        h = K[(kind, R)] * g(phi(c["F"], geo))
        arms[kind] = (x - h, x + h, 2 * (h / LN10) ** 2)
    if PROB:
        mu, b = torch.from_numpy(c["mu"]), torch.from_numpy(c["b"])
        lam = LAM[R]
        lo = U.mix_quantile(mu, b, ALPHA / 2, lam).numpy()
        hi = U.mix_quantile(mu, b, 1 - ALPHA / 2, lam).numpy()
        s2 = c["mu"].var(0) + (2 * (lam * c["b"]) ** 2).mean(0)
        arms["learned"] = (lo, hi, s2)
    for arm, (lo, hi, s2) in arms.items():
        lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
        w = hi - lo
        cv = (y >= lo) & (y <= hi)
        isc = w + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
        s2 = np.asarray(s2, np.float64)
        au = U.ause(err, s2)[0]
        rows.append(dict(base, arm=arm, cov90=cv.mean(), width90=w.mean(), is90=isc.mean(), mae=ae.mean(),
                         ause_norm=au / ae.mean(), uce_e3=1e3 * uce(e2, s2), ence=ence(e2, s2)))
        for b0, b1 in BINS:
            sel = (c["gap"] >= b0) & (c["gap"] <= b1)
            if sel.sum() >= 100:
                brow.append(dict(base, arm=arm, gap_bin=f"{b0}-{b1}", cov90=cv[sel].mean()))
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case.csv"), index=False)
pd.DataFrame(brow).to_csv(os.path.join(OUT, "gapbin_case.csv"), index=False)
MET = ["cov90", "width90", "is90", "mae", "ause_norm", "uce_e3", "ence"]
pp = d.groupby(["arm", "R", "pattern", "patient"], as_index=False)[MET].mean()
rng = np.random.RandomState(0)
out = []
for (arm, R, pat), g in pp.groupby(["arm", "R", "pattern"]):
    r = dict(dataset=a.dataset, seed=a.seed, method=a.method, arm=arm, R=R, pattern=pat, n=len(g))
    for m in MET:
        v = g[m].values
        bs = v[rng.randint(0, len(v), (5000, len(v)))].mean(1)
        r[m], r[m + "_lo"], r[m + "_hi"] = v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    out.append(r)
pd.DataFrame(out).to_csv(os.path.join(OUT, "summary.csv"), index=False)
print(f"[{a.dataset} s{a.seed} {a.method}] done ({time.time() - t0:.0f}s)", flush=True)
