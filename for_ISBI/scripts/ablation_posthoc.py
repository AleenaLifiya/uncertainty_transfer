"""Post-hoc feature ablations (added after the main analysis; exploratory, NOT part of the frozen primary endpoint).

Same data, GLM, fitting (FIT, uniform), calibration (CAL, uniform) and TEST metrics as scripts/analyze.py; only the
feature set changes. Main-effects design for every variant so they differ only in which features enter.
  full        mu, log grad, log |mu-lin|, log |y_l-y_r|, log near, log gap, s_mid
  no_geom     mu, log grad, log |mu-lin|, log |y_l-y_r|           (no nearest-distance / gap / position)
  image_only  mu, log grad, log |mu-lin|                          (no acquired-neighbour difference either)
  hybrid      full + log learned sigma (P means only)
  sigma_geom  log learned sigma + log near, log gap, s_mid        (P means only)
Outputs: results/ablation/{case_metrics,summary,primary}.csv
"""
import glob
import os

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import uq as U

EV = os.path.join(C.ISBI, "runs", "eval")
OUT = os.path.join(C.ISBI, "results", "ablation")
os.makedirs(OUT, exist_ok=True)
EPS = U.EPS
MEANS = ["SAINT_P", "I3Net_P", "linear"]
VARIANTS = {"full": ["mu", "grad", "dlin", "dacq", "near", "gap", "s"], "no_geom": ["mu", "grad", "dlin", "dacq"],
            "image_only": ["mu", "grad", "dlin"], "hybrid": ["mu", "grad", "dlin", "dacq", "near", "gap", "s", "sig"],
            "sigma_geom": ["sig", "near", "gap", "s"]}


def phi(c, m, cols):
    F = c[f"{m}__feat"]
    base = dict(mu=F[:, 0], grad=np.log(F[:, 1] + EPS), dlin=np.log(F[:, 2] + EPS), dacq=np.log(F[:, 3] + EPS),
                near=np.log(F[:, 4]), gap=np.log(F[:, 5]), s=F[:, 6])
    if "sig" in cols:
        base["sig"] = 0.5 * np.log(c[f"{m}__epi"] + c[f"{m}__alea"])
    return np.stack([base[k] for k in cols], 1).astype(np.float64)


class GLM(U.LaplaceGLM):
    """LaplaceGLM on a precomputed design matrix."""

    def fit_X(self, X, e):
        self.m, self.s = X.mean(0), X.std(0) + 1e-8
        Xt = torch.from_numpy((X - self.m) / self.s)
        a = torch.from_numpy(np.abs(e).astype(np.float64))
        th = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        c0 = torch.tensor([float(np.log(np.mean(np.abs(e)) + 1e-6))], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th, c0], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lb = c0 + Xt @ th
            loss = (lb + a * torch.exp(-lb)).mean() + self.ridge * (th ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        self.th, self.c = th.detach().numpy(), float(c0.detach())
        return self

    def predict_X(self, X):
        return np.exp(self.c + ((X - self.m) / self.s) @ self.th).astype(np.float32)


def load(role):
    return [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(EV, role, "cases", "*.npz")))]


fit, cal, test = load("FIT"), load("CAL"), load("TEST")
print(len(fit), len(cal), len(test), flush=True)
rows = []
for m in MEANS:
    for v, cols in VARIANTS.items():
        if "sig" in cols and not m.endswith("_P"):
            continue
        g = GLM(ridge=1e-4).fit_X(np.concatenate([phi(c, m, cols) for c in fit]),
                                  np.concatenate([c["y"] - c[f"{m}__x"] for c in fit]))
        fit_nll = U.laplace_nll(np.concatenate([c["y"] - c[f"{m}__x"] for c in fit]),
                                g.predict_X(np.concatenate([phi(c, m, cols) for c in fit])))
        for R in C.STRIDES:
            cs = [c for c in cal if int(c["R"]) == R]
            pre = [(np.abs(c["y"] - c[f"{m}__x"]), g.predict_X(phi(c, m, cols))) for c in cs]
            kap = U.calibrate_scalar(lambda p, lam: (p[0] <= lam * p[1]).mean(), pre, 0.9, 1e-2, 1e2)
            for c in [c for c in test if int(c["R"]) == R]:
                e = c["y"] - c[f"{m}__x"]
                b = g.predict_X(phi(c, m, cols))
                h = kap * b
                x = c[f"{m}__x"]
                rows.append(dict(patient=c["patient"].item(), pattern=c["pattern"].item(), R=R, rep=int(c["rep"]),
                                 mean=m, variant=v, kappa90=kap, fit_nll=fit_nll,
                                 cov90=float((np.abs(e) <= h).mean()),
                                 width90=float((np.clip(x + h, 0, 1) - np.clip(x - h, 0, 1)).mean()),
                                 spearman=U.spearman(np.abs(e), b),
                                 strat_spearman=U.strat_spearman(e, b, c["gap_mm"], c["near_mm"]),
                                 nll=U.laplace_nll(e, b)))
        print(m, v, "done", flush=True)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case_metrics.csv"), index=False)
pp = d.groupby(["mean", "variant", "R", "pattern", "patient"], as_index=False)[
    ["cov90", "width90", "spearman", "strat_spearman", "nll"]].mean()
summ = pp.groupby(["mean", "variant", "R", "pattern"])[["cov90", "width90", "spearman", "strat_spearman", "nll"]].mean()
summ.to_csv(os.path.join(OUT, "summary.csv"))
# coverage drop uniform -> clustered per patient, with bootstrap CI
rng = np.random.RandomState(0)
prim = []
for (m, v, R), g in pp.groupby(["mean", "variant", "R"]):
    w = g.pivot(index="patient", columns="pattern", values="cov90")
    wd = g.pivot(index="patient", columns="pattern", values="width90")
    dd = (w["clustered"] - w["uniform"]).values
    bs = dd[rng.randint(0, len(dd), (10000, len(dd)))].mean(1)
    prim.append(dict(mean=m, variant=v, R=R, n=len(dd), dcov90=dd.mean(), ci_lo=np.percentile(bs, 2.5),
                     ci_hi=np.percentile(bs, 97.5), width_ratio=(wd["clustered"] / wd["uniform"]).mean(),
                     cov90_uniform=w["uniform"].mean(), cov90_clustered=w["clustered"].mean()))
prim = pd.DataFrame(prim)
prim.to_csv(os.path.join(OUT, "primary.csv"), index=False)
pd.set_option("display.width", 220)
print(prim.round(3).to_string())
print(summ.round(3).to_string())
