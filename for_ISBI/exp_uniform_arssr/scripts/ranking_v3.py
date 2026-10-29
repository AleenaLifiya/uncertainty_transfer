"""Error ranking beyond geometry for the matched setup (CPU, saved outputs only).
Within-(gap, nearest-distance)-stratum Spearman between |error| and the uncertainty score (uq.strat_spearman, as in
scripts/analyze.py), x4 and x8, per pattern, patient = unit. Scores: learned V_pred = V_epi + V_alea (SAINT-P, ArSSR-P-U),
post-hoc scale with geometry ('ph_geo', main-effects GLM fitted on FIT) and without ('ph_nogeo'); nearest distance as
the geometry-only reference (scores 0 by construction). Writes exp_uniform_arssr/results/ranking/{case,summary}.csv.
"""
import glob
import os

import numpy as np
import pandas as pd
import torch

from isbi import common as C
from isbi import uq as U

torch.set_num_threads(8)
EV = os.environ.get("ISBI_EV", os.path.join(C.ISBI, "runs", "eval"))
NATU = os.environ.get("ISBI_NAT_UNIFORM", os.path.join(C.ISBI, "exp_uniform_arssr", "runs", "eval_native", "arssr"))
OUT = os.environ.get("ISBI_RANK_OUT", os.path.join(C.ISBI, "exp_uniform_arssr", "results", "ranking"))
os.makedirs(OUT, exist_ok=True)
EPS = U.EPS


def phi_me(F, geo):
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


def load(role, k, m):
    z = np.load(os.path.join(EV, role, "cases", k))
    c = dict(patient=z["patient"].item(), pattern=z["pattern"].item(), R=int(z["R"]), y=z["y"], gap=z["gap_mm"], near=z["near_mm"])
    if m == "ArSSR_uniform":
        n = np.load(os.path.join(NATU, role, "cases", k))
        c.update(x=n["NAT__x"], F=n["NAT__feat"], vp=n["NAT__epi"] + n["NAT__alea"] if role == "TEST" else None)
    else:
        c.update(x=z[f"{m}__x"], F=z[f"{m}__feat"],
                 vp=(z[f"{m}__epi"] + z[f"{m}__alea"]) if (role == "TEST" and m == "SAINT_P") else None)
    return c


keys = lambda role: sorted(os.path.basename(f) for f in glob.glob(os.path.join(EV, role, "cases", "*.npz")))  # noqa: E731
rows = []
for m in os.environ.get("ISBI_RANK_METHODS", "SAINT_P,ArSSR_uniform,linear").split(","):
    fit = [load("FIT", k, m) for k in keys("FIT")]
    F = np.concatenate([c["F"] for c in fit])
    e = np.concatenate([c["y"] - c["x"] for c in fit])
    G = {kind: (fit_glm(phi_me(F, geo), e), geo) for kind, geo in (("ph_geo", True), ("ph_nogeo", False))}
    for k in keys("TEST"):
        c = load("TEST", k, m)
        err = c["y"] - c["x"]
        scores = {kind: g(phi_me(c["F"], geo)) for kind, (g, geo) in G.items()}
        if c["vp"] is not None:
            scores["learned"] = c["vp"]
        scores["near"] = c["near"] + 1e-6 * np.random.RandomState(0).rand(len(err))
        for sname, sc in scores.items():
            rows.append(dict(method=m, score=sname, patient=c["patient"], pattern=c["pattern"], R=c["R"],
                             strat_spearman=U.strat_spearman(err, sc, c["gap"], c["near"]),
                             spearman=U.spearman(np.abs(err), sc)))
    print(m, "done", flush=True)
d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT, "case.csv"), index=False)
pp = d.groupby(["method", "score", "R", "pattern", "patient"], as_index=False)[["strat_spearman", "spearman"]].mean()
s = pp.groupby(["method", "score", "R", "pattern"])[["strat_spearman", "spearman"]].mean().reset_index()
s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
pd.set_option("display.width", 200)
print(s.pivot_table(index=["method", "score", "R"], columns="pattern", values="strat_spearman").round(3).to_string())
